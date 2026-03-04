"""
Axis Bank Statement Extractor (NEO for Corporates)
==================================================
Handles Axis Bank NEO for Corporates format.
Aligned with HDFC/ICICI/IndusInd/SBI: returns AccountInfo and transactions.

Statement layout:
  - Header: Account name, number, IFSC, period, opening balance
  - Table columns: S.NO | Transaction Date | Value Date | Particulars |
                   Amount(INR) | Debit/Credit | Balance(INR) | Cheque Number | Branch Name
"""

import re
import pdfplumber
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_extractor import (
    BaseBankExtractor, clean_amount, parse_date, normalize_df
)


# ──────────────────────────────────────────────────────────────────────────────
# BankConfig / AccountInfo  (same structure as HDFC/ICICI/IndusInd/SBI)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BankConfig:
    name: str
    date_pattern: str
    opening_balance_pattern: str
    min_columns: int
    date_format: str
    ifsc_pattern: str
    micr_pattern: str


AXIS_CONFIG = BankConfig(
    name="Axis Bank",
    date_pattern=r"\b(\d{2}/\d{2}/\d{4})\b",
    opening_balance_pattern=r"Opening\s+Balance\s*[:\-]?\s*(?:INR)?\s*([\d,]+\.?\d*)",
    min_columns=6,
    date_format="%d/%m/%Y",
    ifsc_pattern=r"\b(UTIB\d{7})\b",
    micr_pattern=r"\bMICR\s*Code\s*[:\-]?\s*(\d{9})\b",
)


@dataclass
class AccountInfo:
    account_number: str = ""
    customer_id: str = ""
    customer_name: str = ""
    account_type: str = ""
    bank_name: str = ""
    ifsc_code: str = ""
    micr_code: str = ""
    branch_name: str = ""
    branch_address: str = ""
    email: str = ""
    phone: str = ""
    pan: str = ""
    address: str = ""
    statement_from: str = ""
    statement_to: str = ""
    txn_start_date: str = ""
    txn_end_date: str = ""
    opening_balance: float = 0.0
    closing_balance: float = 0.0
    account_age_days: int = 0


# Axis transaction columns (matches PDF, aligned with ICICI/IndusInd/SBI)
AXIS_TXN_COLUMNS = [
    "row_id", "sno", "tran_id", "transaction_date", "value_date", "posted_time",
    "narration", "debit", "credit", "balance", "txn_type", "reference",
]


def _normalize_df_with_rowid(df: pd.DataFrame) -> pd.DataFrame:
    """Add row_id and ensure standard columns."""
    for col in AXIS_TXN_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[[c for c in AXIS_TXN_COLUMNS if c != "row_id"]].copy()
    df["row_id"] = range(1, len(df) + 1)
    return df[AXIS_TXN_COLUMNS].reset_index(drop=True)


class AxisBankExtractor(BaseBankExtractor):

    BANK_NAME = "Axis Bank"
    CONFIG = AXIS_CONFIG

    # ── detection ───────────────────────────────────────────────

    def detect(self, first_page_text: str) -> bool:
        indicators = [
            r'AXIS\s*BANK',
            r'UTIB\d{7}',           # IFSC code pattern
            r'neo\s+for\s+corporates',
            r'axisbank\.com',
            r'Statement\s+of\s+Axis\s+Bank',
        ]
        return any(re.search(p, first_page_text, re.I) for p in indicators)

    # ── main extraction (HDFC-style return) ──────────────────────

    def extract(self, pdf_path: str, raw_text: str = None) -> Dict[str, Any]:
        if raw_text is None:
            raw_text = self._extract_raw_text(pdf_path)

        info = self._parse_account_info(raw_text)
        try:
            df = self._extract_transactions(pdf_path, raw_text)
        except Exception as e:
            self._log(f"Transaction extraction failed: {e}")
            df = pd.DataFrame()

        if not df.empty and "transaction_date" in df.columns:
            valid = df["transaction_date"].dropna()
            valid = valid[valid.astype(str).str.match(r'\d{4}-\d{2}-\d{2}', na=False)]
            if len(valid):
                info.txn_start_date = str(valid.min())
                info.txn_end_date = str(valid.max())

        return {
            "account_info": info,
            "transactions": df,
            "config_used": AXIS_CONFIG,
        }

    def run(self, pdf_path: str) -> Dict[str, Any]:
        """Entry point for app.py; returns HDFC-style result."""
        raw_text = self._extract_raw_text(pdf_path)
        if not raw_text.strip():
            info = AccountInfo(bank_name=self.BANK_NAME)
            return {
                "bank": self.BANK_NAME,
                "account_info": info,
                "transactions": pd.DataFrame(),
                "config_used": AXIS_CONFIG,
                "errors": ["No text extracted from PDF"],
                "account_number": info.account_number,
                "account_holder": info.customer_name,
                "opening_balance": info.opening_balance,
                "closing_balance": info.closing_balance,
                "statement_from": info.statement_from,
                "statement_to": info.statement_to,
                "currency": "INR",
            }
        result = self.extract(pdf_path, raw_text)
        result["bank"] = self.BANK_NAME
        result["errors"] = []
        info = result["account_info"]
        result["account_number"] = info.account_number
        result["account_holder"] = info.customer_name
        result["opening_balance"] = info.opening_balance
        result["closing_balance"] = info.closing_balance
        result["statement_from"] = info.statement_from
        result["statement_to"] = info.statement_to
        result["currency"] = "INR"
        return result

    # ── account info (HDFC-style parsing) ────────────────────────

    def _parse_account_info(self, text: str) -> AccountInfo:
        info = AccountInfo(bank_name=self.BANK_NAME)

        m = re.search(r'Account\s+No\s*[:\-]?\s*(\d{10,20})|Axis\s+Bank\s+Account\s+No\s*[:\-]?\s*(\d+)', text, re.I)
        if m:
            info.account_number = (m.group(1) or m.group(2) or '').strip()

        lines = text.strip().split('\n')
        for line in lines[:12]:
            line = line.strip()
            if len(line) > 5 and not any(skip in line.upper() for skip in
                   ['ACCOUNT', 'STATEMENT', 'DATE', 'BRANCH', 'IFSC', 'JOINT', 'SCHEME', 'CUSTOMER']):
                if re.match(r'^[A-Z][A-Z\s]+$', line[:50]) or 'PRIVATE LIMITED' in line.upper():
                    info.customer_name = line
                    break

        m = re.search(r'[Ff]rom\s*[:\-]?\s*(\d{2}/\d{2}/\d{4}).*?[Tt]o\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})', text, re.DOTALL)
        if m:
            info.statement_from = parse_date(m.group(1)) or m.group(1)
            info.statement_to = parse_date(m.group(2)) or m.group(2)

        m = re.search(r'Opening\s+Balance\s*[:\-]?\s*(?:INR)?\s*([\d,]+\.?\d*)', text, re.I)
        if m:
            info.opening_balance = clean_amount(m.group(1)) or 0.0

        m = re.search(r'Closing\s+Balance\s*[:\-]?\s*(?:INR)?\s*([\d,]+\.?\d*)', text, re.I)
        if m:
            info.closing_balance = clean_amount(m.group(1)) or 0.0

        m = re.search(AXIS_CONFIG.ifsc_pattern, text)
        if m:
            info.ifsc_code = m.group(1)

        m = re.search(AXIS_CONFIG.micr_pattern, text, re.I)
        if m:
            info.micr_code = m.group(1)

        return info

    def _parse_account_number(self, text: str) -> str:
        patterns = [
            r'Account\s+No\s*[:\-]?\s*(\d{10,20})',
            r'Axis\s+Bank\s+Account\s+No\s*[:\-]?\s*(\d+)',
            r'(\d{18})',
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return m.group(1).strip()
        return ''

    def _parse_account_holder(self, text: str) -> str:
        # First bold-like line at top is usually the company name
        lines = text.strip().split('\n')
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 5 and line.isupper() and 'AXIS' not in line.upper():
                # Skip lines that are purely header labels
                if not any(skip in line.upper() for skip in
                           ['ACCOUNT', 'STATEMENT', 'DATE', 'BRANCH', 'IFSC']):
                    return line
        return ''

    def _parse_opening_balance(self, text: str) -> Optional[float]:
        patterns = [
            r'Opening\s+Balance\s*[:\-]?\s*(?:INR)?\s*([\d,]+\.?\d*)',
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return clean_amount(m.group(1))
        return None

    def _parse_closing_balance(self, text: str) -> Optional[float]:
        patterns = [
            r'Closing\s+Balance\s*[:\-]?\s*(?:INR)?\s*([\d,]+\.?\d*)',
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return clean_amount(m.group(1))
        return None

    def _parse_period(self, text: str):
        m = re.search(
            r'[Ff]rom\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})'
            r'.*?[Tt]o\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})',
            text, re.DOTALL
        )
        if m:
            return parse_date(m.group(1)), parse_date(m.group(2))
        return None, None

    # ── transaction extraction ──────────────────────────────────

    def _extract_transactions(self, pdf_path: str, raw_text: str) -> pd.DataFrame:
        """
        Strategy:
        1. Try pdfplumber table extraction (best for well-structured PDFs)
        2. Fall back to regex line-by-line parsing of raw text
        """
        df = self._extract_via_pdfplumber(pdf_path)
        if df is not None and len(df) > 0:
            return df

        self._log("pdfplumber table extraction yielded no rows — falling back to regex parser")
        return self._extract_via_regex(raw_text)

    # ── Strategy 1: pdfplumber table ────────────────────────────

    def _extract_via_pdfplumber(self, pdf_path: str) -> Optional[pd.DataFrame]:
        """Axis Bank statements have clear ruled-line tables."""
        all_rows: List[Dict] = []
        saved_col_indices = None  # reuse header from page 1 for continuation pages

        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 4,
            "join_tolerance": 4,
            "edge_min_length": 10,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        }

        COL_MAP = {
            'transaction date (dd/mm/yyyy)': 'transaction_date',
            'transaction date': 'transaction_date',
            'value date (dd/mm/yyyy)': 'value_date',
            'value date': 'value_date',
            'particulars': 'narration',
            'amount(inr)': 'amount_raw',
            'amount (inr)': 'amount_raw',
            'debit/credit': 'txn_type',
            'balance(inr)': 'balance',
            'balance (inr)': 'balance',
            'cheque number': 'reference',
            'branch name(sol)': 'branch',
            'branch name': 'branch',
            's.no': 'sno',
        }

        def norm(h):
            return str(h or '').lower().strip().replace('\n', ' ')

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables(table_settings)
                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    header = [norm(c) for c in table[0]]

                    if self._is_txn_header(header):
                        col_indices = {}
                        for idx, h in enumerate(header):
                            for k, v in COL_MAP.items():
                                if k in h or (h and h in k) and v not in col_indices:
                                    col_indices[v] = idx
                                    break
                        if 'transaction_date' not in col_indices and 'value_date' in col_indices:
                            col_indices['transaction_date'] = col_indices['value_date']
                        if 'transaction_date' in col_indices or 'value_date' in col_indices:
                            saved_col_indices = col_indices

                    if saved_col_indices is None:
                        continue

                    start = 1 if self._is_txn_header(header) else 0
                    for row in table[start:]:
                        if not row:
                            continue
                        parsed = self._parse_row(row, saved_col_indices, header)
                        if parsed:
                            all_rows.append(parsed)

        if not all_rows:
            return None
        return self._finalize_df(pd.DataFrame(all_rows))

    def _is_txn_header(self, header: List[str]) -> bool:
        hset = set(str(h or '').lower().replace('\n', ' ') for h in header)
        required = {'transaction date (dd/mm/yyyy)', 'transaction date',
                    'value date (dd/mm/yyyy)', 'value date'}
        return bool(required.intersection(hset)) or any('transaction' in h and 'date' in h for h in hset)

    def _parse_row(self, row: list, col_indices: dict, header: list) -> Optional[Dict]:
        def get(key):
            idx = col_indices.get(key)
            if idx is not None and idx < len(row):
                val = row[idx]
                return str(val).strip() if val is not None else ''
            return ''

        def clean_cell(s: str) -> str:
            return " ".join(str(s).replace('\n', ' ').split()) if s else ""

        txn_date_raw = get('transaction_date')
        if not txn_date_raw or not re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', txn_date_raw):
            return None

        amount_raw = get('amount_raw')
        txn_type_raw = get('txn_type').upper().strip()
        narration = clean_cell(get('narration'))

        if narration.lower() in ('particulars', 'narration') and not amount_raw:
            return None

        debit, credit = self._parse_cr_dr_amount(amount_raw, txn_type_raw)
        txn_type = txn_type_raw if txn_type_raw in ('CR', 'DR') else ('DR' if debit else 'CR')
        ref = clean_cell(get('reference'))
        sno = clean_cell(get('sno'))

        return {
            'sno': sno,
            'tran_id': ref,
            'transaction_date': parse_date(txn_date_raw),
            'value_date': parse_date(get('value_date')),
            'posted_time': '',  # Axis NEO typically has no time
            'narration': narration,
            'debit': debit,
            'credit': credit,
            'balance': clean_amount(get('balance')),
            'txn_type': txn_type,
            'reference': ref,
        }

    # ── Strategy 2: regex line parser ───────────────────────────

    def _extract_via_regex(self, raw_text: str) -> pd.DataFrame:
        """
        Parse each line looking for the pattern:
        <sno> <dd/mm/yyyy> <dd/mm/yyyy> <narration...> <amount> <CR/DR> <balance> [cheque] [branch]
        
        Axis Bank raw text example line:
        1 01/06/2025 01/06/2025 MIGS PAYMENT AURICYBSRUP0212 DT 31-MAY-25 22,986.99 CR 35,291.84 POWAI, MUMBAI [MH] (100)
        """
        rows = []

        # Regex: captures date, date, narration (greedy), amount, CR/DR, balance
        # Axis amounts use Indian comma format: 1,23,456.78
        AMOUNT_PAT = r'([\d,]+\.\d{2})'
        DATE_PAT = r'(\d{2}/\d{2}/\d{4})'
        TYPE_PAT = r'(CR|DR)'

        # Full line pattern
        line_re = re.compile(
            rf'^\s*\d+\s+'                     # S.NO
            rf'{DATE_PAT}\s+'                  # Transaction Date
            rf'{DATE_PAT}\s+'                  # Value Date
            rf'(.+?)\s+'                        # Narration (non-greedy)
            rf'{AMOUNT_PAT}\s+'                # Amount
            rf'{TYPE_PAT}\s+'                  # CR/DR
            rf'{AMOUNT_PAT}'                   # Balance
            , re.MULTILINE
        )

        # Process multi-line narrations: join continuation lines
        text = self._merge_continuation_lines(raw_text)

        for m in line_re.finditer(text):
            txn_date, val_date, narration, amount, txn_type, balance = (
                m.group(1), m.group(2), m.group(3).strip(),
                m.group(4), m.group(5), m.group(6)
            )

            debit, credit = self._parse_cr_dr_amount(amount, txn_type)

            rows.append({
                'sno': str(len(rows) + 1),
                'tran_id': '',
                'transaction_date': parse_date(txn_date),
                'value_date': parse_date(val_date),
                'posted_time': '',
                'narration': narration,
                'debit': debit,
                'credit': credit,
                'balance': clean_amount(balance),
                'txn_type': txn_type,
                'reference': '',
            })

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return self._finalize_df(df)

    def _merge_continuation_lines(self, text: str) -> str:
        """
        Axis Bank wraps long narrations across lines.
        Merge lines that don't start with a serial number + date pattern.
        """
        lines = text.split('\n')
        merged = []
        buffer = ''
        start_pat = re.compile(r'^\s*\d+\s+\d{2}/\d{2}/\d{4}')

        for line in lines:
            if start_pat.match(line):
                if buffer:
                    merged.append(buffer)
                buffer = line
            else:
                if buffer:
                    buffer = buffer.rstrip() + ' ' + line.strip()
                # else: skip pre-table header lines

        if buffer:
            merged.append(buffer)

        return '\n'.join(merged)

    def _finalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return _normalize_df_with_rowid(df)

        for col in ('debit', 'credit', 'balance'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['transaction_date'], how='all')
        if 'transaction_date' in df.columns:
            df = df[df['transaction_date'].astype(str).str.match(r'\d{4}-\d{2}-\d{2}', na=False)]

        return _normalize_df_with_rowid(df)
