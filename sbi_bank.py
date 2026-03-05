"""
SBI (State Bank of India) Statement Extractor
================================================
Handles SBI "Account Statement" format.
Aligned with HDFC/ICICI/IndusInd: returns AccountInfo and transactions.

Statement layout:
  - Header: Account Number, Name, Branch, IFS Code, Balance as on <date>
  - Table columns: Txn Date | Value Date | Description | Ref No/Cheque No | Branch Code | Debit | Credit | Balance

Notes:
  - Statement period: "Account Statement from DD Mon YYYY to DD Mon YYYY"
  - Dates: DD/MM/YYYY
  - Descriptions can wrap across multiple lines
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
# BankConfig / AccountInfo  (same structure as HDFC/ICICI/IndusInd)
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


SBI_CONFIG = BankConfig(
    name="SBI",
    date_pattern=r"\b(\d{2}/\d{2}/\d{4}|\d{1,2}\s+\w+\s+\d{4})\b",
    opening_balance_pattern=r"Balance\s+as\s+on.*?[:\-]\s*([\d,]+\.?\d*)",
    min_columns=6,
    date_format="%d/%m/%Y",
    ifsc_pattern=r"\b(SBIN\d{7})\b",
    micr_pattern=r"\bMICR\s*[:\-]?\s*(\d{9})\b",
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


# SBI transaction columns (matches PDF, aligned with ICICI/IndusInd)
SBI_TXN_COLUMNS = [
    "row_id", "sno", "tran_id", "transaction_date", "value_date", "posted_time",
    "narration", "debit", "credit", "balance", "txn_type", "reference",
]


def _normalize_df_with_rowid(df: pd.DataFrame) -> pd.DataFrame:
    """Add row_id and ensure standard columns."""
    for col in SBI_TXN_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[[c for c in SBI_TXN_COLUMNS if c != "row_id"]].copy()
    df["row_id"] = range(1, len(df) + 1)
    return df[SBI_TXN_COLUMNS].reset_index(drop=True)


class SBIBankExtractor(BaseBankExtractor):

    BANK_NAME = "SBI"
    CONFIG = SBI_CONFIG

    # ── detection ───────────────────────────────────────────────

    def detect(self, first_page_text: str) -> bool:
        indicators = [
            r'State\s+Bank\s+of\s+India',
            r'\bSBI\b',
            r'SBIN\d{7}',            # SBI IFSC: SBIN0001131
            r'onlinesbi\.com',
            r'sbi\.co\.in',
            r'Book\s+Balance.*Available\s+Balance',
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
            "config_used": SBI_CONFIG,
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
                "config_used": SBI_CONFIG,
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
        text = re.sub(r'\(cid:\d+\)', ' ', text)  # strip PDF encoding artifacts

        m = re.search(r'Account\s+Number\s*[:\-]?\s*(\d{10,20})', text, re.I)
        if m:
            info.account_number = m.group(1).strip()

        m = re.search(r'Name\s*[:\-]?\s*(.+?)(?:\n|Currency|Branch)', text, re.I)
        if m:
            name = " ".join(m.group(1).split()).strip()
            if name and len(name) > 2 and name.lower() != 'null':
                info.customer_name = name

        m = re.search(r'Branch\s*[:\-]?\s*(.+?)(?:\n|Rate of Interest)', text, re.I)
        if m:
            info.branch_name = m.group(1).strip()

        m = re.search(r'Account\s+Statement\s+from\s+(\d{1,2}\s+\w+\s+\d{4})\s+to\s+(\d{1,2}\s+\w+\s+\d{4})', text, re.I)
        if m:
            info.statement_from = parse_date(m.group(1), ['%d %b %Y', '%d %B %Y']) or m.group(1)
            info.statement_to = parse_date(m.group(2), ['%d %b %Y', '%d %B %Y']) or m.group(2)

        m = re.search(r'Balance\s+as\s+on\s+\d{1,2}\s+\w+\s+\d{4}[\s:\-]+([\d,]+\.?\d*)', text, re.I)
        if m:
            info.opening_balance = clean_amount(m.group(1)) or 0.0

        m = re.search(r'Available\s+Balance\s*[:\-]?\s*([\d,]+\.?\d*)', text, re.I)
        if m:
            info.closing_balance = clean_amount(m.group(1)) or 0.0

        m = re.search(SBI_CONFIG.ifsc_pattern, text)
        if m:
            info.ifsc_code = m.group(1)

        return info

    def _parse_account_number(self, text: str) -> str:
        patterns = [
            r'Account\s+Number\s*[:\-]?\s*(\d{10,20})',
            r'(?<!\d)(\d{16,17})(?!\d)',   # SBI account numbers typically 17 digits
            r'(\d{11,17})',
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return m.group(1).strip()
        return ''

    def _parse_account_holder(self, text: str) -> str:
        m = re.search(r'Name\s*[:\-]?\s*(.+?)(?:\n|Currency|Branch)', text, re.I)
        if m:
            name = m.group(1).strip()
            if name and len(name) > 2:
                return name
        return ''

    def _parse_opening_balance(self, text: str) -> Optional[float]:
        # SBI shows "Balance as on DD Mon YYYY : <amount>"
        # Opening balance = first balance shown in table or header
        patterns = [
            r'Opening\s+Bal(?:ance)?\s*[:\-]?\s*([\d,]+\.?\d*)',
            r'Balance\s+as\s+on.*?[:\-]\s*([\d,]+\.?\d*)',
            r'Book\s+Balance\s*[:\-]?\s*([\d,]+\.?\d*)',
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return clean_amount(m.group(1))
        return None

    def _parse_closing_balance(self, text: str) -> Optional[float]:
        patterns = [
            r'Closing\s+Bal(?:ance)?\s*[:\-]?\s*([\d,]+\.?\d*)',
            r'Available\s+Balance\s*[:\-]?\s*([\d,]+\.?\d*)',
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return clean_amount(m.group(1))
        return None

    def _parse_period(self, text: str):
        # "Account Statement from 20 Jul 2025 to 3 Sep 2025"
        patterns = [
            r'Account\s+Statement\s+from\s+(\d{1,2}\s+\w+\s+\d{4})'
            r'\s+to\s+(\d{1,2}\s+\w+\s+\d{4})',
            r'From\s+(\d{2}/\d{2}/\d{4})\s+To\s+(\d{2}/\d{2}/\d{4})',
        ]
        date_fmts = ['%d %b %Y', '%d/%m/%Y', '%d %B %Y']
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return (parse_date(m.group(1), date_fmts),
                        parse_date(m.group(2), date_fmts))
        return None, None

    # ── transaction extraction ──────────────────────────────────

    def _extract_transactions(self, pdf_path: str, raw_text: str) -> pd.DataFrame:
        df = self._extract_via_pdfplumber(pdf_path)
        if df is not None and len(df) > 0:
            return df
        self._log("pdfplumber failed — falling back to regex")
        return self._extract_via_regex(raw_text)

    # ── Strategy 1: pdfplumber ───────────────────────────────────

    def _extract_via_pdfplumber(self, pdf_path: str) -> Optional[pd.DataFrame]:
        all_rows = []
        saved_col_indices = None  # reuse header from page 1 for continuation pages

        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 5,
            "join_tolerance": 5,
        }

        COL_MAP = {
            'txn date': 'transaction_date',
            'transaction date': 'transaction_date',
            'date': 'transaction_date',
            'value date': 'value_date',
            'description': 'narration',
            'particulars': 'narration',
            'ref no./cheque no.': 'reference',
            'ref no/cheque no': 'reference',
            'ref no.': 'reference',
            'cheque no': 'reference',
            'ref no': 'reference',
            'branch code': 'branch',
            'debit': 'debit',
            'credit': 'credit',
            'balance': 'balance',
        }

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables(table_settings)
                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    header = [str(c or '').lower().strip().replace('\n', ' ')
                              for c in table[0]]

                    if self._is_txn_header(header):
                        col_indices = {}
                        for idx, h in enumerate(header):
                            mapped = COL_MAP.get(h)
                            if mapped and mapped not in col_indices:
                                col_indices[mapped] = idx
                        if 'transaction_date' in col_indices or 'value_date' in col_indices:
                            if 'transaction_date' not in col_indices:
                                col_indices['transaction_date'] = col_indices.get('value_date')
                            saved_col_indices = col_indices

                    if saved_col_indices is None:
                        continue

                    start = 1 if self._is_txn_header(header) else 0
                    for row in table[start:]:
                        parsed = self._parse_row(row, saved_col_indices)
                        if parsed:
                            all_rows.append(parsed)

        if not all_rows:
            return None
        return self._finalize_df(pd.DataFrame(all_rows))

    def _is_txn_header(self, header: List[str]) -> bool:
        has_date = any('date' in h or 'txn' in h for h in header)
        has_amount = any(w in h for h in header
                         for w in ['debit', 'credit', 'balance'])
        return has_date and has_amount

    def _parse_row(self, row: list, col_indices: dict) -> Optional[Dict]:
        def get(key):
            idx = col_indices.get(key)
            if idx is not None and idx < len(row):
                v = row[idx]
                return str(v).strip() if v is not None else ''
            return ''

        def clean_cell(s: str) -> str:
            """Merge multi-line text into single line."""
            return " ".join(str(s).replace('\n', ' ').split()) if s else ""

        txn_date_raw = get('transaction_date')
        if not re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', txn_date_raw):
            return None

        debit = clean_amount(get('debit'))
        credit = clean_amount(get('credit'))
        txn_type = 'DR' if debit else ('CR' if credit else '')
        ref = clean_cell(get('reference'))

        return {
            'sno': '',
            'tran_id': ref,
            'transaction_date': parse_date(txn_date_raw),
            'value_date': parse_date(get('value_date')),
            'posted_time': '',  # SBI table typically has no time
            'narration': clean_cell(get('narration')),
            'debit': debit,
            'credit': credit,
            'balance': clean_amount(get('balance')),
            'txn_type': txn_type,
            'reference': ref,
        }

    # ── Strategy 2: regex ────────────────────────────────────────

    def _extract_via_regex(self, raw_text: str) -> pd.DataFrame:
        """
        SBI raw text: date date description branch [debit] [credit] balance
        Narrations can wrap to next line(s).
        """
        rows = []
        DATE_PAT = r'(\d{2}/\d{2}/\d{4})'
        AMT_PAT = r'([\d,]+\.\d{2})'

        line_re = re.compile(
            rf'^{DATE_PAT}\s+{DATE_PAT}\s+'
            rf'(.+?)\s+'
            rf'(\d{4,6})\s+'             # branch code
            rf'(?:{AMT_PAT}\s+)?(?:{AMT_PAT}\s+)?{AMT_PAT}\s*$',
            re.MULTILINE
        )

        txn_start_re = re.compile(r'^\d{2}/\d{2}/\d{4}\s+\d{2}/\d{2}/\d{4}\s+', re.MULTILINE)

        def get_continuation(start: int, end: int) -> str:
            block = raw_text[start:end]
            lines = []
            for line in block.split('\n'):
                s = line.strip()
                if not s or re.match(r'^--\s*\d+\s+of\s+\d+\s*--', s):
                    continue
                if txn_start_re.match(s):
                    break
                lines.append(s)
            return ' '.join(lines) if lines else ''

        matches = list(line_re.finditer(raw_text))
        for i, m in enumerate(matches):
            txn_date, val_date = m.group(1), m.group(2)
            narration = " ".join(m.group(3).split())
            branch = m.group(4)
            g5, g6, g7 = m.group(5), m.group(6), m.group(7)
            debit = clean_amount(g5) if g5 else None
            credit = clean_amount(g6) if g6 else None
            balance = clean_amount(g7)
            txn_type = 'DR' if debit else ('CR' if credit else '')
            next_start = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            cont = get_continuation(m.end(), next_start)
            if cont:
                narration = (narration + ' ' + cont).strip()

            rows.append({
                'sno': '',
                'tran_id': branch,
                'transaction_date': parse_date(txn_date),
                'value_date': parse_date(val_date),
                'posted_time': '',
                'narration': narration,
                'debit': debit,
                'credit': credit,
                'balance': balance,
                'txn_type': txn_type,
                'reference': branch,
            })

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return self._finalize_df(df)

    def _finalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return _normalize_df_with_rowid(df)
        for col in ('debit', 'credit', 'balance'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['transaction_date'])
        if 'transaction_date' in df.columns:
            df = df[df['transaction_date'].astype(str).str.match(r'\d{4}-\d{2}-\d{2}', na=False)]
        return _normalize_df_with_rowid(df)