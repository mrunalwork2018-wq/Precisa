"""
ICICI Bank Statement Extractor
================================
Handles ICICI Bank "Detailed Statement" format.
Aligned with HDFC extractor: returns AccountInfo and transactions.

Statement layout:
  - Header: Name, Address, A/C No, A/C Type, Branch, IFSC, Cust ID
  - Transaction Period, Statement Request Date
  - Table: Sl No | Tran Id | Value Date | Transaction Date | Transaction Posted |
           Cheque no/Ref No | Transaction Remarks | Withdrawal (Dr) | Deposit (Cr) | Balance
"""

import re
import pdfplumber
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_extractor import (
    BaseBankExtractor, clean_amount, parse_date, normalize_df
)


# ──────────────────────────────────────────────────────────────────────────────
# BankConfig / AccountInfo  (same structure as HDFC)
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


ICICI_CONFIG = BankConfig(
    name="ICICI Bank",
    date_pattern=r"\b(\d{2}/\d{2}/\d{2,4}|\d{2}/\w{3}/\d{4})\b",
    opening_balance_pattern=r"Opening\s+Bal(?:ance)?\s*[:\-]?\s*([\-\d,]+\.?\d*)",
    min_columns=7,
    date_format="%d/%m/%Y",
    ifsc_pattern=r"\b(ICIC\d{7})\b",
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


# ICICI transaction columns (matches PDF table)
ICICI_TXN_COLUMNS = [
    "row_id", "sno", "tran_id", "transaction_date", "value_date", "posted_time",
    "narration", "debit", "credit", "balance", "txn_type", "reference",
]


def _normalize_df_with_rowid(df: pd.DataFrame) -> pd.DataFrame:
    """Add row_id and ensure standard columns (incl. tran_id, posted_time from PDF)."""
    for col in ICICI_TXN_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[[c for c in ICICI_TXN_COLUMNS if c != "row_id"]].copy()
    df["row_id"] = range(1, len(df) + 1)
    return df[ICICI_TXN_COLUMNS].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main extractor
# ──────────────────────────────────────────────────────────────────────────────

class ICICIBankExtractor(BaseBankExtractor):

    BANK_NAME = "ICICI Bank"
    CONFIG = ICICI_CONFIG

    # ── detection ───────────────────────────────────────────────

    def detect(self, first_page_text: str) -> bool:
        indicators = [
            r'ICICI\s*BANK',
            r'ICIC\d{7}',
            r'icicibankltd',
            r'Detailed\s+Statement',
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
            "config_used": ICICI_CONFIG,
        }

    def run(self, pdf_path: str) -> Dict[str, Any]:
        """Entry point for app.py; returns HDFC-style result."""
        raw_text = self._extract_raw_text(pdf_path)
        if not raw_text.strip():
            return {
                "bank": self.BANK_NAME,
                "account_info": AccountInfo(bank_name=self.BANK_NAME),
                "transactions": pd.DataFrame(),
                "config_used": ICICI_CONFIG,
                "errors": ["No text extracted from PDF"],
            }
        result = self.extract(pdf_path, raw_text)
        result["bank"] = self.BANK_NAME
        result["errors"] = []
        return result

    # ── account info (HDFC-style parsing) ────────────────────────

    def _parse_account_info(self, text: str) -> AccountInfo:
        info = AccountInfo(bank_name=self.BANK_NAME)

        m = re.search(r'A[/\\]?C\s+No\s*[:\-]?\s*(\d{9,18})', text, re.I)
        if m:
            info.account_number = m.group(1).strip()

        m = re.search(r'Cust\s*ID\s*[:\-]?\s*(\d+)', text, re.I)
        if m:
            info.customer_id = m.group(1).strip()

        m = re.search(r'Name\s*[:\-]?\s*(.+?)(?=\s+Branch\s*[:\-]|$)', text, re.I | re.DOTALL)
        if m:
            info.customer_name = " ".join(m.group(1).split()).strip()

        m = re.search(r'A[/\\]?C\s+Type\s*[:\-]?\s*(\S+)', text, re.I)
        if m:
            info.account_type = m.group(1).strip()

        m = re.search(r'Branch\s*[:\-]?\s*([^A\n]+?)(?=\s+Address\s*[:\-]|Branch\s+Address|$)', text, re.I)
        if m:
            info.branch_name = " ".join(m.group(1).split()).strip().rstrip(',')

        m = re.search(r'Branch\s+Address\s*[:\-]?\s*(.+?)(?=Transaction\s+Date|A[/\\]?C\s+No|IFSC|$)', text, re.I | re.DOTALL)
        if m:
            info.branch_address = " ".join(m.group(1).split()).strip()

        m = re.search(r'Address\s*[:\-]\s*(.+?)(?=Branch\s+Address|Branch\s*[:\-])', text, re.I | re.DOTALL)
        if m:
            info.address = " ".join(m.group(1).split()).strip()

        m = re.search(ICICI_CONFIG.ifsc_pattern, text)
        if m:
            info.ifsc_code = m.group(1)

        m = re.search(ICICI_CONFIG.micr_pattern, text, re.I)
        if m:
            info.micr_code = m.group(1)

        m = re.search(
            r'Transaction\s+Period\s*[:\-]?\s*From\s+(\d{2}/\d{2}/\d{4})\s+To\s+(\d{2}/\d{2}/\d{4})',
            text, re.I
        )
        if m:
            info.statement_from = parse_date(m.group(1)) or m.group(1)
            info.statement_to = parse_date(m.group(2)) or m.group(2)

        m = re.search(r'Opening\s+Bal(?:ance)?\s*[:\-]?\s*([\-\d,]+\.?\d*)', text, re.I)
        if m:
            info.opening_balance = clean_amount(m.group(1)) or 0.0

        m = re.search(r'Closing\s+Bal(?:ance)?\s*[:\-]?\s*([\-\d,]+\.?\d*)', text, re.I)
        if m:
            info.closing_balance = clean_amount(m.group(1)) or 0.0

        if not info.closing_balance and info.opening_balance == 0.0:
            m = re.search(r'Page\s+Total\s+Opening\s+Bal[:\s]+([\-\d,]+\.?\d*)', text, re.I)
            if m:
                info.opening_balance = clean_amount(m.group(1)) or 0.0
            m = re.search(r'Closing\s+Bal\s*[:\s]+([\-\d,]+\.?\d*)', text, re.I)
            if m:
                info.closing_balance = clean_amount(m.group(1)) or 0.0

        return info

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

        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 5,
            "join_tolerance": 5,
        }

        COL_MAP = {
            'sl no': 'sno',
            'tran id': 'tran_id',
            'value date': 'value_date',
            'transaction date': 'transaction_date',
            'transaction posted': 'posted_time',
            'transaction posted date': 'posted_time',
            'cheque no / ref no': 'reference',
            'cheque no / transaction ref no': 'reference',
            'cheque no /transaction ref no': 'reference',
            'transaction ref no': 'reference',
            'transaction remarks': 'narration',
            'remarks': 'narration',
            'withdrawal (dr)': 'debit',
            'withdrawal(dr)': 'debit',
            'withdrawl (dr)': 'debit',
            'withdra wal (dr)': 'debit',
            'deposit (cr)': 'credit',
            'deposit(cr)': 'credit',
            'balance': 'balance',
        }

        saved_col_indices = None

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
                        if 'transaction_date' not in col_indices and 'value_date' in col_indices:
                            col_indices['transaction_date'] = col_indices['value_date']
                        if 'transaction_date' in col_indices or 'value_date' in col_indices:
                            saved_col_indices = col_indices

                    if saved_col_indices is None:
                        continue

                    start = 1 if self._is_txn_header(header) else 0
                    for row in table[start:]:
                        parsed = self._parse_row(row, saved_col_indices)
                        if parsed:
                            all_rows.append(parsed)

        if not all_rows:
            return pd.DataFrame()
        return self._finalize_df(pd.DataFrame(all_rows))

    def _is_txn_header(self, header: List[str]) -> bool:
        has_date = any('date' in h for h in header)
        has_amount = any(w in h for h in header
                         for w in ['withdrawal', 'deposit', 'balance', 'debit', 'credit'])
        return has_date and has_amount

    def _parse_row(self, row: list, col_indices: dict) -> Optional[Dict]:
        def get(key):
            idx = col_indices.get(key)
            if idx is not None and idx < len(row):
                v = row[idx]
                return str(v).strip() if v is not None else ''
            return ''

        def clean_cell(s):
            return " ".join(str(s).replace('\n', ' ').split()) if s else ""

        def fix_value_date(s):
            """Fix split dates like 01/Aug/2 025 -> 01/Aug/2025."""
            if not s:
                return s
            s = clean_cell(s)
            s = re.sub(r'(\d{1,2}/\w{3}/)2\s*0?25\b', r'\g<1>2025', s)
            return s

        txn_date_raw = get('transaction_date')
        if not re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}/\w{3}/\d{4}', txn_date_raw):
            return None

        debit = clean_amount(get('debit'))
        credit = clean_amount(get('credit'))
        txn_type = 'DR' if debit else ('CR' if credit else '')

        balance = clean_amount(clean_cell(get('balance')))

        return {
            'sno': clean_cell(get('sno')),
            'tran_id': clean_cell(get('tran_id')),
            'transaction_date': parse_date(txn_date_raw, ['%d/%b/%Y', '%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d']),
            'value_date': parse_date(fix_value_date(get('value_date')), ['%d/%b/%Y', '%d/%m/%Y', '%d-%m-%Y']),
            'posted_time': clean_cell(get('posted_time')),
            'narration': clean_cell(get('narration')),
            'debit': debit,
            'credit': credit,
            'balance': balance,
            'txn_type': txn_type,
            'reference': clean_cell(get('reference')),
        }

    # ── Strategy 2: regex ────────────────────────────────────────

    def _extract_via_regex(self, raw_text: str) -> pd.DataFrame:
        rows = []
        AMT_PAT = r'([\d,]+\.\d{2})'
        line_re = re.compile(
            r'^\s*\d+\s+'
            r'\S+\s+\d+\s+'
            r'(\d{2}/\w{3}/\d{4})\s+'
            r'(\d{2}/\d{2}/\d{4})\s+'
            r'\d{2}/\d{2}/\d{4}\s+\d+:\d+:\d+\s+[AP]M\s+'
            r'(.+?)\s+'
            r'(?:' + AMT_PAT + r'\s+)?'
            r'(?:' + AMT_PAT + r'\s+)?'
            r'' + AMT_PAT + r'\s*$',
            re.MULTILINE
        )

        for m in line_re.finditer(raw_text):
            val_date = m.group(1)
            txn_date = m.group(2)
            narration = m.group(3).strip()
            g4 = m.group(4)
            g5 = m.group(5)
            g6 = m.group(6)

            debit = clean_amount(g4) if g4 else None
            credit = clean_amount(g5) if g5 else None
            balance = clean_amount(g6)
            txn_type = 'DR' if debit else ('CR' if credit else '')

            rows.append({
                'sno': '',
                'tran_id': '',
                'transaction_date': parse_date(txn_date, ['%d/%b/%Y', '%d/%m/%Y', '%d-%m-%Y']),
                'value_date': parse_date(val_date, ['%d/%b/%Y', '%d/%m/%Y', '%d-%m-%Y']),
                'posted_time': '',
                'narration': narration,
                'debit': debit,
                'credit': credit,
                'balance': balance,
                'txn_type': txn_type,
                'reference': '',
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
