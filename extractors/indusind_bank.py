"""
IndusInd Bank Statement Extractor
====================================
Handles IndusInd Bank "Account Statement" format.
Aligned with HDFC/ICICI: returns AccountInfo and transactions.

Statement layout:
  - Header: Customer Name, Account No, From Date, To Date
  - Table columns:
      Bank Reference | Value Date | Transaction Date & Time | Type |
      Payment Narration | Debit | Credit | Available Balance

Notes:
  - Type column: "Debit" or "Credit" string
  - Balance can be negative (shown as -12,34,567.89)
  - Transaction Date includes time (e.g. '01-APR-25 06:59:44')
  - Value Date format: '01 Apr 2025'
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
# BankConfig / AccountInfo  (same structure as HDFC/ICICI)
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


INDUSIND_CONFIG = BankConfig(
    name="IndusInd Bank",
    date_pattern=r"\b(\d{2}[-/]\w{3}[-/]\d{2,4}|\d{2}\s+\w{3}\s+\d{4})\b",
    opening_balance_pattern=r"Opening\s+Bal(?:ance)?\s*[:\-]?\s*([\-\d,]+\.?\d*)",
    min_columns=7,
    date_format="%d-%b-%y",
    ifsc_pattern=r"\b(INDB\d{7})\b",
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


# IndusInd transaction columns (matches PDF table, aligned with ICICI)
INDUSIND_TXN_COLUMNS = [
    "row_id", "sno", "tran_id", "transaction_date", "value_date", "posted_time",
    "narration", "debit", "credit", "balance", "txn_type", "reference",
]


def _correct_utr_as_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Fix rows where UTR/reference (e.g. 12-digit) was parsed as amount, or balance from wrong column."""
    if df.empty or 'balance' not in df.columns or 'debit' not in df.columns or 'credit' not in df.columns:
        return df
    UTR_THRESHOLD = 1e8  # amounts > 10 crore likely UTR/reference
    BAL_IMPLAUSIBLE = 1e4  # |balance| < 10k when |prev| > 100k suggests wrong-cell extraction
    prev_bal = None
    for i in range(len(df)):
        bal = df.iloc[i].get('balance')
        if bal is None or (isinstance(bal, float) and pd.isna(bal)):
            continue
        try:
            bal = float(bal)
        except (TypeError, ValueError):
            continue
        _dr = df.iloc[i].get('debit')
        _cr = df.iloc[i].get('credit')
        dr = 0.0 if (_dr is None or (isinstance(_dr, float) and pd.isna(_dr))) else float(_dr)
        cr = 0.0 if (_cr is None or (isinstance(_cr, float) and pd.isna(_cr))) else float(_cr)
        if prev_bal is not None:
            expected_delta = prev_bal - bal  # debit - credit
            actual_delta = dr - cr
            computed_bal = prev_bal - dr + cr
            # Fix implausible balance (e.g. 17 when prev was -112M) — wrong cell parsed as balance
            if abs(prev_bal) > 1e5 and abs(bal) < BAL_IMPLAUSIBLE and abs(computed_bal) > BAL_IMPLAUSIBLE:
                df.at[df.index[i], 'balance'] = computed_bal
                bal = computed_bal
            elif abs(actual_delta - expected_delta) > 0.01:
                if dr >= UTR_THRESHOLD and cr == 0 and expected_delta > 0:
                    df.at[df.index[i], 'debit'] = expected_delta
                elif cr >= UTR_THRESHOLD and dr == 0 and expected_delta < 0:
                    df.at[df.index[i], 'credit'] = -expected_delta
        prev_bal = bal
    return df


def _normalize_df_with_rowid(df: pd.DataFrame) -> pd.DataFrame:
    """Add row_id and ensure standard columns (incl. tran_id, posted_time from PDF)."""
    for col in INDUSIND_TXN_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[[c for c in INDUSIND_TXN_COLUMNS if c != "row_id"]].copy()
    df["row_id"] = range(1, len(df) + 1)
    return df[INDUSIND_TXN_COLUMNS].reset_index(drop=True)


class IndusIndBankExtractor(BaseBankExtractor):

    BANK_NAME = "IndusInd Bank"
    CONFIG = INDUSIND_CONFIG

    # ── detection ───────────────────────────────────────────────

    def detect(self, first_page_text: str) -> bool:
        indicators = [
            r'IndusInd\s*Bank',
            r'INDB\d{7}',            # IndusInd IFSC prefix
            r'indusind\.com',
            r'Account\s+Statement',
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
            "config_used": INDUSIND_CONFIG,
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
                "config_used": INDUSIND_CONFIG,
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

        m = re.search(r'Account\s+No\s*[:\-]?\s*(\d{10,18})', text, re.I)
        if m:
            info.account_number = m.group(1).strip()

        m = re.search(r'(?:Customer\s+Name|Account\s+Name)\s*[:\-]?\s*(.+?)(?:\n|Account\s+No|\([A-Z\s]+\)|$)', text, re.I | re.DOTALL)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'\([^)]*\)', '', name).strip()
            info.customer_name = " ".join(name.split())

        m = re.search(
            r'From\s+Date\s*[:\-]?\s*(\d{2}[-/]\w{3}[-/]\d{2,4})'
            r'.*?To\s+Date\s*[:\-]?\s*(\d{2}[-/]\w{3}[-/]\d{2,4})',
            text, re.I | re.DOTALL
        )
        if m:
            info.statement_from = parse_date(m.group(1), ['%d-%b-%y', '%d-%b-%Y', '%d/%m/%Y']) or m.group(1)
            info.statement_to = parse_date(m.group(2), ['%d-%b-%y', '%d-%b-%Y', '%d/%m/%Y']) or m.group(2)

        m = re.search(r'Opening\s+Bal(?:ance)?\s*[:\-]?\s*(-?[\d,]+\.?\d*)', text, re.I)
        if m:
            info.opening_balance = clean_amount(m.group(1)) or 0.0

        m = re.search(r'Closing\s+Bal(?:ance)?\s*[:\-]?\s*(-?[\d,]+\.?\d*)', text, re.I)
        if m:
            info.closing_balance = clean_amount(m.group(1)) or 0.0

        m = re.search(INDUSIND_CONFIG.ifsc_pattern, text)
        if m:
            info.ifsc_code = m.group(1)

        m = re.search(INDUSIND_CONFIG.micr_pattern, text, re.I)
        if m:
            info.micr_code = m.group(1)

        return info

    def _parse_account_number(self, text: str) -> str:
        m = re.search(r'Account\s+No\s*[:\-]?\s*(\d{10,18})', text, re.I)
        if m:
            return m.group(1).strip()
        return ''

    def _parse_account_holder(self, text: str) -> str:
        m = re.search(r'Customer\s+Name\s*[:\-]?\s*(.+?)(?:\n|Account)', text, re.I)
        if m:
            return m.group(1).strip()
        return ''

    def _parse_opening_balance(self, text: str) -> Optional[float]:
        # IndusInd may not explicitly show opening balance
        # Try to derive from first transaction's balance - first debit/credit
        patterns = [
            r'Opening\s+Bal(?:ance)?\s*[:\-]?\s*(-?[\d,]+\.?\d*)',
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return clean_amount(m.group(1))
        return None

    def _parse_closing_balance(self, text: str) -> Optional[float]:
        patterns = [
            r'Closing\s+Bal(?:ance)?\s*[:\-]?\s*(-?[\d,]+\.?\d*)',
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return clean_amount(m.group(1))
        return None

    def _parse_period(self, text: str):
        # "From Date : 01-Apr-25    To Date : 31-May-25"
        patterns = [
            r'From\s+Date\s*[:\-]?\s*(\d{2}[-/]\w{3}[-/]\d{2,4})'
            r'.*?To\s+Date\s*[:\-]?\s*(\d{2}[-/]\w{3}[-/]\d{2,4})',
            r'From\s+Date\s*[:\-]?\s*(\d{2}[-/]\d{2}[-/]\d{2,4})'
            r'.*?To\s+Date\s*[:\-]?\s*(\d{2}[-/]\d{2}[-/]\d{2,4})',
        ]
        date_formats = ['%d-%b-%y', '%d-%b-%Y', '%d/%m/%Y', '%d-%m-%Y', '%d %b %Y']
        for p in patterns:
            m = re.search(p, text, re.I | re.DOTALL)
            if m:
                return (parse_date(m.group(1), date_formats),
                        parse_date(m.group(2), date_formats))
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
            'bank reference': 'reference',
            'value date': 'value_date',
            'transaction date& time': 'transaction_date',
            'transaction date & time': 'transaction_date',
            'transaction date': 'transaction_date',
            'type': 'txn_type',
            'payment narration': 'narration',
            'narration': 'narration',
            'debit': 'debit',
            'credit': 'credit',
            'available balance': 'balance',
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
            return None
        return self._finalize_df(pd.DataFrame(all_rows))

    def _is_txn_header(self, header: List[str]) -> bool:
        has_date = any('date' in h for h in header)
        has_ref = any('reference' in h or 'narration' in h for h in header)
        has_amount = any(w in h for h in header
                         for w in ['debit', 'credit', 'balance'])
        return has_date and (has_ref or has_amount)

    def _parse_row(self, row: list, col_indices: dict) -> Optional[Dict]:
        def get(key):
            idx = col_indices.get(key)
            if idx is not None and idx < len(row):
                v = row[idx]
                return str(v).strip() if v is not None else ''
            return ''

        txn_date_raw = get('transaction_date')
        # IndusInd date in format: '01-APR-25 06:59:44' or '01 Apr 2025'
        date_part = txn_date_raw.split(' ')[0] if txn_date_raw else ''
        if not date_part or not re.match(r'\d{1,2}[-/\s]\w+[-/\s]\d{2,4}', date_part):
            return None

        debit_raw = get('debit')
        credit_raw = get('credit')
        txn_type_raw = get('txn_type').strip()

        debit = clean_amount(debit_raw) if debit_raw else None
        credit = clean_amount(credit_raw) if credit_raw else None

        # Derive txn_type from Type column or amounts
        if txn_type_raw.lower() == 'debit':
            txn_type = 'DR'
        elif txn_type_raw.lower() == 'credit':
            txn_type = 'CR'
        else:
            txn_type = 'DR' if debit else ('CR' if credit else '')

        # IndusInd date formats
        date_formats = ['%d-%b-%y', '%d-%b-%Y', '%d %b %Y', '%d/%m/%Y']
        parsed_txn_date = parse_date(date_part, date_formats)
        value_date_raw = get('value_date')
        parsed_val_date = parse_date(value_date_raw.split(' ')[0] if value_date_raw else '',
                                     date_formats)

        balance_raw = get('balance')
        # Balance can be negative in IndusInd
        balance = None
        if balance_raw:
            balance = clean_amount(balance_raw.replace('−', '-').replace('–', '-'))

        # Extract posted_time from "Transaction Date & Time" (e.g. '01-APR-25 06:59:44')
        posted_time = ''
        if txn_date_raw and ' ' in txn_date_raw:
            parts = txn_date_raw.split(' ', 2)
            if len(parts) >= 3 and re.match(r'\d{1,2}:\d{2}:\d{2}', parts[2]):
                posted_time = parts[2].strip()

        ref = get('reference')
        return {
            'sno': '',
            'tran_id': ref,
            'transaction_date': parsed_txn_date,
            'value_date': parsed_val_date,
            'posted_time': posted_time,
            'narration': get('narration'),
            'debit': debit,
            'credit': credit,
            'balance': balance,
            'txn_type': txn_type,
            'reference': ref,
        }

    # ── Strategy 2: regex ────────────────────────────────────────

    def _extract_via_regex(self, raw_text: str) -> pd.DataFrame:
        """
        IndusInd raw text: narrations can wrap to next line(s).
        Example: "ACH DR INW" on line 1, "PAY/67e803f7cf11b9303ebc3a0c/AXIS BANK/" on line 2.
        """
        rows = []
        AMT_PAT = r'(-?[\d,]+(?:\.\d{1,2})?)'  # 14660, 141804.8, -126621063.04
        DATE_PAT = r'(\d{2}\s+\w{3}\s+\d{4})'     # '01 Apr 2025'
        TXN_DATE_PAT = r"'?(\d{2}-[A-Z]{3}-\d{2,4})"  # '01-APR-25

        line_re = re.compile(
            r'^\s*\*?(\S+)\s+'       # Bank Reference
            r'' + DATE_PAT + r'\s+'  # Value Date
            r'' + TXN_DATE_PAT + r'[\s\d:]+\s+'  # Txn Date & Time
            r'(Debit|Credit)\s+'     # Type
            r'(.+?)\s+'              # Narration
            r'(?:' + AMT_PAT + r'\s+)?'  # optional debit
            r'(?:' + AMT_PAT + r'\s+)?'  # optional credit
            r'' + AMT_PAT + r'\s*$', # balance
            re.MULTILINE | re.I
        )

        # New transaction start: ref-like + date (e.g. S31201400 01 Apr 2025 or 'ICIN... 31 Mar 2025)
        txn_start_re = re.compile(
            r'^\s*[\'*]?[A-Z0-9]+\s+\d{2}\s+\w{3}\s+\d{4}\s+',
            re.MULTILINE | re.I
        )

        def get_continuation_narration(start: int, end: int) -> str:
            """Extract continuation lines between two transaction matches."""
            block = raw_text[start:end]
            lines = []
            for line in block.split('\n'):
                s = line.strip()
                if not s:
                    continue
                if re.match(r'^--\s*\d+\s+of\s+\d+\s*--', s) or re.match(r'^\d+$', s):
                    continue  # skip page footer
                if txn_start_re.match(s):
                    break  # next transaction, stop
                lines.append(s)
            return ' '.join(lines) if lines else ''

        matches = list(line_re.finditer(raw_text))
        date_fmts = ['%d-%b-%y', '%d-%b-%Y', '%d %b %Y']
        for i, m in enumerate(matches):
            ref = m.group(1)
            val_date = parse_date(m.group(2), ['%d %b %Y'])
            txn_date = parse_date(m.group(3), date_fmts)
            txn_type_raw = m.group(4).strip().lower()
            narration = m.group(5).strip()
            # Append continuation lines (narration wraps to next line in PDF)
            next_start = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            cont = get_continuation_narration(m.end(), next_start)
            if cont:
                narration = (narration + ' ' + cont).strip()
            g6, g7, g8 = m.group(6), m.group(7), m.group(8)

            txn_type = 'DR' if txn_type_raw == 'debit' else 'CR'
            # First amount (g6) is debit or credit; g8 is balance (g7 unused for single-amount rows)
            amt = clean_amount(g6) if g6 else None
            g7_val = clean_amount(g7) if g7 else None
            # UTR/reference sanity: PDF may have UTR (12-digit) before actual amount; use smaller
            if amt is not None and g7_val is not None and amt >= 1e8 and g7_val < 1e8:
                amt = g7_val
            debit = amt if txn_type == 'DR' else None
            credit = amt if txn_type == 'CR' else None
            balance = clean_amount(g8)

            # Extract posted_time from txn date/time string (e.g. '01-APR-25 06:59:44')
            posted_time = ''
            txn_dt_match = re.search(r'(\d{1,2}:\d{2}:\d{2})', m.group(0))
            if txn_dt_match:
                posted_time = txn_dt_match.group(1)

            rows.append({
                'sno': '',
                'tran_id': ref,
                'transaction_date': txn_date,
                'value_date': val_date,
                'posted_time': posted_time,
                'narration': narration,
                'debit': debit,
                'credit': credit,
                'balance': balance,
                'txn_type': txn_type,
                'reference': ref,
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
        df = _correct_utr_as_amount(df)
        return _normalize_df_with_rowid(df)
