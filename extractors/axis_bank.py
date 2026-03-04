"""
Axis Bank Statement Extractor (Statement of Account / CA - Business Advantage)
================================================================================
Handles Axis Bank "Statement of Account" format (different from NEO).
Aligned with HDFC/ICICI/IndusInd/SBI: returns AccountInfo and transactions.

Statement layout:
  - Header: Account name, Customer ID, IFSC, period, OPENING BALANCE
  - Table: Tran Date | Chq No | Particulars | Debit | Credit | Balance | Init. Br
  - Dates: dd-mm-yyyy
  - Debit and Credit are separate columns (one per row)
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
# BankConfig / AccountInfo
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


AXIS_SA_CONFIG = BankConfig(
    name="Axis Bank",
    date_pattern=r"\b(\d{2}-\d{2}-\d{4})\b",
    opening_balance_pattern=r"OPENING\s+BALANCE\s+([\d,]+\.?\d*)",
    min_columns=5,
    date_format="%d-%m-%Y",
    ifsc_pattern=r"\b(UTIB\d{7})\b",
    micr_pattern=r"MICR\s+Code\s*[:\-]?\s*(\d{9})",
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


AXIS_SA_TXN_COLUMNS = [
    "row_id", "sno", "tran_id", "transaction_date", "value_date", "posted_time",
    "narration", "debit", "credit", "balance", "txn_type", "reference",
]


def _normalize_df_with_rowid(df: pd.DataFrame) -> pd.DataFrame:
    for col in AXIS_SA_TXN_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[[c for c in AXIS_SA_TXN_COLUMNS if c != "row_id"]].copy()
    df["row_id"] = range(1, len(df) + 1)
    return df[AXIS_SA_TXN_COLUMNS].reset_index(drop=True)


class AxisBankExtractor(BaseBankExtractor):

    BANK_NAME = "Axis Bank"
    CONFIG = AXIS_SA_CONFIG

    def detect(self, first_page_text: str) -> bool:
        indicators = [
            r'Statement\s+of\s+Account\s+No',
            r'UTIB\d{7}',
            r'CA\s*-\s*BUSINESS\s+ADVANTAGE',
            r'Tran\s+Date\s+Chq\s+No\s+Particulars',
        ]
        return any(re.search(p, first_page_text, re.I) for p in indicators)

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
            "config_used": AXIS_SA_CONFIG,
        }

    def run(self, pdf_path: str) -> Dict[str, Any]:
        raw_text = self._extract_raw_text(pdf_path)
        if not raw_text.strip():
            info = AccountInfo(bank_name=self.BANK_NAME)
            return {
                "bank": self.BANK_NAME,
                "account_info": info,
                "transactions": pd.DataFrame(),
                "config_used": AXIS_SA_CONFIG,
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

    def _parse_account_info(self, text: str) -> AccountInfo:
        info = AccountInfo(bank_name=self.BANK_NAME)

        m = re.search(r'Statement\s+of\s+Account\s+No\s*[:\-]?\s*(\d{10,20})', text, re.I)
        if m:
            info.account_number = m.group(1).strip()

        lines = text.strip().split('\n')
        for line in lines[:15]:
            s = line.strip()
            if len(s) > 3 and not re.match(r'^(Joint|GROUND|CHS|RAIGAD|MAHARASHTRA|Nominee|Registered|Scheme|Statement|Tran|OPENING)', s, re.I):
                if any(c in s.upper() for c in ['CORPORATION', 'PRIVATE', 'LIMITED', 'PVT', 'LTD']):
                    info.customer_name = s
                    break

        m = re.search(r'Customer\s+ID\s*[:\-]?\s*(\d+)', text, re.I)
        if m:
            info.customer_id = m.group(1).strip()

        m = re.search(r'[Ff]rom\s*[:\-]?\s*(\d{2}-\d{2}-\d{4}).*?[Tt]o\s*[:\-]?\s*(\d{2}-\d{2}-\d{4})', text, re.DOTALL)
        if m:
            info.statement_from = parse_date(m.group(1), ['%d-%m-%Y']) or m.group(1)
            info.statement_to = parse_date(m.group(2), ['%d-%m-%Y']) or m.group(2)

        m = re.search(r'OPENING\s+BALANCE\s+([\d,]+\.?\d*)', text, re.I)
        if m:
            info.opening_balance = clean_amount(m.group(1)) or 0.0

        m = re.search(r'CLOSING\s+BALANCE\s+([\d,]+\.?\d*)', text, re.I)
        if m:
            info.closing_balance = clean_amount(m.group(1)) or 0.0

        m = re.search(AXIS_SA_CONFIG.ifsc_pattern, text)
        if m:
            info.ifsc_code = m.group(1)

        m = re.search(AXIS_SA_CONFIG.micr_pattern, text)
        if m:
            info.micr_code = m.group(1)

        m = re.search(r'PAN\s*[:\-]?\s*([A-Z0-9]{10})', text)
        if m:
            info.pan = m.group(1).strip()

        return info

    def _extract_transactions(self, pdf_path: str, raw_text: str) -> pd.DataFrame:
        df = self._extract_via_pdfplumber(pdf_path)
        if df is not None and len(df) > 0:
            return df
        df = self._extract_via_pdfplumber(pdf_path, use_text_strategy=True)
        if df is not None and len(df) > 0:
            return df
        self._log("pdfplumber failed — falling back to regex")
        return self._extract_via_regex(raw_text)

    def _extract_via_pdfplumber(self, pdf_path: str, use_text_strategy: bool = False) -> Optional[pd.DataFrame]:
        all_rows = []
        saved_col_indices = None

        if use_text_strategy:
            table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text",
                             "snap_tolerance": 3, "join_tolerance": 3}
        else:
            table_settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines",
                             "snap_tolerance": 5, "join_tolerance": 5}

        COL_MAP = {
            'tran date': 'transaction_date', 'transaction date': 'transaction_date',
            'chq no': 'reference', 'cheque no': 'reference',
            'particulars': 'narration', 'debit': 'debit', 'credit': 'credit',
            'balance': 'balance', 'init. br': 'branch', 'init br': 'branch',
        }

        def norm(h):
            return str(h or '').lower().strip().replace('\n', ' ')

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables(table_settings)
                for table in tables:
                    if not table or len(table) < 1:
                        continue
                    header = [norm(c) for c in table[0]]
                    col_indices = {}
                    for idx, h in enumerate(header):
                        for k, v in COL_MAP.items():
                            if k in h or (h and h in k):
                                col_indices[v] = idx
                                break
                    has_header = (
                        any('date' in h or 'tran' in h for h in header) and
                        any(w in h for h in header for w in ['debit', 'credit', 'balance'])
                    )
                    if has_header:
                        if 'transaction_date' not in col_indices and 'debit' not in col_indices:
                            continue
                        if 'transaction_date' not in col_indices and 'balance' in col_indices:
                            continue
                        if 'transaction_date' in col_indices or 'debit' in col_indices:
                            saved_col_indices = col_indices
                        if saved_col_indices is None:
                            continue
                        start = 1
                    else:
                        if saved_col_indices is None:
                            continue
                        col_indices = saved_col_indices
                        idx_date = saved_col_indices.get('transaction_date', 0)
                        first_cell = str((table[0][idx_date] if idx_date < len(table[0]) else '') or '').strip()
                        if not re.match(r'\d{2}-\d{2}-\d{4}', first_cell):
                            continue
                        start = 0
                    for row in table[start:]:
                        p = self._parse_row(row, saved_col_indices)
                        if p:
                            all_rows.append(p)
        if not all_rows:
            return None
        return self._finalize_df(pd.DataFrame(all_rows))

    def _parse_row(self, row: list, col_indices: dict) -> Optional[Dict]:
        def get(k):
            idx = col_indices.get(k)
            if idx is not None and idx < len(row):
                v = row[idx]
                return str(v).strip() if v else ''
            return ''

        def clean_cell(s):
            return " ".join(str(s).replace('\n', ' ').split()) if s else ""

        txn_date = get('transaction_date')
        if not re.match(r'\d{2}-\d{2}-\d{4}', txn_date):
            return None
        debit = clean_amount(get('debit'))
        credit = clean_amount(get('credit'))
        balance = clean_amount(get('balance'))
        if balance is None and debit is None and credit is None:
            return None
        txn_type = 'DR' if debit else ('CR' if credit else '')
        ref = clean_cell(get('reference'))

        return {
            'sno': '', 'tran_id': ref,
            'transaction_date': parse_date(txn_date, ['%d-%m-%Y']),
            'value_date': parse_date(txn_date, ['%d-%m-%Y']),
            'posted_time': '',
            'narration': clean_cell(get('narration')),
            'debit': debit, 'credit': credit, 'balance': balance,
            'txn_type': txn_type, 'reference': ref,
        }

    def _extract_via_regex(self, raw_text: str) -> pd.DataFrame:
        """Parse: date [chq_no] particulars amt1 amt2 [branch]. Infer debit/credit from balance."""
        rows = []
        text = self._merge_continuation_lines(raw_text)
        DATE_PAT = r'(\d{2}-\d{2}-\d{4})'
        AMT_STRICT = r'([\d,]+\.\d{2})'
        AMT_FLEX = r'([\d,]+(?:\.\d{1,2})?)'
        line_res = [
            re.compile(
                rf'^{DATE_PAT}\s+'
                rf'(?:(\d{4,6})\s+)?'
                rf'(.+?)\s+'
                rf'{AMT_STRICT}\s+{AMT_STRICT}\s*'
                rf'(?:\s+(\d{3,5}))?\s*$',
                re.MULTILINE
            ),
            re.compile(
                rf'^{DATE_PAT}\s+'
                rf'(?:(\d{4,6})\s+)?'
                rf'(.+?)\s+'
                rf'{AMT_FLEX}\s+{AMT_FLEX}\s*'
                rf'(?:\s+(\d{3,5}))?\s*$',
                re.MULTILINE
            ),
        ]

        prev_balance = None
        m_open = re.search(r'OPENING\s+BALANCE\s+([\d,]+\.?\d*)', raw_text, re.I)
        if m_open:
            prev_balance = clean_amount(m_open.group(1))

        seen_lines = set()
        for line_re in line_res:
            for m in line_re.finditer(text):
                txn_date, chq_no, narration, amt1, amt2, branch = (
                    m.group(1), m.group(2), m.group(3).strip(), m.group(4), m.group(5), m.group(6)
                )
                line_key = (txn_date, narration[:30], amt1, amt2)
                if line_key in seen_lines:
                    continue
                seen_lines.add(line_key)
                a1, a2 = clean_amount(amt1), clean_amount(amt2)
                if a1 is None or a2 is None:
                    continue
                balance = a2
                if prev_balance is not None:
                    if abs((prev_balance + a1) - a2) < 0.03:
                        debit, credit = None, a1
                    elif abs((prev_balance - a1) - a2) < 0.03:
                        debit, credit = a1, None
                    else:
                        debit, credit = (a1, None) if a2 < prev_balance else (None, a1)
                else:
                    debit, credit = (a1, None) if a2 < a1 else (None, a1)
                prev_balance = balance
                txn_type = 'DR' if debit else 'CR'

                rows.append({
                    'sno': '', 'tran_id': chq_no or '',
                    'transaction_date': parse_date(txn_date, ['%d-%m-%Y']),
                    'value_date': parse_date(txn_date, ['%d-%m-%Y']),
                    'posted_time': '',
                    'narration': " ".join(narration.split()),
                    'debit': debit, 'credit': credit, 'balance': balance,
                    'txn_type': txn_type, 'reference': chq_no or '',
                })

        return self._finalize_df(pd.DataFrame(rows))

    def _merge_continuation_lines(self, text: str) -> str:
        """Merge: when we see a line starting with dd-mm-yyyy + amounts, prepend pending buffer to it."""
        lines = text.split('\n')
        merged = []
        buffer = ''
        start_pat = re.compile(r'^(\d{2}-\d{2}-\d{4})\s+')
        amount_end_pat = re.compile(r'[\d,]+\.\d{2}\s+[\d,]+\.\d{2}(\s+\d{3,5})?\s*$')
        for line in lines:
            stripped = line.strip()
            m = start_pat.match(stripped)
            if m and amount_end_pat.search(stripped):
                if buffer:
                    merged.append(buffer)
                    buffer = ''
                merged.append(stripped)
            elif m:
                if buffer:
                    merged.append(buffer)
                buffer = stripped
            else:
                if buffer:
                    buffer = buffer.rstrip() + ' ' + stripped
        if buffer and amount_end_pat.search(buffer):
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
