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
    "narration", "debit", "credit", "balance", "txn_type", "reference", "balance_ok",
]


def _correct_utr_as_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Fix rows where UTR/reference (e.g. 12-digit) was parsed as amount, or balance from wrong column."""
    if df.empty or 'balance' not in df.columns or 'debit' not in df.columns or 'credit' not in df.columns:
        return df
    UTR_THRESHOLD = 1e8   # amounts > 10 crore likely UTR/reference
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
            expected_delta = prev_bal - bal
            actual_delta = dr - cr
            computed_bal = prev_bal - dr + cr
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


def _clean_narration(raw: str) -> str:
    """
    Reconstruct a narration string that was split across PDF rendering lines.

    Rules applied per character pair surrounding each '\n':
      1. char_before is ' '          → drop \n, space already present
      2. char_before or char_after is '/' → drop \n, path/segment continuation
      3. both chars are alphanumeric  → drop \n, mid-word PDF wrap
      4. otherwise                    → replace \n with ' ', real word boundary
    """
    def _join(m: re.Match) -> str:
        before, after = m.group(1), m.group(2)
        if before == ' ':
            return before
        if before == '/' or after == '/':
            return before + after
        if before.isalnum() and after.isalnum():
            return before + after
        return before + ' ' + after

    result = re.sub(r'(.)\n(.)', _join, raw)
    return re.sub(r' {2,}', ' ', result).strip()


def _normalize_df_with_rowid(df: pd.DataFrame) -> pd.DataFrame:
    """Add row_id and ensure standard columns (incl. tran_id, posted_time from PDF)."""
    for col in INDUSIND_TXN_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[[c for c in INDUSIND_TXN_COLUMNS if c != "row_id"]].copy()
    df["row_id"] = range(1, len(df) + 1)
    return df[INDUSIND_TXN_COLUMNS].reset_index(drop=True)


def _add_balance_ok(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'balance_ok' boolean column.
    True  = this row's balance is consistent with previous row (within ₹0.02).
    False = balance break (legitimate page-boundary gap or deleted pages).
    The first row is always True.
    """
    ok = [True]
    for i in range(1, len(df)):
        prev_bal = df.iloc[i - 1].get('balance')
        curr_bal = df.iloc[i].get('balance')
        dr = df.iloc[i].get('debit')
        cr = df.iloc[i].get('credit')
        if (prev_bal is None or (isinstance(prev_bal, float) and pd.isna(prev_bal))
                or curr_bal is None or (isinstance(curr_bal, float) and pd.isna(curr_bal))):
            ok.append(True)
            continue
        dr = 0.0 if (dr is None or (isinstance(dr, float) and pd.isna(dr))) else float(dr)
        cr = 0.0 if (cr is None or (isinstance(cr, float) and pd.isna(cr))) else float(cr)
        computed = round(float(prev_bal) - dr + cr, 2)
        ok.append(abs(computed - round(float(curr_bal), 2)) <= 0.02)
    df = df.copy()
    df['balance_ok'] = ok
    return df


class IndusIndBankExtractor(BaseBankExtractor):

    BANK_NAME = "IndusInd Bank"
    CONFIG = INDUSIND_CONFIG

    # ── COL_MAP ─────────────────────────────────────────────────
    # Maps normalised header cell text → internal field name.
    # Both no-space and spaced variants of "Transaction Date & Time" are mapped.
    # Additional aliases for robustness against minor PDF text variations.
    COL_MAP = {
        'bank reference': 'reference',
        'value date': 'value_date',
        'transaction date& time': 'transaction_date',    # no-space variant
        'transaction date & time': 'transaction_date',   # spaced variant (actual PDF)
        'transaction date': 'transaction_date',
        'type': 'txn_type',
        'payment narration': 'narration',
        'narration': 'narration',
        'debit': 'debit',
        'credit': 'credit',
        'available balance': 'balance',
        'balance': 'balance',
    }

    # ── detection ───────────────────────────────────────────────

    def detect(self, first_page_text: str) -> bool:
        indicators = [
            r'IndusInd\s*Bank',
            r'INDB\d{7}',
            r'indusind\.com',
            r'Account\s+Statement',
        ]
        return any(re.search(p, first_page_text, re.I) for p in indicators)

    # ── main extraction ──────────────────────────────────────────

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

    # ── account info ─────────────────────────────────────────────

    def _parse_account_info(self, text: str) -> AccountInfo:
        info = AccountInfo(bank_name=self.BANK_NAME)

        m = re.search(r'Account\s+No\s*[:\-]?\s*(\d{10,18})', text, re.I)
        if m:
            info.account_number = m.group(1).strip()

        m = re.search(
            r'(?:Customer\s+Name|Account\s+Name)\s*[:\-]?\s*(.+?)(?:\n|Account\s+No|\([A-Z\s]+\)|$)',
            text, re.I | re.DOTALL
        )
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
        return m.group(1).strip() if m else ''

    def _parse_account_holder(self, text: str) -> str:
        m = re.search(r'Customer\s+Name\s*[:\-]?\s*(.+?)(?:\n|Account)', text, re.I)
        return m.group(1).strip() if m else ''

    def _parse_opening_balance(self, text: str) -> Optional[float]:
        m = re.search(r'Opening\s+Bal(?:ance)?\s*[:\-]?\s*(-?[\d,]+\.?\d*)', text, re.I)
        return clean_amount(m.group(1)) if m else None

    def _parse_closing_balance(self, text: str) -> Optional[float]:
        m = re.search(r'Closing\s+Bal(?:ance)?\s*[:\-]?\s*(-?[\d,]+\.?\d*)', text, re.I)
        return clean_amount(m.group(1)) if m else None

    def _parse_period(self, text: str):
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
        saved_col_indices = None

        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 5,
            "join_tolerance": 5,
        }

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables(table_settings)
                for table in tables:
                    if not table:
                        continue

                    # ── FIX 8+9: Robust header detection ─────────────────────
                    # Normalise BOTH real \n and escaped \\n so multi-line PDF
                    # header cells ("Available\nBalance") are handled correctly.
                    header_row_idx = None
                    for row_idx, row in enumerate(table):
                        normalised = [
                            str(c or '').lower().strip()
                                        .replace('\n', ' ')   # real newline
                                        .replace('\\n', ' ')  # escaped newline
                            for c in row
                        ]
                        if self._is_txn_header(normalised):
                            header_row_idx = row_idx
                            col_indices = {}
                            for col_idx, h in enumerate(normalised):
                                mapped = self.COL_MAP.get(h)
                                if mapped and mapped not in col_indices:
                                    col_indices[mapped] = col_idx
                            # Fallback: no explicit txn_date → use value_date
                            if ('transaction_date' not in col_indices
                                    and 'value_date' in col_indices):
                                col_indices['transaction_date'] = col_indices['value_date']
                            if ('transaction_date' in col_indices
                                    or 'value_date' in col_indices):
                                saved_col_indices = col_indices
                            break

                    # FIX 2: No column mapping yet → skip table entirely
                    if saved_col_indices is None:
                        continue

                    data_start = (header_row_idx + 1) if header_row_idx is not None else 0

                    # Diagnostic counters per table
                    # parsed_count = 0
                    # skipped_count = 0
                    # for row in table[data_start:]:
                    #     parsed = self._parse_row(row, saved_col_indices, page.page_number)
                    #     if parsed:
                    #         all_rows.append(parsed)
                    #         parsed_count += 1
                    #     else:
                    #         skipped_count += 1

                    # self._log(
                    #     f"[Page {page.page_number}] Table rows: {len(table) - data_start} "
                    #     f"| Parsed: {parsed_count} | Skipped: {skipped_count}"
                    # )

        if not all_rows:
            return None
        return self._finalize_df(pd.DataFrame(all_rows))

    # ── FIX 8: Flat-string header detection ─────────────────────

    def _is_txn_header(self, header: List[str]) -> bool:
        """
        FIX 8: Join ALL header cells into a single flat string before matching.
        This handles cases where pdfplumber splits a multi-line header cell
        across two rows — the individual cells may not match, but the flat
        string will contain the keywords.
        """
        flat = ' '.join(header)
        has_date   = 'date' in flat
        has_ref    = 'reference' in flat or 'narration' in flat
        has_amount = any(w in flat for w in ['debit', 'credit', 'balance'])
        return has_date and (has_ref or has_amount)

    # ── _parse_row ────────────────────────────────────────────────

    def _parse_row(
        self,
        row: list,
        col_indices: dict,
        page_number: int = 0,
    ) -> Optional[Dict]:

        def get(key):
            idx = col_indices.get(key)
            if idx is not None and idx < len(row):
                v = row[idx]
                return str(v).strip() if v is not None else ''
            return ''

        # ── FIX 3: Strip leading apostrophe from date cells ───────
        txn_date_raw   = get('transaction_date').lstrip("'").strip()
        value_date_raw = get('value_date').lstrip("'").strip()

        # ── FIX 7: Fallback to value_date when txn_date is empty ─
        # pdfplumber sometimes splits the "Transaction Date & Time" cell
        # across two lines; in those cases txn_date_raw ends up empty or
        # contains only a time string.  Use value_date as the canonical date.
        active_date_raw = txn_date_raw if txn_date_raw else value_date_raw
        date_part = active_date_raw.split(' ')[0] if active_date_raw else ''

        if not date_part or not re.match(r'\d{1,2}[-/]\w+[-/]\d{2,4}', date_part):
            self._log(
                f"[Page {page_number}] Skipped row (no parseable date) "
                f"txn_date_raw={txn_date_raw!r} value_date_raw={value_date_raw!r} "
                f"row={row}"
            )
            return None

        debit_raw    = get('debit')
        credit_raw   = get('credit')
        txn_type_raw = get('txn_type').strip()

        debit  = clean_amount(debit_raw)  if debit_raw  else None
        credit = clean_amount(credit_raw) if credit_raw else None

        # Derive txn_type from explicit "Type" column; fall back to amounts.
        if txn_type_raw.lower() == 'debit':
            txn_type = 'DR'
        elif txn_type_raw.lower() == 'credit':
            txn_type = 'CR'
        else:
            txn_type = 'DR' if debit else ('CR' if credit else '')

        date_formats = ['%d-%b-%y', '%d-%b-%Y', '%d %b %Y', '%d/%m/%Y']
        parsed_txn_date = parse_date(date_part, date_formats)
        parsed_val_date = parse_date(
            value_date_raw.split(' ')[0] if value_date_raw else '',
            date_formats,
        )

        balance_raw = get('balance')
        balance = None
        if balance_raw:
            balance = clean_amount(
                balance_raw.replace('−', '-').replace('–', '-')
            )

        # ── FIX 4: Extract posted_time correctly ──────────────────
        # After stripping the apostrophe the cell is "01-APR-25 06:59:44".
        # Split on the FIRST space → exactly 2 parts; second part is the time.
        posted_time = ''
        if ' ' in txn_date_raw:
            parts = txn_date_raw.split(' ', 1)
            time_candidate = parts[1].strip() if len(parts) == 2 else ''
            if re.match(r'\d{1,2}:\d{2}:\d{2}', time_candidate):
                posted_time = time_candidate

        ref = get('reference').lstrip("'")
        return {
            'sno':              '',
            'tran_id':          ref,
            'transaction_date': parsed_txn_date,
            'value_date':       parsed_val_date,
            'posted_time':      posted_time,
            'narration':        _clean_narration(get('narration')),
            'debit':            debit,
            'credit':           credit,
            'balance':          balance,
            'txn_type':         txn_type,
            'reference':        ref,
        }

    # ── Strategy 2: regex fallback ───────────────────────────────

    def _extract_via_regex(self, raw_text: str) -> pd.DataFrame:
        rows = []
        AMT_PAT      = r'(-?[\d,]+(?:\.\d{1,2})?)' # 14660, 141804.8
        DATE_PAT     = r'(\d{2}\s+\w{3}\s+\d{4})'  # '01 Apr 2025'
        TXN_DATE_PAT = r"'?(\d{2}-[A-Z]{3}-\d{2,4})"  # '01-APR-25

        line_re = re.compile(
            r'^\s*\*?(\S+)\s+'
            + DATE_PAT + r'\s+'
            + TXN_DATE_PAT + r'[\s\d:]+\s+'
            + r'(Debit|Credit)\s+'
            + r'(.+?)\s+'
            + r'(?:' + AMT_PAT + r'\s+)?'
            + r'(?:' + AMT_PAT + r'\s+)?'
            + AMT_PAT + r'\s*$',
            re.MULTILINE | re.I
        )

        txn_start_re = re.compile(
            r'^\s*[\'*]?[A-Z0-9]+\s+\d{2}\s+\w{3}\s+\d{4}\s+',
            re.MULTILINE | re.I
        )

        def get_continuation_narration(start: int, end: int) -> str:
            block = raw_text[start:end]
            lines = []
            for line in block.split('\n'):
                s = line.strip()
                if not s:
                    continue
                if re.match(r'^--\s*\d+\s+of\s+\d+\s*--', s) or re.match(r'^\d+$', s):
                    continue
                if txn_start_re.match(s):
                    break
                lines.append(s)
            return ' '.join(lines) if lines else ''

        matches    = list(line_re.finditer(raw_text))
        date_fmts  = ['%d-%b-%y', '%d-%b-%Y', '%d %b %Y']

        for i, m in enumerate(matches):
            ref          = m.group(1)
            val_date     = parse_date(m.group(2), ['%d %b %Y'])
            txn_date     = parse_date(m.group(3), date_fmts)
            txn_type_raw = m.group(4).strip().lower()
            narration    = m.group(5).strip()

            next_start = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            cont = get_continuation_narration(m.end(), next_start)
            narration = _clean_narration(narration + '\n' + cont if cont else narration)

            g6, g7, g8 = m.group(6), m.group(7), m.group(8)
            txn_type = 'DR' if txn_type_raw == 'debit' else 'CR'

            amt     = clean_amount(g6) if g6 else None
            g7_val  = clean_amount(g7) if g7 else None
            if amt is not None and g7_val is not None and amt >= 1e8 and g7_val < 1e8:
                amt = g7_val
            debit   = amt if txn_type == 'DR' else None
            credit  = amt if txn_type == 'CR' else None
            balance = clean_amount(g8)

            # FIX 4 (regex path): search for HH:MM:SS anywhere in the match
            posted_time = ''
            t_match = re.search(r'(\d{1,2}:\d{2}:\d{2})', m.group(0))
            if t_match:
                posted_time = t_match.group(1)

            rows.append({
                'sno':              '',
                'tran_id':          ref,
                'transaction_date': txn_date,
                'value_date':       val_date,
                'posted_time':      posted_time,
                'narration':        narration,
                'debit':            debit,
                'credit':           credit,
                'balance':          balance,
                'txn_type':         txn_type,
                'reference':        ref,
            })

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return self._finalize_df(df)

    # ── _finalize_df ──────────────────────────────────────────────

    def _finalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return _normalize_df_with_rowid(df)

        for col in ('debit', 'credit', 'balance'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['transaction_date'])

        # ── FIX 10: Guard against parse_date returning string "NaT" ──
        # convert to string and exclude anything that is not YYYY-MM-DD
        if 'transaction_date' in df.columns:
            df = df[
                df['transaction_date'].astype(str).str.match(
                    r'\d{4}-\d{2}-\d{2}', na=False
                )
            ]

        # FIX 6: Preserve PDF page order — do NOT sort by date.
        df = df.reset_index(drop=True)
        df = _correct_utr_as_amount(df)
        df = _add_balance_ok(df)
        return _normalize_df_with_rowid(df)