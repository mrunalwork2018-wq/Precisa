"""
Base Bank Statement Extractor
==============================
All bank-specific extractors inherit from this base class.
Provides common utilities for PDF text extraction, amount parsing,
date parsing, and dataframe normalization.

Compatible with: pdfplumber, pandas, numpy 1.24+
PaddleOCR hook: override `extract_with_ocr()` in subclass for scanned PDFs.
"""

import re
import pdfplumber
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any


# ─────────────────────────── helpers ───────────────────────────

def clean_amount(val: str) -> Optional[float]:
    """Parse Indian-format amount strings like '1,23,456.78' → 123456.78"""
    if val is None:
        return None
    val = str(val).strip()
    # Remove commas, spaces, currency symbols
    val = re.sub(r'[₹,\s]', '', val)
    # Handle parentheses as negative  e.g. (1234.56)
    if val.startswith('(') and val.endswith(')'):
        val = '-' + val[1:-1]
    try:
        return float(val) if val not in ('', '-', '--', 'Nil', 'N/A') else None
    except ValueError:
        return None


def parse_date(val: str, formats: List[str] = None) -> Optional[str]:
    """Try multiple date formats and return ISO yyyy-mm-dd string."""
    if not val or str(val).strip() in ('', 'nan', 'None'):
        return None
    val = str(val).strip()
    default_formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%d/%m/%y', '%d-%m-%y',
        '%d %b %Y', '%d-%b-%Y', '%d-%b-%y',
        '%Y-%m-%d', '%m/%d/%Y',
    ]
    for fmt in (formats or default_formats):
        try:
            return datetime.strptime(val, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return val   # return as-is if unparseable


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every output DataFrame has these exact columns
    in this exact order — regardless of source bank.
    """
    STANDARD_COLS = [
        'transaction_date',
        'value_date',
        'narration',
        'debit',        # withdrawal / DR
        'credit',       # deposit / CR
        'balance',      # available / closing balance after txn
        'txn_type',     # 'DR' | 'CR' | ''
        'reference',    # cheque no / ref no
    ]
    for col in STANDARD_COLS:
        if col not in df.columns:
            df[col] = None
    return df[STANDARD_COLS].reset_index(drop=True)


# ─────────────────────────── base class ────────────────────────

class BaseBankExtractor(ABC):
    """
    Abstract base for all bank extractors.

    Subclasses must implement:
        - detect(text: str) -> bool          : return True if PDF matches this bank
        - extract(pdf_path: str) -> dict     : full extraction result

    Optional override:
        - extract_with_ocr(pdf_path)         : for scanned / image-based PDFs
          (plug in PaddleOCR here when available)
    """

    BANK_NAME: str = "Unknown"

    # ── public API ──────────────────────────────────────────────

    def run(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main entry point.  Returns:
        {
            'bank'            : str,
            'account_number'  : str,
            'account_holder'  : str,
            'opening_balance' : float,
            'closing_balance' : float,
            'statement_from'  : str (ISO date),
            'statement_to'    : str (ISO date),
            'currency'        : str,
            'transactions'    : pd.DataFrame,
            'raw_text'        : str,
            'errors'          : list[str],
        }
        """
        result = self._empty_result()
        result['bank'] = self.BANK_NAME
        errors = []

        try:
            raw_text = self._extract_raw_text(pdf_path)
            result['raw_text'] = raw_text

            if not raw_text.strip():
                # Possibly scanned — try OCR hook
                result = self.extract_with_ocr(pdf_path)
                result['bank'] = self.BANK_NAME
                return result

            extracted = self.extract(pdf_path, raw_text)
            result.update(extracted)

        except Exception as e:
            errors.append(f"Extraction error: {e}")

        result['errors'] = errors
        return result

    @abstractmethod
    def detect(self, first_page_text: str) -> bool:
        """Return True if this extractor matches the given PDF."""

    @abstractmethod
    def extract(self, pdf_path: str, raw_text: str) -> Dict[str, Any]:
        """
        Bank-specific extraction logic.
        Must return a dict with keys matching _empty_result().
        """

    def extract_with_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """
        Hook for PaddleOCR-based extraction of scanned PDFs.
        Override this in subclass when PaddleOCR is available:

            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en',
                            use_gpu=False, show_log=False)
            # convert PDF pages to images, run OCR, parse lines
        """
        result = self._empty_result()
        result['errors'] = [
            "PDF appears to be scanned/image-based. "
            "PaddleOCR hook not yet implemented. "
            "Install paddlepaddle==2.6.2 + paddleocr==2.7.2 and override extract_with_ocr()."
        ]
        return result

    # ── shared utilities ────────────────────────────────────────

    def _extract_raw_text(self, pdf_path: str) -> str:
        """Extract all text from PDF using pdfplumber."""
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=2, y_tolerance=2)
                if t:
                    text_parts.append(t)
        return '\n'.join(text_parts)

    def _extract_tables_pdfplumber(self, pdf_path: str,
                                   table_settings: dict = None) -> List[pd.DataFrame]:
        """Extract all tables from PDF using pdfplumber."""
        dfs = []
        settings = table_settings or {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        }
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables(settings)
                for tbl in tables:
                    if tbl and len(tbl) > 1:
                        df = pd.DataFrame(tbl[1:], columns=tbl[0])
                        dfs.append(df)
        return dfs

    def _find_balance_in_text(self, text: str,
                               label: str,
                               patterns: List[str] = None) -> Optional[float]:
        """Generic balance finder using regex patterns."""
        default_patterns = [
            rf'{label}\s*[:\-]?\s*(?:INR|Rs\.?)?\s*([\d,]+\.?\d*)',
            rf'{label}\s*(?:INR|Rs\.?)?\s*([\d,]+\.?\d*)',
        ]
        for pat in (patterns or default_patterns):
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return clean_amount(m.group(1))
        return None

    def _parse_cr_dr_amount(self, amount_str: str,
                             type_str: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Given amount + CR/DR indicator, return (debit, credit).
        """
        amt = clean_amount(amount_str)
        if amt is None:
            return None, None
        t = str(type_str).strip().upper()
        if t in ('CR', 'C', 'CREDIT', 'DEP', 'DEPOSIT'):
            return None, amt
        elif t in ('DR', 'D', 'DEBIT', 'WD', 'WITHDRAWAL', 'WITH'):
            return amt, None
        return None, None

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            'bank': '',
            'account_number': '',
            'account_holder': '',
            'opening_balance': None,
            'closing_balance': None,
            'statement_from': None,
            'statement_to': None,
            'currency': 'INR',
            'transactions': pd.DataFrame(),
            'raw_text': '',
            'errors': [],
        }

    def _log(self, msg: str):
        print(f"[{self.BANK_NAME}] {msg}")