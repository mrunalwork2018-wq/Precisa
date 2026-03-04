"""
Bank Statement Extractors Package
===================================
Supported banks:
  - Axis Bank       (NEO for Corporates format)
  - HDFC Bank       (Statement of Account format)
  - ICICI Bank      (Detailed Statement format)
  - IndusInd Bank   (Account Statement format)
  - SBI             (Account Statement format)

Usage:
    from extractors import get_extractor, ALL_EXTRACTORS

    extractor = get_extractor(pdf_path)
    result = extractor.run(pdf_path)
    df = result['transactions']
"""

from .axis_bank import AxisBankExtractor
from .axis_neo_bank import AxisBankExtractor as AxisNeoBankExtractor
from .hdfc_bank import HDFCBankExtractor, AccountInfo as HDFCAccountInfo, HDFC_CONFIG
from .icici_bank import ICICIBankExtractor, AccountInfo as ICICIAccountInfo, ICICI_CONFIG
from .indusind_bank import IndusIndBankExtractor
from .sbi_bank import SBIBankExtractor

# Ordered list — more specific detectors (Axis SA) before generic (Axis NEO)
ALL_EXTRACTORS = [
    AxisBankExtractor,
    AxisNeoBankExtractor,
    HDFCBankExtractor,
    ICICIBankExtractor,
    IndusIndBankExtractor,
    SBIBankExtractor,
]


def get_extractor(pdf_path: str):
    """
    Auto-detect which bank extractor to use for a given PDF.
    Reads only the first page text for detection.
    Returns an instantiated extractor, or None if no match found.
    """
    import pdfplumber

    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page_text = ''
            for page in pdf.pages[:2]:   # check first 2 pages
                t = page.extract_text()
                if t:
                    first_page_text += '\n' + t
    except Exception as e:
        print(f"[AutoDetect] Error reading PDF: {e}")
        return None

    for ExtractorClass in ALL_EXTRACTORS:
        extractor = ExtractorClass()
        if extractor.detect(first_page_text):
            print(f"[AutoDetect] Matched: {extractor.BANK_NAME}")
            return extractor

    print("[AutoDetect] No matching extractor found. Check if bank is supported.")
    return None


__all__ = [
    'AxisBankExtractor',
    'AxisNeoBankExtractor',
    'HDFCBankExtractor',
    'HDFCAccountInfo',
    'HDFC_CONFIG',
    'ICICIBankExtractor',
    'ICICIAccountInfo',
    'ICICI_CONFIG',
    'IndusIndBankExtractor',
    'SBIBankExtractor',
    'ALL_EXTRACTORS',
    'get_extractor',
]