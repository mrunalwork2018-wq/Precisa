import streamlit as st
import pdfplumber
from pdfplumber.utils.exceptions import PdfminerException
import pandas as pd
import numpy as np
import re
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
import logging
import io
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Extractors package — preferred over legacy parsers
from extractors import (
    get_extractor,
    AxisBankExtractor,
    AxisNeoBankExtractor,
    HDFCBankExtractor,
    ICICIBankExtractor,
    IndusIndBankExtractor,
    SBIBankExtractor,
)

# Map UI bank keys to extractor classes (for manual selection)
EXTRACTOR_MAP = {
    "AXIS": AxisBankExtractor,
    "AXIS_NEO": AxisNeoBankExtractor,
    "HDFC": HDFCBankExtractor,
    "ICICI": ICICIBankExtractor,
    "SBI": SBIBankExtractor,
    "INDUSIND": IndusIndBankExtractor,
}

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="PCREDInsight Pro - Bank Statement Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        max-width: 98%;
    }
    
    h1 {
        color: #1e3a8a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    
    h2 {
        color: #1e40af;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #3730a3;
        margin-top: 1.5rem;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 15px;
        padding: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: #f8fafc;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .dataframe {
        font-size: 0.9rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .badge-info {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .stExpander {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# CONFIGURATION & LOGGING
# ==================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_df_for_display(df: pd.DataFrame, formats: dict) -> pd.DataFrame:
    """
    Format DataFrame columns for display without using .style (avoids Python 3.12+
    pandas Styler bug: NameError: name '_imp' is not defined).
    """
    out = df.copy()
    for col, fmt in formats.items():
        if col not in out.columns:
            continue
        try:
            if "₹" in fmt and "," in fmt:
                out[col] = out[col].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x)
            elif "%" in fmt:
                out[col] = out[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and isinstance(x, (int, float)) else x)
            elif "," in fmt and ".0f" in fmt:
                out[col] = out[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else x)
            elif "," in fmt:
                out[col] = out[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x)
        except (TypeError, ValueError):
            pass
    return out


# ==================================================
# DATA CLASSES
# ==================================================
@dataclass
class BankConfig:
    """Bank-specific parsing configurations"""
    name: str
    date_pattern: str
    opening_balance_pattern: str
    min_columns: int
    date_format: str
    ifsc_pattern: str = r"IFSC.*?([A-Z]{4}0[A-Z0-9]{6})"
    micr_pattern: str = r"MICR.*?(\d{9})"


@dataclass
class AccountInfo:
    """Account holder information"""
    account_number: str = ""
    customer_id: str = ""
    customer_name: str = ""
    account_type: str = "Current Account"
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


@dataclass
class MonthlyMetrics:
    """Monthly financial metrics"""
    month: str
    opening_balance: float = 0.0
    closing_balance: float = 0.0
    debit_txns: float = 0.0
    credit_txns: float = 0.0
    debit_count: int = 0
    credit_count: int = 0
    min_balance: float = 0.0
    max_balance: float = 0.0
    avg_balance: float = 0.0
    min_eod_balance: float = 0.0
    max_eod_balance: float = 0.0
    avg_eod_balance: float = 0.0
    cash_deposit: float = 0.0
    cash_withdrawal: float = 0.0
    cash_deposit_count: int = 0
    cash_withdrawal_count: int = 0
    cheque_deposit: float = 0.0
    cheque_deposit_count: int = 0
    cheque_issue: float = 0.0
    cheque_issue_count: int = 0
    inward_cheque_bounce: int = 0
    outward_cheque_bounce: int = 0
    inward_bounce_pct: float = 0.0
    outward_bounce_pct: float = 0.0
    minimum_balance_charges: float = 0.0
    loan_credit: float = 0.0
    emi_debit: float = 0.0
    emi_count: int = 0
    salary_income: float = 0.0
    interest_paid: float = 0.0
    interest_received: float = 0.0
    bank_charges: float = 0.0
    bank_charges_count: int = 0
    penalty_charges: float = 0.0
    penalty_charges_count: int = 0
    internal_debit: float = 0.0
    internal_credit: float = 0.0
    internal_debit_count: int = 0
    internal_credit_count: int = 0
    self_deposit: float = 0.0
    self_withdrawal: float = 0.0
    net_debit: float = 0.0
    net_credit: float = 0.0
    net_debit_count: int = 0
    net_credit_count: int = 0
    foir_score: float = 0.0
    fixed_obligations: float = 0.0
    neft_returns: int = 0
    ecs_nach_count: int = 0
    ecs_nach_amount: float = 0.0
    balance_1st: float = 0.0
    balance_14th: float = 0.0
    balance_last: float = 0.0
    abb_1_14_30: float = 0.0
    daily_balance_change_pct: float = 0.0


@dataclass
class IrregularityFlags:
    """Irregularity detection flags"""
    suspicious_estatements: List[str] = field(default_factory=list)
    rtgs_below_2l: List[Dict] = field(default_factory=list)
    cheque_on_holiday: List[Dict] = field(default_factory=list)
    cash_on_holiday: List[Dict] = field(default_factory=list)
    more_cash_than_salary: bool = False
    round_tax_payments: List[Dict] = field(default_factory=list)
    equal_debit_credit: List[Dict] = field(default_factory=list)
    atm_above_20k: List[Dict] = field(default_factory=list)
    negative_computed_balance: List[Dict] = field(default_factory=list)
    balance_mismatch: List[Dict] = field(default_factory=list)
    immediate_big_debit_after_salary: List[Dict] = field(default_factory=list)
    unchanged_salary: bool = False
    circular_parties: List[str] = field(default_factory=list)


# ==================================================
# BANK STATEMENT PARSER
# ==================================================
class EnhancedBankStatementParser:
    """Production-grade parser with comprehensive extraction"""
    
    BANK_CONFIGS = {
        "AXIS": BankConfig(
            name="Axis Bank",
            date_pattern=r"\d{2}-\d{2}-\d{4}",
            opening_balance_pattern=r"OPENING BALANCE\s+([\d,]+\.\d{2})",
            min_columns=6,
            date_format="%d-%m-%Y"
        ),
        "SBI": BankConfig(
            name="State Bank of India",
            date_pattern=r"\d{2}/\d{2}/\d{4}",          # 20/07/2025
            opening_balance_pattern=r"Balance as on.*?([\d,]+\.\d{2})",
            min_columns=7,
            date_format="%d/%m/%Y",
            ifsc_pattern=r"IFS Code\s*:\s*([A-Z]{4}0[A-Z0-9]{6})",
            micr_pattern=r"MICR.*?(\d{9})",
        ),
        "HDFC": BankConfig(
            name="HDFC Bank",
            date_pattern=r"\d{2}/\d{2}/\d{2}",           # 08/04/25
            opening_balance_pattern=r"Opening Balance.*?([\d,]+\.?\d*)",
            min_columns=6,
            date_format="%d/%m/%y",
            ifsc_pattern=r"RTGS/NEFT IFSC\s*:\s*(HDFC\w+)",
            micr_pattern=r"MICR\s*:\s*(\d{9})",
        ),
        "ICICI": BankConfig(
            name="ICICI Bank",
            date_pattern=r"\d{2}/[A-Za-z]{3}/\d{2,4}",   # 01/Aug/2025
            opening_balance_pattern=r"Opening Balance.*?([\d,]+\.?\d*)",
            min_columns=8,
            date_format="%d/%b/%Y",
            ifsc_pattern=r"IFSC\s*:\s*([A-Z]{4}0[A-Z0-9]{6})",
            micr_pattern=r"MICR.*?(\d{9})",
        ),
        "INDUSIND": BankConfig(
            name="IndusInd Bank",
            date_pattern=r"\d{2}-[A-Za-z]{3}-\d{2,4}",   # 01-Apr-25 or 01-APR-2025
            opening_balance_pattern=r"(?:Opening|Available)\s*Balance.*?([\d,\-.]+)",
            min_columns=7,
            date_format="%d-%b-%y",
            ifsc_pattern=r"IFSC.*?([A-Z]{4}0[A-Z0-9]{6})",
            micr_pattern=r"MICR.*?(\d{9})",
        ),
        "AXIS_NEO": BankConfig(
            name="Axis Bank Neo",
            date_pattern=r"\d{2}/\d{2}/\d{4}",           # 01/06/2025
            opening_balance_pattern=r"Opening Balance\s*:?\s*([\d,]+\.?\d*)",
            min_columns=7,
            date_format="%d/%m/%Y",
            ifsc_pattern=r"IFSC Code\s*[:\s]*([A-Z]{4}0[A-Z0-9]{6})",
            micr_pattern=r"MICR Code\s*[:\s]*(\d{9})",
        )
    }
    
    # Category keywords
    CATEGORY_KEYWORDS = {
        "Salary": ["SALARY", "IFT/SALARY", "PAYROLL", "WAGE"],
        "UPI": ["UPI", "PHONEPE", "PAYTM", "GPAY", "BHIM"],
        "NEFT": ["NEFT"],
        "RTGS": ["RTGS"],
        "IMPS": ["IMPS"],
        "Cheque": ["CHQ", "CLG", "CHEQUE", "BRN-CLG"],
        "Cash Deposit": ["CASH DEP", "SAK/CASH DEP", "BY CASH DEPOSIT"],
        "Cash Withdrawal": ["CASH WDL", "SAK/CASH WDL", "CWDR", "ATM"],
        "EMI": ["EMI", "LOAN"],
        "Interest": ["INT.PD", "INT PAID", "INTEREST", "INT.COLL"],
        "Bank Charges": ["CHARGE", "FEE", "GST", "SERVICE", "ANNUAL"],
        "Penalty": ["PENALTY", "CHQ RTN", "CHEQUE RETURN", "BOUNCE"],
        "Transfer": ["TRF", "TRANSFER", "SELFFT", "TPFT"],
        "Utility": ["ELECTRIC", "WATER", "GAS", "MSEDCL"],
        "Tax": ["TAX", "TDS", "GST"],
        "Insurance": ["INSURANCE", "LIC", "POLICY"],
        "Investment": ["MUTUAL FUND", "SIP", "EQUITY", "STOCK"],
        "ECS/NACH": ["ECS", "NACH", "MANDATE"],
    }
    
    def __init__(self, bank_type: str = "AXIS"):
        self.config = self.BANK_CONFIGS.get(bank_type, self.BANK_CONFIGS["AXIS"])
        self.transactions = []
        self.account_info = AccountInfo()
        self.account_info.bank_name = self.config.name
        
    def extract_account_info(self, text: str):
        """Extract comprehensive account information"""
        patterns = {
            "account_number": r"Account No\s*:?\s*(\d+)",
            "customer_id": r"Customer ID\s*:?\s*(\d+)",
            "ifsc_code": r"IFSC Code\s*:?\s*([A-Z]{4}0[A-Z0-9]{6})",
            "micr_code": r"MICR Code\s*:?\s*(\d{9})",
            "customer_name": r"(?:MARUTI SALES CORPORATION|Customer Name|Account Holder)\s*:?\s*([A-Z\s&]+)",
            "email": r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "phone": r"(?:Mobile|Phone).*?(\d{10})",
            "pan": r"PAN\s*:?\s*([A-Z]{5}\d{4}[A-Z])",
            "branch_name": r"(?:Branch|BRANCH ADDRESS).*?([A-Z\s,]+(?:KARJAT|MUMBAI|DELHI))",
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                setattr(self.account_info, key, value)
        
        # Extract statement period
        period_match = re.search(
            r"period.*?(\d{2}-\d{2}-\d{4}).*?(\d{2}-\d{2}-\d{4})", 
            text, 
            re.IGNORECASE
        )
        if period_match:
            self.account_info.statement_from = period_match.group(1)
            self.account_info.statement_to = period_match.group(2)
        
        # Extract address
        address_match = re.search(
            r"(GROUND FLOOR.*?(?:\d{6}))", 
            text, 
            re.IGNORECASE | re.DOTALL
        )
        if address_match:
            self.account_info.address = address_match.group(1).strip()[:200]
    
    def clean_amount(self, value: str) -> float:
        """Clean and convert amounts"""
        if pd.isna(value) or value == "" or value is None:
            return 0.0
        
        cleaned = str(value).replace(",", "").replace("₹", "").replace(" ", "").strip()
        
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0
    
    def categorize_transaction(self, narration: str) -> str:
        """Categorize transaction based on narration"""
        narration_upper = narration.upper()
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in narration_upper for kw in keywords):
                return category
        
        return "Others"
    
    def extract_counterparty(self, narration: str) -> str:
        """Extract counterparty name from narration"""
        narration = narration.upper()
        
        # Patterns to extract counterparty
        patterns = [
            r"NEFT/[^/]+/([^/]+)/",
            r"UPI/[^/]+/[^/]+/([^/]+)/",
            r"RTGS/[^/]+/([^/]+)/",
            r"IMPS/[^/]+/[^/]+/([^/]+)/",
            r"CLG/\d+/\d+/[^/]+/([^/]+)",
            r"TRF/([^/]+)/",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, narration)
            if match:
                return match.group(1).strip()[:50]
        
        # If no pattern matches, return first 50 chars
        return narration[:50]
    
    def parse_pdf(self, pdf_file) -> Tuple[pd.DataFrame, AccountInfo]:
        """Main parsing function"""
        opening_balance = None
        all_text = ""
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    all_text += text + "\n"
                    
                    if opening_balance is None and page_num == 1:
                        match = re.search(self.config.opening_balance_pattern, text)
                        if match:
                            opening_balance = self.clean_amount(match.group(1))
                    
                    tables = page.extract_tables()
                    for table in tables:
                        self._process_table(table)
            
            self.extract_account_info(all_text)
            if opening_balance is not None:
                self.account_info.opening_balance = opening_balance
            
            df = pd.DataFrame(self.transactions)
            
            if df.empty:
                return df, self.account_info
            
            df = self._clean_dataframe(df)
            df = self._add_derived_columns(df, opening_balance)
            
            # Set additional account info
            if not df.empty:
                self.account_info.txn_start_date = df['Date'].min().strftime("%d-%m-%Y")
                self.account_info.txn_end_date = df['Date'].max().strftime("%d-%m-%Y")
                self.account_info.closing_balance = df.iloc[-1]['Balance']
                
                start_date = df['Date'].min()
                end_date = df['Date'].max()
                self.account_info.account_age_days = (end_date - start_date).days
            
            return df, self.account_info
            
        except Exception as e:
            logger.error(f"Parsing error: {str(e)}")
            raise
    
    def _process_table(self, table: List[List]):
        """Process table rows"""
        for row in table:
            if not row or len(row) < self.config.min_columns:
                continue
            
            date_str = str(row[0]).strip()
            
            if not re.match(self.config.date_pattern, date_str):
                continue
            
            narration = str(row[2]) if len(row) > 2 else ""
            debit = str(row[3]) if len(row) > 3 else "0"
            credit = str(row[4]) if len(row) > 4 else "0"
            balance = str(row[5]) if len(row) > 5 else "0"
            
            self.transactions.append({
                "_row_id": len(self.transactions),
                "Date": date_str,
                "Narration": narration,
                "Debit": debit,
                "Credit": credit,
                "Balance": balance,
                "Chq_No": str(row[1]) if len(row) > 1 else ""
            })
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame"""
        df["Date"] = pd.to_datetime(df["Date"], format=self.config.date_format, errors="coerce")
        
        for col in ["Debit", "Credit", "Balance"]:
            df[col] = df[col].apply(self.clean_amount)
        
        df = df.dropna(subset=["Date"])
        df = df[df["Balance"] >= 0]
        df = df.sort_values("_row_id").reset_index(drop=True)
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame, opening_balance: float) -> pd.DataFrame:
        """Add calculated columns"""
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
        df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
        df["Year"] = df["Date"].dt.year
        df["Day_of_Week"] = df["Date"].dt.day_name()
        df["Day_of_Month"] = df["Date"].dt.day
        df["Transaction_Type"] = df.apply(
            lambda x: "Credit" if x["Credit"] > 0 else "Debit", axis=1
        )
        df["Category"] = df["Narration"].apply(self.categorize_transaction)
        df["Counterparty"] = df["Narration"].apply(self.extract_counterparty)
        df["Calculated_Balance"] = opening_balance + (df["Credit"] - df["Debit"]).cumsum()
        df["Balance_Match"] = np.isclose(df["Balance"], df["Calculated_Balance"], atol=1)
        
        # UPI App detection
        df["UPI_App"] = df["Narration"].apply(self._detect_upi_app)
        
        # Tags
        df["Tags"] = df.apply(self._generate_tags, axis=1)
        
        return df
    
    def _detect_upi_app(self, narration: str) -> str:
        """Detect UPI app from narration"""
        narration_upper = narration.upper()
        
        upi_apps = {
            "PhonePe": ["PHONEPE"],
            "Paytm": ["PAYTM"],
            "Google Pay": ["GPAY", "GOOGLE PAY"],
            "BHIM": ["BHIM"],
            "Amazon Pay": ["AMAZON PAY"],
            "WhatsApp": ["WHATSAPP"],
        }
        
        for app, keywords in upi_apps.items():
            if any(kw in narration_upper for kw in keywords):
                return app
        
        if "UPI" in narration_upper:
            return "UPI-Other"
        
        return ""
    
    def _generate_tags(self, row) -> str:
        """Generate tags for transaction"""
        tags = []
        
        narration_upper = row['Narration'].upper()
        
        if "SALARY" in narration_upper:
            tags.append("Salary")
        
        if row['Category'] == "Cash Deposit" and row['Credit'] >= 100000:
            tags.append("High Value Cash Deposit")
        
        if row['Category'] == "Cash Withdrawal" and row['Debit'] >= 20000:
            tags.append("High Value Cash Withdrawal")
        
        if "BOUNCE" in narration_upper or "RETURN" in narration_upper:
            tags.append("Bounced")
        
        if row['Category'] == "EMI":
            tags.append("Loan Repayment")
        
        if not row['Balance_Match']:
            tags.append("Balance Mismatch")
        
        return ", ".join(tags)




# ==================================================
# SBI PARSER
# Columns: Txn Date | Value Date | Description | Ref No./Cheque No. |
#           Branch Code | Debit | Credit | Balance
# ==================================================
class SBIParser(EnhancedBankStatementParser):
    """
    SBI Statement Parser
    Table structure (0-indexed):
      0 → Txn Date        (dd/mm/yyyy)
      1 → Value Date      (dd/mm/yyyy)
      2 → Description     (narration)
      3 → Ref No./Cheque  
      4 → Branch Code
      5 → Debit
      6 → Credit
      7 → Balance
    Account meta:
      - Account Number  : "Account Number : <num>"
      - IFS Code        : "IFS Code : SBIN0001131"
      - Book/Avail Bal  : "Book Balance : 8678.36"
      - Opening Bal     : "Balance as on 20 Jul 2025 : 3,615.80"
    """

    def __init__(self):
        super().__init__(bank_type="SBI")                     # base init

    # ── account-info extraction ─────────────────────────────────────────────
    def extract_account_info(self, text: str):
        patterns = {
            "account_number": r"Account Number\s*:\s*(\d+)",
            "customer_name":  r"Name\s*:\s*([A-Z\s]+?)(?:\n|Currency)",
            "ifsc_code":      r"IFS Code\s*:\s*([A-Z]{4}0[A-Z0-9]{6})",
            "micr_code":      r"MICR.*?(\d{9})",
            "branch_name":    r"Branch\s*:\s*(.+?)(?:\n|$)",
            "email":          r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "phone":          r"(?:Mobile|Phone).*?(\d{10})",
            "pan":            r"PAN\s*:\s*([A-Z]{5}\d{4}[A-Z])",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                setattr(self.account_info, key, m.group(1).strip())

        # Statement period from header
        period = re.search(
            r"Account Statement from\s+(\d{2}\s+\w+\s+\d{4})\s+to\s+(\d{2}\s+\w+\s+\d{4})",
            text, re.IGNORECASE
        )
        if period:
            self.account_info.statement_from = period.group(1)
            self.account_info.statement_to   = period.group(2)

        # Opening balance ("Balance as on DD Mon YYYY : X,XXX.XX")
        ob = re.search(r"Balance as on.*?:\s*([\d,]+\.\d{2})", text, re.IGNORECASE)
        if ob:
            self.account_info.opening_balance = self.clean_amount(ob.group(1))

    # ── table row processor ─────────────────────────────────────────────────
    def _process_table(self, table):
        for row in table:
            if not row or len(row) < self.config.min_columns:
                continue

            date_str = str(row[0]).strip()
            if not re.match(self.config.date_pattern, date_str):
                continue

            # SBI has separate Debit / Credit columns
            self.transactions.append({
                "_row_id":   len(self.transactions),
                "Date":      date_str,
                "Narration": str(row[2]) if len(row) > 2 else "",
                "Chq_No":    str(row[3]) if len(row) > 3 else "",
                "Debit":     str(row[5]) if len(row) > 5 else "0",
                "Credit":    str(row[6]) if len(row) > 6 else "0",
                "Balance":   str(row[7]) if len(row) > 7 else "0",
            })

    def _clean_dataframe(self, df):
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        for col in ["Debit", "Credit", "Balance"]:
            df[col] = df[col].apply(self.clean_amount)
        df = df.dropna(subset=["Date"])
        df = df[df["Balance"] >= 0]
        df = df.sort_values("_row_id").reset_index(drop=True)
        return df


# ==================================================
# HDFC PARSER
# Columns: Date | Narration | Chq./Ref.No. | Value Dt |
#           Withdrawal Amt. | Deposit Amt. | Closing Balance
# ==================================================
class HDFCParser(EnhancedBankStatementParser):
    """
    HDFC Statement Parser
    Table structure (0-indexed):
      0 → Date            (dd/mm/yy)
      1 → Narration
      2 → Chq./Ref.No.
      3 → Value Dt        (dd/mm/yy)
      4 → Withdrawal Amt. (Debit)
      5 → Deposit Amt.    (Credit)
      6 → Closing Balance
    Account meta:
      - Account No  : "Account No  0086940700120"
      - IFSC        : "RTGS/NEFT IFSC  HDFC0000086"
      - MICR        : "MICR : 400240021"
      - Opening Bal : from Statement Summary row
    """

    def __init__(self):
        super().__init__(bank_type="HDFC")
        # self.config = BANK_CONFIGS["HDFC"]
        # self.account_info.bank_name = self.config.name

    def extract_account_info(self, text: str):
        patterns = {
            "account_number": r"Account No\.?\s*[:\s]*(\d{6,20})",
            "customer_name":  r"M/S\.?\s*([A-Z\s]+?)(?:\n|\r|\d{3})",
            "ifsc_code":      r"RTGS/NEFT IFSC\s*[:\s]*(HDFC\w+)",
            "micr_code":      r"MICR\s*[:\s]*(\d{9})",
            "email":          r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "phone":          r"(?:Phone|Mobile)\s*[:\s]*(\d{10,})",
            "pan":            r"PAN\s*[:\s]*([A-Z]{5}\d{4}[A-Z])",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                setattr(self.account_info, key, m.group(1).strip())

        # Statement period "From : 01/04/2025  To : 30/04/2025"
        period = re.search(
            r"From\s*:\s*(\d{2}/\d{2}/\d{4})\s+To\s*:\s*(\d{2}/\d{2}/\d{4})",
            text, re.IGNORECASE
        )
        if period:
            self.account_info.statement_from = period.group(1)
            self.account_info.statement_to   = period.group(2)

        # Opening balance from Statement Summary
        ob = re.search(r"Opening Balance\s*[\n\r\s]*(\d[\d,\.]*)", text, re.IGNORECASE)
        if ob:
            self.account_info.opening_balance = self.clean_amount(ob.group(1))

    def _process_table(self, table):
        for row in table:
            if not row or len(row) < self.config.min_columns:
                continue

            date_str = str(row[0]).strip()
            if not re.match(self.config.date_pattern, date_str):
                continue

            self.transactions.append({
                "_row_id":   len(self.transactions),
                "Date":      date_str,
                "Narration": str(row[1]) if len(row) > 1 else "",
                "Chq_No":    str(row[2]) if len(row) > 2 else "",
                "Debit":     str(row[4]) if len(row) > 4 else "0",   # Withdrawal
                "Credit":    str(row[5]) if len(row) > 5 else "0",   # Deposit
                "Balance":   str(row[6]) if len(row) > 6 else "0",
            })

    def _clean_dataframe(self, df):
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce")
        for col in ["Debit", "Credit", "Balance"]:
            df[col] = df[col].apply(self.clean_amount)
        df = df.dropna(subset=["Date"])
        df = df[df["Balance"] >= 0]
        df = df.sort_values("_row_id").reset_index(drop=True)
        return df


# ==================================================
# ICICI PARSER
# Columns: Sl No | Tran Date | Value Date | Transaction Posted Date |
#           Cheque no / Txn Ref No | Transaction Remarks |
#           Withdrawal (Dr) | Deposit (Cr) | Balance
# ==================================================
class ICICIParser(EnhancedBankStatementParser):
    """
    ICICI Statement Parser
    Table structure (0-indexed):
      0 → Sl No
      1 → Tran Date           (01/Aug/2025 or 01-08-2025)
      2 → Value Date
      3 → Transaction Posted Date
      4 → Cheque no / Txn Ref No
      5 → Transaction Remarks (narration)
      6 → Withdrawal (Dr)
      7 → Deposit (Cr)
      8 → Balance
    Account meta:
      - A/C No     : "A/C No: 150105500142"
      - IFSC       : "IFSC Code : ICIC0001501"
      - Name       : "Name: SURYAKSHA DAIRY"
    """

    # ICICI uses both "01/Aug/2025" and "01-08-2025" date styles
    _DATE_FORMATS = ["%d/%b/%Y", "%d/%b/%y", "%d-%m-%Y"]

    def __init__(self):
        super().__init__(bank_type="ICICI")
        # self.config = BANK_CONFIGS["ICICI"]
        # self.account_info.bank_name = self.config.name

    def extract_account_info(self, text: str):
        patterns = {
            "account_number": r"A/C No\.?\s*[:\s]*(\d+)",
            "customer_name":  r"Name\s*[:\s]*([A-Z\s]+?)(?:\n|A/C)",
            "ifsc_code":      r"IFSC\s*(?:Code)?\s*[:\s]*([A-Z]{4}0[A-Z0-9]{6})",
            "micr_code":      r"MICR.*?(\d{9})",
            "branch_name":    r"A/C Branch\s*[:\s]*(.+?)(?:\n|$)",
            "email":          r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "pan":            r"PAN\s*[:\s]*([A-Z]{5}\d{4}[A-Z])",
            "customer_id":    r"Cust ID\s*[:\s]*(\d+)",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                setattr(self.account_info, key, m.group(1).strip())

        # Statement period "From: 01/08/2025 To: 31/08/2025"
        period = re.search(
            r"(?:Transaction Period|From)\s*[:\s]*(\d{2}/\d{2}/\d{4})\s+(?:To|-)\s+(\d{2}/\d{2}/\d{4})",
            text, re.IGNORECASE
        )
        if period:
            self.account_info.statement_from = period.group(1)
            self.account_info.statement_to   = period.group(2)

    def _parse_date(self, date_str: str):
        """Try multiple date formats."""
        for fmt in self._DATE_FORMATS:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except Exception:
                pass
        return pd.NaT

    def _process_table(self, table):
        for row in table:
            if not row or len(row) < self.config.min_columns:
                continue

            # col 1 is Tran Date; col 0 is Sl No (numeric)
            date_str = str(row[1]).strip()
            # Allow "01/Aug/2025" or "01-08-2025"
            if not re.match(r"\d{2}[/\-][A-Za-z0-9]{2,3}[/\-]\d{2,4}", date_str):
                continue

            self.transactions.append({
                "_row_id":   len(self.transactions),
                "Date":      date_str,
                "Narration": str(row[5]) if len(row) > 5 else "",
                "Chq_No":    str(row[4]) if len(row) > 4 else "",
                "Debit":     str(row[6]) if len(row) > 6 else "0",   # Withdrawal
                "Credit":    str(row[7]) if len(row) > 7 else "0",   # Deposit
                "Balance":   str(row[8]) if len(row) > 8 else "0",
            })

    def _clean_dataframe(self, df):
        df["Date"] = df["Date"].apply(self._parse_date)
        for col in ["Debit", "Credit", "Balance"]:
            df[col] = df[col].apply(self.clean_amount)
        df = df.dropna(subset=["Date"])
        df = df[df["Balance"] >= 0]
        df = df.sort_values("_row_id").reset_index(drop=True)
        return df


# ==================================================
# INDUSIND PARSER
# Columns: Bank Reference | Value Date | Transaction Date & Time |
#           Type | Payment Narration | Debit | Credit | Available Balance
# ==================================================
class IndusIndParser(EnhancedBankStatementParser):
    """
    IndusInd Statement Parser
    Table structure (0-indexed):
      0 → Bank Reference
      1 → Value Date          (01 Apr 2025  or  31 Mar 2025)
      2 → Transaction Date & Time  ('01-APR-25 06:59:44')
      3 → Type                (Debit / Credit)
      4 → Payment Narration
      5 → Debit
      6 → Credit
      7 → Available Balance
    Account meta:
      - Account No  : "Account No : 650014044804"
      - Customer Name: "CARGOSOL LOGISTICS PVT LTD"
    """

    _DATE_FORMATS = ["%d %b %Y", "%d-%b-%y", "%d-%b-%Y", "%d %B %Y"]

    def __init__(self):
        super().__init__(bank_type="INDUSIND")
        # self.config = BANK_CONFIGS["INDUSIND"]
        # self.account_info.bank_name = self.config.name

    def extract_account_info(self, text: str):
        patterns = {
            "account_number": r"Account No\s*[:\s]*(\d+)",
            "customer_name":  r"Customer Name.*?\n([A-Z\s]+?)(?:\n|Account)",
            "ifsc_code":      r"IFSC.*?([A-Z]{4}0[A-Z0-9]{6})",
            "micr_code":      r"MICR.*?(\d{9})",
            "email":          r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "pan":            r"PAN\s*[:\s]*([A-Z]{5}\d{4}[A-Z])",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                setattr(self.account_info, key, m.group(1).strip())

        # "From Date: 01-Apr-25  To Date: 31-May-25"
        period = re.search(
            r"From Date\s*[:\s]*(\S+)\s+To Date\s*[:\s]*(\S+)",
            text, re.IGNORECASE
        )
        if period:
            self.account_info.statement_from = period.group(1)
            self.account_info.statement_to   = period.group(2)

    def _parse_date(self, date_str: str):
        date_str = date_str.strip()
        for fmt in self._DATE_FORMATS:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except Exception:
                pass
        try:
            return pd.to_datetime(date_str, infer_datetime_format=True)
        except Exception:
            return pd.NaT

    def _process_table(self, table):
        for row in table:
            if not row or len(row) < self.config.min_columns:
                continue

            # Value Date is col 1; format "01 Apr 2025" or "31 Mar 2025"
            date_str = str(row[1]).strip()
            if not re.match(r"\d{1,2}[\s\-][A-Za-z]{3}[\s\-]\d{2,4}", date_str):
                continue

            txn_type = str(row[3]).strip().lower()   # "Debit" or "Credit"
            amount   = str(row[5]).strip() if len(row) > 5 else "0"

            # IndusInd has separate Debit / Credit columns
            debit_val  = str(row[5]) if len(row) > 5 else "0"
            credit_val = str(row[6]) if len(row) > 6 else "0"

            self.transactions.append({
                "_row_id":   len(self.transactions),
                "Date":      date_str,
                "Narration": str(row[4]) if len(row) > 4 else "",
                "Chq_No":    str(row[0]) if len(row) > 0 else "",   # Bank Reference
                "Debit":     debit_val,
                "Credit":    credit_val,
                "Balance":   str(row[7]) if len(row) > 7 else "0",
            })

    def _clean_dataframe(self, df):
        df["Date"] = df["Date"].apply(self._parse_date)
        for col in ["Debit", "Credit", "Balance"]:
            df[col] = df[col].apply(self.clean_amount)
        df = df.dropna(subset=["Date"])
        # IndusInd may show negative balance (overdraft accounts)
        df = df.sort_values("_row_id").reset_index(drop=True)
        return df


# ==================================================
# AXIS NEO PARSER
# Columns: S.NO | Transaction Date | Value Date | Particulars |
#           Amount(INR) | Debit/Credit | Balance(INR) |
#           Cheque Number | Branch Name(SOL)
# NOTE: Amount + DR/CR flag instead of separate Debit/Credit columns
# ==================================================
class AxisNeoParser(EnhancedBankStatementParser):
    """
    Axis Bank Neo / MCA Registered Entity Statement Parser
    Table structure (0-indexed):
      0 → S.NO
      1 → Transaction Date  (dd/mm/yyyy)
      2 → Value Date
      3 → Particulars       (narration)
      4 → Amount(INR)
      5 → Debit/Credit      ("CR" / "DR")
      6 → Balance(INR)
      7 → Cheque Number
      8 → Branch Name(SOL)
    Account meta:
      - Account No   : "Axis Bank Account No : 923020060299587"
      - IFSC Code    : "IFSC Code : UTIB0000246"
      - Customer No  : "Customer No : 956889166"
      - Opening Bal  : "Opening Balance: INR 12,304.86"
    """

    def __init__(self):
        super().__init__(bank_type="AXIS_NEO")
        # self.config = BANK_CONFIGS["AXIS_NEO"]
        # self.account_info.bank_name = self.config.name

    def extract_account_info(self, text: str):
        patterns = {
            "account_number": r"(?:Account No|Axis Bank Account No)\s*[:\s]*(\d+)",
            "customer_name":  r"([A-Z\s]+(?:PRIVATE LIMITED|PVT\.?\s*LTD|LLP|LIMITED))\s*(?:\n|Joint)",
            "customer_id":    r"Customer No\s*[:\s]*(\d+)",
            "ifsc_code":      r"IFSC Code\s*[:\s]*([A-Z]{4}0[A-Z0-9]{6})",
            "micr_code":      r"MICR Code\s*[:\s]*(\d{9})",
            "email":          r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "pan":            r"PAN\s*[:\s]*([A-Z]{5}\d{4}[A-Z])",
            "branch_name":    r"POWAI|ANDHERI|MUMBAI",   # fallback branch from SOL column
        }
        for key, pattern in patterns.items():
            if key == "branch_name":
                continue
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                setattr(self.account_info, key, m.group(1).strip())

        # "Statement of Axis Bank Account No : XXXXXXX for the period ( From : 01/06/2025 To : 04/12/2025 )"
        period = re.search(
            r"[Ff]rom\s*[:\s]*(\d{2}/\d{2}/\d{4})\s+[Tt]o\s*[:\s]*(\d{2}/\d{2}/\d{4})",
            text, re.IGNORECASE
        )
        if period:
            self.account_info.statement_from = period.group(1)
            self.account_info.statement_to   = period.group(2)

        # Opening balance "Opening Balance: INR 12,304.86"
        ob = re.search(r"Opening Balance\s*[:\s]*(?:INR)?\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
        if ob:
            self.account_info.opening_balance = self.clean_amount(ob.group(1))

    def _process_table(self, table):
        for row in table:
            if not row or len(row) < self.config.min_columns:
                continue

            date_str = str(row[1]).strip()
            if not re.match(r"\d{2}/\d{2}/\d{4}", date_str):
                continue

            amount   = self.clean_amount(str(row[4]) if len(row) > 4 else "0")
            dr_cr    = str(row[5]).strip().upper() if len(row) > 5 else "CR"

            # Convert single-amount + DR/CR flag → separate debit/credit
            debit  = amount if "DR" in dr_cr else 0.0
            credit = amount if "CR" in dr_cr else 0.0

            self.transactions.append({
                "_row_id":   len(self.transactions),
                "Date":      date_str,
                "Narration": str(row[3]) if len(row) > 3 else "",
                "Chq_No":    str(row[7]) if len(row) > 7 else "",
                "Debit":     str(debit),
                "Credit":    str(credit),
                "Balance":   str(row[6]) if len(row) > 6 else "0",
            })

    def _clean_dataframe(self, df):
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        for col in ["Debit", "Credit", "Balance"]:
            df[col] = df[col].apply(self.clean_amount)
        df = df.dropna(subset=["Date"])
        df = df.sort_values("_row_id").reset_index(drop=True)
        return df


def get_parser(bank_type: str):
    """
    Factory function that returns the correct parser instance.
    Legacy fallback — prefer run_extractor() when using extractors package.
    """
    parsers = {
        "AXIS":      lambda: EnhancedBankStatementParser("AXIS"),
        "HDFC":      HDFCParser,
        "ICICI":     ICICIParser,
        "SBI":       SBIParser,
        "INDUSIND":  IndusIndParser,
        "AXIS_NEO":  AxisNeoParser,
    }
    cls = parsers.get(bank_type.upper())
    if cls is None:
        raise ValueError(f"Unsupported bank: {bank_type}")
    return cls()


def run_extractor(pdf_path: str, bank_key: Optional[str] = None) -> Tuple[pd.DataFrame, AccountInfo]:
    """
    Run bank extractor and convert output to (df, account_info) format
    expected by ComprehensiveStatementAnalyzer.
    """
    if bank_key and str(bank_key).upper() != "AUTO":
        extractor_cls = EXTRACTOR_MAP.get(str(bank_key).upper())
        if extractor_cls is None:
            raise ValueError(f"Unsupported bank: {bank_key}")
        extractor = extractor_cls()
    else:
        extractor = get_extractor(pdf_path)
        if extractor is None:
            raise ValueError("Could not auto-detect bank. Please select bank manually.")
    result = extractor.run(pdf_path)
    return _convert_extractor_result(result)


def _convert_extractor_result(result: dict) -> Tuple[pd.DataFrame, AccountInfo]:
    """Convert extractor result to analyzer format (df, AccountInfo)."""
    df = result.get("transactions", pd.DataFrame())
    if df.empty:
        info = AccountInfo(bank_name=result.get("bank", ""))
        info.account_number = str(result.get("account_number", ""))
        info.customer_name = str(result.get("account_holder", ""))
        info.opening_balance = float(result.get("opening_balance") or 0.0)
        info.closing_balance = float(result.get("closing_balance") or 0.0)
        info.statement_from = str(result.get("statement_from", ""))
        info.statement_to = str(result.get("statement_to", ""))
        return df, info

    # Map extractor columns to analyzer columns
    col_map = {
        "transaction_date": "Date",
        "narration": "Narration",
        "debit": "Debit",
        "credit": "Credit",
        "balance": "Balance",
        "reference": "Chq_No",
        "row_id": "_row_id",
    }
    out = pd.DataFrame()
    for src, dst in col_map.items():
        if src in df.columns:
            out[dst] = df[src].values
        else:
            out[dst] = None if dst == "_row_id" else (0.0 if dst in ("Debit", "Credit", "Balance") else "")
    if "Chq_No" in out.columns and "tran_id" in df.columns:
        empty_chq = out["Chq_No"].isna() | (out["Chq_No"].astype(str).str.strip() == "")
        if empty_chq.all():
            out["Chq_No"] = df["tran_id"].fillna("").astype(str).values

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["Debit", "Credit", "Balance"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["Chq_No"] = out["Chq_No"].fillna("").astype(str)
    out["Narration"] = out["Narration"].fillna("").astype(str)
    if "_row_id" not in out.columns or out["_row_id"].isna().all():
        out["_row_id"] = range(len(out))
    out = out.dropna(subset=["Date"])
    out = out.sort_values("_row_id").reset_index(drop=True)

    opening = float(result.get("opening_balance") or 0.0)
    if not out.empty and opening == 0 and "Balance" in out.columns:
        first_bal = out.iloc[0]["Balance"]
        first_cr = out.iloc[0].get("Credit", 0) or 0
        first_dr = out.iloc[0].get("Debit", 0) or 0
        if first_cr > 0:
            opening = first_bal - first_cr
        elif first_dr > 0:
            opening = first_bal + first_dr

    parser = EnhancedBankStatementParser("AXIS")
    out = _add_derived_columns(out, opening, parser)

    info = AccountInfo(bank_name=result.get("bank", ""))
    info.account_number = str(result.get("account_number", ""))
    info.customer_name = str(result.get("account_holder", ""))
    info.opening_balance = opening
    info.closing_balance = float(out.iloc[-1]["Balance"]) if len(out) > 0 else 0.0
    info.statement_from = str(result.get("statement_from", ""))
    info.statement_to = str(result.get("statement_to", ""))
    if not out.empty:
        info.txn_start_date = out["Date"].min().strftime("%d-%m-%Y")
        info.txn_end_date = out["Date"].max().strftime("%d-%m-%Y")
        info.account_age_days = (out["Date"].max() - out["Date"].min()).days
    ex_info = result.get("account_info")
    if ex_info is not None:
        for attr in ["ifsc_code", "micr_code", "branch_name", "customer_id", "account_type"]:
            if hasattr(ex_info, attr):
                setattr(info, attr, str(getattr(ex_info, attr) or ""))

    return out, info


def _add_derived_columns(df: pd.DataFrame, opening_balance: float, parser: "EnhancedBankStatementParser") -> pd.DataFrame:
    """Add Category, Counterparty, UPI_App, Tags, etc."""
    df = df.copy()
    df["Narration"] = df["Narration"].fillna("").astype(str)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Year"] = df["Date"].dt.year
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["Day_of_Month"] = df["Date"].dt.day
    df["Transaction_Type"] = df.apply(lambda x: "Credit" if x["Credit"] > 0 else "Debit", axis=1)
    df["Category"] = df["Narration"].apply(parser.categorize_transaction)
    df["Counterparty"] = df["Narration"].apply(parser.extract_counterparty)
    df["Calculated_Balance"] = opening_balance + (df["Credit"] - df["Debit"]).cumsum()
    df["Balance_Match"] = np.isclose(df["Balance"].fillna(0), df["Calculated_Balance"], atol=1)
    df["UPI_App"] = df["Narration"].apply(parser._detect_upi_app)
    df["Tags"] = df.apply(parser._generate_tags, axis=1)
    df["Tags"] = df["Tags"].fillna("").astype(str)
    return df




# ==================================================
# ADVANCED ANALYZER
# ==================================================
class ComprehensiveStatementAnalyzer:
    """Comprehensive analytics engine"""
    
    def __init__(self, df: pd.DataFrame, account_info: AccountInfo):
        self.df = df
        self.account_info = account_info
        self.monthly_metrics = {}
        self.irregularities = IrregularityFlags()
        
    def analyze_all(self):
        """Run all analyses"""
        self._calculate_monthly_metrics()
        self._detect_irregularities()
        self._analyze_loans()
        self._analyze_counterparties()
        self._analyze_circular_transactions()
        self._analyze_recurring_payments()
    
    def _calculate_monthly_metrics(self):
        """Calculate comprehensive monthly metrics"""
        prev_closing = self.account_info.opening_balance
        
        for month, g in self.df.groupby("Month", sort=True):
            g = g.sort_values(["Date", "_row_id"])
            
            metrics = MonthlyMetrics(month=month)
            
            # Basic metrics
            metrics.opening_balance = prev_closing
            metrics.closing_balance = g.iloc[-1]["Balance"]
            metrics.debit_txns = g["Debit"].sum()
            metrics.credit_txns = g["Credit"].sum()
            metrics.debit_count = (g["Debit"] > 0).sum()
            metrics.credit_count = (g["Credit"] > 0).sum()
            metrics.min_balance = g["Balance"].min()
            metrics.max_balance = g["Balance"].max()
            metrics.avg_balance = g["Balance"].mean()
            
            # EOD metrics (simplified - using balance)
            metrics.min_eod_balance = g["Balance"].min()
            metrics.max_eod_balance = g["Balance"].max()
            metrics.avg_eod_balance = g["Balance"].mean()
            
            # Cash transactions
            cash_deposits = g[g["Category"] == "Cash Deposit"]
            cash_withdrawals = g[g["Category"] == "Cash Withdrawal"]
            metrics.cash_deposit = cash_deposits["Credit"].sum()
            metrics.cash_withdrawal = cash_withdrawals["Debit"].sum()
            metrics.cash_deposit_count = len(cash_deposits)
            metrics.cash_withdrawal_count = len(cash_withdrawals)
            
            # Cheque transactions
            cheques = g[g["Category"] == "Cheque"]
            metrics.cheque_deposit = cheques["Credit"].sum()
            metrics.cheque_deposit_count = (cheques["Credit"] > 0).sum()
            metrics.cheque_issue = cheques["Debit"].sum()
            metrics.cheque_issue_count = (cheques["Debit"] > 0).sum()
            
            # Bounces
            bounces = g[g["Tags"].str.contains("Bounced", na=False)]
            metrics.inward_cheque_bounce = ((bounces["Credit"] > 0) | (bounces["Debit"] < 0)).sum()
            metrics.outward_cheque_bounce = ((bounces["Debit"] > 0) | (bounces["Credit"] < 0)).sum()
            
            if metrics.cheque_deposit_count > 0:
                metrics.inward_bounce_pct = (metrics.inward_cheque_bounce / metrics.cheque_deposit_count) * 100
            if metrics.cheque_issue_count > 0:
                metrics.outward_bounce_pct = (metrics.outward_cheque_bounce / metrics.cheque_issue_count) * 100
            
            # Charges
            charges = g[g["Category"] == "Bank Charges"]
            metrics.bank_charges = charges["Debit"].sum()
            metrics.bank_charges_count = len(charges)
            
            penalties = g[g["Category"] == "Penalty"]
            metrics.penalty_charges = penalties["Debit"].sum()
            metrics.penalty_charges_count = len(penalties)
            
            min_bal_charges = g[g["Narration"].str.contains("MINIMUM BALANCE", case=False, na=False)]
            metrics.minimum_balance_charges = min_bal_charges["Debit"].sum()
            
            # Loan & EMI
            emi = g[g["Category"] == "EMI"]
            metrics.emi_debit = emi["Debit"].sum()
            metrics.emi_count = len(emi)
            metrics.loan_credit = g[g["Narration"].str.contains("LOAN", case=False, na=False)]["Credit"].sum()
            
            # Salary
            salary = g[g["Tags"].str.contains("Salary", na=False)]
            metrics.salary_income = salary["Credit"].sum()
            
            # Interest
            interest_paid = g[g["Narration"].str.contains("INT|INTEREST", case=False, na=False)]
            metrics.interest_paid = interest_paid["Debit"].sum()
            metrics.interest_received = interest_paid["Credit"].sum()
            
            # Internal transfers
            internal = g[g["Category"] == "Transfer"]
            metrics.internal_debit = internal["Debit"].sum()
            metrics.internal_credit = internal["Credit"].sum()
            metrics.internal_debit_count = (internal["Debit"] > 0).sum()
            metrics.internal_credit_count = (internal["Credit"] > 0).sum()
            
            # Self transactions
            self_txns = g[g["Narration"].str.contains("SELF", case=False, na=False)]
            metrics.self_deposit = self_txns["Credit"].sum()
            metrics.self_withdrawal = self_txns["Debit"].sum()
            
            # Net transactions
            metrics.net_debit = metrics.debit_txns - metrics.internal_debit
            metrics.net_credit = metrics.credit_txns - metrics.internal_credit
            metrics.net_debit_count = metrics.debit_count - metrics.internal_debit_count
            metrics.net_credit_count = metrics.credit_count - metrics.internal_credit_count
            
            # FOIR (simplified)
            metrics.fixed_obligations = metrics.emi_debit
            if metrics.salary_income > 0:
                metrics.foir_score = (metrics.fixed_obligations / metrics.salary_income) * 100
            
            # ECS/NACH
            ecs_nach = g[g["Category"] == "ECS/NACH"]
            metrics.ecs_nach_count = len(ecs_nach)
            metrics.ecs_nach_amount = ecs_nach["Debit"].sum()
            
            # Balance on specific dates
            day_1 = g[g["Day_of_Month"] == 1]
            if not day_1.empty:
                metrics.balance_1st = day_1.iloc[-1]["Balance"]
            
            day_14 = g[g["Day_of_Month"] == 14]
            if not day_14.empty:
                metrics.balance_14th = day_14.iloc[-1]["Balance"]
            
            last_day = g[g["Day_of_Month"] == g["Day_of_Month"].max()]
            if not last_day.empty:
                metrics.balance_last = last_day.iloc[-1]["Balance"]
            
            # ABB (Average of balances on 1st, 14th, last day)
            balances = [b for b in [metrics.balance_1st, metrics.balance_14th, metrics.balance_last] if b > 0]
            if balances:
                metrics.abb_1_14_30 = sum(balances) / len(balances)
            
            # Daily balance change
            if len(g) > 1:
                balance_changes = g["Balance"].diff().abs()
                metrics.daily_balance_change_pct = (balance_changes.mean() / g["Balance"].mean()) * 100 if g["Balance"].mean() > 0 else 0
            
            # NEFT Returns
            neft_returns = g[g["Narration"].str.contains("NEFT.*RETURN", case=False, na=False)]
            metrics.neft_returns = len(neft_returns)
            
            self.monthly_metrics[month] = metrics
            prev_closing = metrics.closing_balance
    
    def _detect_irregularities(self):
        """Detect irregularities"""
        # RTGS below 2 lakhs
        rtgs_txns = self.df[self.df["Category"] == "RTGS"]
        rtgs_below_2l = rtgs_txns[(rtgs_txns["Debit"] > 0) & (rtgs_txns["Debit"] < 200000)]
        self.irregularities.rtgs_below_2l = rtgs_below_2l.to_dict('records')
        
        # ATM withdrawals above 20k
        atm_high = self.df[(self.df["Category"] == "Cash Withdrawal") & (self.df["Debit"] > 20000)]
        self.irregularities.atm_above_20k = atm_high.to_dict('records')
        
        # Round tax payments
        tax_txns = self.df[self.df["Category"] == "Tax"]
        round_tax = tax_txns[(tax_txns["Debit"] % 1000 == 0) & (tax_txns["Debit"] > 0)]
        self.irregularities.round_tax_payments = round_tax.to_dict('records')
        
        # Balance mismatches
        mismatches = self.df[~self.df["Balance_Match"]]
        self.irregularities.balance_mismatch = mismatches.to_dict('records')
        
        # Negative computed balance
        negative_balance = self.df[self.df["Calculated_Balance"] < 0]
        self.irregularities.negative_computed_balance = negative_balance.to_dict('records')
        
        # Immediate big debit after salary
        salary_txns = self.df[self.df["Tags"].str.contains("Salary", na=False)]
        for idx in salary_txns.index:
            if idx + 1 < len(self.df):
                next_txn = self.df.iloc[idx + 1]
                if next_txn["Debit"] > 0.5 * salary_txns.loc[idx, "Credit"]:
                    self.irregularities.immediate_big_debit_after_salary.append({
                        "Date": next_txn["Date"],
                        "Salary": salary_txns.loc[idx, "Credit"],
                        "Debit": next_txn["Debit"],
                        "Narration": next_txn["Narration"]
                    })
        
        # Unchanged salary
        if len(salary_txns) > 1:
            salary_amounts = salary_txns["Credit"].unique()
            if len(salary_amounts) == 1:
                self.irregularities.unchanged_salary = True
        
        # More cash deposits than salary
        total_cash_deposits = self.df[self.df["Category"] == "Cash Deposit"]["Credit"].sum()
        total_salary = salary_txns["Credit"].sum()
        if total_cash_deposits > total_salary:
            self.irregularities.more_cash_than_salary = True
    
    def _analyze_loans(self):
        """Analyze loan transactions"""
        # This is a simplified version
        # In production, you'd track individual loans
        self.loan_summary = {
            "total_emi": self.df[self.df["Category"] == "EMI"]["Debit"].sum(),
            "emi_count": len(self.df[self.df["Category"] == "EMI"]),
            "loan_credits": self.df[self.df["Narration"].str.contains("LOAN", case=False, na=False)]["Credit"].sum()
        }
    
    def _analyze_counterparties(self):
        """Analyze counterparties"""
        self.counterparty_summary = self.df.groupby("Counterparty").agg({
            "Credit": "sum",
            "Debit": "sum",
            "Date": "count"
        }).rename(columns={"Date": "Count"}).sort_values("Debit", ascending=False)
    
    def _analyze_circular_transactions(self):
        """Detect circular transactions"""
        # Parties in both credits and debits
        credit_parties = set(self.df[self.df["Credit"] > 0]["Counterparty"].unique())
        debit_parties = set(self.df[self.df["Debit"] > 0]["Counterparty"].unique())
        
        circular = credit_parties.intersection(debit_parties)
        self.irregularities.circular_parties = list(circular)
    
    def _analyze_recurring_payments(self):
        """Detect recurring payments"""
        # Group by counterparty and look for regular patterns
        self.recurring_payments = []
        
        for counterparty, g in self.df.groupby("Counterparty"):
            if len(g) < 3:
                continue
            
            debit_txns = g[g["Debit"] > 0].sort_values("Date")
            if len(debit_txns) < 3:
                continue
            
            # Check if amounts are similar
            amounts = debit_txns["Debit"].values

            if len(amounts) < 3 or np.std(amounts) == 0:
                continue

            if np.std(amounts) / np.mean(amounts) < 0.1:  # Low variation
                # Check intervals
                dates = debit_txns["Date"].values
                intervals = [int((dates[i+1] - dates[i]) / np.timedelta64(1, 'D')) 
                             for i in range(len(dates)-1)]
                
                if not intervals:
                    continue
                
                avg_interval = np.mean(intervals)
                if 25 <= avg_interval <= 35:  # Monthly
                    self.recurring_payments.append({
                        "Counterparty": counterparty,
                        "Amount": np.mean(amounts),
                        "Count": len(debit_txns),
                        "Interval": "Monthly",
                        "Start_Date": pd.Timestamp(dates[0]).strftime("%d-%m-%Y"),
                        "End_Date": pd.Timestamp(dates[-1]).strftime("%d-%m-%Y")
                    })
    
    def get_monthly_summary_df(self) -> pd.DataFrame:
        """Get monthly summary as DataFrame"""
        rows = []
        for month, metrics in self.monthly_metrics.items():
            rows.append({
                "Month": month,
                "Opening Balance": metrics.opening_balance,
                "Debit Txns": metrics.debit_txns,
                "Credit Txns": metrics.credit_txns,
                "Closing Balance": metrics.closing_balance,
                "Debit Count": metrics.debit_count,
                "Credit Count": metrics.credit_count,
                "Min Balance": metrics.min_balance,
                "Max Balance": metrics.max_balance,
                "Avg Balance": metrics.avg_balance,
                "Cash Deposit": metrics.cash_deposit,
                "Cash Withdrawal": metrics.cash_withdrawal,
                "Cheque Deposit": metrics.cheque_deposit,
                "Cheque Issue": metrics.cheque_issue,
                "I/W Bounce": metrics.inward_cheque_bounce,
                "O/W Bounce": metrics.outward_cheque_bounce,
                "EMI": metrics.emi_debit,
                "Salary": metrics.salary_income,
                "Bank Charges": metrics.bank_charges,
                "FOIR %": metrics.foir_score,
                "Balance 1st": metrics.balance_1st,
                "Balance 14th": metrics.balance_14th,
                "Balance Last": metrics.balance_last,
            })
        
        return pd.DataFrame(rows)
    
    def get_cashflow_summary(self) -> pd.DataFrame:
        """Get cashflow summary"""
        summary = self.df.groupby("Month").agg({
            "Credit": "sum",
            "Debit": "sum",
            "Balance": "mean",
        }).rename(columns={
            "Credit": "Inflow",
            "Debit": "Outflow",
            "Balance": "Monthly Avg Balance"
        })
        
        summary["Net Cash Flow"] = summary["Inflow"] - summary["Outflow"]
        
        # Add counts
        credit_counts = self.df[self.df["Credit"] > 0].groupby("Month").size()
        debit_counts = self.df[self.df["Debit"] > 0].groupby("Month").size()
        
        summary["Inflow Txn Count"] = credit_counts
        summary["Outflow Txn Count"] = debit_counts
        
        return summary.reset_index()


# ==================================================
# MAIN APP
# ==================================================
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("💰 PCREDInsight Pro")
        st.markdown("""
        <p style='text-align: center; color: #64748b; font-size: 1.1rem;'>
            Bank Statement Analysis Platform<br>
            <span style='font-size: 0.9rem;'>Complete Financial Intelligence Suite</span>
        </p>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        bank_type = st.selectbox(
            "Select Bank",
            ["AUTO", "AXIS", "AXIS_NEO", "HDFC", "ICICI", "SBI", "INDUSIND"],
            format_func=lambda x: "Auto-detect" if x == "AUTO" else x.replace("_", " "),
            help="Auto-detect reads PDF to guess bank, or choose manually"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Modules")
        modules = [
            "✅ Account Details",
            "✅ Monthly Metrics (40+ KPIs)",
            "✅ Cashflow Analysis",
            "✅ Irregularity Detection",
            "✅ Loan Analysis",
            "✅ Counterparty Analysis",
            "✅ Recurring Payments",
            "✅ Circular Transactions",
            "✅ Category Breakdown",
            "✅ Cheque Analysis",
        ]
        for module in modules:
            st.markdown(f"<p style='margin: 0.2rem 0; font-size: 0.85rem;'>{module}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File upload
    st.markdown("## 📂 Upload Bank Statement")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Upload PDF Statement",
            type=["pdf"],
            help="Upload your bank statement in PDF format"
        )
    
    if not uploaded_file:
        st.info("👆 Please upload your bank statement to begin analysis")
        return
    
    # Process file (extractors require file path)
    with st.spinner("🔄 Processing statement... This may take a minute"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name
            try:
                df, account_info = run_extractor(pdf_path, bank_key=bank_type)
            finally:
                try:
                    os.unlink(pdf_path)
                except OSError:
                    pass

            if df.empty:
                st.error("❌ No transactions found. Please check the PDF format or try selecting the bank manually.")
                return

            analyzer = ComprehensiveStatementAnalyzer(df, account_info)
            analyzer.analyze_all()

        except PdfminerException as e:
            orig = e.args[0] if e.args else None
            is_password = orig and type(orig).__name__ == "PDFPasswordIncorrect"
            if is_password:
                st.error(
                    "❌ **Password-protected PDF detected.** Please remove the password and upload an "
                    "unprotected copy, or use 'Print to PDF' to create an unsecured version."
                )
            else:
                st.error("❌ PDF processing error. The file may be corrupted or in an unsupported format.")
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            return
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            return
    
    bank_display = account_info.bank_name or "Statement"
    st.success(f"✅ Successfully extracted {len(df):,} transactions from **{bank_display}**!")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋 Account Info",
        "📊 Monthly Metrics",
        "💰 Cashflow",
        "🔍 Irregularities",
        "📈 Analytics",
        "💳 Transactions",
        "🏦 Loans & EMIs",
        "📥 Export"
    ])
    
    # ==================================================
    # TAB 1: ACCOUNT INFO
    # ==================================================
    with tab1:
        st.markdown("## 📋 Account Holder Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Bank Details")
            st.write(f"**Bank Name:** {account_info.bank_name}")
            st.write(f"**Account Number:** {account_info.account_number}")
            st.write(f"**Account Type:** {account_info.account_type}")
            st.write(f"**IFSC Code:** {account_info.ifsc_code}")
            st.write(f"**MICR Code:** {account_info.micr_code}")
            st.write(f"**Customer ID:** {account_info.customer_id}")
        
        with col2:
            st.markdown("### Personal Details")
            st.write(f"**Name:** {account_info.customer_name}")
            st.write(f"**Email:** {account_info.email}")
            st.write(f"**Phone:** {account_info.phone}")
            st.write(f"**PAN:** {account_info.pan}")
        
        with col3:
            st.markdown("### Statement Period")
            st.write(f"**Statement From:** {account_info.statement_from}")
            st.write(f"**Statement To:** {account_info.statement_to}")
            st.write(f"**Transaction Start:** {account_info.txn_start_date}")
            st.write(f"**Transaction End:** {account_info.txn_end_date}")
            st.write(f"**Account Age:** {account_info.account_age_days} days")
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Opening Balance", f"₹ {account_info.opening_balance:,.2f}")
        col2.metric("Closing Balance", f"₹ {account_info.closing_balance:,.2f}")
        col3.metric("Net Change", f"₹ {account_info.closing_balance - account_info.opening_balance:,.2f}")
        col4.metric("Total Transactions", f"{len(df):,}")
    
    # ==================================================
    # TAB 2: MONTHLY METRICS
    # ==================================================
    with tab2:
        st.markdown("## 📊 Comprehensive Monthly Metrics")
        
        monthly_df = analyzer.get_monthly_summary_df()
        
        # Display DataFrame
        # Separate columns properly
        amount_cols = [
            "Opening Balance", "Debit Txns", "Credit Txns", "Closing Balance",
            "Min Balance", "Max Balance", "Avg Balance",
            "Cash Deposit", "Cash Withdrawal",
            "Cheque Deposit", "Cheque Issue",
            "EMI", "Salary", "Bank Charges",
            "Balance 1st", "Balance 14th", "Balance Last"
        ]

        count_cols = [
            "Debit Count", "Credit Count",
            "I/W Bounce", "O/W Bounce"
        ]

        percent_cols = ["FOIR %"]

        # Apply correct formatting (avoid .style to prevent Python 3.12+ pandas bug)
        fmt_map = (
            {col: "₹{:,.2f}" for col in amount_cols if col in monthly_df.columns} |
            {col: "{:,.0f}" for col in count_cols if col in monthly_df.columns} |
            {col: "{:.2f}%" for col in percent_cols if col in monthly_df.columns}
        )
        st.dataframe(
            _format_df_for_display(monthly_df, fmt_map),
            width="stretch",
            height=400
        )
        
        # Visualizations
        st.markdown("### 📈 Visual Analysis")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Balance Trend", "Cash Flow", "EMI vs Salary", "Cheque Analysis"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Balance trend
        fig.add_trace(
            go.Scatter(x=monthly_df["Month"], y=monthly_df["Avg Balance"], name="Avg Balance", mode="lines+markers"),
            row=1, col=1
        )
        
        # Cash flow
        fig.add_trace(
            go.Bar(x=monthly_df["Month"], y=monthly_df["Credit Txns"], name="Credits"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=monthly_df["Month"], y=monthly_df["Debit Txns"], name="Debits"),
            row=1, col=2
        )
        
        # EMI vs Salary
        fig.add_trace(
            go.Bar(x=monthly_df["Month"], y=monthly_df["Salary"], name="Salary"),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=monthly_df["Month"], y=monthly_df["EMI"], name="EMI"),
            row=2, col=1
        )
        
        # Cheque analysis
        fig.add_trace(
            go.Bar(x=monthly_df["Month"], y=monthly_df["Cheque Deposit"], name="Cheque Deposits"),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(x=monthly_df["Month"], y=monthly_df["Cheque Issue"], name="Cheque Issues"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, width="stretch")
    
    # ==================================================
    # TAB 3: CASHFLOW
    # ==================================================
    with tab3:
        st.markdown("## 💰 Cashflow Analysis")
        
        cashflow_df = analyzer.get_cashflow_summary()
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_inflow = cashflow_df["Inflow"].sum()
        total_outflow = cashflow_df["Outflow"].sum()
        net_cashflow = total_inflow - total_outflow
        avg_balance = cashflow_df["Monthly Avg Balance"].mean()
        
        col1.metric("Total Inflow", f"₹ {total_inflow:,.2f}")
        col2.metric("Total Outflow", f"₹ {total_outflow:,.2f}")
        col3.metric("Net Cashflow", f"₹ {net_cashflow:,.2f}")
        col4.metric("Avg Monthly Balance", f"₹ {avg_balance:,.2f}")
        
        st.markdown("---")
        
        # Cashflow table
        st.markdown("### 📊 Monthly Cashflow")
        st.dataframe(
            _format_df_for_display(cashflow_df, {
                "Inflow": "₹{:,.2f}",
                "Outflow": "₹{:,.2f}",
                "Net Cash Flow": "₹{:,.2f}",
                "Monthly Avg Balance": "₹{:,.2f}"
            }),
            width="stretch"
        )
        
        # Waterfall chart
        st.markdown("### 💦 Cashflow Waterfall")
        fig = go.Figure(go.Waterfall(
            x=cashflow_df["Month"],
            y=cashflow_df["Net Cash Flow"],
            text=cashflow_df["Net Cash Flow"].apply(lambda x: f"₹{x:,.0f}"),
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#ef4444"}},
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
        
        # Category breakdown
        st.markdown("### 📂 Category Analysis")
        
        category_summary = df.groupby("Category").agg({
            "Credit": "sum",
            "Debit": "sum",
            "Date": "count"
        }).rename(columns={"Date": "Count"})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Credit Categories**")
            credit_cats = category_summary[category_summary["Credit"] > 0].sort_values("Credit", ascending=False)
            fig = px.pie(
                values=credit_cats["Credit"],
                names=credit_cats.index,
                title="Credits by Category"
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("**Debit Categories**")
            debit_cats = category_summary[category_summary["Debit"] > 0].sort_values("Debit", ascending=False)
            fig = px.pie(
                values=debit_cats["Debit"],
                names=debit_cats.index,
                title="Debits by Category"
            )
            st.plotly_chart(fig, width="stretch")
    
    # ==================================================
    # TAB 4: IRREGULARITIES
    # ==================================================
    with tab4:
        st.markdown("## 🔍 Irregularity Detection")
        
        irregularities_found = False
        
        # RTGS below 2L
        if analyzer.irregularities.rtgs_below_2l:
            irregularities_found = True
            st.warning(f"⚠️ **RTGS Transactions Below ₹2 Lakhs:** {len(analyzer.irregularities.rtgs_below_2l)} found")
            with st.expander("View Details"):
                st.dataframe(pd.DataFrame(analyzer.irregularities.rtgs_below_2l))
        
        # ATM above 20k
        if analyzer.irregularities.atm_above_20k:
            irregularities_found = True
            st.warning(f"⚠️ **ATM Withdrawals Above ₹20,000:** {len(analyzer.irregularities.atm_above_20k)} found")
            with st.expander("View Details"):
                st.dataframe(pd.DataFrame(analyzer.irregularities.atm_above_20k))
        
        # Balance mismatches
        if analyzer.irregularities.balance_mismatch:
            irregularities_found = True
            st.error(f"❌ **Balance Mismatches:** {len(analyzer.irregularities.balance_mismatch)} found")
            with st.expander("View Details"):
                st.dataframe(pd.DataFrame(analyzer.irregularities.balance_mismatch))
        
        # Immediate debit after salary
        if analyzer.irregularities.immediate_big_debit_after_salary:
            irregularities_found = True
            st.warning(f"⚠️ **Large Debits After Salary:** {len(analyzer.irregularities.immediate_big_debit_after_salary)} found")
            with st.expander("View Details"):
                st.dataframe(pd.DataFrame(analyzer.irregularities.immediate_big_debit_after_salary))
        
        # Circular transactions
        if analyzer.irregularities.circular_parties:
            irregularities_found = True
            st.warning(f"⚠️ **Circular Transaction Parties:** {len(analyzer.irregularities.circular_parties)} found")
            with st.expander("View Details"):
                st.write(analyzer.irregularities.circular_parties)
        
        # More cash than salary
        if analyzer.irregularities.more_cash_than_salary:
            irregularities_found = True
            st.warning("⚠️ **More Cash Deposits than Salary Income**")
        
        # Unchanged salary
        if analyzer.irregularities.unchanged_salary:
            irregularities_found = True
            st.info("ℹ️ **Salary Amount Unchanged Throughout Period**")
        
        # Round tax payments
        if analyzer.irregularities.round_tax_payments:
            irregularities_found = True
            st.warning(f"⚠️ **Round Figure Tax Payments:** {len(analyzer.irregularities.round_tax_payments)} found")
            with st.expander("View Details"):
                st.dataframe(pd.DataFrame(analyzer.irregularities.round_tax_payments))
        
        if not irregularities_found:
            st.success("✅ No major irregularities detected!")
    
    # ==================================================
    # TAB 5: ANALYTICS
    # ==================================================
    with tab5:
        st.markdown("## 📈 Advanced Analytics")
        
        # Counterparty analysis
        st.markdown("### 👥 Top Counterparties")
        
        top_credit = analyzer.counterparty_summary.nlargest(10, "Credit")
        top_debit = analyzer.counterparty_summary.nlargest(10, "Debit")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Credit Counterparties**")
            st.dataframe(
                _format_df_for_display(top_credit, {"Credit": "₹{:,.2f}", "Debit": "₹{:,.2f}"}),
                width="stretch"
            )
        
        with col2:
            st.markdown("**Top Debit Counterparties**")
            st.dataframe(
                _format_df_for_display(top_debit, {"Credit": "₹{:,.2f}", "Debit": "₹{:,.2f}"}),
                width="stretch"
            )
        
        # Recurring payments
        st.markdown("### 🔄 Recurring Payments")
        if analyzer.recurring_payments:
            recurring_df = pd.DataFrame(analyzer.recurring_payments)
            st.dataframe(
                _format_df_for_display(recurring_df, {"Amount": "₹{:,.2f}"}),
                width="stretch"
            )
        else:
            st.info("No recurring payments detected")
        
        # Transaction patterns
        st.markdown("### 📅 Transaction Patterns")
        
        day_of_week_analysis = df.groupby("Day_of_Week").agg({
            "Credit": ["sum", "count"],
            "Debit": ["sum", "count"]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=day_of_week_analysis.index,
            y=day_of_week_analysis[("Credit", "count")],
            name="Credit Txns"
        ))
        fig.add_trace(go.Bar(
            x=day_of_week_analysis.index,
            y=day_of_week_analysis[("Debit", "count")],
            name="Debit Txns"
        ))
        fig.update_layout(title="Transactions by Day of Week", barmode="group", height=400)
        st.plotly_chart(fig, width="stretch")
    
    # ==================================================
    # TAB 6: TRANSACTIONS
    # ==================================================
    with tab6:
        st.markdown("## 💳 All Transactions")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_categories = st.multiselect(
                "Filter by Category",
                options=df["Category"].unique(),
                default=None
            )
        
        with col2:
            min_amount = st.number_input("Min Amount", value=0.0, step=1000.0)
        
        with col3:
            max_amount = st.number_input("Max Amount", value=float(df[["Credit", "Debit"]].max().max()), step=1000.0)
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_categories:
            filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
        
        filtered_df = filtered_df[
            ((filtered_df["Credit"] >= min_amount) & (filtered_df["Credit"] <= max_amount)) |
            ((filtered_df["Debit"] >= min_amount) & (filtered_df["Debit"] <= max_amount))
        ]
        
        st.markdown(f"**Showing {len(filtered_df):,} transactions**")
        
        # Display
        display_cols = ["Date", "Narration", "Category", "Counterparty", "Credit", "Debit", "Balance", "Tags"]
        st.dataframe(
            _format_df_for_display(filtered_df[display_cols], {
                "Credit": "₹{:,.2f}",
                "Debit": "₹{:,.2f}",
                "Balance": "₹{:,.2f}"
            }),
            width="stretch",
            height=600
        )
        
        # Cheque analysis
        st.markdown("---")
        st.markdown("### 📝 Cheque Analysis")
        
        cheque_txns = df[df["Category"] == "Cheque"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Inward Cheques (Credits)**")
            inward_cheques = cheque_txns[cheque_txns["Credit"] > 0]
            st.dataframe(
                _format_df_for_display(inward_cheques[["Date", "Narration", "Chq_No", "Credit"]], {"Credit": "₹{:,.2f}"}),
                width="stretch"
            )
        
        with col2:
            st.markdown("**Outward Cheques (Debits)**")
            outward_cheques = cheque_txns[cheque_txns["Debit"] > 0]
            st.dataframe(
                _format_df_for_display(outward_cheques[["Date", "Narration", "Chq_No", "Debit"]], {"Debit": "₹{:,.2f}"}),
                width="stretch"
            )
    
    # ==================================================
    # TAB 7: LOANS & EMIs
    # ==================================================
    with tab7:
        st.markdown("## 🏦 Loan & EMI Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total EMI Paid", f"₹ {analyzer.loan_summary['total_emi']:,.2f}")
        col2.metric("EMI Transactions", analyzer.loan_summary['emi_count'])
        col3.metric("Loan Credits", f"₹ {analyzer.loan_summary['loan_credits']:,.2f}")
        
        st.markdown("---")
        
        # EMI transactions
        st.markdown("### 💳 EMI Transactions")
        emi_txns = df[df["Category"] == "EMI"]
        
        if not emi_txns.empty:
            st.dataframe(
                _format_df_for_display(emi_txns[["Date", "Narration", "Counterparty", "Debit"]], {"Debit": "₹{:,.2f}"}),
                width="stretch"
            )
            
            # Monthly EMI chart
            monthly_emi = emi_txns.groupby("Month")["Debit"].sum()
            fig = px.bar(
                x=monthly_emi.index,
                y=monthly_emi.values,
                title="Monthly EMI Payments",
                labels={"x": "Month", "y": "EMI Amount (₹)"}
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No EMI transactions found")
    
    # ==================================================
    # TAB 8: EXPORT
    # ==================================================
    with tab8:
        st.markdown("## 📥 Export Reports")
        
        st.markdown("""
        Download comprehensive analysis reports in various formats:
        - **Excel Workbook**: Multiple sheets with all analysis
        - **CSV Files**: Individual datasets
        - **Summary Report**: Text summary
        """)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        # Excel export
        with col1:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='All Transactions', index=False)
                monthly_df.to_excel(writer, sheet_name='Monthly Metrics', index=False)
                cashflow_df.to_excel(writer, sheet_name='Cashflow', index=False)
                analyzer.counterparty_summary.to_excel(writer, sheet_name='Counterparties')
                
                if analyzer.recurring_payments:
                    pd.DataFrame(analyzer.recurring_payments).to_excel(writer, sheet_name='Recurring Payments', index=False)
            
            st.download_button(
                label="📊 Download Excel Report",
                data=buffer.getvalue(),
                file_name=f"bank_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.ms-excel",
                width="stretch"
            )
        
        # CSV export
        with col2:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Transactions CSV",
                data=csv,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch"
            )
        
        # Summary report
        with col3:
            summary_text = f"""
BANK STATEMENT ANALYSIS REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ACCOUNT INFORMATION
{'='*50}
Bank Name: {account_info.bank_name}
Account Number: {account_info.account_number}
Account Type: {account_info.account_type}
Customer Name: {account_info.customer_name}
IFSC Code: {account_info.ifsc_code}

STATEMENT PERIOD
{'='*50}
From: {account_info.statement_from}
To: {account_info.statement_to}
Transaction Period: {account_info.txn_start_date} to {account_info.txn_end_date}
Account Age: {account_info.account_age_days} days

FINANCIAL SUMMARY
{'='*50}
Opening Balance: ₹{account_info.opening_balance:,.2f}
Closing Balance: ₹{account_info.closing_balance:,.2f}
Net Change: ₹{account_info.closing_balance - account_info.opening_balance:,.2f}

Total Credits: ₹{total_inflow:,.2f}
Total Debits: ₹{total_outflow:,.2f}
Net Cashflow: ₹{net_cashflow:,.2f}

Total Transactions: {len(df):,}
Credit Transactions: {len(df[df['Credit'] > 0]):,}
Debit Transactions: {len(df[df['Debit'] > 0]):,}

LOAN & EMI
{'='*50}
Total EMI Paid: ₹{analyzer.loan_summary['total_emi']:,.2f}
EMI Transactions: {analyzer.loan_summary['emi_count']}
Loan Credits: ₹{analyzer.loan_summary['loan_credits']:,.2f}

IRREGULARITIES
{'='*50}
RTGS below ₹2L: {len(analyzer.irregularities.rtgs_below_2l)}
ATM above ₹20k: {len(analyzer.irregularities.atm_above_20k)}
Balance Mismatches: {len(analyzer.irregularities.balance_mismatch)}
Circular Parties: {len(analyzer.irregularities.circular_parties)}

{'='*50}
End of Report
"""
            
            st.download_button(
                label="📝 Download Summary",
                data=summary_text,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                width="stretch"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 1rem 0;'>
        <p style='margin: 0;'>Made with ❤️ by PCREDInsight Pro</p>
        <p style='margin: 0; font-size: 0.9rem;'>© 2026 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
