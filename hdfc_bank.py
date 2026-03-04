import re
import pdfplumber
import pandas as pd
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────────
# BankConfig / AccountInfo  (unchanged)
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


HDFC_CONFIG = BankConfig(
    name="HDFC Bank",
    date_pattern=r"\b(\d{2}/\d{2}/\d{2})\b",
    opening_balance_pattern=(
        r"Opening\s+Bal(?:ance)?\s*"
        r"(?:Dr\s+Count\s+Cr\s+Count\s+Debits\s+Credits\s+Closing\s+Bal\s*)?"
        r"([\d,]+\.\d{2})"
    ),
    min_columns=7,
    date_format="%d/%m/%y",
    ifsc_pattern=r"\b(HDFC\d{7})\b",
    micr_pattern=r"\bMICR\s*[:\-]?\s*(\d{9})\b",
)


@dataclass
class AccountInfo:
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fix_sci_notation(text: str) -> str:
    """
    pdfplumber.extract_text() converts long digit strings (ref numbers,
    account numbers inside narrations) into scientific notation.
    e.g.  3304220240426785  →  3.304220240426785e+15
    This function reverses that conversion so downstream code sees correct strings.
    """
    def _replace(m):
        try:
            val = Decimal(m.group(0))
            if val == val.to_integral_value():
                return str(int(val))
        except Exception:
            pass
        return m.group(0)
    return re.sub(r'\d+\.\d+[eE][+\-]\d+', _replace, text)


def _clean_amount(text: str) -> Optional[float]:
    if not text:
        return None
    cleaned = re.sub(r"[,\s]", "", str(text).strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_date(text: str, fmt: str = "%d/%m/%y") -> Optional[date]:
    if not text:
        return None
    for f in [fmt, "%d/%m/%Y", "%d-%m-%Y", "%d-%m-%y"]:
        try:
            return datetime.strptime(text.strip(), f).date()
        except ValueError:
            continue
    return None


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "row_id", "transaction_date", "value_date", "narration",
        "debit", "credit", "balance", "txn_type", "reference",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = None
    df = df[required].reset_index(drop=True)
    df["row_id"] = range(1, len(df) + 1)
    return df


def _infer_txn_type(debit, credit, prev_bal, curr_bal) -> str:
    if debit is not None and credit is None:
        return "DR"
    if credit is not None and debit is None:
        return "CR"
    if debit is not None and credit is not None:
        return "DR" if debit > credit else "CR"
    if prev_bal is not None and curr_bal is not None:
        return "DR" if curr_bal < prev_bal else "CR"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Line-classification regexes  (used by the state-machine parser)
# ──────────────────────────────────────────────────────────────────────────────

# Matches the START of a transaction: "DD/MM/YY <rest of narration>"
_TXN_START_RE = re.compile(r"^\s*(\d{2}/\d{2}/\d{2,4})\s+(.+)$")

# Matches a ref-number + value-date line: "<ref> DD/MM/YY"
# The ref is any non-space token; value date must be dd/mm/yy or dd/mm/yyyy
_REF_VDATE_RE = re.compile(r"^\s*(\S+)\s+(\d{2}/\d{2}/\d{2,4})\s*$")

# Matches a standalone amount (possibly negative for OD balances)
_AMT_ONLY_RE  = re.compile(r"^\s*-?([\d,]+\.\d{2})\s*$")

# HDFC first line: ends with [ValueDate] Withdrawal Deposit Closing
# 2 amounts: ValueDate Amt1 Amt2  (Amt1=withdrawal or deposit, Amt2=closing)
# 3 amounts: ValueDate Withdrawal Deposit Closing
_TAIL_3AMT_RE = re.compile(
    r"\s+(\d{2}/\d{2}/\d{2,4})\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})\s*$"
)
_TAIL_2AMT_RE = re.compile(
    r"\s+(\d{2}/\d{2}/\d{2,4})\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})\s*$"
)

# Page/header content to strip from narration (partial matches)
_PAGE_POLLUTION = re.compile(
    r"(?:HDFCBANKLIMITED|PageNo\.\s*:\s*\d+|Statementofaccount|"
    r"AccountBranch\s*:|CustID\s*:|AccountNo\s*:|A/COpenDate|"
    r"thisstatement\.|hdfcbank\.com|Stateaccountbranch|"
    r"Contentsofthisstatement|RegisteredOfficeAddress|"
    r"Generated\s+On|Phoneno\.\s*:|ODLimit|Currency|ProductCode|"
    r"AccountStatus|BranchCode|JOINTHOLDERS|Nomination|"
    r"Address\s*:|City\s*:|State\s*:|Email\s*:|RTGS|MICR).*",
    re.I,
)


def _parse_first_line_rest(rest: str) -> Dict[str, Any]:
    """
    Parse "Narration Ref ValueDate Amt1 Amt2 [Amt3]" from first line rest.
    Returns dict with: narration, ref, value_date_str, amounts, debit, credit.
    For 3 amounts: explicit withdrawal=debit, deposit=credit.
    For 2 amounts: amounts=[flow, closing], debit/credit inferred in _flush.
    """
    out = {
        "narration": rest,
        "ref": "",
        "value_date_str": "",
        "amounts": [],
        "debit": None,
        "credit": None,
    }
    m = _TAIL_3AMT_RE.search(rest)
    if m:
        out["value_date_str"] = m.group(1)
        wdr = _clean_amount(m.group(2))
        dep = _clean_amount(m.group(3))
        closing = _clean_amount(m.group(4))
        out["amounts"] = [closing]
        out["debit"] = wdr if wdr and wdr > 0 else None
        out["credit"] = dep if dep and dep > 0 else None
        head = rest[: m.start()].rstrip()
    else:
        m2 = _TAIL_2AMT_RE.search(rest)
        if m2:
            out["value_date_str"] = m2.group(1)
            amt1 = m2.group(2)
            amt2 = m2.group(3)
            out["amounts"] = [amt1, amt2]
            head = rest[: m2.start()].rstrip()
        else:
            return out

    parts = head.split()
    ref_parts = []
    while parts:
        tok = parts[-1]
        if "." in tok or "_" in tok:
            break
        if tok.replace(",", "").replace("-", "").isdigit() and len(tok) >= 10:
            ref_parts.insert(0, parts.pop())
        elif re.match(r"^[A-Z]?[\d\-]+$", tok, re.I) and len(tok) >= 8:
            ref_parts.insert(0, parts.pop())
        else:
            break
        if len(ref_parts) >= 3:
            break
    out["ref"] = " ".join(ref_parts) if ref_parts else ""
    out["narration"] = " ".join(parts) if parts else head
    return out

# Lines that are definitely page-header/footer — never transaction data
_SKIP_RE = re.compile(
    r"^(?:Date|Narration|Withdrawal|Deposit|Balance|Page\s+No|"
    r"Statement\s+(?:From|of\s+account)|Opening|Closing|HDFC\s+BANK|M/S|"
    r"Generated|STATEMENT\s+SUMMARY|Cust(?:omer)?\s+ID|Account\s+No|"
    r"JOINT|Nomination|A/C\s+Open|Account\s+Branch|Address|City|State|"
    r"Phone|OD\s+Limit|Currency|Email|RTGS|MICR|Branch\s+Code|"
    r"Product\s+Code|Imperia|\*Closing|Contents|Registered|"
    r"State\s+account|To\s*:|hdfcbank\.com|This\s+is\s+a\s+computer)",
    re.I,
)


# ──────────────────────────────────────────────────────────────────────────────
# State-machine block parser
# ──────────────────────────────────────────────────────────────────────────────

def _parse_hdfc_blocks(raw_text: str) -> pd.DataFrame:
    """
    HDFC extract_text() lays every transaction across multiple lines:

      Line 1  DD/MM/YY <narration_part1>              ← TXN_START
      Line 2  <ref_no> DD/MM/YY                       ← REF + VALUE_DATE
      Line 3  <amount1>                                ← withdrawal OR deposit
      Line 4  <amount2>                                ← closing balance
      Line 5+ <narration_continuation ...>             ← optional overflow

    We walk lines with a 4-state machine:
      state=1  → waiting for ref+value_date
      state=2  → waiting for first amount
      state=3  → waiting for second amount (closing balance)
      state=4  → row complete, collecting narration overflow
    """
    rows: List[Dict] = []

    def _flush(cur):
        if cur is None:
            return
        amounts = cur["amounts"]
        full_narr = (cur["narration"] + " " + " ".join(cur["narr_extra"])).strip()
        full_narr = re.sub(r"\s{2,}", " ", full_narr)
        full_narr = _PAGE_POLLUTION.sub("", full_narr).strip()
        full_narr = re.sub(r"\s{2,}", " ", full_narr)

        bal = _clean_amount(amounts[-1]) if amounts else None
        prev_bal = rows[-1]["balance"] if rows else None

        if cur.get("parsed_debit") is not None or cur.get("parsed_credit") is not None:
            debit = cur.get("parsed_debit")
            credit = cur.get("parsed_credit")
        elif len(amounts) >= 2:
            amt1 = _clean_amount(amounts[0])
            if prev_bal is not None and bal is not None:
                if bal > prev_bal:
                    debit, credit = None, amt1
                else:
                    debit, credit = amt1, None
            else:
                narr_up = full_narr.upper()
                if any(k in narr_up for k in ("NEFT CR", "NEFT CR-", "ACH C-", "ACHCR",
                                               "INTEREST CREDIT", "FT - CR", "FT-CR",
                                               "CHQ DEP", "CHQDEP", "HDFC BANK DIV",
                                               "HDFCBANKDIV", "AUTO_REDEMPTION",
                                               "QUARTERLYINTERESTCREDIT")):
                    debit, credit = None, amt1
                else:
                    debit, credit = amt1, None
        else:
            debit, credit = None, None

        txn_type = _infer_txn_type(debit, credit, prev_bal, bal)

        rows.append({
            "transaction_date": _parse_date(cur["date_str"]),
            "value_date":       _parse_date(cur["value_date_str"]),
            "narration":        full_narr,
            "reference":        cur["ref"],
            "debit":            debit,
            "credit":           credit,
            "balance":          bal,
            "txn_type":         txn_type,
        })

    cur = None

    for raw_line in raw_text.split("\n"):
        # Fix scientific notation BEFORE any other processing
        line     = _fix_sci_notation(raw_line)
        stripped = line.strip()

        if not stripped:
            continue
        if _SKIP_RE.match(stripped):
            continue

        # ── New transaction start ──────────────────────────────────────────
        m = _TXN_START_RE.match(line)
        if m:
            _flush(cur)
            parsed = _parse_first_line_rest(m.group(2).strip())
            cur = {
                "date_str":       m.group(1),
                "narration":      parsed["narration"],
                "ref":            parsed["ref"],
                "value_date_str": parsed["value_date_str"],
                "amounts":        list(parsed["amounts"]),
                "narr_extra":     [],
                "state":          1,
                "parsed_debit":   parsed.get("debit"),
                "parsed_credit":  parsed.get("credit"),
            }
            continue

        if cur is None:
            continue

        # ── State 1: expect ref + value_date ──────────────────────────────
        if cur["state"] == 1:
            m_ref = _REF_VDATE_RE.match(line)
            if m_ref:
                cur["ref"]            = m_ref.group(1)
                cur["value_date_str"] = m_ref.group(2)
                cur["state"]          = 2
                continue
            # Could be an amount already (edge case)
            if _AMT_ONLY_RE.match(stripped):
                cur["amounts"].append(stripped)
                cur["state"] = 3
                continue
            # Extra narration text on the first line
            cur["narration"] += " " + stripped
            continue

        # ── State 2: expect first amount (withdrawal or deposit) ──────────
        if cur["state"] == 2:
            if _AMT_ONLY_RE.match(stripped):
                cur["amounts"].append(stripped)
                cur["state"] = 3
                continue
            cur["narr_extra"].append(stripped)
            continue

        # ── State 3: expect closing balance ───────────────────────────────
        if cur["state"] == 3:
            if _AMT_ONLY_RE.match(stripped):
                cur["amounts"].append(stripped)
                cur["state"] = 4
                continue
            cur["narr_extra"].append(stripped)
            continue

        # ── State 4: row complete — collect narration overflow ────────────
        if cur["state"] == 4:
            if _AMT_ONLY_RE.match(stripped):
                # Extra amount line (shouldn't happen — ignore)
                continue
            cur["narr_extra"].append(stripped)
            continue

    _flush(cur)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ("debit", "credit", "balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["transaction_date"])
    df = df[df["transaction_date"].apply(
        lambda x: x is not None and isinstance(x, date)
    )]
    df = df.drop_duplicates(subset=["transaction_date", "narration", "balance"])
    return _normalize_df(df)


# ──────────────────────────────────────────────────────────────────────────────
# Main extractor
# ──────────────────────────────────────────────────────────────────────────────

class HDFCBankExtractor:

    BANK_NAME = "HDFC Bank"
    CONFIG    = HDFC_CONFIG

    # ── detect ─────────────────────────────────────────────────────────────────

    def detect(self, first_page_text: str) -> bool:
        indicators = [
            r"HDFC\s*BANK",
            r"hdfcbank\.com",
            r"HDFC\d{7}",
            r"We\s+understand\s+your\s+world",
        ]
        return any(re.search(p, first_page_text, re.I) for p in indicators)

    # ── public entry point ─────────────────────────────────────────────────────

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        with pdfplumber.open(pdf_path) as pdf:
            raw_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        info         = self._parse_account_info(raw_text)
        transactions = _parse_hdfc_blocks(raw_text)  # ← single strategy now

        if not transactions.empty and "transaction_date" in transactions.columns:
            valid = transactions["transaction_date"].dropna()
            if len(valid):
                info.txn_start_date = str(valid.min())
                info.txn_end_date   = str(valid.max())

        return {
            "account_info": info,
            "transactions": transactions,
            "config_used":  HDFC_CONFIG,
        }

    def run(self, pdf_path: str) -> Dict[str, Any]:
        """Entry point for app.py; returns flat result dict like BaseBankExtractor."""
        result = self.extract(pdf_path)
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

    # ── account info ───────────────────────────────────────────────────────────

    def _parse_account_info(self, raw_text: str) -> AccountInfo:
        # Fix sci-notation in the header text too
        text = _fix_sci_notation(raw_text)
        info = AccountInfo(bank_name=self.BANK_NAME)

        m = re.search(r"Account\s*No\s*[:\-]?\s*:?\s*(\d{9,18})", text, re.I)
        if m:
            info.account_number = m.group(1).strip()

        m = re.search(r"Cust\s*ID\s*[:\-]?\s*:?\s*(\d+)", text, re.I)
        if m:
            info.customer_id = m.group(1).strip()

        m = re.search(r"M[/\\]?S\.?\s+(.+?)(?:\n|$)", text, re.I)
        if m:
            info.customer_name = m.group(1).strip()

        if re.search(r"Imperia", text, re.I):
            info.account_type = "Current Account - Imperia"

        m = re.search(HDFC_CONFIG.ifsc_pattern, text)
        if m:
            info.ifsc_code = m.group(1)

        m = re.search(HDFC_CONFIG.micr_pattern, text, re.I)
        if m:
            info.micr_code = m.group(1)

        m = re.search(r"Account\s+Branch\s*[:\-]\s*(.+?)(?:\n|$)", text, re.I)
        if m:
            info.branch_name = m.group(1).strip()

        m = re.search(r"Address\s*[:\-]\s*(.+?)(?=City\s*[:\-])", text, re.I | re.DOTALL)
        if m:
            info.branch_address = " ".join(m.group(1).split()).strip()

        m = re.search(r"Email\s*[:\-]\s*([\w.\-+]+@[\w.\-]+)", text, re.I)
        if m:
            info.email = m.group(1).strip()

        m = re.search(r"Phone\s+no\.?\s*[:\-]\s*([\d/]+)", text, re.I)
        if m:
            info.phone = m.group(1).strip()

        m = re.search(
            r"Statement\s+From\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})\s+To\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
            text, re.I,
        )
        if m:
            info.statement_from = m.group(1)
            info.statement_to   = m.group(2)

        # Opening / Closing from STATEMENT SUMMARY block
        m = re.search(
            r"Opening\s+Balance\s*\n"
            r"(?:Dr\s+Count.*?Closing\s+Bal\s*\n)?"
            r"([\d,]+\.\d{2})",
            text, re.I | re.DOTALL,
        )
        if m:
            info.opening_balance = _clean_amount(m.group(1)) or 0.0

        m = re.search(r"([\-\d,]+\.\d{2})\s*\nGenerated\s+On", text, re.I)
        if m:
            info.closing_balance = _clean_amount(m.group(1)) or 0.0

        m = re.search(r"A[/\\]C\s+Open\s+Date\s*[:\-]\s*(\d{2}/\d{2}/\d{4})", text, re.I)
        if m:
            open_date = _parse_date(m.group(1), "%d/%m/%Y")
            to_date   = _parse_date(info.statement_to, "%d/%m/%Y") if info.statement_to else None
            if open_date and to_date:
                info.account_age_days = (to_date - open_date).days

        return info


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pprint

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else r"d:\llm-project\hdfcaprmar.pdf"

    extractor = HDFCBankExtractor()
    result    = extractor.extract(pdf_path)

    info: AccountInfo  = result["account_info"]
    txns: pd.DataFrame = result["transactions"]

    print("=" * 70)
    print("ACCOUNT INFO")
    print("=" * 70)
    pprint.pprint(vars(info))

    print("\n" + "=" * 70)
    print(f"TRANSACTIONS  ({len(txns)} rows)")
    print("=" * 70)
    if not txns.empty:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 80)
        print(txns.to_string(index=False))
    else:
        print("No transactions extracted.")
