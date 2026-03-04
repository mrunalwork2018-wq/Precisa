from extractors.icici_bank import ICICIBankExtractor, AccountInfo, ICICI_CONFIG

extractor = ICICIBankExtractor()
result = extractor.extract("D:\llm-project\icici.pdf")

info = result["account_info"]        # AccountInfo dataclass
txns = result["transactions"]        # DataFrame with row_id, debit, credit, etc.
config = result["config_used"]   

print("info: ", info)
print("txns: ", txns)
print("config: ", config)