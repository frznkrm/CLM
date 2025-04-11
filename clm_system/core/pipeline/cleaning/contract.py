# clm_system/core/pipeline/cleaning/contract.py
from ..base import BaseCleaner

class ContractCleaner(BaseCleaner, doc_type="contract"):
    def process(self, data: dict) -> dict:
        # e.g. strip whitespace, normalize dates, remove PII
        # if already clean, return as‑is
        return data
