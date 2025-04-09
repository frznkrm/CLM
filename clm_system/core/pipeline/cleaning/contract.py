# clm_system/core/pipeline/cleaning/contract.py
from .base import CleanerABC

class ContractCleaner(CleanerABC):
    def clean(self, data: dict) -> dict:
        # e.g. strip whitespace, normalize dates, remove PII
        # if already clean, return as‑is
        return data
