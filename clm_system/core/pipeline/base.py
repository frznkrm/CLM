# clm_system/core/pipeline/base.py
from typing import Any, Dict

class IngestorABC:
    def ingest(self, raw: Any) -> Dict[str, Any]:
        """Normalize raw input into our contract dict."""
        raise NotImplementedError

class CleanerABC:
    def clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply PII scrubbing, standardization, etc."""
        raise NotImplementedError

class ChunkerABC:
    def chunk(self, text: str) -> list[str]:
        """Split long text into embedding‑ready chunks."""
        raise NotImplementedError
