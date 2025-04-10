# clm_system/core/pipeline/ingestion/deal.py
import uuid
from datetime import datetime
from .base import IngestorABC

class DealIngestor(IngestorABC):
    def ingest(self, raw: dict) -> dict:
        """
        Process raw oil industry deal data into a standardized format.
        
        Args:
            raw: Raw deal data from input source
            
        Returns:
            Standardized deal document with metadata
        """
        data = raw.copy()
        
        # Ensure required fields
        data.setdefault("id", str(uuid.uuid4()))
        now = datetime.utcnow()
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        
        # Ensure deal-specific metadata
        if "metadata" not in data:
            data["metadata"] = {}
            
        # Set document type if not present
        data["metadata"].setdefault("document_type", "deal")
        
        # Ensure deal-specific fields
        data["metadata"].setdefault("deal_type", data.get("deal_type", "unspecified"))
        data["metadata"].setdefault("volume", data.get("volume"))
        data["metadata"].setdefault("price_per_unit", data.get("price_per_unit"))
        data["metadata"].setdefault("total_value", data.get("total_value"))
        data["metadata"].setdefault("location", data.get("location"))
        data["metadata"].setdefault("counterparties", data.get("counterparties", []))
        
        return data