# clm_system/core/pipeline/ingestion/contract.py
import uuid
from datetime import datetime
from .base import IngestorABC

class ContractIngestor(IngestorABC):
    def ingest(self, raw: dict) -> dict:
        
        data = raw.copy()
        data.setdefault("id", str(uuid.uuid4()))
        now = datetime.utcnow()
        data["created_at"] = now
        data["updated_at"] = now
        return data
