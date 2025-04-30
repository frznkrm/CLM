# clm_system/core/pipeline/ingestion/contract.py
import uuid
from datetime import datetime
from ..base import BaseIngestor
from typing import Dict, Any

class ContractIngestor(BaseIngestor, doc_type="contract"):
    def process(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        data = raw.copy()
        data.setdefault("id", str(uuid.uuid4()))
        now = datetime.utcnow()
        data["created_at"] = now
        data["updated_at"] = now
        
        # Ensure metadata exists and add document_type
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["document_type"] = "contract"
        
        return data