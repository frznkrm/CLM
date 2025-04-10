# clm_system/core/pipeline/ingestion/recap.py
import uuid
from datetime import datetime
from .base import IngestorABC

class RecapIngestor(IngestorABC):
    def ingest(self, raw: dict) -> dict:
        """
        Process raw recap/summary data into a standardized format.
        
        Args:
            raw: Raw recap data from input source
            
        Returns:
            Standardized recap document with metadata
        """
        data = raw.copy()
        
        # Ensure required fields
        data.setdefault("id", str(uuid.uuid4()))
        now = datetime.utcnow()
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        
        # Ensure metadata exists
        if "metadata" not in data:
            data["metadata"] = {}
            
        # Set document type
        data["metadata"]["document_type"] = "recap"
        
        # Extract recap-specific metadata
        data["metadata"].setdefault("recap_type", data.get("recap_type", "general"))
        data["metadata"].setdefault("meeting_date", data.get("meeting_date"))
        data["metadata"].setdefault("author", data.get("author"))
        data["metadata"].setdefault("participants", data.get("participants", []))
        data["metadata"].setdefault("related_documents", data.get("related_documents", []))
        
        # Handle content sections
        if "sections" in data and isinstance(data["sections"], list):
            # Convert sections to clauses if not already present
            if "clauses" not in data:
                data["clauses"] = []
                for i, section in enumerate(data["sections"]):
                    if isinstance(section, dict):
                        data["clauses"].append({
                            "id": f"{data['id']}_section_{i}",
                            "title": section.get("heading", f"Section {i+1}"),
                            "type": section.get("type", "recap_section"),
                            "text": section.get("content", ""),
                            "position": i,
                            "metadata": {
                                "section_type": section.get("type", "general")
                            }
                        })
            # Remove sections from top level after storing in clauses
            data.pop("sections", None)
            
        return data