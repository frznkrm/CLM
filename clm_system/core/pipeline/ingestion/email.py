# clm_system/core/pipeline/ingestion/email.py
import uuid
from datetime import datetime
from email.utils import parsedate_to_datetime
from .base import IngestorABC

class EmailIngestor(IngestorABC):
    def ingest(self, raw: dict) -> dict:
        """
        Process raw email data into a standardized format.
        
        Args:
            raw: Raw email data from input source
            
        Returns:
            Standardized email document with metadata
        """
        data = raw.copy()
        
        # Ensure required fields
        data.setdefault("id", str(uuid.uuid4()))
        now = datetime.utcnow()
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        
        # Set title to email subject if not specified
        if "subject" in data and "title" not in data:
            data["title"] = data["subject"]
        
        # Ensure metadata exists
        if "metadata" not in data:
            data["metadata"] = {}
        
        # Set document type
        data["metadata"]["document_type"] = "email"
        
        # Extract and normalize email metadata
        data["metadata"].setdefault("from", data.get("from"))
        data["metadata"].setdefault("to", data.get("to", []))
        data["metadata"].setdefault("cc", data.get("cc", []))
        data["metadata"].setdefault("bcc", data.get("bcc", []))
        
        # Parse email date if present
        if "date" in data and isinstance(data["date"], str):
            try:
                parsed_date = parsedate_to_datetime(data["date"])
                data["metadata"]["email_date"] = parsed_date
            except Exception:
                # If parsing fails, use ingestion date
                data["metadata"]["email_date"] = now
        
        # Handle email body
        if "body" in data:
            # Store original body in metadata
            data["metadata"]["original_format"] = "text" if isinstance(data["body"], str) else "html"
            
            # Convert body to clauses
            if "clauses" not in data:
                data["clauses"] = [{
                    "id": f"{data['id']}_body",
                    "type": "email_body",
                    "text": data["body"],
                    "position": 0
                }]
            
            # Remove body from top level after storing in clauses
            data.pop("body", None)
            
        # Handle attachments metadata
        if "attachments" in data:
            data["metadata"]["has_attachments"] = True
            data["metadata"]["attachment_count"] = len(data["attachments"])
            data["metadata"]["attachment_names"] = [
                att.get("filename", f"attachment_{i}") 
                for i, att in enumerate(data["attachments"])
            ]
            # Remove attachments from top level
            data.pop("attachments", None)
        else:
            data["metadata"]["has_attachments"] = False
            
        return data