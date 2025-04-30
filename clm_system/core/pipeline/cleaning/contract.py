from ..base import BaseCleaner
from typing import Dict, Any, List
import re

class ContractCleaner(BaseCleaner, doc_type="contract"):
    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not isinstance(text, str):
            return str(text)
        # Remove extra whitespace and normalize Unicode
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace('\u2013', '-')  # Replace en-dash with hyphen
        text = text.replace('\u2019', "'")  # Replace right single quote with apostrophe
        return text

    def clean_content(self, content: List[Any]) -> List[str]:
        """Clean clause content, handling strings and tables."""
        cleaned = []
        for item in content:
            if isinstance(item, str):
                cleaned.append(self.clean_text(item))
            elif isinstance(item, dict) and "table" in item:
                # Flatten table to text (basic handling for now)
                table = item["table"]
                table_text = " ".join(
                    " ".join(str(cell) for cell in row) for row in table
                )
                cleaned.append(self.clean_text(table_text))
            else:
                cleaned.append(self.clean_text(str(item)))
        return cleaned

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean the contract data."""
        cleaned_data = data.copy()
        
        # Clean title
        if "title" in cleaned_data:
            cleaned_data["title"] = self.clean_text(cleaned_data["title"])
        
        # Clean header
        if "header" in cleaned_data:
            header = cleaned_data["header"]
            for key in ["to", "e-mail", "re", "sellers ref"]:
                if key in header:
                    header[key] = self.clean_text(header[key])
            if "notes" in header:
                header["notes"] = [self.clean_text(note) for note in header["notes"]]
        
        # Clean clauses
        if "clauses" in cleaned_data:
            for clause in cleaned_data["clauses"]:
                clause["title"] = self.clean_text(clause["title"])
                clause["content"] = self.clean_content(clause["content"])
        
        return cleaned_data