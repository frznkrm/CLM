# clm_system/core/pipeline/cleaning/recap.py
import re
from .base import CleanerABC

class RecapCleaner(CleanerABC):
    def clean(self, data: dict) -> dict:
        """
        Clean and normalize recap/summary data.
        
        Args:
            data: Raw recap data
            
        Returns:
            Cleaned recap data
        """
        cleaned = data.copy()
        
        # Normalize recap type
        if "metadata" in cleaned and "recap_type" in cleaned["metadata"]:
            cleaned["metadata"]["recap_type"] = cleaned["metadata"]["recap_type"].lower().replace(" ", "_")
        
        # Clean title
        if "title" in cleaned:
            cleaned["title"] = self._clean_title(cleaned["title"])
        
        # Clean participant names if present
        if "metadata" in cleaned and "participants" in cleaned["metadata"]:
            if isinstance(cleaned["metadata"]["participants"], list):
                cleaned["metadata"]["participants"] = [
                    self._clean_name(participant) for participant in cleaned["metadata"]["participants"]
                ]
        
        # Clean author name
        if "metadata" in cleaned and "author" in cleaned["metadata"]:
            cleaned["metadata"]["author"] = self._clean_name(cleaned["metadata"]["author"])
                
        # Clean clauses text
        if "clauses" in cleaned:
            for clause in cleaned["clauses"]:
                if "text" in clause:
                    clause["text"] = self._clean_text(clause["text"])
                if "title" in clause:
                    clause["title"] = self._clean_title(clause["title"])
        
        return cleaned
    
    def _clean_title(self, title):
        """Clean recap title."""
        if not title:
            return title
            
        # Remove unnecessary prefixes
        prefixes = ["RECAP:", "SUMMARY:", "MINUTES:", "Meeting:"]
        for prefix in prefixes:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        
        # Remove excessive whitespace
        title = re.sub(r'\s+', ' ', title)
        
        # Ensure first letter is capitalized
        if title:
            title = title[0].upper() + title[1:]
            
        return title.strip()
    
    def _clean_name(self, name):
        """Clean person names."""
        if not name or not isinstance(name, str):
            return name
            
        # Remove titles
        titles = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]
        for title in titles:
            if name.startswith(title):
                name = name[len(title):].strip()
        
        # Remove parenthetical information (like departments)
        name = re.sub(r'\([^)]*\)', '', name)
        
        # Normalize capitalization
        name = name.title()
        
        return name.strip()
    
    def _clean_text(self, text):
        """Clean recap text content."""
        if not text or not isinstance(text, str):
            return text
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'CONFIDENTIAL', '', text, flags=re.IGNORECASE)
        
        # Remove bullet point decorations but keep content
        text = re.sub(r'•\s*', '- ', text)
        text = re.sub(r'★\s*', '- ', text)
        text = re.sub(r'✓\s*', '- ', text)
        
        # Fix common OCR errors
        corrections = {
            "0ption": "Option",
            "decisíon": "decision",
            "actíon": "action",
            "revíew": "review",
        }
        
        for error, correction in corrections.items():
            text = text.replace(error, correction)
            
        return text.strip()