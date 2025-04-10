# clm_system/core/pipeline/cleaning/deal.py
import re
from .base import CleanerABC

class DealCleaner(CleanerABC):
    def clean(self, data: dict) -> dict:
        """
        Clean and normalize oil industry deal data.
        
        Args:
            data: Raw deal data
            
        Returns:
            Cleaned deal data
        """
        cleaned = data.copy()
        
        # Normalize deal types (lowercase, replace spaces with underscores)
        if "metadata" in cleaned and "deal_type" in cleaned["metadata"]:
            cleaned["metadata"]["deal_type"] = cleaned["metadata"]["deal_type"].lower().replace(" ", "_")
        
        # Standardize location formatting if present
        if "metadata" in cleaned and "location" in cleaned["metadata"]:
            cleaned["metadata"]["location"] = self._normalize_location(cleaned["metadata"]["location"])
            
        # Clean monetary values
        for field in ["price_per_unit", "total_value"]:
            if "metadata" in cleaned and field in cleaned["metadata"]:
                cleaned["metadata"][field] = self._normalize_monetary_value(cleaned["metadata"][field])
        
        # Clean volume 
        if "metadata" in cleaned and "volume" in cleaned["metadata"]:
            cleaned["metadata"]["volume"] = self._normalize_volume(cleaned["metadata"]["volume"])
                
        # Clean clauses text
        if "clauses" in cleaned:
            for clause in cleaned["clauses"]:
                if "text" in clause:
                    clause["text"] = self._clean_text(clause["text"])
        
        return cleaned
    
    def _normalize_location(self, location: str) -> str:
        """Normalize location strings."""
        if not location or not isinstance(location, str):
            return location
            
        # Convert to title case
        location = location.title()
        
        # Handle common abbreviations
        abbreviations = {
            " Usa": " USA",
            " Uk": " UK",
            " Uae": " UAE",
        }
        for abbr, replacement in abbreviations.items():
            location = location.replace(abbr, replacement)
            
        return location
    
    def _normalize_monetary_value(self, value):
        """Normalize monetary values to numeric format."""
        if isinstance(value, (int, float)):
            return value
            
        if not value or not isinstance(value, str):
            return value
            
        # Remove currency symbols and commas
        value = re.sub(r'[$£€,]', '', value)
        
        # Extract numeric part
        match = re.search(r'([\d.]+)', value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
                
        return value
    
    def _normalize_volume(self, volume):
        """Normalize volume values."""
        if isinstance(volume, (int, float)):
            return volume
            
        if not volume or not isinstance(volume, str):
            return volume
            
        # Remove commas, standardize units
        volume = re.sub(r'[,]', '', volume)
        
        # Match number and unit
        match = re.search(r'([\d.]+)\s*([a-zA-Z]+)', volume)
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2).lower()
                
                # Standardize units
                unit_mapping = {
                    'bbl': 'bbl',
                    'barrel': 'bbl',
                    'barrels': 'bbl',
                    'bbls': 'bbl',
                    'mcf': 'mcf',
                    'mmcf': 'mmcf',
                    'boe': 'boe',
                }
                
                if unit in unit_mapping:
                    return f"{value} {unit_mapping[unit]}"
                return volume
            except ValueError:
                pass
                
        return volume
    
    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        if not text or not isinstance(text, str):
            return text
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors in oil & gas terminology
        corrections = {
            "0il": "Oil",
            "0ffshore": "Offshore",
            "C0ntract": "Contract",
            "Petr0leum": "Petroleum",
        }
        
        for error, correction in corrections.items():
            text = text.replace(error, correction)
            
        return text.strip()