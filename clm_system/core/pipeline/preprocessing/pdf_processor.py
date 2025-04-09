# clm_system/core/preprocessing/pdf_processor.py
import fitz  # PyMuPDF
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processes PDF files into contract JSON structure."""
    
    def __init__(self):
        # Initialize any needed resources
        pass
        
    async def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF file into a contract structure.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Contract object in dictionary form
        """
        try:
            # Extract text from PDF
            text = await self._extract_text(file_path)
            
            # Process the text into contract structure
            contract = await self._structure_contract(text, file_path)
            
            return contract
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    async def _extract_text(self, file_path: str) -> str:
        """Extract text from PDF."""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    async def _structure_contract(self, text: str, file_name: str) -> Dict[str, Any]:
        """
        Convert raw text into contract structure.
        Uses heuristics and patterns to identify contract components.
        """
        # Basic structure
        contract = {
            "title": self._extract_title(text, file_name),
            "metadata": self._extract_metadata(text),
            "clauses": self._extract_clauses(text)
        }
        
        return contract
    
    def _extract_title(self, text: str, file_name: str) -> str:
        """Extract contract title from text or use filename."""
        # Look for title patterns (often in first few lines)
        first_lines = text.strip().split('\n')[:5]
        for line in first_lines:
            if re.search(r'agreement|contract|terms', line.lower()):
                return line.strip()
        
        # Fallback to filename
        base_name = os.path.basename(file_name)
        name_without_ext = os.path.splitext(base_name)[0]
        return name_without_ext.replace('_', ' ').title()
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract contract metadata."""
        metadata = {
            "contract_type": self._identify_contract_type(text),
            "parties": self._identify_parties(text),
            "status": "draft",  # Default status
            "tags": []  # Can be filled based on content analysis
        }
        
        # Extract dates
        dates = self._extract_dates(text)
        if dates.get("effective_date"):
            metadata["effective_date"] = dates["effective_date"]
        if dates.get("expiration_date"):
            metadata["expiration_date"] = dates["expiration_date"]
            
        return metadata
    
    def _identify_contract_type(self, text: str) -> str:
        """Identify contract type from text."""
        type_patterns = {
            "license": r'license|licensing|licensor|licensee',
            "service": r'service|services|provider|customer',
            "nda": r'confidential|non-disclosure|nda',
            "employment": r'employment|employer|employee|hire',
            "purchase": r'purchase|procurement|buyer|seller'
        }
        
        for contract_type, pattern in type_patterns.items():
            if re.search(pattern, text.lower()):
                return contract_type
                
        return "other"
    
    def _identify_parties(self, text: str) -> List[Dict[str, str]]:
        """Extract party information."""
        parties = []
        
        # Look for common party patterns
        party_patterns = [
            r'between\s+([^,]+),?\s+(?:a|an)\s+([^,]+)',
            r'(?:party|client|customer|vendor|supplier):\s*([^\n]+)',
            r'(?:hereinafter\s+referred\s+to\s+as\s+["\'])([^"\']+)'
        ]
        
        for pattern in party_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for i, match in enumerate(matches):
                party_name = match.group(1).strip()
                if party_name and len(party_name) < 100:  # Sanity check
                    parties.append({
                        "name": party_name,
                        "id": f"party-{len(parties)+1:03d}"
                    })
        
        return parties
    
    from dateutil import parser  # Add this import

    def _extract_dates(self, text: str) -> Dict[str, str]:
        dates = {}
        
        # Existing regex patterns
        effective_match = re.search(r'effective\s+date.*?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text, re.IGNORECASE)
        expiration_match = re.search(r'(?:expiration|termination|end)\s+date.*?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text, re.IGNORECASE)

        # Parse dates with flexible format handling
        try:
            if effective_match:
                dates["effective_date"] = parser.parse(effective_match.group(1)).isoformat()
        except:
            logger.warning(f"Could not parse effective date: {effective_match.group(1)}")

        try:
            if expiration_match:
                dates["expiration_date"] = parser.parse(expiration_match.group(1)).isoformat()
        except:
            logger.warning(f"Could not parse expiration date: {expiration_match.group(1)}")

        return dates
    
    def _extract_clauses(self, text: str) -> List[Dict[str, Any]]:
        """Extract clauses from contract text."""
        clauses = []
        
        # Split text into possible clauses (simplified)
        # In production, would need more sophisticated section detection
        section_pattern = r'(?:\n|\r\n)(\d+\.?\s+[A-Z][^\.]+)\.?(?:\n|\r\n)'
        sections = re.split(section_pattern, text)
        
        position = 1
        for i in range(1, len(sections), 2):
            if i < len(sections):
                title = sections[i].strip()
                content = sections[i+1].strip() if i+1 < len(sections) else ""
                
                if title and content:
                    clause_type = self._identify_clause_type(title, content)
                    clauses.append({
                        "id": f"clause-{position:03d}",
                        "title": title,
                        "type": clause_type,
                        "text": content,
                        "position": position
                    })
                    position += 1
        
        return clauses
    
    def _identify_clause_type(self, title: str, content: str) -> str:
        """Identify clause type from title and content."""
        # Simple mapping of common clause types
        type_patterns = {
            "term": r'term|duration|period',
            "payment": r'payment|fee|compensation|price',
            "termination": r'terminat|cancel|end',
            "confidentiality": r'confidential|secret|disclosure',
            "liability": r'liab|indemnif|harmless',
            "warranty": r'warrant|guarantee',
            "grant": r'grant|licens|right',
            "governing_law": r'governing law|jurisdiction|venue'
        }
        
        combined_text = (title + " " + content[:200]).lower()
        
        for clause_type, pattern in type_patterns.items():
            if re.search(pattern, combined_text):
                return clause_type
                
        return "other"