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
            logger.debug(f"Extracted raw text:\n{text[:1000]}...")  # First 1000 chars
            # Process the text into contract structure
            contract = await self._structure_contract(text, file_path)
            logger.debug("Structured contract:", contract)
            
            return contract
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    async def _extract_text(self, file_path: str) -> str:
        """Extract and normalize text from PDF."""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
            
            # Normalize text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace
            text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix hyphenated words
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
        logger.debug(f"Structured contract: {contract}")
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
        logger.debug(f"Raw text for clause extraction:\n{text[:2000]}...")  # First 2000 chars
        
        # Enhanced section pattern with multiple variations
        section_pattern = r'''
            (?:\n|\r\n)  # Section starts with newline
            (?:          # Section number formats:
            (?:\d+[\.\)]?|          # 1, 2., 3)
            [A-Z][\.\)]|           # A., B)
            ARTICLE\s+[IVXLCDM]+|  # ARTICLE I
            SECTION\s+[\dA-Z]+|    # SECTION 1, SECTION A
            Clause\s+\d+|          # Clause 1
            §\s?[\dA-Z]+|          # §1, §A
            [IVXLCDM]+[\.\)]       # I., II)
            )
            )
            \s+          # Whitespace after number
            ([A-Z][^\.:\n]{5,})  # Title (capitalized, min 5 chars)
            [\.:]?       # Optional ending punctuation
            (?=\n|\r\n|$)  # Lookahead for newline or end
        '''
        
        sections = re.split(section_pattern, text, flags=re.IGNORECASE|re.VERBOSE)
        
        # Debug found sections
        logger.debug(f"Split sections: {sections[:10]}")  # First 10 sections
        
        position = 1
        for i in range(1, len(sections)-1, 2):
            title = sections[i].strip()
            content = sections[i+1].strip()
            
            if len(content) > 50:  # Increased minimum content length
                clause_type = self._identify_clause_type(title, content)
                clauses.append({
                    "id": f"clause-{position:03d}",
                    "title": title,
                    "type": clause_type,
                    "text": content,
                    "position": position
                })
                position += 1
                
        # Fallback: Split by paragraph if no sections found
        if not clauses:
            logger.warning("No sections found, falling back to paragraph split")
            paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 100]
            for i, para in enumerate(paragraphs):
                clauses.append({
                    "id": f"para-{i+1:03d}",
                    "title": f"Paragraph {i+1}",
                    "type": "uncategorized",
                    "text": para,
                    "position": i+1
                })
        
        logger.debug(f"Found {len(clauses)} clauses: {clauses}")
        logger.debug(f"Pre-filtered clauses: {len(clauses)} items")
        valid_clauses = [c for c in clauses if c.get("text")]
        logger.debug(f"Post-filtered clauses: {len(valid_clauses)} items")
                
        return [
        clause for clause in clauses
        if clause.get("text") and len(clause["text"]) > 50
    ]