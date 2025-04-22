
import docx
import re
import logging
import asyncio
from typing import Dict, Any, List
from PIL import Image
import pytesseract
import io
import os

# Explicitly set Tesseract path for WSL
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

logger = logging.getLogger(__name__)

class DOCXProcessor:
    """Processes DOCX files into contract JSON structure."""
    
    def __init__(self):
        # Check Tesseract availability
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is properly configured.")
        except Exception as e:
            logger.warning(f"Tesseract not found or misconfigured: {e}. Please ensure Tesseract is installed.")
    
    async def process_docx(self, file_path: str) -> Dict[str, Any]:
        """
        Process a DOCX file into a contract structure.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Contract object in dictionary form
        """
        try:
            # Extract text and images from DOCX
            raw_data = await self._extract_content(file_path)
            text = raw_data["text"]
            images = raw_data["images"]
            logger.debug(f"Extracted raw text:\n{text[:1000]}...")
            
            # Process the text into contract structure
            contract = await self._structure_contract(text, images, file_path)
            logger.debug(f"Structured contract: {contract}")
            
            return contract
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            raise
    
    async def _extract_content(self, file_path: str) -> Dict[str, Any]:
        """Extract text and images from DOCX."""
        text_lines = []
        images = []  # List of (paragraph_index, image_data) tuples
        try:
            doc = docx.Document(file_path)
            for i, para in enumerate(doc.paragraphs):
                text = para.text.strip()
                if text:
                    text_lines.append(text)
                
                # Check for images in the paragraph
                for run in para.runs:
                    if run.element.xpath('.//wp:inline'):
                        blip = run.element.xpath('.//a:blip')[0]
                        rid = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        part = doc.part.related_parts[rid]
                        image = Image.open(io.BytesIO(part.image.blob))
                        images.append((i, image))
            
            full_text = "\n".join(text_lines)
            return {"text": full_text, "images": images}
        except Exception as e:
            logger.error(f"Error extracting content from DOCX: {e}")
            raise
    
    async def _structure_contract(self, text: str, images: List[tuple], file_name: str) -> Dict[str, Any]:
        """
        Convert raw text and images into contract structure.
        """
        lines = text.split("\n")
        header = {}
        sections = []
        current_section = None
        section_pattern = re.compile(r'^\d+\.\s*.+$')
        image_idx = 0
        
        # Process lines with logging
        for i, line in enumerate(lines):
            line = line.strip()
            logger.debug(f"Line {i}: '{line}'")
            if not line:
                continue
                
            if section_pattern.match(line):
                logger.debug(f"Line {i} matched section pattern: {line}")
                if current_section:
                    sections.append(current_section)
                current_section = {"title": line, "content": []}
            else:
                if current_section:
                    current_section["content"].append(line)
                else:
                    # Header parsing before first section
                    if ":" in line:
                        key, value = line.split(":", 1)
                        header[key.strip().lower()] = value.strip()
                    else:
                        header.setdefault("notes", []).append(line)
            
            # Handle images based on their position
            while image_idx < len(images) and images[image_idx][0] <= i:
                _, image = images[image_idx]
                table_text = await self._extract_table_text(image)
                if current_section:
                    current_section["content"].append({"table": table_text})
                else:
                    header.setdefault("notes", []).append({"table": table_text})
                image_idx += 1
        
        if current_section:
            sections.append(current_section)
        
        # Structure the contract with dynamic party extraction
        contract = {
            "title": header.get("re", os.path.basename(file_name).replace(".docx", "").replace("_", " ").title()),
            "metadata": {
                "contract_type": "purchase",
                "parties": self._extract_parties(sections, header),
                "status": "draft",
                "tags": ["trading", "gasoline", "naphtha"]
            },
            "header": header,
            "clauses": sections
        }
        return contract
    
    def _extract_parties(self, sections: List[Dict], header: Dict) -> List[Dict]:
        """Extract party information dynamically from sections or fall back to header."""
        parties = []
        buyer = None
        seller = None
        
        for section in sections:
            title = section["title"].lower()
            if "buyer" in title and not buyer:
                buyer_content = " ".join(section["content"]).strip()
                buyer = {"name": buyer_content or "Unknown Buyer", "id": "party-001"}
            elif "seller" in title and not seller:
                seller_content = " ".join(section["content"]).strip()
                seller = {"name": seller_content or "Unknown Seller", "id": "party-002"}
        
        # Fallback to header if sections don't provide party info
        if not buyer:
            buyer = {"name": header.get("to", "Unknown Buyer"), "id": "party-001"}
        if not seller:
            seller = {"name": "TRADECO SA", "id": "party-002"}
        
        if buyer:
            parties.append(buyer)
        if seller:
            parties.append(seller)
        
        return parties
    
    async def _extract_table_text(self, image: Image.Image) -> List[List[str]]:
        """Extract text from image (table) using OCR and attempt basic structuring."""
        try:
            # Use OCR with automatic page segmentation for tables
            text = pytesseract.image_to_string(image, config='--psm 3')
            logger.debug(f"Raw OCR text from table: {text}")
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Basic table structuring: split by lines and spaces
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            table = [re.split(r'\s{2,}', line) for line in lines]
            return table if table else [["No table data extracted"]]
        except Exception as e:
            logger.error(f"Error extracting table text: {e}")
            return [["Error extracting table"]]
    
# Example usage (for testing)
if __name__ == "__main__":
    async def test():
        processor = DOCXProcessor()
        file_path = "sample_contract.docx"
        result = await processor.process_docx(file_path)
        with open("contract.json", "w") as f:
            import json
            json.dump(result, f, indent=4)
    
    asyncio.run(test())