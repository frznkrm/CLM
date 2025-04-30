import docx
import re
import logging
import asyncio
import requests
import json
import base64
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageEnhance
import pytesseract
import io
import os
import pandas as pd
from io import StringIO
import numpy as np
import cv2

# Explicitly set Tesseract path for WSL (fallback method)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DOCXProcessor:
    """Processes DOCX files into contract JSON structure with VLM-enhanced table parsing."""
    
    def __init__(self, use_vlm=True, vlm_endpoint="http://192.168.10.1:1234/v1"):
        """
        Initialize the processor with VLM support.
        
        Args:
            use_vlm: Whether to use VLM for table parsing (default: True)
            vlm_endpoint: Base URL for the LM Studio API endpoint
        """
        self.use_vlm = use_vlm
        self.vlm_endpoint = vlm_endpoint.rstrip('/')  # Remove trailing slash if present
        self.vlm_model = None
        
        # Check if LM Studio is accessible
        if self.use_vlm:
            try:
                response = requests.get(f"{self.vlm_endpoint}/models")
                if response.status_code == 200:
                    models = response.json()
                    logger.info(f"LM Studio is accessible. Available models: {models}")
                    # Set default VLM model based on available models
                    if 'data' in models and len(models['data']) > 0:
                        for model in models['data']:
                            if 'llava' in model['id'].lower():  # Prefer LLaVA model if available
                                self.vlm_model = model['id']
                                break
                        # If no LLaVA model found, use first available model
                        if not self.vlm_model and len(models['data']) > 0:
                            self.vlm_model = models['data'][0]['id']
                else:
                    logger.warning(f"LM Studio returned status code {response.status_code}. API may not be fully compatible.")
            except requests.RequestException as e:
                logger.warning(f"Could not connect to LM Studio API: {e}. Will fall back to OCR if needed.")
        
        # Check Tesseract availability (fallback)
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is properly configured as fallback.")
        except Exception as e:
            logger.warning(f"Tesseract not found or misconfigured: {e}. Please ensure Tesseract is installed for fallback.")
    
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
        # Improved regex to match section titles more flexibly
        section_pattern = re.compile(r'^\d+\.\s*(?:[A-Z\s/]+|\w+\s*[\w\s]*(?:M3|M³).*)$')
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
                if self.use_vlm:
                    table_text = await self._extract_table_with_vlm(image)
                else:
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
            seller = {"name": "TRADECO", "id": "party-002"}
        
        if buyer:
            parties.append(buyer)
        if seller:
            parties.append(seller)
        
        return parties
    
    async def _extract_table_with_vlm(self, image: Image.Image) -> List[List[str]]:
        """
        Extract table data from an image using a Vision Language Model via LM Studio.
        
        Args:
            image: PIL Image containing a table
            
        Returns:
            Structured table as a list of rows, each row being a list of cell texts
        """
        try:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare the prompt for the VLM, combining system and user instructions
            system_instruction = "You are an expert at extracting data from table images. Return only the extracted table as a JSON array of arrays, with no additional text."
            user_prompt = "Extract all data from this table. Format the result as a JSON array of arrays (rows and columns). Be precise with numbers, units, and technical specifications."
            combined_prompt = f"{system_instruction}\n\n{user_prompt}"
            
            try:
                # Get model information
                models_response = requests.get(f"{self.vlm_endpoint}/models")
                if models_response.status_code != 200:
                    logger.warning("Could not get model information from LM Studio")
                    raise ValueError("Failed to get model info")
                
                model_info = models_response.json()
                logger.info(f"Models info: {model_info}")
                
                # Use the model we determined during initialization or fallback to local-model
                model_name = self.vlm_model or "local-model"
                
                # Use chat completions endpoint
                chat_endpoint = f"{self.vlm_endpoint}/chat/completions"
                logger.info(f"Using chat endpoint: {chat_endpoint}")
                
                # Updated payload with only user role
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": combined_prompt
                        }
                    ],
                    "temperature": 0.1,
                    "image": img_str  # Add image directly as a parameter
                }
                
                # Log the request for debugging (excluding the image data)
                debug_payload = payload.copy()
                if "image" in debug_payload:
                    debug_payload["image"] = "[BASE64_IMAGE_DATA]"
                logger.info(f"Sending request to VLM API: {json.dumps(debug_payload, indent=2)}")
                
                headers = {
                    "Content-Type": "application/json"
                }
                
                response = requests.post(chat_endpoint, json=payload, headers=headers)
                
                # If chat completions fails, try the completions endpoint
                if response.status_code != 200:
                    logger.warning(f"Chat completions format failed with status {response.status_code}: {response.text}")
                    
                    # Use a simpler format for completions endpoint
                    simple_payload = {
                        "model": model_name,
                        "prompt": f"{combined_prompt}\n\n[Image data for table analysis]",
                        "image": img_str,
                        "temperature": 0.1,
                        "max_tokens": 1000
                    }
                    
                    logger.info("Trying direct completions endpoint")
                    response = requests.post(f"{self.vlm_endpoint}/completions", json=simple_payload, headers=headers)
                
                if response.status_code == 200:
                    logger.info("VLM request successful!")
                    result = response.json()
                    
                    # Extract content based on response structure
                    if "choices" in result and len(result["choices"]) > 0:
                        if "message" in result["choices"][0]:
                            # Chat completion format
                            response_content = result["choices"][0]["message"]["content"]
                        elif "text" in result["choices"][0]:
                            # Completion format
                            response_content = result["choices"][0]["text"]
                        else:
                            logger.warning("Unexpected response format")
                            response_content = str(result)
                    else:
                        logger.warning("No choices in response")
                        response_content = str(result)
                    
                    logger.info(f"Raw VLM response: {response_content}")
                    
                    # Try to extract JSON array from the response
                    json_matches = re.search(r'\[\s*\[.*\]\s*\]', response_content, re.DOTALL)
                    if json_matches:
                        json_str = json_matches.group(0)
                        try:
                            table_data = json.loads(json_str)
                            if isinstance(table_data, list) and all(isinstance(row, list) for row in table_data):
                                logger.info(f"Successfully parsed table with {len(table_data)} rows")
                                return table_data
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON parsing error: {e}")
                    
                    # Try parsing code blocks or direct JSON
                    try:
                        code_block_matches = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_content)
                        if code_block_matches:
                            json_str = code_block_matches.group(1)
                            table_data = json.loads(json_str)
                            if isinstance(table_data, list) and all(isinstance(row, list) for row in table_data):
                                return table_data
                        
                        table_data = json.loads(response_content)
                        if isinstance(table_data, list) and all(isinstance(row, list) for row in table_data):
                            return table_data
                    except json.JSONDecodeError:
                        logger.warning("Could not parse response as JSON")
                
                else:
                    logger.warning(f"VLM request failed with status {response.status_code}: {response.text}")
            
            except Exception as e:
                logger.error(f"Error in VLM API request: {e}")
            
            # Fall back to OCR if VLM fails
            logger.info("VLM extraction failed, falling back to OCR method")
            return await self._extract_table_text(image)
            
        except Exception as e:
            logger.error(f"Error in VLM table extraction: {e}")
            logger.info("Exception in VLM processing, falling back to OCR")
            return await self._extract_table_text(image)
    
    async def _extract_table_text(self, image: Image.Image) -> List[List[str]]:
        """
        Enhanced method to extract text from table images using advanced OCR techniques.
        
        Args:
            image: PIL Image containing a table
            
        Returns:
            Structured table as a list of rows, each row being a list of cell texts
        """
        try:
            # Convert PIL Image to OpenCV format
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing
            thresh = cv2.adaptiveThreshold(gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations to enhance text connectivity
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Optimized OCR config with preserved spaces for better column detection
            custom_config = r'--oem 1 --psm 6 -c preserve_interword_spaces=1 tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/%°C-.,:()'
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            logger.debug(f"Raw OCR text from table:\n{text}")
            
            # Process into structured table
            table_data = self._parse_table_data(text)
            
            # Apply post-processing for common OCR errors
            table_data = self._enhance_table_quality(table_data)
            
            return table_data
            
        except Exception as e:
            logger.error(f"Error extracting table text: {e}")
            return [["Error extracting table"]]
    
    def _parse_table_data(self, text: str) -> List[List[str]]:
        """
        Parse OCR text into structured table data.
        
        Args:
            text: Raw OCR text
            
        Returns:
            List of rows, each containing a list of cell values
        """
        # Split rows and clean data
        rows = [row.strip() for row in text.split('\n') if row.strip()]
        
        # Process rows into table structure
        table = []
        for row in rows:
            # Split on multiple spaces to identify columns
            cells = re.split(r'\s{2,}', row)
            # Clean empty cells
            cells = [cell.strip() for cell in cells if cell.strip()]
            
            if cells:  # Skip empty rows
                table.append(cells)
        
        # Table structure validation
        if len(table) < 1:
            return [["Table structure not recognized"]]
        
        # Attempt to align columns based on header/first row
        num_columns = max([len(row) for row in table[:3]], default=0)
        if num_columns > 0:
            aligned_table = []
            for row in table:
                # Pad rows with missing columns
                while len(row) < num_columns:
                    row.append('')
                # Trim excess columns
                aligned_table.append(row[:num_columns])
            return aligned_table
        
        return table
    
    def _enhance_table_quality(self, table: List[List[str]]) -> List[List[str]]:
        """
        Apply post-processing to improve table data quality.
        
        Args:
            table: Structured table data
            
        Returns:
            Enhanced table data
        """
        # Common OCR error corrections
        replacements = {
            r'(\d)\s?([°%])': r'\1\2',  # Fix spaced symbols
            'rn': 'm',                  # Common misread
            'iii': 'in',                # Common misread
            'Coriyon': 'Corrosion',     # Domain-specific corrections
            'Tensioii': 'Tension',
            'om3': 'cm³',
            'kg/rn3': 'kg/m³',
            'inl': 'ml',
            '\*C': '°C',
            'iG': '°C',
            '96': '%',
            'vol\s': 'vol%'
        }
        
        # Process each cell
        for i, row in enumerate(table):
            for j, cell in enumerate(row):
                # Normalize whitespace
                cell = ' '.join(cell.split())
                
                # Apply replacements
                for pattern, replacement in replacements.items():
                    cell = re.sub(pattern, replacement, cell)
                
                # Fix unit spacing
                cell = re.sub(r'(\d)([A-Za-z])', r'\1 \2', cell)
                
                # Handle hyphenated values correctly
                cell = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', cell)
                
                table[i][j] = cell
        
        return table

# Example usage (for testing)
if __name__ == "__main__":
    async def test():
        processor = DOCXProcessor(use_vlm=True)
        file_path = "sample_contract.docx"
        result = await processor.process_docx(file_path)
        with open("contract.json", "w") as f:
            import json
            json.dump(result, f, indent=4)
    
    asyncio.run(test())