# clm_system/core/pipeline/chunking/deal.py
from typing import List
import re
from ..base import BaseChunker
from clm_system.config import settings
from typing import Dict, Any
class DealChunker(BaseChunker, doc_type="deal"):
    """Chunker for oil industry deal documents."""
    
    def chunk(self, text: str) -> List[str]:
        """
        Split deal text into semantic chunks for embedding.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Get configuration
        size = settings.chunk_size
        overlap = settings.chunk_overlap
        
        # Split on section boundaries first
        sections = self._split_sections(text)
        
        # Process each section
        chunks = []
        for section in sections:
            # If section is already small enough, keep as is
            if len(section.split()) <= size:
                chunks.append(section)
                continue
                
            # Otherwise, chunk using sliding window
            tokens = section.split()
            for i in range(0, len(tokens), size - overlap):
                chunk_tokens = tokens[i:min(i + size, len(tokens))]
                if chunk_tokens:
                    chunks.append(" ".join(chunk_tokens))
        
        # Ensure each chunk is non-empty and deduplicate
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_sections(self, text: str) -> List[str]:
        """Split text into logical sections based on headers/markers."""
        # Common section markers in oil & gas deals
        section_markers = [
            r'DEAL SUMMARY',
            r'PRICING TERMS',
            r'VOLUME DETAILS',
            r'DELIVERY TERMS',
            r'QUALITY SPECIFICATIONS',
            r'PAYMENT TERMS',
            r'SPECIAL PROVISIONS',
            r'FORCE MAJEURE',
            r'REGULATORY COMPLIANCE',
        ]
        
        # Create a regex pattern matching any of the section markers
        pattern = '|'.join(f'({marker})' for marker in section_markers)
        
        # Find all matches
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            # No section markers found, return whole text
            return [text]
        
        # Extract sections using the positions of the matches
        sections = []
        
        # First section (from start to first match)
        if matches[0].start() > 0:
            sections.append(text[:matches[0].start()].strip())
        
        # Middle sections
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i+1].start() if i < len(matches) - 1 else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append(section_text)
        
        return sections
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data  # Implement if additional processing is needed