# clm_system/core/pipeline/chunking/email.py
from typing import List
import re
from .base import ChunkerABC
from clm_system.config import settings

class EmailChunker(ChunkerABC):
    """Chunker for email documents."""
    
    def chunk(self, text: str) -> List[str]:
        """
        Split email text into semantic chunks for embedding.
        
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
        
        # First identify logical parts (greeting, body, signature, etc.)
        parts = self._split_email_parts(text)
        
        # Process each part
        chunks = []
        for part in parts:
            # Short parts can be kept as-is
            if len(part.split()) <= size:
                chunks.append(part)
                continue
                
            # For longer parts, chunk with sliding window
            tokens = part.split()
            for i in range(0, len(tokens), size - overlap):
                chunk_tokens = tokens[i:min(i + size, len(tokens))]
                if chunk_tokens:  # Ensure non-empty
                    chunks.append(" ".join(chunk_tokens))
        
        # Ensure each chunk is non-empty and deduplicate
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_email_parts(self, text: str) -> List[str]:
        """Split email into logical parts (greeting, body, signature, etc.)."""
        # Try to identify parts based on common patterns
        
        # Check for greeting
        greeting_patterns = [
            r'^(Dear\s+[^,\n]+[,\n])',
            r'^(Hi\s+[^,\n]+[,\n])',
            r'^(Hello\s+[^,\n]+[,\n])',
            r'^(Good\s+(morning|afternoon|evening)[^,\n]*[,\n])'
        ]
        
        # Check for signature
        signature_patterns = [
            r'(Best\s+regards,?\s*\n+.+)$',
            r'(Regards,?\s*\n+.+)$',
            r'(Thanks,?\s*\n+.+)$',
            r'(Thank\s+you,?\s*\n+.+)$',
            r'(Sincerely,?\s*\n+.+)$',
            r'(--\s*\n+.+)$'
        ]
        
        # Initialize parts
        parts = []
        remaining_text = text
        
        # Extract greeting
        for pattern in greeting_patterns:
            match = re.search(pattern, remaining_text, re.IGNORECASE | re.MULTILINE)
            if match:
                greeting = match.group(1).strip()
                if greeting:
                    parts.append(greeting)
                remaining_text = remaining_text[match.end():].strip()
                break
        
        # Extract signature
        signature = None
        for pattern in signature_patterns:
            match = re.search(pattern, remaining_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                signature = match.group(1).strip()
                remaining_text = remaining_text[:match.start()].strip()
                break
        
        # Process the body (remaining text)
        if remaining_text:
            # Try to split by paragraphs
            paragraphs = re.split(r'\n{2,}', remaining_text)
            
            # Add each significant paragraph as a separate part
            for paragraph in paragraphs:
                cleaned = paragraph.strip()
                if cleaned and len(cleaned) > 10:  # Avoid tiny fragments
                    parts.append(cleaned)
        
        # Add signature at the end if found
        if signature:
            parts.append(signature)
        
        # If no parts were identified, return original text
        if not parts:
            return [text]
            
        return parts