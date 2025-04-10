# clm_system/core/pipeline/cleaning/email.py
import re
from html import unescape
from bs4 import BeautifulSoup
from .base import CleanerABC

class EmailCleaner(CleanerABC):
    def clean(self, data: dict) -> dict:
        """
        Clean and normalize email data.
        
        Args:
            data: Raw email data
            
        Returns:
            Cleaned email data
        """
        cleaned = data.copy()
        
        # Clean email addresses
        for field in ["from", "to", "cc", "bcc"]:
            if "metadata" in cleaned and field in cleaned["metadata"]:
                if isinstance(cleaned["metadata"][field], list):
                    cleaned["metadata"][field] = [
                        self._normalize_email(email) for email in cleaned["metadata"][field]
                    ]
                else:
                    cleaned["metadata"][field] = self._normalize_email(cleaned["metadata"][field])
        
        # Clean title/subject
        if "title" in cleaned:
            cleaned["title"] = self._clean_subject(cleaned["title"])
            
        # Clean clauses text (usually email body)
        if "clauses" in cleaned:
            for clause in cleaned["clauses"]:
                if "text" in clause:
                    # Check if this is HTML content
                    if clause.get("metadata", {}).get("format") == "html" or \
                       (clause["text"].startswith("<") and ">" in clause["text"]):
                        clause["text"] = self._clean_html(clause["text"])
                    else:
                        clause["text"] = self._clean_text(clause["text"])
        
        return cleaned
    
    def _normalize_email(self, email_address):
        """Normalize email addresses."""
        if not email_address:
            return email_address
            
        if isinstance(email_address, str):
            # Convert to lowercase
            email_address = email_address.lower()
            
            # Extract actual email from "Name <email>" format
            match = re.search(r'<([^>]+)>', email_address)
            if match:
                email_address = match.group(1)
                
            # Remove whitespace
            email_address = email_address.strip()
            
        return email_address
    
    def _clean_subject(self, subject):
        """Clean email subject/title."""
        if not subject:
            return subject
            
        # Remove common prefixes
        prefixes = ["RE:", "FW:", "FWD:", "Re:", "Fw:", "Fwd:"]
        cleaned_subject = subject
        for prefix in prefixes:
            if cleaned_subject.startswith(prefix):
                cleaned_subject = cleaned_subject[len(prefix):].strip()
                
        # Remove excessive whitespace
        cleaned_subject = re.sub(r'\s+', ' ', cleaned_subject)
        
        return cleaned_subject.strip()
    
    def _clean_html(self, html_content):
        """Convert HTML content to plain text."""
        if not html_content:
            return ""
            
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove scripts, styles, and hidden divs
            for tag in soup(["script", "style", "meta", "head"]):
                tag.extract()
                
            # Get text content
            text = soup.get_text(separator=' ')
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Replace HTML entities
            text = unescape(text)
            
            # Remove email signature markers and footers
            signature_markers = ["--", "Best regards", "Regards", "Thanks,", "Thank you,", "Sincerely,"]
            for marker in signature_markers:
                if marker in text:
                    # Find position and truncate (keeping some context)
                    pos = text.find(marker)
                    # Keep the marker but limit what comes after
                    signature_limit = min(pos + 100, len(text))
                    text = text[:signature_limit]
                    break
                    
            return text.strip()
        except Exception:
            # If HTML parsing fails, fall back to simple cleanup
            return self._clean_text(html_content)
    
    def _clean_text(self, text):
        """Clean plain text email content."""
        if not text:
            return ""
            
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove common email quotes (lines starting with >)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith('>')]
        text = '\n'.join(cleaned_lines)
        
        # Remove common signature separators
        text = re.sub(r'-{2,}|_{2,}|={2,}', '', text)
        
        # Remove common email footers
        footer_patterns = [
            r'Sent from my iPhone',
            r'Sent from my mobile device',
            r'Get Outlook for',
            r'CONFIDENTIALITY NOTICE:',
            r'DISCLAIMER:',
            r'PRIVILEGED AND CONFIDENTIAL',
        ]
        
        for pattern in footer_patterns:
            match = re.search(pattern, text)
            if match:
                text = text[:match.start()]
                
        return text.strip()