# clm_system/core/pipeline/chunking/contract.py
from typing import List
from spacy.lang.en import English
import spacy
import logging
from ..base import BaseChunker
from clm_system.config import settings
from typing import Dict, Any
logger = logging.getLogger(__name__)

class ContractChunker(BaseChunker, doc_type="contract"):
    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        logger.info("SpaCy pipeline components: %s", self.nlp.pipe_names)

    def chunk(self, text: str) -> List[str]:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        size = settings.chunk_size
        overlap = settings.chunk_overlap
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = sentence.split()
            sentence_len = len(sentence_tokens)
            
            if sentence_len > size:
                for i in range(0, sentence_len, size - overlap):
                    chunk_tokens = sentence_tokens[i:min(i + size, sentence_len)]
                    chunks.append(" ".join(chunk_tokens))
            else:
                if current_length + sentence_len > size:
                    chunks.append(" ".join(current_chunk))
                    overlap_tokens = min(overlap, len(current_chunk))
                    current_chunk = current_chunk[-overlap_tokens:] if overlap_tokens > 0 else []
                    current_length = len(current_chunk)
                
                current_chunk.extend(sentence_tokens)
                current_length += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data  # Implement if additional processing is needed