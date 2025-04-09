# clm_system/core/pipeline/chunking/contract.py
from typing import List
import spacy
from .base import ChunkerABC
from clm_system.config import settings

class ContractChunker(ChunkerABC):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes("parser", "ner")  # Keep only tokenizer/sentence boundary detection

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
                # Handle very long sentences
                for i in range(0, sentence_len, size - overlap):
                    chunk_tokens = sentence_tokens[i:min(i + size, sentence_len)]
                    chunks.append(" ".join(chunk_tokens))
            else:
                if current_length + sentence_len > size:
                    # Finalize current chunk
                    chunks.append(" ".join(current_chunk))
                    # Start new chunk with overlap tokens from previous chunk
                    overlap_tokens = min(overlap, len(current_chunk))
                    current_chunk = current_chunk[-overlap_tokens:] if overlap_tokens > 0 else []
                    current_length = len(current_chunk)
                
                current_chunk.extend(sentence_tokens)
                current_length += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks