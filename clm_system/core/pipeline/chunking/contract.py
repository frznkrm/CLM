# clm_system/core/pipeline/chunking/contract.py
from .base import ChunkerABC
from clm_system.config import settings

class ContractChunker(ChunkerABC):
    def chunk(self, text: str) -> list[str]:
        # simple sliding‑window chunker
        size, overlap = settings.chunk_size, settings.chunk_overlap
        tokens = text.split()
        chunks = []
        for i in range(0, len(tokens), size - overlap):
            chunk = " ".join(tokens[i : i + size])
            chunks.append(chunk)
            if i + size >= len(tokens):
                break
        return chunks
