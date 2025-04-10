# file: core/pipeline/chunking/base.py 
from abc import ABC, abstractmethod

class ChunkerABC(ABC):
    @abstractmethod
    def chunk(self, raw: dict) -> dict:
        pass
