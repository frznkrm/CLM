# file: core/pipeline/ingestion/base.py 
from abc import ABC, abstractmethod

class IngestorABC(ABC):
    @abstractmethod
    def ingest(self, raw: dict) -> dict:
        pass
