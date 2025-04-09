from abc import ABC, abstractmethod

class CleanerABC(ABC):
    @abstractmethod
    def clean(self, raw: dict) -> dict:
        pass
