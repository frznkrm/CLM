from abc import ABC, abstractmethod

class PreprocessorABC(ABC):
    @abstractmethod
    def process(self, file_path: str) -> dict:
        pass