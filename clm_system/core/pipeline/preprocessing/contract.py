# clm_system/core/pipeline/preprocessing/contract.py
import asyncio
from typing import Dict, Any
from .base import PreprocessorABC
from .docx_processor import DOCXProcessor

class ContractPreprocessor(PreprocessorABC):
    """Preprocessor for contract DOCX files, using DOCXProcessor."""
    
    def __init__(self):
        self.processor = DOCXProcessor()

    async def process(self, file_path: str) -> Dict[str, Any]:
        """
        Process a DOCX file into a contract JSON structure using DOCXProcessor.

        Args:
            file_path (str): Path to the DOCX file.

        Returns:
            Dict[str, Any]: A dictionary representing the contract.
        """
        return await self.processor.process_docx(file_path)