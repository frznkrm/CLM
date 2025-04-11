# clm_system/core/pipeline/base.py
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Type, Any, Optional

class BaseStep(ABC):
    """Abstract base class for all pipeline steps with auto-registration"""
    _registry: ClassVar[Dict[str, Dict[str, Type["BaseStep"]]]] = {
        'ingestor': {},
        'cleaner': {},
        'chunker': {}
    }
    _step_type: str  # Must be set by subclasses (ingestor, cleaner, chunker)

    def __init_subclass__(cls, *, doc_type: Optional[str] = None, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Skip direct subclasses of BaseStep (abstract base classes)
        if BaseStep in cls.__bases__:
            return
        
        # Check if the class is abstract (has any abstract methods)
        is_abstract = bool(getattr(cls, '__abstractmethods__', None))
        
        if doc_type is None:
            if not is_abstract:
                raise TypeError(f"Concrete {cls._step_type} subclasses must provide 'doc_type'")
            return
        
        registry = cls._registry[cls._step_type]
        if doc_type in registry:
            raise ValueError(f"Duplicate {cls._step_type} registration for doc_type: {doc_type}")
        registry[doc_type] = cls
        cls.doc_type = doc_type

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the document data"""
        pass

class BaseIngestor(BaseStep, ABC):
    """Base class for document ingestion"""
    _step_type = 'ingestor'

class BaseCleaner(BaseStep, ABC):
    """Base class for document cleaning"""
    _step_type = 'cleaner'

class BaseChunker(BaseStep, ABC):
    """Base class for document chunking"""
    _step_type = 'chunker'
    
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks"""
        pass