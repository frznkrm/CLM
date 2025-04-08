

# File: clm_system/core/utils/embeddings.py
import logging
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

from clm_system.config import settings

logger = logging.getLogger(__name__)

# Global cache for embedding model
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """
    Gets or initializes the embedding model.
    
    Returns:
        SentenceTransformer model instance
    """
    global _embedding_model
    
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        try:
            _embedding_model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    return _embedding_model

def compute_embedding(text: str, model: Optional[SentenceTransformer] = None) -> List[float]:
    """
    Computes embedding for a given text.
    
    Args:
        text: Input text to embed
        model: Optional pre-loaded model (if not provided, will get from cache)
        
    Returns:
        List of floats representing the text embedding
    """
    if model is None:
        model = get_embedding_model()
    
    try:
        # Compute embedding
        embedding = model.encode(text)
        
        # Convert to list if it's a tensor or numpy array
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.tolist()
        elif hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        
        return embedding
    except Exception as e:
        logger.error(f"Error computing embedding: {str(e)}")
        raise