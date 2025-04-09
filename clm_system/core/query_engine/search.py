# File: clm_system/core/queryEngine/search.py
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union

from clm_system.config import settings
from clm_system.core.database.elasticsearch_client import ElasticsearchClient
from clm_system.core.database.qdrant_client import QdrantClient
from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding
from .query_classifier import QueryClassifier
from .helpers import reciprocal_rank_fusion  # Add this import

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Routes queries to either structured search (Elasticsearch) or 
    semantic search (Qdrant) based on query analysis.
    """
    
    def __init__(self):
        self.es_client = ElasticsearchClient()
        self.qdrant_client = QdrantClient()
        self.embedding_model = get_embedding_model()
        self.top_k = settings.default_top_k
        self.classifier = QueryClassifier()
    
    async def route_query(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyzes the query and routes it to the appropriate search engine.
        
        Args:
            query: User's search query
            filters: Optional metadata filters
            top_k: Number of results to return
            
        Returns:
            Dict containing search results and metadata
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = self.top_k
        
        # Determine query type using classifier
        query_type = await self.classifier.classify(query)
        logger.info(f"Query classified as {query_type}: {query}")
        
        results = []
        
        if query_type == "structured":
            # Structured search using Elasticsearch
            results = await self.es_client.search(query, filters, top_k)
        elif query_type == "semantic":
            # Semantic search using Qdrant
            query_embedding = compute_embedding(query, self.embedding_model)
            results = await self.qdrant_client.search(query_embedding, filters, top_k)
        else:  # hybrid
            # Compute embedding here before passing to search
            query_embedding = compute_embedding(query, self.embedding_model)
            
            # Run searches in parallel
            es_results, qdrant_results = await asyncio.gather(
                self.es_client.search(query, filters, top_k * 2),
                self.qdrant_client.search(query_embedding, filters, top_k * 2)
            )
            
            # Combine results using RRF
            results = reciprocal_rank_fusion(
                es_results,
                qdrant_results,
                k=60,
                weight_a=0.4,  # Elasticsearch weight
                weight_b=0.6   # Vector search weight
            )[:top_k]
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "metadata": {
                "query_type": query_type,
                "filters_applied": filters is not None
            },
            "execution_time_ms": execution_time
        }
    
    # This method is no longer needed since we're using the QueryClassifier
    # It can be removed or kept as a fallback
    def _heuristic_classify(self, query: str) -> str:
        """
        Classifies a query as structured, semantic, or hybrid using heuristics.
        Used as a fallback when classifier is unavailable.
        
        Args:
            query: The user's search query
            
        Returns:
            String indicating query type: "structured", "semantic", or "hybrid"
        """
        structured_keywords = [
            "date:", "type:", "status:", "party:", "before:", "after:",
            "contract type", "effective date", "expiration date", "status is"
        ]
        
        has_structured = any(keyword in query.lower() for keyword in structured_keywords)
        
        if len(query.split()) <= 3 and not has_structured:
            return "semantic"
        
        if len(query.split()) > 3 and has_structured:
            return "hybrid"
        
        if has_structured:
            return "structured"
        
        return "semantic"