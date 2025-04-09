
# File: clm_system/core/search.py
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union

from clm_system.config import settings
from clm_system.core.database.elasticsearch_client import ElasticsearchClient
from clm_system.core.database.qdrant_client import QdrantClient
from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding

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
        
        # Determine if this is a structured or semantic query
        query_type = self._classify_query(query)
        logger.info(f"Query classified as {query_type}: {query}")
        
        if top_k is None:
            top_k = self.top_k
        
        results = []
        
        if query_type == "structured":
            # Structured search using Elasticsearch
            results = await self.es_client.search(query, filters, top_k)
        elif query_type == "semantic":
            # Semantic search using Qdrant
            query_embedding = compute_embedding(query, self.embedding_model)
            results = await self.qdrant_client.search(query_embedding, filters, top_k)
        else:  # hybrid
            # Hybrid search - combine results from both
            es_results = await self.es_client.search(query, filters, top_k)
            
            query_embedding = compute_embedding(query, self.embedding_model)
            qdrant_results = await self.qdrant_client.search(query_embedding, filters, top_k)
            
            # Combine and deduplicate results (simple approach for MVP)
            combined_results = es_results + qdrant_results
            seen_ids = set()
            results = []
            
            for result in combined_results:
                if result["clause_id"] not in seen_ids:
                    seen_ids.add(result["clause_id"])
                    results.append(result)
                    if len(results) >= top_k:
                        break
        
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
    
    def _classify_query(self, query: str) -> str:
        """
        Classifies a query as structured, semantic, or hybrid.
        
        For MVP, using a simple heuristic approach. In a more advanced implementation,
        this would use a trained classifier or LLM.
        
        Args:
            query: The user's search query
            
        Returns:
            String indicating query type: "structured", "semantic", or "hybrid"
        """
        # MVP implementation - simple keyword-based heuristics
        structured_keywords = [
            "date:", "type:", "status:", "party:", "before:", "after:",
            "contract type", "effective date", "expiration date", "status is"
        ]
        
        # Check if query contains structured search patterns
        has_structured = any(keyword in query.lower() for keyword in structured_keywords)
        
        # If query is short and has no structured keywords, assume it's semantic
        if len(query.split()) <= 3 and not has_structured:
            return "semantic"
        
        # If query is longer and has structured keywords, assume it's hybrid
        if len(query.split()) > 3 and has_structured:
            return "hybrid"
        
        # If query has structured keywords, assume it's structured
        if has_structured:
            return "structured"
        
        # Default to semantic search
        return "semantic"
