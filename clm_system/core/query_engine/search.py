# File: clm_system/core/query_engine/search.py
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
        classification = await self.classifier.classify(query)
        query_type = classification.get('query_type', 'semantic')
        logger.info(f"Query classified as {query_type}: {query}")
        
        # Convert document_type filter to list if needed
        if filters and 'metadata.document_type' in filters:
            doc_types = [filters['metadata.document_type']]
        else:
            doc_types = classification.get('doc_types', ['contract', 'email', 'recap', 'deal'])
        
        # Parallel searches per document type
        search_tasks = []
        for doc_type in doc_types:
            task = self._search_by_type(query, query_type, filters, top_k, doc_type)
            search_tasks.append(task)
        
        # Gather results from all searches
        type_results = await asyncio.gather(*search_tasks)
        
        # Merge results from different document types
        results = self._merge_results(type_results, top_k)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "metadata": {
                "query_type": query_type,
                "filters_applied": filters is not None,
                "document_types": doc_types
            },
            "execution_time_ms": execution_time
        }
    
    async def _search_by_type(self, query: str, query_type: str, 
                             filters: Optional[Dict[str, Any]], 
                             top_k: int, doc_type: str) -> List[Dict[str, Any]]:
        """
        Perform search for a specific document type.
        
        Args:
            query: Search query
            query_type: Classification of query (structured, semantic, hybrid)
            filters: Metadata filters
            top_k: Number of results to return
            doc_type: Document type to search for
            
        Returns:
            List of search results for this document type
        """
        # Add document type to filters
        type_filters = filters.copy() if filters else {}
        type_filters["metadata.document_type"] = doc_type
        #query_type = "structured"
        if query_type == "structured":
            # Structured search using Elasticsearch
            results = await self.es_client.search(query, type_filters, top_k)
        elif query_type == "semantic":
            # Semantic search using Qdrant
            query_embedding = compute_embedding(query, self.embedding_model)
            results = await self.qdrant_client.search(embedding=query_embedding, filters=type_filters, top_k=top_k)
        else:  # hybrid
            # Compute embedding here before passing to search
            query_embedding = compute_embedding(query, self.embedding_model)
            
            # Run searches in parallel
            es_results, qdrant_results = await asyncio.gather(
                self.es_client.search(query, type_filters, top_k * 2),
                self.qdrant_client.search(embedding=query_embedding, filters=type_filters, top_k=top_k * 2)
            )
            
            # Combine results using RRF
            results = reciprocal_rank_fusion(
                es_results,
                qdrant_results,
                k=60,
                weight_a=0.4,  # Elasticsearch weight
                weight_b=0.6   # Vector search weight
            )[:top_k]
        
        # Add document type to results metadata
        for result in results:
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["document_type"] = doc_type
        
        return results
    
    def _merge_results(self, type_results: List[List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
        """
        Merge results from different document types, sorting by relevance.
        
        Args:
            type_results: List of result lists from different document types
            top_k: Maximum number of results to return
            
        Returns:
            Combined and sorted list of results
        """
        # Flatten results from all document types
        all_results = []
        for results in type_results:
            all_results.extend(results)
        
        # Sort by relevance score
        sorted_results = sorted(all_results, key=lambda x: -x["relevance_score"])
        
        # Return top-k
        return sorted_results[:top_k]