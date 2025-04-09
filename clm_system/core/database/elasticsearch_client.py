
# File: clm_system/core/database/elasticsearch_client.py
import logging
from typing import Dict, List, Any, Optional

from elasticsearch import AsyncElasticsearch, NotFoundError

from clm_system.config import settings

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    """Client for interacting with Elasticsearch."""
    
    def __init__(self):
        self.client = AsyncElasticsearch(settings.elasticsearch_uri)
        self.index_name = "contracts"
    
    async def ensure_index(self):
        """Ensures the contracts index exists with proper mappings."""
        try:
            # Check if index exists
            exists = await self.client.indices.exists(index=self.index_name)
            if not exists:
                # Create index with mappings
                mappings = {
                    "mappings": {
                        "properties": {
                            "id": {"type": "keyword"},
                            "title": {"type": "text", "analyzer": "standard"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "metadata": {
                                "properties": {
                                    "contract_type": {"type": "keyword"},
                                    "effective_date": {"type": "date"},
                                    "expiration_date": {"type": "date"},
                                    "parties": {
                                        "type": "nested",
                                        "properties": {
                                            "name": {"type": "text", "analyzer": "standard"},
                                            "id": {"type": "keyword"}
                                        }
                                    },
                                    "status": {"type": "keyword"},
                                    "tags": {"type": "keyword"}
                                }
                            },
                            "clauses": {
                                "type": "nested",
                                "properties": {
                                    "id": {"type": "keyword"},
                                    "title": {"type": "text", "analyzer": "standard"},
                                    "type": {"type": "keyword"},
                                    "text": {"type": "text", "analyzer": "standard"},
                                    "position": {"type": "integer"},
                                    "metadata": {"type": "object"}
                                }
                            }
                        }
                    }
                }
                await self.client.indices.create(index=self.index_name, body=mappings)
                logger.info(f"Created Elasticsearch index {self.index_name}")
        except Exception as e:
            logger.error(f"Error ensuring Elasticsearch index: {str(e)}")
            raise
    
    async def index_contract(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """
        Indexes a contract in Elasticsearch.
        
        Args:
            contract: Contract data
            
        Returns:
            Elasticsearch response
        """
        try:
            # Ensure index exists
            await self.ensure_index()
            
            # Index document
            response = await self.client.index(
                index=self.index_name,
                id=contract["id"],
                document=contract,
                refresh=True  # Make document immediately searchable
            )
            return response
        except Exception as e:
            logger.error(f"Elasticsearch index error: {str(e)}")
            raise
    
    async def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches for contracts in Elasticsearch.
        
        Args:
            query: Search query
            filters: Query filters
            top_k: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            await self.ensure_index()
            
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "nested": {
                                    "path": "clauses",
                                    "query": {
                                        "multi_match": {
                                            "query": query,
                                            "fields": ["clauses.title^1.5", "clauses.text"]
                                        }
                                    },
                                    "inner_hits": {
                                        "highlight": {
                                            "fields": {
                                                "clauses.text": {"fragment_size": 150}
                                            }
                                        },
                                        "_source": True,
                                        "size": 3  # Maximum number of clause matches per contract
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^2"]
                                }
                            }
                        ]
                    }
                },
                "size": top_k
            }

            # Add filters if provided
            if filters:
                filter_clauses = []
                for field, value in filters.items():
                    if field.startswith("metadata."):
                        filter_clauses.append({"term": {field: value}})
                    elif field.startswith("clauses."):
                        filter_clauses.append({
                            "nested": {
                                "path": "clauses",
                                "query": {"term": {field: value}}
                            }
                        })
                    else:
                        filter_clauses.append({"term": {field: value}})
                
                if filter_clauses:
                    search_query["query"]["bool"]["filter"] = filter_clauses

            response = await self.client.search(
                index=self.index_name,
                body=search_query
            )

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                inner_hits = hit.get("inner_hits", {}).get("clauses", {}).get("hits", {}).get("hits", [])
                
                for clause_hit in inner_hits:
                    clause_source = clause_hit["_source"]
                    highlights = clause_hit.get("highlight", {}).get("clauses.text", [])
                    highlight_text = highlights[0] if highlights else clause_source.get("text", "")[:150]
                    
                    results.append({
                        "clause_id": clause_source["id"],
                        "contract_id": source["id"],
                        "contract_title": source["title"],
                        "clause_type": clause_source["type"],
                        "clause_title": clause_source.get("title"),
                        "content": highlight_text,
                        "relevance_score": hit["_score"] * clause_hit["_score"],
                        "metadata": {
                            **source.get("metadata", {}),
                            **clause_source.get("metadata", {})
                        }
                    })
            
            return sorted(results, key=lambda x: -x["relevance_score"])[:top_k]
        except Exception as e:
            logger.error(f"Elasticsearch search error: {str(e)}")
            raise
 