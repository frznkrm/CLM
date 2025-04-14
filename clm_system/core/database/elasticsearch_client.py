import logging
from typing import Dict, List, Optional, Any
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    def __init__(self, host: str = "localhost", port: int = 9200):
        self.client = AsyncElasticsearch([f"http://{host}:{port}"])
        self.index_name = "documents"

    async def ensure_index(self):
        """Ensure the Elasticsearch index exists with appropriate mappings."""
        try:
            index_exists = await self.client.indices.exists(index=self.index_name)
            if not index_exists:
                mappings = {
                    "mappings": {
                        "properties": {
                            "id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "content": {"type": "text"},
                            "metadata": {
                                "type": "object",
                                "properties": {
                                    "document_type": {"type": "keyword"},
                                    "contract_type": {"type": "keyword"},
                                    "status": {"type": "keyword"},
                                    "jurisdiction": {"type": "keyword"},
                                    "parties": {
                                        "type": "nested",
                                        "properties": {
                                            "name": {"type": "text"},
                                            "id": {"type": "keyword"},
                                            "role": {"type": "keyword"}
                                        }
                                    },
                                    "tags": {"type": "keyword"}
                                }
                            },
                            "clauses": {
                                "type": "nested",
                                "properties": {
                                    "id": {"type": "keyword"},
                                    "title": {"type": "text"},
                                    "type": {"type": "keyword"},
                                    "text": {"type": "text"},
                                    "position": {"type": "integer"}
                                }
                            }
                        }
                    }
                }
                await self.client.indices.create(index=self.index_name, body=mappings)
                logger.info(f"Created Elasticsearch index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to ensure Elasticsearch index: {str(e)}")
            raise

    async def index_document(self, document: Dict[str, Any]):
        """Index a document in Elasticsearch."""
        try:
            await self.ensure_index()
            doc_id = document.get("id")
            await self.client.index(
                index=self.index_name,
                id=doc_id,
                body=document
            )
            logger.info(f"Indexed document {doc_id} in Elasticsearch")
        except Exception as e:
            logger.error(f"Elasticsearch indexing failed: {str(e)}")
            raise

    async def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches for documents in Elasticsearch.
        """
        try:
            await self.ensure_index()
            doc_type = filters.get("metadata.document_type", "contract") if filters else "contract"
            
            search_query = {
                "query": {
                    "bool": {
                        "must": [],
                        "filter": []
                    }
                },
                "size": top_k
            }
            
            if ":" in query:
                field, value = query.split(":", 1)
                value = value.strip()
                if field == "type":
                    search_query["query"]["bool"]["must"].append({
                        "nested": {
                            "path": "clauses",
                            "query": {
                                "match": {"clauses.type": value}
                            },
                            "inner_hits": {
                                "highlight": {
                                    "fields": {
                                        "clauses.text": {"fragment_size": 150}
                                    }
                                },
                                "_source": True,
                                "size": 3
                            }
                        }
                    })
                elif field == "party":
                    search_query["query"]["bool"]["must"].append({
                        "nested": {
                            "path": "metadata.parties",
                            "query": {
                                "match": {"metadata.parties.name": value}
                            }
                        }
                    })
                elif field.startswith("clauses."):
                    subfield = field.split(".", 1)[1]
                    search_query["query"]["bool"]["must"].append({
                        "nested": {
                            "path": "clauses",
                            "query": {
                                "match": {f"clauses.{subfield}": value}
                            },
                            "inner_hits": {
                                "highlight": {
                                    "fields": {
                                        "clauses.text": {"fragment_size": 150}
                                    }
                                },
                                "_source": True,
                                "size": 3
                            }
                        }
                    })
                else:
                    search_query["query"]["bool"]["must"].append({
                        "multi_match": {
                            "query": value,
                            "fields": ["title^2", "clauses.text"]
                        }
                    })
            else:
                search_query["query"]["bool"]["must"].append({
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "clauses.title^1.5", "clauses.text"]
                    }
                })
            
            if filters:
                for field, value in filters.items():
                    search_query["query"]["bool"]["filter"].append({
                        "term": {field: value}
                    })
            
            logger.debug(f"Elasticsearch query DSL for '{query}': {search_query}")
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
                        "clause_id": clause_source.get("id"),
                        "contract_id": source.get("id"),
                        "contract_title": source.get("title"),
                        "clause_type": clause_source.get("type"),
                        "clause_title": clause_source.get("title"),
                        "content": highlight_text,
                        "relevance_score": hit["_score"] * clause_hit["_score"],
                        "metadata": {
                            **source.get("metadata", {}),
                            **clause_source.get("metadata", {})
                        }
                    })
                
                if not inner_hits:
                    results.append({
                        "clause_id": None,
                        "contract_id": source.get("id"),
                        "contract_title": source.get("title"),
                        "clause_type": None,
                        "clause_title": None,
                        "content": source.get("title", "")[:150],
                        "relevance_score": hit["_score"],
                        "metadata": source.get("metadata", {})
                    })
            
            logger.debug(f"Elasticsearch results for '{query}': {results}")
            return sorted(results, key=lambda x: -x["relevance_score"])[:top_k]
        except Exception as e:
            logger.error(f"Elasticsearch search error for '{query}': {str(e)}")
            return []
    
    async def close(self):
        """Close the Elasticsearch client."""
        try:
            await self.client.close()
            logger.info("Closed Elasticsearch client")
        except Exception as e:
            logger.error(f"Failed to close Elasticsearch client: {str(e)}")