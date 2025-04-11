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
        self.index_name = "documents"  # Changed from 'contracts' to 'documents'
    
    async def ensure_index(self):
        """Ensures the documents index exists with proper mappings for all document types."""
        try:
            # Check if index exists
            exists = await self.client.indices.exists(index=self.index_name)
            if not exists:
                # Create index with mappings for all document types
                mappings = {
                    "mappings": {
                        "dynamic": "strict",
                        "properties": {
                            "id": {"type": "keyword"},
                            "title": {"type": "text", "analyzer": "standard"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "metadata": {
                                "properties": {
                                    "document_type": {"type": "keyword"},
                                    "tags": {"type": "keyword"},
                                    "status": {"type": "keyword"},
                                    "contract_type": {"type": "keyword"},
                                    "deal_type": {"type": "keyword"},
                                    "effective_date": {"type": "date"},
                                    "expiration_date": {"type": "date"},
                                    "parties": {
                                        "type": "nested",
                                        "properties": {
                                            "name": {"type": "text", "analyzer": "standard"},
                                            "id": {"type": "keyword"}
                                        }
                                    },
                                    "from_address": {"type": "keyword"},
                                    "to": {"type": "keyword"},
                                    "cc": {"type": "keyword"},
                                    "bcc": {"type": "keyword"},
                                    "subject": {"type": "text"},
                                    "has_attachments": {"type": "boolean"},
                                    "attachment_count": {"type": "integer"},  # Add this line
                                    "attachment_count": {"type": "integer"},  # Add this line
                                    "attachment_names": {"type": "keyword"},  # Add this line
                                    "sent_date": {"type": "date"},
                                    "volume": {"type": "keyword"},
                                    "price_per_unit": {"type": "float"},
                                    "meeting_date": {"type": "date"},
                                    "participants": {"type": "keyword"},
                                    "decisions": {"type": "text"},
                                    "action_items": {"type": "text"}
                                }
                            },  # Added missing comma here
                            # Content fields for different types
                            "content": {"type": "text"},  # Email content
                            "summary": {"type": "text"},  # Recap summary
                            "key_points": {"type": "text"},  # Recap key points
                            "financial_terms": {"type": "object"},  # Deal terms
                            "clauses": {
                                "type": "nested",
                                "properties": {
                                    "id": {"type": "keyword"},
                                    "title": {"type": "text", "analyzer": "standard"},
                                    "type": {"type": "keyword"},
                                    "text": {"type": "text", "analyzer": "standard"},
                                    "position": {"type": "integer"},
                                    "metadata": {
                                        "type": "object",
                                        "properties": {
                                            "attachment_names": {"type": "keyword"},  # Add this line
                                            "section_type": {"type": "keyword"}
                                        }
                                    }
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
    
    async def index_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Indexes a document in Elasticsearch.
        
        Args:
            document: Document data (any type)
            
        Returns:
            Elasticsearch response
        """
        try:
            # Ensure index exists
            await self.ensure_index()
            
            # Index document
            response = await self.client.index(
                index=self.index_name,
                id=document["id"],
                document=document,
                refresh=True  # Make document immediately searchable
            )
            return response
        except Exception as e:
            logger.error(f"Elasticsearch index error: {str(e)}")
            raise
    
    # Alias for backward compatibility
    async def index_contract(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for index_document for backward compatibility."""
        return await self.index_document(contract)
    
    async def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches for documents in Elasticsearch.
        
        Args:
            query: Search query
            filters: Query filters
            top_k: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            await self.ensure_index()
            
            # Determine document type for type-specific field mapping
            doc_type = filters.get("metadata.document_type", "contract") if filters else "contract"
            
            # Base query structure
            search_query = {
                "query": {
                    "bool": {
                        "must": []
                    }
                },
                "size": top_k
            }
            
            # Add type-specific queries
            if doc_type == "contract" or doc_type == "deal":
                # For contract/deal, search in clauses and title
                search_query["query"]["bool"]["must"] = [
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
            elif doc_type == "email":
                # For email, search in content and subject
                search_query["query"]["bool"]["must"] = [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "metadata.subject^2"]
                        }
                    }
                ]
            elif doc_type == "recap":
                # For recap, search in summary and key points
                search_query["query"]["bool"]["must"] = [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["summary^2", "key_points"]
                        }
                    }
                ]

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
                doc_type = source.get("metadata", {}).get("document_type", "contract")
                
                # Process results based on document type
                if doc_type in ["contract", "deal"]:
                    # Process contract/deal with clauses
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
                elif doc_type == "email":
                    # Process email results
                    content_preview = source.get("content", "")[:150]
                    results.append({
                        "clause_id": f"{source['id']}_content",
                        "contract_id": source["id"],
                        "contract_title": source.get("metadata", {}).get("subject", "Email"),
                        "clause_type": "email_content",
                        "clause_title": None,
                        "content": content_preview,
                        "relevance_score": hit["_score"],
                        "metadata": source.get("metadata", {})
                    })
                elif doc_type == "recap":
                    # Process recap results
                    summary_preview = source.get("summary", "")[:150]
                    results.append({
                        "clause_id": f"{source['id']}_summary",
                        "contract_id": source["id"],
                        "contract_title": source.get("title", "Meeting Recap"),
                        "clause_type": "recap_summary",
                        "clause_title": None,
                        "content": summary_preview,
                        "relevance_score": hit["_score"],
                        "metadata": source.get("metadata", {})
                    })
            
            return sorted(results, key=lambda x: -x["relevance_score"])[:top_k]
        except Exception as e:
            logger.error(f"Elasticsearch search error: {str(e)}")
            raise