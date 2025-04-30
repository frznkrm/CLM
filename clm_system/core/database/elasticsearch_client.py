# file: core/database/elasticsearch_client.py
import logging
from typing import Dict, List, Optional, Any
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
from openai import AsyncOpenAI  # Added for the new function
import ipdb


# Configure logging at the module level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
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
                                    "title": {
                                        "type": "text",
                                        "fields": {
                                            "keyword": {"type": "keyword", "ignore_above": 256}
                                        }
                                    },
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
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        use_llm: bool = True
    ) -> List[Dict[str, Any]]:
        try:
            await self.ensure_index()
            doc_type = filters.get("metadata.document_type", "contract") if filters else "contract"

            # If use_llm is True, generate the structured query using the LLM
            if use_llm:
                #ipdb.set_trace()
                structured_query = await self.test_generate_search_query(query)
                if structured_query is None:
                    logger.warning("LLM query generation failed, falling back to original query")
                    structured_query = query
            else:
                structured_query = query

            # Build the search query using the structured query
            search_query = {
                "query": {
                    "bool": {
                        "must": [],
                        "filter": []
                    }
                },
                "size": top_k
            }

            if ":" in structured_query:
                field, value = structured_query.split(":", 1)
                value = value.strip()

                if field == "type":
                    # normalize 'SLA' -> 'sla' for keyword match
                    normalized = value.lower()
                    search_query["query"]["bool"]["must"].append({
                        "nested": {
                            "path": "clauses",
                            "query": {
                                "term": {"clauses.type": normalized}
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
                                "match_phrase": {"metadata.parties.name": value}
                            }
                        }
                    })

                elif field.startswith("clauses."):
                    subfield = field.split(".", 1)[1]
                    # if they explicitly query clauses.type, lowercase it too
                    if subfield == "type":
                        value = value.lower()

                    if subfield == "title":
                        search_query["query"]["bool"]["must"].append({
                            "nested": {
                                "path": "clauses",
                                "query": {
                                    "term": {"clauses.title.keyword": value}
                                },
                                "inner_hits": {"size": 1}
                            }
                        })
                    else:
                        search_query["query"]["bool"]["must"].append({
                            "nested": {
                                "path": "clauses",
                                "query": {
                                    "term": {f"clauses.{subfield}": value}
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
                        "query": structured_query,
                        "fields": ["title^2", "clauses.title^1.5", "clauses.text"]
                    }
                })

            if filters:
                for field, value in filters.items():
                    search_query["query"]["bool"]["filter"].append({
                        "term": {field: value}
                    })

            logger.debug(f"Elasticsearch query DSL for '{structured_query}': {search_query}")
            response = await self.client.search(
                index=self.index_name,
                body=search_query
            )

            # After getting the search response
            logger.debug(f"Full Elasticsearch response: {response}")
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

            logger.debug(f"Elasticsearch results for '{structured_query}': {results}")
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
    async def test_generate_search_query(self, user_input: str) -> Optional[str]:
        """
        Generate a search query based on user input using the LLM for testing purposes.
        
        Args:
            user_input (str): The natural language input from the user.
        
        Returns:
            Optional[str]: The generated search query, or None if generation fails.
        """
        # Initialize the AsyncOpenAI client with the same configuration as the original code
        client = AsyncOpenAI(
            base_url="http://192.168.10.1:1234/v1",
            api_key="qwen2.5-coder-14b-instruct",
            timeout=30.0
        )
        
        # System message instructing the LLM on how to generate search queries
        system_message = """
You are a search query generator for a legal document database. 
Based on the user's natural language input, generate a search query compatible with an Elasticsearch search system.
Use field specifiers like 'type:' for clause type, 'party:' for party name, or 'clauses.title:' for clause title when appropriate.
If no specific field is implied, provide a general keyword query.
Respond with only the generated search query, without additional text or explanation.

Examples:
- User: "Find confidentiality clauses"
  Query: "type:confidentiality"
- User: "Contracts involving CompanyX"
  Query: "party:CompanyX"
- User: "Documents about merger agreements"
  Query: "merger agreements"
"""
        
        try:
            # Send the request to the LLM
            response = await client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=50    # Limit output length, as queries are typically short
            )
            
            # Extract and clean the generated query
            generated_query = response.choices[0].message.content.strip()
            # Remove surrounding quotes if present
            if generated_query.startswith('"') and generated_query.endswith('"'):
                generated_query = generated_query[1:-1]
            logger.info(f"Generated search query for '{user_input}': {generated_query}")
            print(f"Generated search query: {generated_query}")
            return generated_query
        
        except Exception as e:
            logger.error(f"Failed to generate search query for '{user_input}': {str(e)}")
            return None