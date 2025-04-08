
# File: clm_system/core/database/qdrant_client.py
import logging
from typing import Dict, List, Any, Optional

from qdrant_client import QdrantClient, models  # Import directly from the external library

from clm_system.config import settings

logger = logging.getLogger(__name__)

class  CustomQdrantClient:
    """Client for interacting with Qdrant vector database."""
    
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_uri)  # You might want to differentiate here if desired
        self.collection_name = "contract_clauses"
        self.vector_size = settings.vector_dimension
    
    async def ensure_collection(self):
        """Ensures the vector collection exists."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                
                # Create payload indexes for fast filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="contract_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="clause_type",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                
                logger.info(f"Created Qdrant collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {str(e)}")
            raise

    
    async def store_embedding(
        self,
        contract_id: str,
        contract_title: str,
        clause_id: str,
        clause_type: str,
        content: str,
        embedding: List[float],
        clause_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Stores a clause embedding in Qdrant.
        
        Args:
            contract_id: ID of the contract
            contract_title: Title of the contract
            clause_id: ID of the clause
            clause_type: Type of the clause
            content: Text content of the clause
            embedding: Vector embedding of the clause text
            clause_title: Optional title of the clause
            metadata: Additional metadata
            
        Returns:
            ID of the stored point
        """
        try:
            # Ensure collection exists
            await self.ensure_collection()
            
            # Build payload
            payload = {
                "contract_id": contract_id,
                "contract_title": contract_title,
                "clause_id": clause_id,
                "clause_type": clause_type,
                "content": content
            }
            
            if clause_title:
                payload["clause_title"] = clause_title
                
            if metadata:
                payload["metadata"] = metadata
            
            # Store point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=clause_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            return clause_id
        except Exception as e:
            logger.error(f"Qdrant store error: {str(e)}")
            raise
    
    async def search(
        self, 
        embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches for similar clauses in Qdrant.
        
        Args:
            embedding: Query vector embedding
            filters: Optional filters
            top_k: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Ensure collection exists
            await self.ensure_collection()
            
            # Build filter
            filter_query = None
            if filters:
                conditions = []
                for field, value in filters.items():
                    if field == "contract_id":
                        conditions.append(
                            models.FieldCondition(
                                key="contract_id",
                                match=models.MatchValue(value=value)
                            )
                        )
                    elif field == "clause_type":
                        conditions.append(
                            models.FieldCondition(
                                key="clause_type",
                                match=models.MatchValue(value=value)
                            )
                        )
                
                if conditions:
                    filter_query = models.Filter(
                        must=conditions
                    )
            
            # Execute search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                query_filter=filter_query,
                limit=top_k
            )
            
            # Process results
            results = []
            for hit in search_results:
                results.append({
                    "clause_id": hit.payload["clause_id"],
                    "contract_id": hit.payload["contract_id"],
                    "contract_title": hit.payload["contract_title"],
                    "clause_type": hit.payload["clause_type"],
                    "clause_title": hit.payload.get("clause_title"),
                    "content": hit.payload["content"],
                    "relevance_score": hit.score,
                    "metadata": hit.payload.get("metadata", {})
                })
            
            return results
        except Exception as e:
            logger.error(f"Qdrant search error: {str(e)}")
            raise
