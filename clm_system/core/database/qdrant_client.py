# File: clm_system/core/database/qdrant_client.py
import logging
import uuid
from typing import Dict, List, Any, Optional

#from qdrant_client import QdrantClient as QdrantClientLib, models
from qdrant_client import AsyncQdrantClient, models
from clm_system.config import settings

logger = logging.getLogger(__name__)

class QdrantClient:
    """Client for interacting with Qdrant vector database with multi-document support"""
    
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.qdrant_uri)
        self.collection_name = "document_chunks"  # Generic collection name
        self.vector_size = settings.vector_dimension
    
    async def ensure_collection(self):
        """Ensures the vector collection exists with updated schema"""
        try:
            collections_response = await self.client.get_collections() # Await the call first
            collections = collections_response.collections           # THEN access the attribute
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )

                # Create generic payload indexes
                index_config = [
                    ("document_id", models.PayloadSchemaType.KEYWORD),
                    ("chunk_type", models.PayloadSchemaType.KEYWORD),
                    ("document_type", models.PayloadSchemaType.KEYWORD),
                    ("tags", models.PayloadSchemaType.KEYWORD),
                    ("metadata.has_attachments", models.PayloadSchemaType.BOOL),  # Index for boolean metadata
                ]

                for field, schema_type in index_config:
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=schema_type
                    )

                logger.info(f"Created Qdrant collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {str(e)}")
            raise

    async def store_embedding(
        self,
        document_id: str,
        document_title: str,
        chunk_id: str,
        chunk_type: str,
        content: str,
        embedding: List[float],
        document_type: str,
        chunk_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Stores any type of document chunk embedding
        """
        try:
            await self.ensure_collection()

            # Ensure metadata is a dictionary
            metadata_dict = metadata or {}
            
            # Ensure boolean values remain as booleans, not strings
            if "has_attachments" in metadata_dict:
                # Explicitly convert to boolean if it's not already
                if not isinstance(metadata_dict["has_attachments"], bool):
                    metadata_dict["has_attachments"] = bool(metadata_dict["has_attachments"])
                logger.debug(f"has_attachments is set to {metadata_dict['has_attachments']} ({type(metadata_dict['has_attachments']).__name__})")

            payload = {
                "document_id": document_id,
                "document_title": document_title,
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "document_type": document_type,
                "content": content,
                "metadata": metadata_dict,
            }

            if chunk_title:
                payload["chunk_title"] = chunk_title

            # Generate unique ID across all document types
            unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{chunk_id}"))

            # Log payload for debugging
            logger.debug(f"Storing point with payload: {payload}")

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=unique_id,
                    vector=embedding,
                    payload=payload
                )]
            )
            return chunk_id
        except Exception as e:
            logger.error(f"Storage error: {str(e)}")
            raise

    async def search(
    self,
    embedding: List[float],
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
        """Generic search across all document types with filters"""
        try:
            await self.ensure_collection()
            query_filter = self._build_filter(filters)

            response = await self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )

            return [self._format_result(hit) for hit in response]
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
    
    async def scroll(self, document_id: str) -> List[Dict[str, Any]]:
        """Helper method to verify points are stored in Qdrant"""
        try:
            scroll_filter = models.Filter(
                must=[models.FieldCondition(
                    key="document_id", 
                    match=models.MatchValue(value=document_id)
                )]
            )
            
            points, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Log the first point's payload structure for debugging
            if points:
                logger.debug(f"Sample point payload structure: {points[0].payload}")
                # Check if has_attachments exists and log its type
                if 'metadata' in points[0].payload and 'has_attachments' in points[0].payload['metadata']:
                    val = points[0].payload['metadata']['has_attachments']
                    logger.debug(f"has_attachments value: {val}, type: {type(val).__name__}")
            
            return [self._format_result(point) for point in points]
        except Exception as e:
            logger.error(f"Scroll error: {str(e)}")
            raise
    
    async def debug_points(self, document_id: str):
        """Debug method to print detailed point information"""
        try:
            points, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="document_id", 
                        match=models.MatchValue(value=document_id)
                    )]
                ),
                limit=10,
                with_payload=True
            )
            
            debug_info = []
            for i, point in enumerate(points):
                debug_info.append(f"Point {i+1}:")
                debug_info.append(f"  ID: {point.id}")
                debug_info.append(f"  Payload: {point.payload}")
                
                # Check metadata structure
                if 'metadata' in point.payload:
                    metadata = point.payload['metadata']
                    debug_info.append(f"  Metadata:")
                    for key, value in metadata.items():
                        debug_info.append(f"    {key}: {value} (type: {type(value).__name__})")
            
            return "\n".join(debug_info)
        except Exception as e:
            logger.error(f"Debug error: {str(e)}")
            return f"Error debugging points: {str(e)}"
    
    def _format_result(self, hit) -> Dict[str, Any]:
        """Standardizes result format across document types"""
        payload = hit.payload
        return {
            "document_id": payload["document_id"],
            "document_title": payload["document_title"],
            "document_type": payload["document_type"],
            "chunk_id": payload["chunk_id"],
            "chunk_type": payload["chunk_type"],
            "content": payload["content"],
            "relevance_score": hit.score if hasattr(hit, "score") else 1.0,
            "metadata": payload.get("metadata", {}),
            "chunk_title": payload.get("chunk_title")
        }
    
    def _build_filter(self, filters: Optional[Dict[str, Any]] = None) -> Optional[models.Filter]:
        """Convert filters dictionary to Qdrant Filter object"""
        if not filters:
            return None

        conditions = []
        for field, value in filters.items():
            # Handle nested metadata filters
            if field.startswith("metadata."):
                key = field.split(".", 1)[1]
                # Ensure booleans are correctly typed
                if isinstance(value, bool) or value in (True, False):
                    conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=bool(value))
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value)
                        )
                    )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value)
                    )
                )

        return models.Filter(must=conditions) if conditions else None
    async def aclose(self):
        """Asynchronously close the Qdrant client connection."""
        try:
            # AsyncQdrantClient uses close() which returns an awaitable
            await self.client.close()
            logger.info("Closed AsyncQdrantClient")
        except Exception as e:
            logger.error(f"Failed to close AsyncQdrantClient: {str(e)}")