# clm_system/core/database/mongodb_client.py
import logging
from dateutil import parser
from typing import Any, Dict, List, Optional, Tuple  # Add Tuple to imports
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from motor.motor_asyncio import AsyncIOMotorClient
from clm_system.config import settings

logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self):
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_database]
        self.documents_collection = self.db.documents  # Generic collection name
    
    async def insert_document(self, document: Dict[str, Any]) -> str:
        """
        Inserts any type of document into MongoDB.
        
        Args:
            document: Document data (contract, deal, email, etc.)
            
        Returns:
            ID of the inserted document
        """
        try:
            # Convert datetime to string for MongoDB
            doc = document.copy()
            self._normalize_datetimes(doc)
            
            # Validate required fields
            if "id" not in doc:
                raise ValueError("Document must contain an 'id' field")
            if "metadata" not in doc or "document_type" not in doc["metadata"]:
                raise ValueError("Document metadata must contain 'document_type'")
                
            result = await self.documents_collection.insert_one(doc)
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"MongoDB insert error: {str(e)}")
            raise
        
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves any document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document data or None if not found
        """
        try:
            document = await self.documents_collection.find_one({"id": document_id})
            return self._denormalize_datetimes(document)
        except PyMongoError as e:
            logger.error(f"MongoDB get error: {str(e)}")
            raise
    
    async def get_documents(
    self,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    skip: int = 0,
    sort: Optional[List[Tuple[str, int]]] = None  # Now properly typed
) -> List[Dict[str, Any]]:
        """
        Retrieves documents matching the given filters.
        
        Args:
            filters: Query filters
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            sort: List of (field, direction) tuples to sort by
            
        Returns:
            List of documents
        """
        try:
            query = filters or {}
            cursor = self.documents_collection.find(query)
            
            if sort:
                cursor = cursor.sort(sort)
                
            documents = await cursor.skip(skip).limit(limit).to_list(length=None)
            return [self._denormalize_datetimes(doc) for doc in documents]
        except PyMongoError as e:
            logger.error(f"MongoDB get_documents error: {str(e)}")
            raise
    def _normalize_datetimes(self, document: Dict[str, Any]):
        """No longer convert datetime fields to strings"""
        pass

    def _denormalize_datetimes(self, document: Optional[Dict[str, Any]]):
        """No longer needed as dates are stored as Date objects"""
        return document

    # def _normalize_datetimes(self, document: Dict[str, Any]):
    #     """Convert datetime fields to ISO strings for MongoDB storage"""
    #     for field in ['created_at', 'updated_at']:
    #         if field in document:
    #             document[field] = document[field].isoformat()
                
    #     # Handle metadata dates
    #     if 'metadata' in document:
    #         for dt_field in ['effective_date', 'expiration_date', 'sent_date', 'meeting_date']:
    #             if dt_field in document['metadata'] and document['metadata'][dt_field] is not None:
    #                 document['metadata'][dt_field] = document['metadata'][dt_field].isoformat()

    # def _denormalize_datetimes(self, document: Optional[Dict[str, Any]]):
    #     """Convert ISO strings back to datetime objects"""
    #     if not document:
    #         return None
            
    #     for field in ['created_at', 'updated_at']:
    #         if field in document:
    #             document[field] = parser.isoparse(document[field])
                
    #     # Handle metadata dates
    #     if 'metadata' in document:
    #         for dt_field in ['effective_date', 'expiration_date', 'sent_date', 'meeting_date']:
    #             if dt_field in document['metadata'] and isinstance(document['metadata'][dt_field], str):
    #                 document['metadata'][dt_field] = parser.isoparse(document['metadata'][dt_field])
                    
    #     return document