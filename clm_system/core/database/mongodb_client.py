
# File: clm_system/core/database/mongodb_client.py
import logging
from typing import Dict, List, Any, Optional

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from clm_system.config import settings

logger = logging.getLogger(__name__)

class MongoDBClient:
    """Client for interacting with MongoDB."""
    
    def __init__(self):
        self.client = MongoClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_database]
        self.contracts_collection = self.db["contracts"]
    
    async def insert_contract(self, contract: Dict[str, Any]) -> str:
        """
        Inserts a contract into the MongoDB collection.
        
        Args:
            contract: Contract data
            
        Returns:
            ID of the inserted contract
        """
        try:
            # Convert datetime to string for MongoDB
            contract = contract.copy()
            if "created_at" in contract:
                contract["created_at"] = contract["created_at"].isoformat()
            if "updated_at" in contract:
                contract["updated_at"] = contract["updated_at"].isoformat()
                
            result = self.contracts_collection.insert_one(contract)
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"MongoDB insert error: {str(e)}")
            raise
        
    async def get_contract(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a contract by ID.
        
        Args:
            contract_id: ID of the contract to retrieve
            
        Returns:
            Contract data or None if not found
        """
        try:
            contract = self.contracts_collection.find_one({"id": contract_id})
            return contract
        except PyMongoError as e:
            logger.error(f"MongoDB get error: {str(e)}")
            raise
    
    async def get_contracts(
        self, filters: Optional[Dict[str, Any]] = None, limit: int = 100, skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieves contracts matching the given filters.
        
        Args:
            filters: Query filters
            limit: Maximum number of contracts to return
            skip: Number of contracts to skip
            
        Returns:
            List of contracts
        """
        try:
            query = filters or {}
            contracts = list(
                self.contracts_collection
                .find(query)
                .limit(limit)
                .skip(skip)
            )
            return contracts
        except PyMongoError as e:
            logger.error(f"MongoDB get_contracts error: {str(e)}")
            raise
