# File: clm_system/main.py
import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from clm_system.api.routes import router as api_router
from clm_system.config import settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CLM Smart Search",
    description="Contract Lifecycle Management with Smart Search capabilities",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "clm_system.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


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
