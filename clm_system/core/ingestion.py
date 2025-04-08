
# File: clm_system/core/ingestion.py
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from clm_system.config import settings
from clm_system.core.database.elasticsearch_client import ElasticsearchClient
from clm_system.core.database.mongodb_client import MongoDBClient
from clm_system.core.database.qdrant_client import QdrantClient
from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding

logger = logging.getLogger(__name__)

class ContractIngestionService:
    """
    Service for ingesting contracts into the system.
    Handles storing the contract in MongoDB, indexing metadata in Elasticsearch,
    and computing embeddings for clauses to store in Qdrant.
    """
    
    def __init__(self):
        self.mongodb_client = MongoDBClient()
        self.es_client = ElasticsearchClient()
        self.qdrant_client = QdrantClient()
        self.embedding_model = get_embedding_model()
    
    async def ingest_contract(self, contract_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingests a contract into the system.
        
        Args:
            contract_data: Dictionary containing contract data
            
        Returns:
            Dictionary with ingestion result information
        """
        try:
            # Generate contract ID if not provided
            if "id" not in contract_data:
                contract_data["id"] = str(uuid.uuid4())
            
            # Add timestamps
            current_time = datetime.utcnow()
            contract_data["created_at"] = current_time
            contract_data["updated_at"] = current_time
            
            # Store in MongoDB
            await self.mongodb_client.insert_contract(contract_data)
            logger.info(f"Stored contract {contract_data['id']} in MongoDB")
            
            # Index metadata in Elasticsearch
            await self.es_client.index_contract(contract_data)
            logger.info(f"Indexed contract {contract_data['id']} metadata in Elasticsearch")
            
            # Process clauses and store embeddings in Qdrant
            for clause in contract_data.get("clauses", []):
                if "id" not in clause:
                    clause["id"] = str(uuid.uuid4())
                
                # Compute embedding for clause text
                embedding = compute_embedding(clause["text"], self.embedding_model)
                
                # Store in Qdrant
                await self.qdrant_client.store_embedding(
                    contract_id=contract_data["id"],
                    contract_title=contract_data["title"],
                    clause_id=clause["id"],
                    clause_type=clause["type"],
                    clause_title=clause.get("title"),
                    content=clause["text"],
                    metadata={
                        **contract_data["metadata"].dict(),
                        **clause.get("metadata", {})
                    },
                    embedding=embedding
                )
            
            logger.info(f"Stored {len(contract_data.get('clauses', []))} clause embeddings in Qdrant")
            
            # Return result
            return {
                "id": contract_data["id"],
                "title": contract_data["title"],
                "metadata": contract_data["metadata"],
                "created_at": current_time,
                "updated_at": current_time,
                "clauses_count": len(contract_data.get("clauses", [])),
                "status": "indexed"
            }
        
        except Exception as e:
            logger.error(f"Error ingesting contract: {str(e)}")
            raise
