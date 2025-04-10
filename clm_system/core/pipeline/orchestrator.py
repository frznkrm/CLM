# clm_system/core/pipeline/orchestrator.py
import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional

from clm_system.core.pipeline.ingestion.contract import ContractIngestor
from clm_system.core.pipeline.ingestion.deal import DealIngestor
from clm_system.core.pipeline.ingestion.email import EmailIngestor
from clm_system.core.pipeline.ingestion.recap import RecapIngestor

from clm_system.core.pipeline.cleaning.contract import ContractCleaner
from clm_system.core.pipeline.cleaning.deal import DealCleaner
from clm_system.core.pipeline.cleaning.email import EmailCleaner
from clm_system.core.pipeline.cleaning.recap import RecapCleaner

from clm_system.core.pipeline.chunking.contract import ContractChunker
from clm_system.core.pipeline.chunking.deal import DealChunker
from clm_system.core.pipeline.chunking.email import EmailChunker

from clm_system.core.database.mongodb_client import MongoDBClient
from clm_system.core.database.elasticsearch_client import ElasticsearchClient
from clm_system.core.database.qdrant_client import QdrantClient
from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding

logger = logging.getLogger(__name__)

class PipelineService:
    def __init__(self):
        # Document type registry with processing components
        self.ingestors = {
            "contract": ContractIngestor(),
            "deal": DealIngestor(),
            "email": EmailIngestor(),
            "recap": RecapIngestor(),
        }
        
        self.cleaners = {
            "contract": ContractCleaner(),
            "deal": DealCleaner(),
            "email": EmailCleaner(),
            "recap": RecapCleaner(),
        }
        
        # Note: No RecapChunker since recaps are a component of deals
        self.chunkers = {
            "contract": ContractChunker(),
            "deal": DealChunker(),
            "email": EmailChunker(),
        }
        
        # Initialize database connections
        self.mongo = MongoDBClient()
        self.es = ElasticsearchClient()
        self.qdrant = QdrantClient()
        
        # Load embedding model
        self.embedding_model = get_embedding_model()

    async def process_document(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main pipeline execution flow:
        1. Determine document type from metadata
        2. Run through type-specific processing components
        3. Store in databases and vector store
        4. Return processing results
        """
        try:
            # Determine document type (default to contract if missing)
            doc_type = raw.get("metadata", {}).get("document_type", "contract")
            
            # Validate document type
            if doc_type not in self.ingestors:
                raise ValueError(f"Unsupported document type: {doc_type}")
            
            # Get processing components
            ingestor = self.ingestors[doc_type]
            cleaner = self.cleaners[doc_type]
            
            # Pipeline execution - steps 1 & 2
            normalized = ingestor.ingest(raw)      # Structure raw data
            cleaned = cleaner.clean(normalized)    # Sanitize and normalize
            
            # Special handling for recaps - don't chunk them
            if doc_type == "recap":
                # Store recaps without chunking
                await self._persist_document(cleaned)
                
                logger.info(f"Pipeline complete for recap {cleaned['id']}")
                return {
                    "id": cleaned["id"],
                    "title": cleaned.get("title", "Untitled Recap"),
                    "status": "indexed",
                    "document_type": doc_type
                }
            else:
                # For other document types, proceed with chunking
                chunker = self.chunkers[doc_type]
                
                # Step 3: Chunk & embed based on document type structure
                chunks = self._generate_chunks_for_type(cleaned, chunker, doc_type)
                
                # Step 4: Persist in MongoDB and Elasticsearch
                await self._persist_document(cleaned)
                
                # Step 5: Store embeddings in Qdrant if there are chunks
                if chunks:
                    await self._store_embeddings(cleaned, chunks, doc_type)
                
                logger.info(f"Pipeline complete for {doc_type} {cleaned['id']}")
                
                # Return appropriate result based on document type
                result = {
                    "id": cleaned["id"],
                    "title": cleaned.get("title", f"Untitled {doc_type.title()}"),
                    "status": "indexed",
                    "document_type": doc_type
                }
                
                # Add type-specific fields
                if doc_type == "contract" and "clauses" in cleaned:
                    result["clauses_count"] = len(cleaned["clauses"])
                elif doc_type == "email":
                    if "metadata" in cleaned and "has_attachments" in cleaned["metadata"]:
                        result["has_attachments"] = cleaned["metadata"]["has_attachments"]
                elif doc_type == "deal":
                    if "metadata" in cleaned and "deal_type" in cleaned["metadata"]:
                        result["deal_type"] = cleaned["metadata"]["deal_type"]
                
                return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
        finally:
            await self._cleanup_connections()

    async def _cleanup_connections(self):
        """Clean up all database connections"""
        # Elasticsearch: Close asynchronously
        if hasattr(self.es, 'client'):
            await self.es.client.close()

        # Qdrant: Close synchronously (remove 'await')
        if hasattr(self.qdrant, 'client'):
          self.qdrant.client.close()
    def _generate_chunks_for_type(self, 
                                  document: Dict[str, Any], 
                                  chunker: Any, 
                                  doc_type: str) -> List[Tuple[Dict[str, Any], str, List[float]]]:
        """Generate chunks based on document type structure"""
        chunks = []
        
        if doc_type == "contract":
            # Contracts have clauses
            for clause in document.get("clauses", []):
                for text in chunker.chunk(clause["text"]):
                    emb = compute_embedding(text, self.embedding_model)
                    chunks.append((clause, text, emb))
                    
        elif doc_type == "email":
            # Emails may have a body or content field instead of clauses
            if "body" in document:
                # Email with direct body field
                for text in chunker.chunk(document["body"]):
                    emb = compute_embedding(text, self.embedding_model)
                    # Create a pseudo-clause for consistent handling
                    pseudo_clause = {
                        "id": f"{document['id']}_body",
                        "type": "email_body",
                        "metadata": {}
                    }
                    chunks.append((pseudo_clause, text, emb))
            elif "content" in document:
                # Email with content field
                for text in chunker.chunk(document["content"]):
                    emb = compute_embedding(text, self.embedding_model)
                    pseudo_clause = {
                        "id": f"{document['id']}_content",
                        "type": "email_content",
                        "metadata": {}
                    }
                    chunks.append((pseudo_clause, text, emb))
            elif "clauses" in document:
                # Email already processed into clauses
                for clause in document["clauses"]:
                    for text in chunker.chunk(clause["text"]):
                        emb = compute_embedding(text, self.embedding_model)
                        chunks.append((clause, text, emb))
                        
        elif doc_type == "deal":
            # Deals may have sections or clauses
            if "sections" in document:
                for i, section in enumerate(document["sections"]):
                    if isinstance(section, dict) and "content" in section:
                        for text in chunker.chunk(section["content"]):
                            emb = compute_embedding(text, self.embedding_model)
                            pseudo_clause = {
                                "id": f"{document['id']}_section_{i}",
                                "type": section.get("type", "deal_section"),
                                "metadata": {
                                    "section_title": section.get("title", f"Section {i+1}")
                                }
                            }
                            chunks.append((pseudo_clause, text, emb))
            elif "clauses" in document:
                # Deal already processed into clauses
                for clause in document["clauses"]:
                    for text in chunker.chunk(clause["text"]):
                        emb = compute_embedding(text, self.embedding_model)
                        chunks.append((clause, text, emb))
            elif "terms" in document and isinstance(document["terms"], list):
                # Deal with terms structure
                for i, term in enumerate(document["terms"]):
                    if isinstance(term, dict) and "description" in term:
                        for text in chunker.chunk(term["description"]):
                            emb = compute_embedding(text, self.embedding_model)
                            pseudo_clause = {
                                "id": f"{document['id']}_term_{i}",
                                "type": "deal_term",
                                "metadata": {
                                    "term_name": term.get("name", f"Term {i+1}")
                                }
                            }
                            chunks.append((pseudo_clause, text, emb))
                            
        return chunks
    
    async def _persist_document(self, document: Dict[str, Any]):
        """Store document in MongoDB and Elasticsearch"""
        # MongoDB - main document storage
        # Ensure async connections
        await self.mongo.client.admin.command('ping')
        await self.es.client.info()
        
        # MongoDB - main document storage
        inserted_id = await self.mongo.insert_contract(document)
        logger.debug("Inserted MongoDB ID: %s", inserted_id)
        
        # Elasticsearch - searchable content
        es_doc = document.copy()
        es_doc.pop("_id", None)
        es_response = await self.es.index_contract(es_doc)
        logger.debug("ES response: %s", es_response)
        
    async def _store_embeddings(self, 
                               document: Dict[str, Any], 
                               chunks: List[Tuple[Dict[str, Any], str, List[float]]], 
                               doc_type: str):
        """Store embeddings in vector database"""
        for clause, text, emb in chunks:
            # Get metadata from the document and clause
            doc_metadata = document.get("metadata", {})
            clause_metadata = clause.get("metadata", {})
            
            # Combine metadata, with clause metadata taking precedence
            combined_metadata = {**doc_metadata, **clause_metadata}
            
            # Add document type to metadata
            combined_metadata["document_type"] = doc_type
            
            await self.qdrant.store_embedding(
                contract_id=document["id"],
                contract_title=document.get("title", f"Untitled {doc_type.title()}"),
                clause_id=clause["id"],
                clause_type=clause.get("type", "generic"),
                content=text,
                metadata=combined_metadata,
                embedding=emb
            )