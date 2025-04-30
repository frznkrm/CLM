import asyncio
import logging
from typing import Dict, Any, List, Optional

from clm_system.core.database.mongodb_client import MongoDBClient
from clm_system.core.database.elasticsearch_client import ElasticsearchClient
#from clm_system.core.database.qdrant_client import QdrantClient
from clm_system.core.database.qdrant_client import AsyncQdrantClient

from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding

# Import base classes to ensure registration
from .base import BaseIngestor, BaseCleaner, BaseChunker

# Import ZenML pipeline
from clm_system.zenml_pipelines.document_processing import document_processing_pipeline

logger = logging.getLogger(__name__)

class PipelineService:
    def __init__(self):
        # Keep database clients for compatibility
        self.mongo = MongoDBClient()
        self.es = ElasticsearchClient()
        self.qdrant = AsyncQdrantClient()
        self.embedding_model = get_embedding_model()

    async def process_document(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document using ZenML pipeline.
        """
        try:
            doc_type = raw.get("metadata", {}).get("document_type", "contract")
            # Run ZenML pipeline
            result = document_processing_pipeline(raw=raw, doc_type=doc_type)
            return result
        except Exception as e:
            logger.error(f"ZenML pipeline failed: {str(e)}")
            raise

    # # Original implementation (commented out for reference)
    # """
    # async def process_document(self, raw: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Generic document processing pipeline:
    #     1. Determine document type
    #     2. Run through registered components
    #     3. Store results
    #     """
    #     try:
    #         doc_type = raw.get("metadata", {}).get("document_type", "contract")
            
    #         # Validate document type has registered components
    #         if not self._validate_components(doc_type):
    #             raise ValueError(f"No processors registered for document type: {doc_type}")

    #         # 1. Ingestion
    #         ingestor = BaseIngestor._registry['ingestor'][doc_type]()
    #         normalized = ingestor.process(raw)

    #         # 2. Cleaning
    #         cleaner = BaseCleaner._registry['cleaner'][doc_type]()
    #         cleaned = cleaner.process(normalized)

    #         # 3. Chunking (if chunker exists for type)
    #         chunks = []
    #         if doc_type in BaseChunker._registry['chunker']:
    #             chunker = BaseChunker._registry['chunker'][doc_type]()
    #             chunks = self._generate_chunks(cleaned, chunker)

    #         # 4. Persistence
    #         await self._persist_document(cleaned)
            
    #         if chunks:
    #             await self._store_embeddings(cleaned, chunks)

    #         return self._format_result(cleaned, doc_type, chunks)
            
    #     except KeyError as e:
    #         raise ValueError(f"Missing processor for document type {doc_type}") from e
    #     except Exception as e:
    #         logger.error(f"Document processing failed: {str(e)}")
    #         raise
    #     finally:
    #         await self._cleanup_connections()

    # def _validate_components(self, doc_type: str) -> bool:
    #     """Check required components exist for document type"""
    #     has_ingestor = doc_type in BaseIngestor._registry['ingestor']
    #     has_cleaner = doc_type in BaseCleaner._registry['cleaner']
    #     return has_ingestor and has_cleaner

    # def _generate_chunks(self, document: Dict, chunker: BaseChunker) -> List:
    #     """Generic chunk generation across document types"""
    #     chunks = []
    #     content_sources = [
    #         ("clauses", "text"),
    #         ("sections", "content"),
    #         ("body", None),
    #         ("content", None)
    #     ]

    #     for source_field, text_field in content_sources:
    #         if source := document.get(source_field):
    #             for item in self._iter_content_items(source, text_field):
    #                 for chunk_text in chunker.chunk(item["text"]):
    #                     chunks.append({
    #                         "source": source_field,
    #                         "text": chunk_text,
    #                         "metadata": item.get("metadata", {})
    #                     })

    #     return chunks

    # def _iter_content_items(self, source, text_field: Optional[str]):
    #     """Yield text content from various document structures"""
    #     if isinstance(source, list):
    #         for item in source:
    #             if text_field:
    #                 yield {"text": item.get(text_field, ""), "metadata": item.get("metadata", {})}
    #             else:
    #                 yield {"text": str(item), "metadata": {}}
    #     elif isinstance(source, dict):
    #         yield {"text": source.get(text_field, "") if text_field else str(source), "metadata": {}}
    #     else:
    #         yield {"text": str(source), "metadata": {}}

    # async def _persist_document(self, document: Dict[str, Any]):
    #     """Store document in databases"""
    #     # MongoDB
    #     inserted_id = await self.mongo.insert_document(document)
    #     logger.debug(f"Inserted MongoDB document: {inserted_id}")
        
    #     # Elasticsearch
    #     es_response = await self.es.index_document(document)
    #     logger.debug(f"Elasticsearch response: {es_response}")

    # async def _store_embeddings(self, document: Dict[str, Any], chunks: List):
    #     """Store chunks in vector database"""
    #     doc_id = document["id"]
    #     doc_title = document.get("title", "Untitled Document")
    #     doc_type = document["metadata"]["document_type"]
    #     doc_metadata = document.get("metadata", {})

    #     for chunk in chunks:
    #         combined_metadata = {
    #             **doc_metadata,
    #             **chunk.get("metadata", {})
    #         }   
    #         embedding = compute_embedding(chunk["text"], self.embedding_model)
    #         await self.qdrant.store_embedding(
    #             document_id=document["id"],
    #             document_title=document.get("title", "Untitled"),
    #             chunk_id=f"{document['id']}_{hash(chunk['text'])}",
    #             chunk_type=chunk["source"],
    #             content=chunk["text"],
    #             document_type=document["metadata"]["document_type"],
    #             metadata=combined_metadata,
    #             embedding=compute_embedding(chunk["text"], self.embedding_model)
    #         )

    # def _format_result(self, document: Dict, doc_type: str, chunks: List) -> Dict:
    #     """Create standardized result format"""
    #     result = {
    #         "id": document["id"],
    #         "title": document.get("title", f"Untitled {doc_type.title()}"),
    #         "document_type": doc_type,
    #         "status": "indexed",
    #         "chunks_processed": len(chunks)
    #     }

    #     # Add type-specific metadata
    #     metadata = document.get("metadata", {})
    #     result.update({
    #         k: metadata[k] 
    #         for k in ["contract_type", "deal_type", "has_attachments"]
    #         if k in metadata
    #     })

    #     return result

    # async def _cleanup_connections(self):
    #     """Close database connections"""
    #     await self.es.client.close()
    #     self.qdrant.client.close()
    # """