import time
import json
from typing import Dict, Any, List, Optional
from zenml import pipeline, step
from zenml.logger import get_logger
from comet_ml import Experiment
from datetime import datetime
import pymongo
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient as QdrantSyncClient
from bson import ObjectId

from clm_system.core.pipeline.base import BaseIngestor, BaseChunker
from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding

logger = get_logger(__name__)

# CometML experiment tracker name
COMET_TRACKER = "comet_tracker"

def serialize_datetime(obj):
    """Convert datetime objects to ISO format strings for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def make_json_serializable(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure dictionary is JSON-serializable by converting non-serializable types."""
    result = json.loads(json.dumps(data, default=serialize_datetime))
    if "_id" in result:
        del result["_id"]
    return result

@step(experiment_tracker=COMET_TRACKER)
def ingest_step(raw: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
    """Ingest raw data into normalized JSON using BaseIngestor."""
    start_time = time.time()
    try:
        ingestor = BaseIngestor._registry['ingestor'][doc_type]()
        normalized = ingestor.process(raw)
        ingestion_time = time.time() - start_time

        experiment = Experiment()
        experiment.log_metric("ingestion_time", ingestion_time)
        experiment.log_metric("normalized_json_size", len(json.dumps(normalized, default=serialize_datetime)))
        logger.info(f"Ingested document {normalized.get('id')} for type {doc_type}")
        return normalized
    except KeyError as e:
        logger.error(f"Missing ingestor for document type {doc_type}")
        raise ValueError(f"Missing ingestor for document type {doc_type}") from e

@step(experiment_tracker=COMET_TRACKER)
def chunk_step(normalized: Dict[str, Any], doc_type: str) -> List[Dict[str, Any]]:
    """Generate chunks from normalized document using BaseChunker."""
    start_time = time.time()
    chunks = []
    if doc_type in BaseChunker._registry['chunker']:
        chunker = BaseChunker._registry['chunker'][doc_type]()
        content_sources = [
            ("clauses", "text"),
            ("sections", "content"),
            ("body", None),
            ("content", None)
        ]

        for source_field, text_field in content_sources:
            if source := normalized.get(source_field):
                for item in _iter_content_items(source, text_field):
                    for chunk_text in chunker.chunk(item["text"]):
                        chunk_data = {
                            "source": source_field,
                            "text": chunk_text,
                            "metadata": item.get("metadata", {})
                        }
                        # Include clause ID for clauses
                        if source_field == "clauses" and "id" in item:
                            chunk_data["id"] = item["id"]
                        chunks.append(chunk_data)

    chunking_time = time.time() - start_time
    experiment = Experiment()
    experiment.log_metric("chunking_time", chunking_time)
    experiment.log_metric("chunk_count", len(chunks))
    logger.info(f"Generated {len(chunks)} chunks for document {normalized.get('id')}")
    return chunks

def _iter_content_items(source, text_field: Optional[str]):
    """Yield text content from various document structures."""
    if isinstance(source, list):
        for item in source:
            if text_field:
                yield {"text": item.get(text_field, ""), "metadata": item.get("metadata", {})}
            else:
                yield {"text": str(item), "metadata": {}}
    elif isinstance(source, dict):
        yield {"text": source.get(text_field, "") if text_field else str(source), "metadata": {}}
    else:
        yield {"text": str(source), "metadata": {}}

import uuid



@step(experiment_tracker=COMET_TRACKER)
def store_step(normalized: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Store normalized JSON in MongoDB/Elasticsearch and chunks in Qdrant."""
    # Synchronous clients
    mongo = pymongo.MongoClient("mongodb://localhost:27017")
    db = mongo["clm_db"]
    collection = db["documents"]
    es = Elasticsearch(["http://localhost:9200"])
    qdrant = QdrantSyncClient("http://localhost:6333")
    embedding_model = get_embedding_model()

    try:
        # Store in MongoDB
        start_time = time.time()
        mongo_doc = normalized.copy()
        inserted_id = collection.insert_one(mongo_doc).inserted_id
        mongo_time = time.time() - start_time
        logger.debug(f"Inserted MongoDB document: {inserted_id}")

        # Store in Elasticsearch
        start_time = time.time()
        es_doc = make_json_serializable(normalized)
        es_response = es.index(index="documents", id=normalized["id"], document=es_doc)
        es_time = time.time() - start_time
        logger.debug(f"Elasticsearch response: {es_response}")

        # Store embeddings in Qdrant
        start_time = time.time()
        doc_id = normalized["id"]
        doc_title = normalized.get("title", "Untitled Document")
        doc_type = normalized["metadata"]["document_type"]
        doc_metadata = normalized.get("metadata", {})
        chunk_count = len(chunks)

        # Ensure collection exists
        collections = qdrant.get_collections().collections
        if not any(c.name == "document_chunks" for c in collections):
            qdrant.create_collection(
                collection_name="document_chunks",
                vectors_config={
                    "size": 384,  # Match sentence-transformers/all-MiniLM-L6-v2
                    "distance": "Cosine"
                }
            )

        # Batch upsert points
        points = []
        for chunk in chunks:
            combined_metadata = {
                **doc_metadata,
                **chunk.get("metadata", {})
            }
            embedding = compute_embedding(chunk["text"], embedding_model)
            # Use chunk ID if available (e.g., clause ID), else generate UUID
            chunk_id = chunk.get("id", str(uuid.uuid4()))
            points.append({
                "id": chunk_id,  # Use chunk_id as point ID
                "vector": embedding,
                "payload": {
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "chunk_id": chunk_id,  # Include chunk_id in payload
                    "chunk_type": chunk["source"],
                    "content": chunk["text"],
                    "document_type": doc_type,
                    "metadata": combined_metadata
                }
            })
        if points:
            qdrant.upsert(
                collection_name="document_chunks",
                points=points
            )
        qdrant_time = time.time() - start_time

        experiment = Experiment()
        experiment.log_metric("mongo_store_time", mongo_time)
        experiment.log_metric("elasticsearch_store_time", es_time)
        experiment.log_metric("qdrant_store_time", qdrant_time)
        experiment.log_metric("chunk_count_stored", chunk_count)
        experiment.log_metric("storage_success", 1)

        result = {
            "id": doc_id,
            "title": doc_title,
            "document_type": doc_type,
            "status": "indexed",
            "chunks_processed": chunk_count
        }
        logger.info(f"Stored document {doc_id} with {chunk_count} chunks")
        return result

    except Exception as e:
        logger.error(f"Storage failed: {str(e)}")
        experiment = Experiment()
        experiment.log_metric("storage_success", 0)
        raise
    finally:
        mongo.close()
        es.close()
        qdrant.close()
        
@pipeline
def document_processing_pipeline(raw: Dict[str, Any], doc_type: str):
    """ZenML pipeline for document ingestion, chunking, and storage."""
    normalized = ingest_step(raw, doc_type)
    chunks = chunk_step(normalized, doc_type)
    result = store_step(normalized, chunks)
    return result

if __name__ == "__main__":
    sample_raw = {
        "id": "test_doc_001",
        "metadata": {"document_type": "contract"},
        "text": "Sample contract text",
        "clauses": [{"text": "Clause 1 content"}, {"text": "Clause 2 content"}]
    }
    pipeline_run = document_processing_pipeline(raw=sample_raw, doc_type="contract")