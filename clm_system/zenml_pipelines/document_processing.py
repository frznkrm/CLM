import time
import json
import asyncio  # Added import
from typing import Dict, Any, List, Optional, Annotated
from zenml import pipeline, step
from zenml.logger import get_logger
from comet_ml import Experiment
from datetime import datetime
import pymongo
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient as QdrantSyncClient
from bson import ObjectId
import uuid

from clm_system.core.pipeline.base import BaseIngestor, BaseChunker, BaseCleaner
from clm_system.core.pipeline.preprocessing.contract import ContractPreprocessor
from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding

logger = get_logger(__name__)
comet_tracker = "comet_tracker"

def serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def make_json_serializable(data: Dict[str, Any]) -> Dict[str, Any]:
    result = json.loads(json.dumps(data, default=serialize_datetime))
    if "_id" in result:
        del result["_id"]
    return result

def clause_content_to_text(content: List[Any]) -> str:
    """Convert clause or header content to a single text string."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "table" in item:
                table = item["table"]
                table_text = "\n".join(" ".join(str(cell) for cell in row) for row in table)
                text_parts.append(table_text)
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)
    else:
        return str(content)

@step(experiment_tracker=comet_tracker, enable_cache=False)
def preprocess_step(file_path: str, doc_type: str) -> Annotated[Dict[str, Any], "preprocessed"]:
    """Preprocess a DOCX file into JSON format."""
    start_time = time.time()
    try:
        preprocessor = ContractPreprocessor()
        # Use get_event_loop().run_until_complete instead of asyncio.run
        loop = asyncio.get_event_loop()
        preprocessed = loop.run_until_complete(preprocessor.process(file_path))
        
        if not isinstance(preprocessed, dict) or not preprocessed:
            raise ValueError(f"Preprocessing returned invalid output: {preprocessed}")
        preprocess_time = time.time() - start_time
        experiment = Experiment()
        experiment.log_metric("preprocess_time", preprocess_time)
        logger.info(f"Preprocessed document from {file_path} for type {doc_type}")
        return preprocessed
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

@step(experiment_tracker=comet_tracker, enable_cache=False)
def ingest_step(raw: Dict[str, Any], doc_type: str) -> Annotated[Dict[str, Any], "output"]:
    start_time = time.time()
    try:
        ingestor = BaseIngestor._registry['ingestor'][doc_type]()
        normalized = ingestor.process(raw)
        if not isinstance(normalized, dict) or not normalized:
            raise ValueError(f"BaseIngestor.process returned invalid output: {normalized}")
        ingestion_time = time.time() - start_time
        experiment = Experiment()
        experiment.log_metric("ingestion_time", ingestion_time)
        experiment.log_metric("normalized_json_size", len(json.dumps(normalized, default=serialize_datetime)))
        logger.info(f"Ingested document {normalized.get('id')} for type {doc_type}")
        return normalized
    except KeyError as e:
        raise ValueError(f"Missing ingestor for document type {doc_type}") from e

@step(experiment_tracker=comet_tracker)
def clean_step(ingested: Dict[str, Any], doc_type: str) -> Annotated[Dict[str, Any], "cleaned"]:
    """Clean the ingested data."""
    start_time = time.time()
    try:
        cleaner = BaseCleaner._registry['cleaner'][doc_type]()
        cleaned = cleaner.process(ingested)
        if not isinstance(cleaned, dict) or not cleaned:
            raise ValueError(f"Cleaning returned invalid output: {cleaned}")
        clean_time = time.time() - start_time
        experiment = Experiment()
        experiment.log_metric("clean_time", clean_time)
        logger.info(f"Cleaned document {cleaned.get('id')} for type {doc_type}")
        return cleaned
    except KeyError as e:
        raise ValueError(f"Missing cleaner for document type {doc_type}") from e

@step(experiment_tracker=comet_tracker)
def chunk_step(normalized: Dict[str, Any], doc_type: str) -> List[Dict[str, Any]]:
    start_time = time.time()
    chunks = []
    if doc_type in BaseChunker._registry['chunker']:
        chunker = BaseChunker._registry['chunker'][doc_type]()
        content_sources = [
            ("clauses", None),
            ("header", None),
        ]

        for source_field, _ in content_sources:
            if source := normalized.get(source_field):
                for item in _iter_content_items(source, None, source_field):
                    for chunk_text in chunker.chunk(item["text"]):
                        chunk_data = {
                            "source": source_field,
                            "text": chunk_text,
                            "metadata": item.get("metadata", {})
                        }
                        chunks.append(chunk_data)

    chunking_time = time.time() - start_time
    experiment = Experiment()
    experiment.log_metric("chunking_time", chunking_time)
    experiment.log_metric("chunk_count", len(chunks))
    logger.info(f"Generated {len(chunks)} chunks for document {normalized.get('id')}")
    return chunks

def _iter_content_items(source, text_field: Optional[str], source_field: str):
    if source_field == "clauses":
        for clause in source:
            content = clause.get("content", [])
            clause_text_content = clause_content_to_text(content)
            clause_title = clause.get("title", "") # Get the title

            # --- CHANGE: Prepend title to the text being chunked ---
            # Add a clear separator like a newline.
            text_to_chunk = f"Clause Title: {clause_title}\n\n{clause_text_content}"

            # Keep title in metadata as well for potential filtering/display
            metadata = {"clause_title": clause_title}
            yield {"text": text_to_chunk, "metadata": metadata} # Yield text with title prepended
    elif source_field == "header":
        header = source
        header_text = (
            f"To: {header.get('to', '')}\n"
            f"E-mail: {header.get('e-mail', '')}\n"
            f"Re: {header.get('re', '')}\n"
            f"Sellers Ref: {header.get('sellers ref', '')}\n"
            f"Notes: {' '.join(header.get('notes', []))}"
        )
        yield {"text": header_text, "metadata": {}}
    else:
        if isinstance(source, list):
            for item in source:
                yield {"text": item.get(text_field, "") if text_field else str(item), "metadata": {}}
        elif isinstance(source, dict):
            yield {"text": source.get(text_field, "") if text_field else str(source), "metadata": {}}
        else:
            yield {"text": str(source), "metadata": {}}

@step(experiment_tracker=comet_tracker)
def store_step(normalized: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Store normalized JSON in MongoDB/Elasticsearch and chunks in Qdrant."""
    mongo = pymongo.MongoClient("mongodb://localhost:27017")
    db = mongo["clm_db"]
    collection = db["documents"]
    es = Elasticsearch(["http://localhost:9200"])
    qdrant = QdrantSyncClient("http://localhost:6333")
    embedding_model = get_embedding_model()

    try:
        mongo.admin.command("ping")
        logger.debug(f"Connected to MongoDB: database={db.name}, collection={collection.name}")

        start_time = time.time()
        mongo_doc = normalized.copy()
        logger.debug(f"Attempting to insert document with id={mongo_doc['id']} into MongoDB")
        result = collection.insert_one(mongo_doc)
        inserted_id = result.inserted_id
        mongo_time = time.time() - start_time
        logger.debug(f"Inserted MongoDB document: id={mongo_doc['id']}, inserted_id={inserted_id}")

        start_time = time.time()
        es_doc = make_json_serializable(normalized)
        es_response = es.index(index="documents", id=normalized["id"], document=es_doc)
        es_time = time.time() - start_time
        logger.debug(f"Elasticsearch response: {es_response}")

        start_time = time.time()
        doc_id = normalized["id"]
        doc_title = normalized.get("title", "Untitled Document")
        doc_type = normalized["metadata"]["document_type"]
        doc_metadata = normalized.get("metadata", {})
        chunk_count = len(chunks)

        collections = qdrant.get_collections().collections
        if not any(c.name == "document_chunks" for c in collections):
            qdrant.create_collection(
                collection_name="document_chunks",
                vectors_config={
                    "size": 384,
                    "distance": "Cosine"
                }
            )

        points = []
        for chunk in chunks:
            combined_metadata = {
                **doc_metadata,
                **chunk.get("metadata", {})
            }
            embedding = compute_embedding(chunk["text"], embedding_model)
            chunk_id = chunk.get("id", str(uuid.uuid4()))
            points.append({
                "id": chunk_id,
                "vector": embedding,
                "payload": {
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "chunk_id": chunk_id,
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
def document_processing_pipeline(file_path: str, doc_type: str):
    """ZenML pipeline for document preprocessing, ingestion, cleaning, chunking, and storage."""
    preprocessed = preprocess_step(file_path, doc_type)
    ingested = ingest_step(preprocessed, doc_type)
    cleaned = clean_step(ingested, doc_type)
    chunks = chunk_step(cleaned, doc_type)
    result = store_step(cleaned, chunks)
    return result

if __name__ == "__main__":
    pipeline_run = document_processing_pipeline(file_path="sample_contract.docx", doc_type="contract")