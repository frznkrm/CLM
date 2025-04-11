import pytest
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from clm_system.core.database.mongodb_client import MongoDBClient
from clm_system.core.database.qdrant_client import QdrantClient
from clm_system.core.pipeline.orchestrator import PipelineService
from clm_system.core.utils.embeddings import compute_embedding
from qdrant_client import models

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_email():
    return {
        "id": "email_123",
        "title": "Updated Oil Shipment Schedule",
        "metadata": {
            "document_type": "email",
            "from_address": "logistics@oilcorp.com",
            "to": ["procurement@client.com", "manager@client.com"],
            "cc": ["archive@oilcorp.com"],
            "subject": "Revised Crude Oil Delivery Schedule",
            "sent_date": datetime(2023, 3, 15, 14, 30),
            "tags": ["urgent", "shipment"]
        },
        "attachments": [  # Add this key
            {"filename": "bol_revised.pdf"},
            {"filename": "customs_docs.zip"}
        ],
        "clauses": [
            {
                "id": "body",
                "type": "email_body",
                "text": """Dear Procurement Team,
                
We're updating the March crude oil shipment schedule:
- Tanker OC_Voyager delayed until March 25th
- Increased volume on OC_Explorer to 500,000 barrels
- New pricing at $78/barrel for Brent crude

Attached find the revised Bill of Lading and customs documentation.

Best regards,
OilCorp Logistics Team""",
                "position": 0,
                "metadata": {
                    "attachment_names": ["bol_revised.pdf", "customs_docs.zip"]
                }
            }
        ]
    }
@pytest.mark.asyncio
async def test_email_workflow(sample_email):
    """End-to-end test of email processing and search"""
    
    # Initialize clients
    mongo = MongoDBClient()
    qdrant = QdrantClient()
    pipeline = PipelineService()

    # --- Phase 1: Ingestion ---
    # Process email through pipeline
    processed_email = await pipeline.process_document(sample_email)
    
    # Verify MongoDB insertion
    db_email = await mongo.get_document("email_123")
    assert db_email is not None
    assert db_email["metadata"]["document_type"] == "email"
    assert db_email["metadata"]["subject"] == "Revised Crude Oil Delivery Schedule"
    assert len(db_email["clauses"]) == 1
    
    # --- Phase 2: Embedding Storage ---
    # Verify points are stored in Qdrant
    stored_points = await qdrant.scroll("email_123")
    assert len(stored_points) > 0, "No points stored in Qdrant for email_123"
    logger.info(f"Found {len(stored_points)} points in Qdrant for email_123")
    
    # Debug point structure
    #debug_info = await qdrant.debug_points("email_123")
    #logger.debug(f"Point details:\n{debug_info}")
    
    
    # After verifying points are stored
    logger.info(f"Found {len(stored_points)} points in Qdrant for email_123")

    # Add debugging output
    for point in stored_points:
        logger.info(f"Document ID: {point['document_id']}")
        logger.info(f"Content: {point['content'][:100]}...")
        logger.info(f"Metadata: {point['metadata']}")
    
    # --- Phase 3: Search Verification ---
    test_query = "Current oil shipment prices and schedules"
    query_embedding = compute_embedding(test_query)

    # Search across all document types
    search_results = await qdrant.search(
        embedding=query_embedding,
        filters={
            "document_type": "email",  # Top-level field
            "metadata.has_attachments": True  # Nested under metadata
        },
        top_k=3
    )

    # If still no results, try broader search
    if not search_results:
        backup_results = await qdrant.search(
            embedding=query_embedding,
            filters={"document_type": "email"},
            top_k=3
        )
        logger.info(f"Backup search results: {backup_results}")

    assert len(search_results) > 0, "No search results returned from Qdrant"
    
    # Verify email-specific metadata
    email_result = next(
        (r for r in search_results if r["document_id"] == "email_123"),
        None
    )
    
    assert email_result is not None, "Email result not found in search results"
    assert email_result["document_type"] == "email"
    assert "oil shipment" in email_result["content"].lower()
    assert email_result["metadata"].get("attachment_names") == ["bol_revised.pdf", "customs_docs.zip"]
    
    # --- Phase 4: Temporal Filtering ---
    # Date range search
    date_filtered = await mongo.get_documents(
        filters={
            "metadata.document_type": "email",
            "metadata.sent_date": {
                "$gte": datetime(2023, 3, 1),
                "$lte": datetime(2023, 3, 31)
            }
        },
        sort=[("metadata.sent_date", -1)]
    )
    
    assert len(date_filtered) >= 1
    assert date_filtered[0]["id"] == "email_123"
    
    # --- Phase 5: Cleanup ---
    await mongo.documents_collection.delete_one({"id": "email_123"})
    # qdrant.client.delete(
    #     collection_name="document_chunks",
    #     points_selector=models.FilterSelector(
    #         filter=models.Filter(
    #             must=[
    #                 models.FieldCondition(
    #                     key="document_id",
    #                     match=models.MatchValue(value="email_123")
    #                 )
    #             ]
    #         )
    #     )
    # )