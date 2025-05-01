import ipdb
import asyncio
import pytest
import logging
import os
from dotenv import load_dotenv
from zenml.client import Client
from qdrant_client import models
from clm_system.core.pipeline.orchestrator import PipelineService
from clm_system.core.database.mongodb_client import MongoDBClient
from clm_system.core.database.elasticsearch_client import ElasticsearchClient
from clm_system.core.database.qdrant_client import QdrantClient
from clm_system.zenml_pipelines.document_processing import document_processing_pipeline
from clm_system.zenml_pipelines.search_inference_pipeline import search_inference_pipeline
from comet_ml import Experiment

# Load .env for database credentials
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
class ModuleFilter(logging.Filter):
    def __init__(self, module_name_prefix):
        super().__init__()
        self.module_name_prefix = module_name_prefix
    def filter(self, record):
        return record.name.startswith(self.module_name_prefix)
logging.getLogger().handlers.clear()
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
handler.addFilter(ModuleFilter("clm_system"))
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger("clm_system")
logger.debug("Initialized logger for clm_system tests")
# --- End Logging Configuration ---

@pytest.fixture(autouse=True)
def comet_experiment():
    experiment = Experiment(
        project_name="clm-system-experiments",
        display_summary_level=0
    )
    yield experiment
    experiment.end()

@pytest.fixture
def sample_docx_file():
    docx_path = "sample_contract.docx"
    assert os.path.exists(docx_path), f"DOCX file not found at {docx_path}"
    return docx_path

@pytest.mark.asyncio
async def test_zenml_search_workflow(sample_docx_file):
    # --- Test Setup ---
    logger.info("Starting test_zenml_search_workflow")
    import nest_asyncio
    nest_asyncio.apply()  # Apply nest_asyncio for running async within sync pytest

    mongo = MongoDBClient()
    es = ElasticsearchClient()
    qdrant = QdrantClient()

    test_doc_type = "contract"

    # --- Initial Cleanup ---
    logger.info(f"Cleaning up any previous test data for doc_type: {test_doc_type}")
    try:
        # Clean Elasticsearch
        await es.client.indices.delete(index=es.index_name, ignore=[400, 404])
        await es.ensure_index()
        logger.info(f"Ensured Elasticsearch index '{es.index_name}' exists and is empty/clean.")

        # Clean MongoDB
        delete_mongo_result = await mongo.documents_collection.delete_many(
            {"metadata.document_type": test_doc_type}
        )
        logger.info(f"Deleted {delete_mongo_result.deleted_count} docs from MongoDB for type '{test_doc_type}'")
        logger.debug(f"delete_mongo_result type: {type(delete_mongo_result)}, value: {delete_mongo_result}")
        logger.info(f"Cleaned MongoDB: {delete_mongo_result.deleted_count} documents deleted for type '{test_doc_type}'.")

        # Clean Qdrant
        try:
            await qdrant.client.get_collection(collection_name=qdrant.collection_name)
            delete_qdrant_result = await qdrant.client.delete(
                collection_name=qdrant.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_type",
                                match=models.MatchValue(value=test_doc_type)
                            )
                        ]
                    )
                )
            )
            status = getattr(delete_qdrant_result, 'status', 'N/A')
            logger.info(f"Cleaned Qdrant: status={status} for type '{test_doc_type}'.")
        except Exception as q_err:
            logger.warning(f"Could not clean Qdrant (collection '{qdrant.collection_name}' might not exist yet): {q_err}")

    except Exception as cleanup_err:
        logger.error(f"Error during initial cleanup: {cleanup_err}", exc_info=True)

    # ========================================================================
    # === PHASE 1: PROCESS DOCX FILE ===
    # ========================================================================
    logger.info(f"--- PHASE 1: Processing DOCX file: {sample_docx_file} ---")
    try:
        client = Client()
        pipeline_name = "document_processing_pipeline"
        logger.info(f"Clearing previous ZenML runs for pipeline: {pipeline_name}")
    except Exception as zenml_err:
        logger.warning(f"Could not clear ZenML runs: {zenml_err}")

    pipeline_run = document_processing_pipeline(
        file_path=sample_docx_file,
        doc_type=test_doc_type
    )
    store_step_output = pipeline_run.steps['store_step'].outputs['output'].load()
    processed_doc_id = store_step_output.get("id", "UnknownID")
    logger.info(f"--- Phase 1 completed. Processed document ID (for reference): {processed_doc_id} ---")

    # ========================================================================
    # === VERIFICATION ===
    # ========================================================================
    logger.info(f"--- VERIFICATION: Checking data for doc_type: {test_doc_type} ---")
    verification_passed = True
    try:
        # Verify MongoDB storage
        logger.info("Verifying MongoDB storage...")
        db_contract = await mongo.documents_collection.find_one(
            {"metadata.document_type": test_doc_type}
        )
        logger.debug(f"db_contract type: {type(db_contract)}, value: {db_contract}")
        assert db_contract is not None, f"No document found in MongoDB for type '{test_doc_type}'"
        assert "title" in db_contract, "Title missing in MongoDB document"
        assert "clauses" in db_contract, "Clauses missing in MongoDB document"
        logger.info("MongoDB storage verified.")

        # Verify Elasticsearch indexing
        
        logger.info("Verifying Elasticsearch indexing...")
        mapping = await es.client.indices.get_mapping(index=es.index_name)
        logger.debug(f"Elasticsearch mapping for index '{es.index_name}': {mapping}")
        await es.client.indices.refresh(index=es.index_name) # Force refresh
        es_search_result = await es.client.search(
            index=es.index_name,
            query={"term": {"metadata.document_type": test_doc_type}},
            size=1,
            ignore=[404]
        )
        assert es_search_result['hits']['total']['value'] > 0, f"No document found in Elasticsearch for type '{test_doc_type}'"
        es_doc_source = es_search_result['hits']['hits'][0]['_source']
        assert "title" in es_doc_source, "Title missing in Elasticsearch document"
        logger.info("Elasticsearch indexing verified.")

        # Verify Qdrant storage
        logger.info("Verifying Qdrant storage...")
        qdrant_points, _ = await qdrant.client.scroll(
            collection_name=qdrant.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value=test_doc_type)
                    )
                ]
            ),
            limit=10,
            with_payload=False,
            with_vectors=False
        )
        assert len(qdrant_points) > 0, f"No points found in Qdrant for type '{test_doc_type}'"
        logger.info(f"Qdrant storage verified with at least {len(qdrant_points)} points.")

    except AssertionError as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        verification_passed = False
    except Exception as e:
        logger.error(f"Error during verification: {e}", exc_info=True)
        verification_passed = False

    if verification_passed:
        logger.info("--- Verification completed successfully. ---")
    else:
        logger.error("--- Verification FAILED. ---")
        pytest.fail("Verification of ingested data failed.")

    # ========================================================================
    # === PHASE 2: STRUCTURED QUERIES ===
    # ========================================================================
    logger.info(f"--- PHASE 2: Testing Structured Queries (doc_type: {test_doc_type}) ---")
    structured_queries = [
        {
            "query": "clauses.title:9. PAYMENT TERMS",
            "description": "Search for the payment terms clause title",
            "expected": "9. PAYMENT TERMS",
            "field": "clause_title"
        },
        {
            "query": "party:COP1CO TRADING SA",
            "description": "Search for the buyer party name",
            "expected": "COP1CO TRADING SA",
            "field": "metadata.parties.name"
        },
        {
            "query": "clauses.title:6. DELIVERY",
            "description": "Search for the delivery clause",
            "expected": "6. DELIVERY",
            "field": "clause_title"
        }
    ]

    for test in structured_queries:
        logger.info(f"Testing structured query: '{test['query']}' - {test['description']}")
        search_run = search_inference_pipeline(
            query=test['query'],
            filters={"metadata.document_type": test_doc_type},
            top_k=5
        )

        result_list = search_run.steps["merge_results"].outputs["output"].load()
        logger.debug(f"Structured query result list: {result_list}")
        assert result_list, f"No results returned for query: {test['query']}"

        found_expected = False
        for result_item in result_list:
            try:
                if test["field"].startswith("metadata.parties"):
                    parties = result_item.get("metadata", {}).get("parties", [])
                    party_names = [p.get("name") for p in parties if isinstance(p, dict)]
                    if test["expected"] in party_names:
                        found_expected = True
                        logger.debug(f"Found expected party name '{test['expected']}' in parties: {party_names}")
                        break
                elif test["field"] == "clause_title":
                    clause_title = result_item.get("clause_title")
                    if clause_title == test["expected"]:
                        found_expected = True
                        logger.debug(f"Found expected clause title '{test['expected']}' in clause_title: {clause_title}")
                        break
                    elif test["expected"] in result_item.get("content", ""):
                        logger.warning(f"Found expected title '{test['expected']}' in content, not clause_title field.")
                        found_expected = True
                        break
                else:
                    field_value = result_item.get(test["field"])
                    if not field_value and "metadata" in result_item:
                        field_value = result_item["metadata"].get(test["field"])
                    if field_value == test["expected"]:
                        found_expected = True
                        logger.debug(f"Found expected value '{test['expected']}' in field '{test['field']}': {field_value}")
                        break
            except Exception as e:
                logger.warning(f"Error accessing field '{test['field']}' in result item: {result_item}. Error: {e}")
                continue

        if not found_expected:
            logger.error(f"Expected value '{test['expected']}' for field '{test['field']}' not found. Results: {result_list}")
        assert found_expected, f"Expected value '{test['expected']}' for field '{test['field']}' not found in top results for query: {test['query']}"
        logger.info(f"Structured query test passed for: {test['query']}")

    logger.info("--- Phase 2 completed. ---")
    # ========================================================================
    # === PHASE 3: SEMANTIC QUERIES ===
    # ========================================================================
    logger.info(f"--- PHASE 3: Testing Semantic Queries (doc_type: {test_doc_type}) ---")
    semantic_queries = [
        {
            "query": "When are payments due?",
            "description": "Question about payment due dates",
            "expected": "thirty (30) calendar days after the date of notice of readiness",
            "field": "content"
        },
        {
            "query": "What are the conditions for terminating the contract?",
            "description": "Question about termination conditions",
            "expected": "insolvency default",
            "field": "content"
        }
    ]

    test_top_k = 15
    for test in semantic_queries:
        logger.info(f"Testing semantic query: '{test['query']}' - {test['description']} (top_k={test_top_k})")
        search_run = search_inference_pipeline(
            query=test['query'],
            filters={"metadata.document_type": test_doc_type},
            top_k=test_top_k
        )

        results = search_run.steps["merge_results"].outputs["output"].load()
        logger.debug(f"Semantic query raw results (top {len(results)}): {results}")
        assert results, f"No results for query: {test['query']}"

        found_expected = False
        correct_chunk_content = None
        for result_item in results:
            field_value = result_item.get(test["field"], "")
            if test["expected"].lower() in field_value.lower():
                found_expected = True
                correct_chunk_content = field_value
                logger.info(f"Found expected text in chunk: {correct_chunk_content[:200]}...")
                break

        if not found_expected:
            logger.warning(f"Expected text '{test['expected']}' NOT found in top {len(results)} results.")
        assert found_expected, f"Expected content '{test['expected']}' not found in top {len(results)} results for query: {test['query']}"
        logger.info(f"Semantic query test passed for: {test['query']}")

    logger.info("--- Phase 3 completed. ---")

    # --- Final Cleanup ---
    try:
        logger.info(f"--- FINAL CLEANUP: Removing data for doc_type: {test_doc_type} ---")
        # MongoDB Cleanup
        logger.debug(f"Attempting to delete docs with type '{test_doc_type}' from MongoDB...")
        try:
            delete_mongo_result = await mongo.documents_collection.delete_many(
                {"metadata.document_type": test_doc_type}
            )
            logger.debug(f"delete_mongo_result type: {type(delete_mongo_result)}, value: {delete_mongo_result}")
            logger.info(f"Cleaned MongoDB: {delete_mongo_result.deleted_count} documents deleted for type '{test_doc_type}'.")
            logger.info(f"Deleted {delete_mongo_result.deleted_count} docs from MongoDB for type '{test_doc_type}'")
        except Exception as mongo_del_err:
            logger.error(f"Error during MongoDB cleanup: {mongo_del_err}", exc_info=True)

        # Elasticsearch Cleanup
        logger.debug(f"Attempting to delete docs with type '{test_doc_type}' from Elasticsearch...")
        try:
            if es.client and es.client.transport and not es.client.transport.closed:
                es_delete_response = await es.client.delete_by_query(
                    index=es.index_name,
                    query={"term": {"metadata.document_type": test_doc_type}},
                    conflicts='proceed',
                    refresh=True,
                    ignore=[400, 404]
                )
                deleted_count = es_delete_response.get('deleted', 0)
                logger.info(f"Elasticsearch delete_by_query completed. Docs deleted: {deleted_count} for type '{test_doc_type}'. Response: {es_delete_response}")
            else:
                logger.warning("Elasticsearch client was closed or unavailable for cleanup.")
        except Exception as es_del_err:
            logger.error(f"Error during Elasticsearch delete_by_query: {es_del_err}", exc_info=True)

        # Qdrant Cleanup
        logger.debug(f"Attempting to delete points with type '{test_doc_type}' from Qdrant...")
        try:
            await qdrant.client.get_collection(collection_name=qdrant.collection_name)
            delete_qdrant_result = await qdrant.client.delete(
                collection_name=qdrant.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_type",
                                match=models.MatchValue(value=test_doc_type)
                            )
                        ]
                    )
                )
            )
            status = getattr(delete_qdrant_result, 'status', 'N/A')
            logger.info(f"Qdrant delete result: status={status} for type '{test_doc_type}'")
        except Exception as q_del_err:
            logger.warning(f"Could not delete from Qdrant (collection '{qdrant.collection_name}' might not exist or filter failed): {q_del_err}", exc_info=True)

        # Close Clients
        logger.debug("Attempting to close database clients...")
        try:
            if es.client and es.client.transport and not es.client.transport.closed:
                await es.close()
                logger.info("Closed Elasticsearch client.")
            else:
                logger.info("Elasticsearch client already closed or not initialized.")
        except Exception as es_close_err:
            logger.error(f"Error closing Elasticsearch client: {es_close_err}", exc_info=True)

        try:
            if qdrant and qdrant.client:
                closed_qdrant = False
                if hasattr(qdrant, 'close') and asyncio.iscoroutinefunction(qdrant.close):
                    await qdrant.close()
                    logger.info("Closed Qdrant client wrapper (async).")
                    closed_qdrant = True
                elif hasattr(qdrant.client, 'close') and asyncio.iscoroutinefunction(qdrant.client.close):
                    await qdrant.client.close()
                    logger.info("Closed underlying Qdrant client (async).")
                    closed_qdrant = True
                if not closed_qdrant:
                    logger.warning("Could not determine async close method for Qdrant client or it was already closed.")
            else:
                logger.info("Qdrant client not initialized or already closed.")
        except Exception as q_close_err:
            logger.error(f"Error closing Qdrant client: {q_close_err}", exc_info=True)

    except Exception as e:
        logger.error(f"Error during final cleanup setup: Type={type(e).__name__}, Message={str(e)}", exc_info=True)
    finally:
        logger.info("--- Final cleanup attempt finished. ---")
        logger.info("--- Test test_zenml_search_workflow finished. ---")