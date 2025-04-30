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

# Configure logging
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

@pytest.fixture(autouse=True)
def comet_experiment():
    # Initialize Comet.ml with summary logging disabled
    experiment = Experiment(
        project_name="clm-system-experiments",
        display_summary_level=0  # Disables summary output
    )
    yield experiment
    experiment.end()

@pytest.fixture
def sample_docx_file():
    # Path to a real or mocked DOCX file
    docx_path = "sample_contract.docx"
    assert os.path.exists(docx_path), f"DOCX file not found at {docx_path}"
    return docx_path

@pytest.mark.asyncio
async def test_zenml_search_workflow(sample_docx_file):
    # Clear previous pipeline runs and artifacts to avoid caching
    logger.info("Clearing previous ZenML pipeline runs and artifacts")
    client = Client()
    pipeline_name = "document_processing_pipeline"
    for run in client.list_pipeline_runs(pipeline_name=pipeline_name):
        client.delete_pipeline_run(run.id)
    for artifact in client.list_artifacts():
        client.delete_artifact(artifact.id)
    logger.info("ZenML pipeline runs and artifacts cleared")

    import nest_asyncio
    nest_asyncio.apply()
    mongo = MongoDBClient()
    es = ElasticsearchClient()
    qdrant = QdrantClient()
    pipeline = PipelineService()
    doc_id = None  # Initialize doc_id to avoid reference error in finally block

    # Clear existing data in Elasticsearch
    await es.client.indices.delete(index=es.index_name, ignore_unavailable=True)
    await es.ensure_index()

    try:
        # --- Phase 1: Process DOCX File ---
        logger.info(f"Processing DOCX file: {sample_docx_file}")
        pipeline_run = document_processing_pipeline(
            file_path=sample_docx_file,
            doc_type="contract"
        )  # Explicitly run the pipeline
        
        # Correctly access step outputs
        store_step_output = pipeline_run.steps['store_step'].outputs['output'].load()
        doc_id = store_step_output["id"]
        logger.info(f"Processed document ID: {doc_id}")

        # Verify ingestion in MongoDB
        logger.info("Verifying MongoDB storage")
        db_contract = await mongo.get_document(doc_id)
        assert db_contract is not None, "Contract not found in MongoDB"
        assert "title" in db_contract, "Title missing in MongoDB document"
        assert "clauses" in db_contract, "Clauses missing in MongoDB document"
        logger.info("MongoDB storage verified")

        # Verify in Elasticsearch
        logger.info("Verifying Elasticsearch indexing")
        es_contract = await es.client.get(index="documents", id=doc_id)
        assert es_contract["found"], "Contract not found in Elasticsearch"
        assert "title" in es_contract["_source"], "Title missing in Elasticsearch document"
        logger.info("Elasticsearch indexing verified")

        # Verify in Qdrant
        logger.info("Verifying Qdrant storage")
        qdrant_points = await qdrant.scroll(doc_id)
        assert len(qdrant_points) > 0, "No chunks found in Qdrant"
        logger.info(f"Qdrant storage verified with {len(qdrant_points)} points")

        # --- Phase 2: Structured Queries ---
        # Queries adapted to the contract between TRADECO SA and COP1CO TRADING SA
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
                filters={"metadata.document_type": "contract"},
                top_k=1
            )  # Explicitly run the pipeline

            # Get the actual results from the pipeline output
            result = search_run.steps["merge_results"].outputs["output"].load()

            logger.info(f"Structured query result: {result}")
            assert result, f"No results for query: {test['query']}"

            if test["field"].startswith("metadata.parties"):
                party_names = [party["name"] for party in result[0]["metadata"].get("parties", [])]
                assert test["expected"] in party_names, f"Expected party '{test['expected']}' not found for query: {test['query']}"
            else:
                field_value = result[0].get(test["field"])
                assert field_value == test["expected"], f"Expected clause '{test['expected']}', got '{field_value}' for query: {test['query']}"
            logger.info(f"Structured query test passed for: {test['query']}")

        # --- Phase 3: Semantic Queries ---
        # Queries adapted to the contract's content
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

        for test in semantic_queries:
            logger.info(f"Testing semantic query: '{test['query']}' - {test['description']}")
            #ipdb.set_trace()
            search_run = search_inference_pipeline(
                query=test['query'],
                filters={"metadata.document_type": "contract"},
                top_k=1
            )  # Explicitly run the pipeline

            # Get the actual results from the pipeline output
            result = search_run.steps["merge_results"].outputs["output"].load()

            logger.info(f"Semantic query result: {result}")
            assert result, f"No results for query: {test['query']}"
            field_value = result[0].get(test["field"], "")
            assert test["expected"] in field_value.lower(), f"Expected content '{test['expected']}' not found in '{field_value}' for query: {test['query']}"
            logger.info(f"Semantic query test passed for: {test['query']}")

    finally:
        logger.info("Cleaning up test data")
        try:
            # Get the current event loop for the executor
            loop = asyncio.get_running_loop()

            if doc_id:
                # Run sync mongo delete in executor
                logger.debug(f"Attempting to delete doc {doc_id} from MongoDB...")
                await loop.run_in_executor(
                    None,  # Use default executor
                    mongo.documents_collection.delete_one, # Function to run
                    {"id": doc_id} # Arguments for the function
                )
                logger.info(f"Deleted doc {doc_id} from MongoDB")

                # Elasticsearch deletion (async)
                logger.debug(f"Attempting to delete doc {doc_id} from Elasticsearch...")
                await es.client.options(ignore_status=[404]).delete(
                    index=es.index_name, id=doc_id
                )
                logger.info(f"Deleted doc {doc_id} from Elasticsearch")

                # Qdrant deletion (async) - Assuming qdrant is the async client wrapper
                logger.debug(f"Attempting to delete doc {doc_id} points from Qdrant...")
                # Ensure you are using the correct collection name from your wrapper/config
                delete_result = await qdrant.client.delete(
                    collection_name=qdrant.collection_name, # Use collection name from your client instance
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="document_id",
                                    match=models.MatchValue(value=doc_id)
                                )
                            ]
                        )
                    )
                )
                # Log the result type and status to understand what delete returns
                status = getattr(delete_result, 'status', 'N/A') # Qdrant UpdateResult has 'status'
                logger.info(f"Qdrant delete result: type={type(delete_result).__name__}, status={status}")
                logger.info(f"Deleted doc {doc_id} points from Qdrant")

            # Close Elasticsearch (async)
            logger.debug("Attempting to close Elasticsearch client...")
            await es.close()
            logger.info("Closed Elasticsearch client")

            # Close Qdrant async client correctly
            logger.debug("Attempting to close Qdrant client...")
            # Check if your wrapper QdrantClient has an async close method
            if hasattr(qdrant, 'aclose') and asyncio.iscoroutinefunction(qdrant.aclose):
                 await qdrant.aclose()
                 logger.info("Closed Qdrant client wrapper (async)")
            # Or check if the underlying library client needs async closing
            elif hasattr(qdrant.client, 'aclose') and asyncio.iscoroutinefunction(qdrant.client.aclose):
                 await qdrant.client.aclose()
                 logger.info("Closed underlying Qdrant client (async)")
            # Fallback for synchronous close if needed (less likely for async test)
            elif hasattr(qdrant, 'close') and not asyncio.iscoroutinefunction(qdrant.close):
                 logger.warning("Closing Qdrant client synchronously.")
                 qdrant.close()
            else:
                 logger.warning("Could not determine async close method for Qdrant client.")

            # Close MongoDB client (if needed, usually managed globally or via context manager)
            # mongo.client.close() # Typically synchronous

        except Exception as e:
            # Log the actual exception type and message clearly
            logger.error(f"Error during cleanup: Type={type(e).__name__}, Message={str(e)}", exc_info=True) # Add exc_info=True
        logger.info("Cleanup completed")