import asyncio
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
import ipdb
import argparse
import nest_asyncio

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
logger.debug("Initialized logger for clm_system workflow")

async def run_workflow(docx_path: str):
    # Check if DOCX file exists
    if not os.path.exists(docx_path):
        logger.error(f"DOCX file not found at {docx_path}")
        raise FileNotFoundError(f"DOCX file not found at {docx_path}")

    # Initialize Comet.ml experiment
    experiment = Experiment(
        project_name="clm-system-experiments",
        display_summary_level=0  # Disables summary output
    )

    try:
        # Clear previous pipeline runs and artifacts to avoid caching
        logger.info("Clearing previous ZenML pipeline runs and artifacts")
        client = Client()
        pipeline_name = "document_processing_pipeline"
        for run in client.list_pipeline_runs(pipeline_name=pipeline_name):
            client.delete_pipeline_run(run.id)
        for artifact in client.list_artifacts():
            client.delete_artifact(artifact.id)
        logger.info("ZenML pipeline runs and artifacts cleared")

        nest_asyncio.apply()
        mongo = MongoDBClient()
        es = ElasticsearchClient()
        qdrant = QdrantClient()
        pipeline = PipelineService()
        doc_id = None  # Initialize doc_id to avoid reference error in finally block

        # Clear existing data in Elasticsearch
        await es.client.indices.delete(index=es.index_name, ignore_unavailable=True)
        await es.ensure_index()

        # --- Phase 1: Process DOCX File ---
        logger.info(f"Processing DOCX file: {docx_path}")
        pipeline_run = document_processing_pipeline(
            file_path=docx_path,
            doc_type="contract"
        )  # Explicitly run the pipeline
        
        # Correctly access step outputs
        store_step_output = pipeline_run.steps['store_step'].outputs['output'].load()
        doc_id = store_step_output["id"]
        logger.info(f"Processed document ID: {doc_id}")

        # Verify ingestion in MongoDB
        logger.info("Verifying MongoDB storage")
        db_contract = await mongo.get_document(doc_id)
        if db_contract is None:
            raise AssertionError("Contract not found in MongoDB")
        if "title" not in db_contract:
            raise AssertionError("Title missing in MongoDB document")
        if "clauses" not in db_contract:
            raise AssertionError("Clauses missing in MongoDB document")
        logger.info("MongoDB storage verified")

        # Verify in Elasticsearch
        logger.info("Verifying Elasticsearch indexing")
        es_contract = await es.client.get(index="documents", id=doc_id)
        if not es_contract["found"]:
            raise AssertionError("Contract not found in Elasticsearch")
        if "title" not in es_contract["_source"]:
            raise AssertionError("Title missing in Elasticsearch document")
        logger.info("Elasticsearch indexing verified")

        # Verify in Qdrant
        logger.info("Verifying Qdrant storage")
        qdrant_points = await qdrant.scroll(doc_id)
        if len(qdrant_points) == 0:
            raise AssertionError("No chunks found in Qdrant")
        logger.info(f"Qdrant storage verified with {len(qdrant_points)} points")

        # --- Phase 2: Structured Queries ---
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
            if not result:
                raise AssertionError(f"No results for query: {test['query']}")

            if test["field"].startswith("metadata.parties"):
                party_names = [party["name"] for party in result[0]["metadata"].get("parties", [])]
                if test["expected"] not in party_names:
                    raise AssertionError(f"Expected party '{test['expected']}' not found for query: {test['query']}")
            else:
                field_value = result[0].get(test["field"])
                if field_value != test["expected"]:
                    raise AssertionError(f"Expected clause '{test['expected']}', got '{field_value}' for query: {test['query']}")
            logger.info(f"Structured query test passed for: {test['query']}")

        # --- Phase 3: Semantic Queries ---
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
            if not result:
                raise AssertionError(f"No results for query: {test['query']}")
            field_value = result[0].get(test["field"], "")
            if test["expected"] not in field_value.lower():
                raise AssertionError(f"Expected content '{test['expected']}' not found in '{field_value}' for query: {test['query']}")
            logger.info(f"Semantic query test passed for: {test['query']}")

    # Inside the finally block of run_workflow
    finally:
        logger.info("Cleaning up test data")
        try:
            if doc_id:
                # MongoDB deletion (synchronous) - NO await
                logger.debug(f"Deleting MongoDB document: {doc_id}")
                delete_result = mongo.documents_collection.delete_one({"id": doc_id})
                logger.debug(f"MongoDB delete result: {delete_result.deleted_count}")

                # Elasticsearch deletion (async) - Keep await
                logger.debug(f"Deleting Elasticsearch document: {doc_id}")
                await es.client.options(ignore_status=[404]).delete(
                    index=es.index_name, id=doc_id
                )

                # Qdrant deletion (async) - Keep await
                logger.debug(f"Deleting Qdrant points for document: {doc_id}")
                await qdrant.client.delete(
                    collection_name="document_chunks",
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
            # Close async clients properly
            if es and es.client:
                logger.debug("Closing Elasticsearch client")
                await es.close()
            if qdrant and qdrant.client:
                logger.debug("Closing Qdrant client")
                # Qdrant client might not have an async close, check its docs
                # qdrant.client.close() # If synchronous
                pass # Or handle appropriately
            # Close MongoDB client if necessary (usually managed by motor)

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True) # Add traceback
        logger.info("Cleanup completed")
        # End Comet.ml experiment
        if experiment: # Check if experiment was initialized
            experiment.end()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ZenML workflow")
    parser.add_argument("docx_path", type=str, help="Path to the DOCX file")
    args = parser.parse_args()
    asyncio.run(run_workflow(args.docx_path))