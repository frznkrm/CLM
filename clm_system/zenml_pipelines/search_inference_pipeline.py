# File: clm_system/core/zenml_pipelines/search_inference_pipeline.py
import logging
import os
from typing import Dict, List, Any, Optional, Annotated
from zenml import step, pipeline
from zenml.integrations.comet.flavors.comet_experiment_tracker_flavor import CometExperimentTrackerSettings
from clm_system.core.query_engine.search import QueryRouter # Keep QueryRouter
# Import clients directly if needed for explicit closing
from clm_system.core.query_engine.query_classifier import QueryClassifier
from clm_system.core.database.elasticsearch_client import ElasticsearchClient
from clm_system.core.database.qdrant_client import QdrantClient
from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding
from clm_system.config import settings
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from zenml.client import Client

logger = logging.getLogger(__name__)

comet_tracker = "comet_tracker"

# --- Helper to run async code safely ---
def run_async_in_thread(coro):
    """Runs an async coroutine in a separate thread with its own loop."""
    result = None
    exception = None

    def run():
        nonlocal result, exception
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
        except Exception as e:
            exception = e
        finally:
            try:
                # Gracefully shutdown pending tasks
                tasks = asyncio.all_tasks(loop)
                if tasks:
                    for task in tasks:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                # Shutdown async generators
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception as shutdown_exc:
                 logger.error(f"Error during loop shutdown: {shutdown_exc}")
            finally:
                 loop.close()
                 logger.debug("Asyncio event loop closed.")


    import threading
    thread = threading.Thread(target=run)
    thread.start()
    thread.join() # Wait for the thread to complete

    if exception:
        raise exception
    return result

# --- classify_query ---
@step(enable_cache=False)
def classify_query(query: str) -> Annotated[Dict[str, Any], "output"]:
    """Classify the query using QueryClassifier."""
    start_time = time.time()
    logger.info(f"Executing classify_query for query: {query}")

    async def _classify_and_close():
        # Instantiate classifier here to manage its lifecycle within the async context
        classifier = QueryClassifier()
        try:
            result = await classifier.classify(query)
            logger.debug(f"Query classification result: {result}")
            return result
        finally:
            # Explicitly close the underlying client if necessary and possible
            # Note: AsyncOpenAI might rely on httpx context management.
            # If it has an aclose(), call it here. Check library docs.
            if hasattr(classifier, 'client') and hasattr(classifier.client, 'aclose'):
                 logger.debug("Attempting to close QueryClassifier client")
                 await classifier.client.aclose()
            else:
                 logger.debug("QueryClassifier client does not have explicit aclose or not needed.")


    try:
        # Use the helper to run the async function
        classification = run_async_in_thread(_classify_and_close())

        if not isinstance(classification, dict):
            logger.error(f"Invalid classification result: {classification}")
            raise ValueError("Classification must return a dictionary")

        logger.info(f"Finished classify_query for '{query}' as {classification.get('query_type')}")
        return classification

    except Exception as e:
        logger.error(f"Classification step failed for '{query}': {str(e)}")
        raise

# --- execute_search ---
@step(enable_cache=False)
def execute_search(
    classification: Dict[str, Any],
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Execute search based on classification."""
    start_time = time.time()
    top_k = top_k or settings.default_top_k # Use settings for default
    classification_data = classification
    logger.info(f"Executing execute_search for query: {query}")

    if filters and "metadata.document_type" in filters:
        doc_types = [filters["metadata.document_type"]]
    else:
        doc_types = classification_data.get("doc_types", ["contract", "email", "recap", "deal"])

    all_results = []

    async def _search_single_type(doc_type):
        # Instantiate clients within the async function for proper lifecycle management
        es_client = ElasticsearchClient()
        qdrant_client = QdrantClient() # Assuming sync usage is okay, or use async version if available
        embedding_model = get_embedding_model() # Assuming this is lightweight

        type_filters = filters.copy() if filters else {}
        type_filters["metadata.document_type"] = doc_type
        query_type = classification_data["query_type"]
        results = []

        try:
            if query_type == "structured":
                results = await es_client.search(query, type_filters, top_k)
            elif query_type == "semantic":
                query_embedding = compute_embedding(query, embedding_model)
                # Assuming QdrantClient's search is async or wrapped appropriately
                results = await qdrant_client.search(query_embedding, type_filters, top_k)
            else:  # hybrid
                query_embedding = compute_embedding(query, embedding_model)
                es_results, qdrant_results = await asyncio.gather(
                    es_client.search(query, type_filters, top_k * 2),
                    qdrant_client.search(query_embedding, type_filters, top_k * 2)
                )
                # Assuming reciprocal_rank_fusion is synchronous
                from .helpers import reciprocal_rank_fusion # Ensure import
                results = reciprocal_rank_fusion(
                    es_results, qdrant_results, k=60, weight_a=0.4, weight_b=0.6
                )[:top_k]

            # Add metadata
            for result in results:
                if "metadata" not in result: result["metadata"] = {}
                result["metadata"]["document_type"] = doc_type
            return results

        finally:
            # Ensure clients used in this scope are closed
            logger.debug(f"Closing clients for doc_type: {doc_type}")
            await es_client.close()
            # Close Qdrant if it's async and needs closing
            if hasattr(qdrant_client, 'close') and asyncio.iscoroutinefunction(qdrant_client.close):
                 await qdrant_client.close()
            elif hasattr(qdrant_client, 'close'): # Sync close
                 qdrant_client.close()


    # Run searches for each doc_type sequentially using the helper
    # Running concurrently might hit resource limits or complicate debugging
    for doc_type in doc_types:
         logger.info(f"Running search for doc_type: {doc_type}")
         try:
             type_results = run_async_in_thread(_search_single_type(doc_type))
             all_results.extend(type_results)
         except Exception as e:
             logger.error(f"Search failed for doc_type {doc_type}: {e}")


    # --- Merging (moved out of async execution) ---
    # Use QueryRouter just for merging logic if needed, or implement directly
    # router = QueryRouter() # Avoid instantiating if only merge logic is needed
    # merged_results = router._merge_results([all_results], top_k) # If using router's merge

    # Simple merge and sort:
    merged_results = sorted(all_results, key=lambda x: -x.get("relevance_score", 0.0))[:top_k]


    logger.info(f"Finished execute_search for '{query}', found {len(merged_results)} results.")
    return merged_results


# --- merge_results: Now simpler, just receives final list ---
@step(enable_cache=False) # Keep tracker here
def merge_results(
    search_results: List[Dict[str, Any]], # Receives already merged list from execute_search
    top_k: int # Keep top_k for logging/consistency if needed
) -> Annotated[List[Dict[str, Any]], "output"]:
    """Logs final merged results."""
    start_time = time.time()
    # The results are already merged and sorted by execute_search
    final_results = search_results
    logger.info(f"Executing merge_results step with {len(final_results)} results.")

    execution_time = (time.time() - start_time) * 1000
    try:
        client = Client()
        experiment_tracker = client.active_stack.experiment_tracker
        if experiment_tracker:
             experiment_tracker.log_metrics({"merge_step_time_ms": execution_time}) # Renamed metric
             experiment_tracker.log_params({"final_results_count": len(final_results)})
        else:
             logger.warning("No active experiment tracker found in merge_results.")
    except Exception as e:
         logger.error(f"Failed to log to experiment tracker in merge_results: {e}")

    return final_results # Return the results passed in

# --- Pipeline Definition ---
@pipeline(
    name="search_inference_pipeline",
    settings={
        "experiment_tracker.comet": CometExperimentTrackerSettings(
            workspace=settings.comet_workspace,
            project_name=settings.comet_project_name
        )
    }
)
def search_inference_pipeline(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None
):
    """Search and inference pipeline."""
    classification = classify_query(query)
    # Execute search now handles merging across types
    search_results = execute_search(classification, query, filters, top_k)
    # Merge step now just logs and passes through
    final_results = merge_results(search_results, top_k or settings.default_top_k)
    return final_results

# --- Main block remains the same ---
if __name__ == "__main__":
    # Example usage:
    # query = "clauses.title:9. PAYMENT TERMS"
    query = "When are payments due?"
    filters = {"metadata.document_type": "contract"}
    top_k = 5

    # Run the pipeline
    pipeline_instance = search_inference_pipeline(query=query, filters=filters, top_k=top_k)
    # pipeline_instance.run() # If running outside pytest

    # If running directly for testing:
    results = pipeline_instance # The return value if called directly
    logger.info(f"Pipeline results: {results}")