import logging
import os
from typing import Dict, List, Any, Optional
from zenml import step, pipeline
from zenml.integrations.comet.flavors.comet_experiment_tracker_flavor import CometExperimentTrackerSettings
from opik import track
from clm_system.core.query_engine.search import QueryRouter
from clm_system.config import settings
import time
from zenml.client import Client

logger = logging.getLogger(__name__)

# Set Opik environment variables from settings
os.environ["OPIK_API_KEY"] = settings.opik_api_key
os.environ["OPIK_WORKSPACE"] = settings.opik_workspace
os.environ["OPIK_PROJECT_NAME"] = settings.opik_project_name

COMET_TRACKER = "comet_tracker"

@step(experiment_tracker=COMET_TRACKER)
@track
def classify_query(query: str) -> Dict[str, Any]:
    """Classify the query using QueryClassifier with isolation from existing event loops."""
    router = QueryRouter()
    start_time = time.time()
    
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    def run_async_classification():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(router.classifier.classify(query))
        finally:
            loop.close()
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        classification = executor.submit(run_async_classification).result()

    execution_time = (time.time() - start_time) * 1000
    experiment_tracker = Client().active_stack.experiment_tracker
    experiment_tracker.log_metrics({"classification_time_ms": execution_time})
    experiment_tracker.log_params({"query": query, "query_type": classification["query_type"]})
    return classification

@step(experiment_tracker=COMET_TRACKER)
def execute_search(
    classification: Dict[str, Any],
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Execute search based on classification."""
    router = QueryRouter()
    start_time = time.time()
    top_k = top_k or router.top_k

    if filters and "metadata.document_type" in filters:
        doc_types = [filters["metadata.document_type"]]
    else:
        doc_types = classification.get("doc_types", ["contract", "email", "recap", "deal"])

    search_results = []
    
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def run_async_search(doc_type):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                router._search_by_type(query, classification["query_type"], filters, top_k, doc_type)
            )
        finally:
            loop.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_async_search, doc_type) for doc_type in doc_types]
        for future in futures:
            search_results.extend(future.result())

    execution_time = (time.time() - start_time) * 1000
    experiment_tracker = Client().active_stack.experiment_tracker
    experiment_tracker.log_metrics({"search_time_ms": execution_time})
    experiment_tracker.log_params({"doc_types": doc_types, "total_results": len(search_results)})
    return search_results

@step(experiment_tracker=COMET_TRACKER)
def merge_results(
    search_results: List[Dict[str, Any]],
    top_k: int
) -> List[Dict[str, Any]]:
    """Merge and sort search results."""
    router = QueryRouter()
    start_time = time.time()
    merged_results = router._merge_results([search_results], top_k)
    execution_time = (time.time() - start_time) * 1000
    experiment_tracker = Client().active_stack.experiment_tracker
    experiment_tracker.log_metrics({"merge_time_ms": execution_time})
    experiment_tracker.log_params({"final_results_count": len(merged_results)})
    return merged_results

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
    """Search and inference pipeline integrating ZenML, CometML, and Opik."""
    classification = classify_query(query)
    search_results = execute_search(classification, query, filters, top_k)
    merged_results = merge_results(search_results, top_k or settings.default_top_k)
    return merged_results

if __name__ == "__main__":
    query = "Find confidentiality clauses in active contracts"
    filters = {"metadata.status": "active"}
    top_k = 5
    results = search_inference_pipeline(query, filters, top_k)
    logger.info(f"Pipeline results: {results}")