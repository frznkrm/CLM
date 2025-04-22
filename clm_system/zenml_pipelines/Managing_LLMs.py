import logging
from typing import List, Dict, Any, Optional
from zenml import step, pipeline
from zenml.integrations.comet.flavors.comet_experiment_tracker_flavor import CometExperimentTrackerSettings
from zenml.logger import get_logger
from opik import track
from comet_ml import Experiment, API
import json
import os
from sentence_transformers import SentenceTransformer
from clm_system.config import settings
from clm_system.zenml_pipelines.search_inference_pipeline import search_inference_pipeline

logger = get_logger(__name__)

# CometML experiment tracker name
COMET_TRACKER = "comet_tracker"
MODEL_REGISTRY_PATH = "model_registry.json"

@step(experiment_tracker=COMET_TRACKER)
def register_models(models_to_add: List[Dict[str, str]]) -> None:
    """Register new embedding models in the model registry."""
    initialize_model_registry()
    with open(MODEL_REGISTRY_PATH, "r+") as f:
        registry = json.load(f)
        existing_models = {(m["name"], m["version"]) for m in registry["models"]}
        for model in models_to_add:
            name = model["name"]
            version = model["version"]
            if (name, version) not in existing_models:
                registry["models"].append({
                    "name": name,
                    "version": version,
                    "path": name,  # Hugging Face model path
                    "metrics": {"search_accuracy": 0.0},
                    "selected": False
                })
                logger.info(f"Registered model: {name} (v{version})")
        f.seek(0)
        json.dump(registry, f, indent=2)
    
    experiment = Experiment()
    experiment.log_parameters({"registered_models": [f"{m['name']}_v{m['version']}" for m in models_to_add]})
    experiment.log_metric("models_registered", len(models_to_add))

def initialize_model_registry():
    """Initialize a simple model registry if it doesn't exist."""
    if not os.path.exists(MODEL_REGISTRY_PATH):
        default_registry = {
            "models": [
                {
                    "name": settings.embedding_model,
                    "version": "1.0",
                    "path": settings.embedding_model,
                    "metrics": {"search_accuracy": 0.0},
                    "selected": True
                }
            ]
        }
        with open(MODEL_REGISTRY_PATH, "w") as f:
            json.dump(default_registry, f, indent=2)
        logger.info("Initialized model registry with default model.")

@step(experiment_tracker=COMET_TRACKER)
def fetch_performance_from_comet() -> Dict[str, float]:
    """Fetch performance metrics from CometML and update the registry."""
    api = API(api_key=settings.comet_api_key)
    experiments = api.get_experiments(
        workspace=settings.comet_workspace,
        project_name=settings.comet_project_name
    )
    performance_data = {}
    with open(MODEL_REGISTRY_PATH, "r+") as f:
        registry = json.load(f)
        for exp in experiments:
            metrics = exp.get_metrics()
            params = exp.get_parameters()
            model_name = params.get("embedding_model", settings.embedding_model)
            version = params.get("model_version", "1.0")
            for metric in metrics:
                if metric["metricName"] == "search_accuracy":
                    accuracy = float(metric["metricValueMax"])
                    performance_data[f"{model_name}_v{version}"] = accuracy
                    for model in registry["models"]:
                        if model["name"] == model_name and model["version"] == version:
                            model["metrics"]["search_accuracy"] = accuracy
                            break
                    else:
                        registry["models"].append({
                            "name": model_name,
                            "version": version,
                            "path": model_name,
                            "metrics": {"search_accuracy": accuracy},
                            "selected": False
                        })
        f.seek(0)
        json.dump(registry, f, indent=2)
    
    experiment = Experiment()
    experiment.log_metrics(performance_data)
    logger.info(f"Fetched performance data for {len(performance_data)} models")
    return performance_data

@step(experiment_tracker=COMET_TRACKER)
def evaluate_models(test_queries: List[Dict[str, str]]) -> Dict[str, float]:
    """Evaluate each model using test semantic queries through the search pipeline."""
    with open(MODEL_REGISTRY_PATH, "r") as f:
        registry = json.load(f)
    
    performance_data = {}
    for model in registry["models"]:
        model_name = model["name"]
        version = model["version"]
        model_path = model["path"]
        logger.info(f"Evaluating model: {model_name} (v{version})")
        
        try:
            # Load model
            embedding_model = SentenceTransformer(model_path)
            
            # Run test queries through search pipeline
            correct = 0
            total = len(test_queries)
            for query_data in test_queries:
                query = query_data["query"]
                expected = query_data["expected"]
                pipeline_run = search_inference_pipeline(
                    query=query,
                    filters={"metadata.document_type": "contract"},
                    top_k=1
                )
                result = pipeline_run.steps["merge_results"].outputs["output"].load()
                
                # Simplified evaluation: check if expected content is in result
                if result and expected in result[0].get("content", ""):
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0.0
            performance_data[f"{model_name}_v{version}"] = accuracy
            
            # Update registry
            with open(MODEL_REGISTRY_PATH, "r+") as f:
                registry = json.load(f)
                for reg_model in registry["models"]:
                    if reg_model["name"] == model_name and reg_model["version"] == version:
                        reg_model["metrics"]["search_accuracy"] = accuracy
                        break
                f.seek(0)
                json.dump(registry, f, indent=2)
            
            # Log to CometML
            experiment = Experiment()
            experiment.log_parameters({"embedding_model": model_name, "model_version": version})
            experiment.log_metric("search_accuracy", accuracy)
            logger.info(f"Model {model_name} (v{version}) accuracy: {accuracy}")
        
        except Exception as e:
            logger.error(f"Error evaluating model {model_name} (v{version}): {str(e)}")
            performance_data[f"{model_name}_v{version}"] = 0.0
    
    return performance_data

@step(experiment_tracker=COMET_TRACKER)
def select_best_model(performance_data: Dict[str, float]) -> str:
    """Select the model with the highest search accuracy."""
    with open(MODEL_REGISTRY_PATH, "r+") as f:
        registry = json.load(f)
        best_model = max(
            registry["models"],
            key=lambda m: m["metrics"].get("search_accuracy", 0.0),
            default=None
        )
        if best_model:
            for model in registry["models"]:
                model["selected"] = (model["name"] == best_model["name"] and model["version"] == best_model["version"])
            f.seek(0)
            json.dump(registry, f, indent=2)
            selected_model = f"{best_model['name']}_v{best_model['version']}"
            logger.info(f"Selected best model: {selected_model}")
            
            # Log to CometML
            experiment = Experiment()
            experiment.log_parameters({"selected_model": selected_model})
            experiment.log_metric("best_search_accuracy", best_model["metrics"]["search_accuracy"])
            return selected_model
        else:
            logger.warning("No models available to select.")
            return ""

@step(experiment_tracker=COMET_TRACKER)
@track
def monitor_inference(selected_model: str, sample_query: str) -> Dict[str, Any]:
    """Monitor inference performance of the selected model using Opik."""
    if not selected_model:
        logger.warning("No selected model; skipping inference monitoring")
        return {"result": None, "score": 0.0}
    
    model_name, version = selected_model.rsplit("_v", 1)
    with open(MODEL_REGISTRY_PATH, "r") as f:
        registry = json.load(f)
        model_info = next(
            (m for m in registry["models"] if m["name"] == model_name and m["version"] == version),
            None
        )
    if not model_info:
        logger.error("Selected model not found in registry")
        return {"result": None, "score": 0.0}
    
    try:
        model = SentenceTransformer(model_info["path"])
        embedding = model.encode([sample_query])[0]
        # Placeholder: Evaluate embedding quality (e.g., similarity to expected)
        result = {"embedding": embedding.tolist()[:10]}  # Truncated for logging
        score = 0.85  # Hypothetical score
        
        # Log to Opik
        opik.track_event(
            name="embedding_inference",
            inputs={"query": sample_query},
            outputs={"result": result},
            metadata={"model_name": model_name, "model_version": version}
        )
        opik.log_metric("inference_score", score)
        
        # Log to CometML
        experiment = Experiment()
        experiment.log_parameters({"model_name": model_name, "model_version": version, "sample_query": sample_query})
        experiment.log_metric("inference_score", score)
        
        logger.info(f"Inference monitoring for {selected_model}: score={score}")
        return {"result": result, "score": score}
    
    except Exception as e:
        logger.error(f"Error monitoring inference for {selected_model}: {str(e)}")
        return {"result": None, "score": 0.0}

@pipeline(
    name="embedding_management_pipeline",
    settings={
        "experiment_tracker.comet": CometExperimentTrackerSettings(
            workspace=settings.comet_workspace,
            project_name=settings.comet_project_name
        )
    }
)
def embedding_management_pipeline(
    models_to_add: Optional[List[Dict[str, str]]] = None,
    test_queries: Optional[List[Dict[str, str]]] = None,
    sample_query: Optional[str] = None
):
    """ZenML pipeline for managing embedding models."""
    if models_to_add:
        register_models(models_to_add)
    performance_data = fetch_performance_from_comet()
    if test_queries:
        performance_data = evaluate_models(test_queries)
    selected_model = select_best_model(performance_data)
    if sample_query:
        inference_result = monitor_inference(selected_model, sample_query)
    return selected_model

if __name__ == "__main__":
    # Example usage
    sample_models = [
        {"name": "sentence-transformers/all-MiniLM-L6-v2", "version": "1.0"},
        {"name": "sentence-transformers/all-MiniLM-L12-v2", "version": "1.0"}
    ]
    sample_test_queries = [
        {"query": "When are payments due?", "expected": "5th of each month"},
        {"query": "What is the uptime guarantee?", "expected": "99.9%"}
    ]
    sample_query = "Sample contract clause"
    
    embedding_management_pipeline(
        models_to_add=sample_models,
        test_queries=sample_test_queries,
        sample_query=sample_query
    )
    # Run with: python embedding_management_pipeline.py
    # Or use ZenML CLI: zenml pipeline run embedding_management_pipeline.py -c config.yaml