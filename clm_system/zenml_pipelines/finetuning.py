# File: clm_system/core/zenml_pipelines/managing_finetuning_LLMs.py
import logging
import os
from typing import Dict, List, Any, Optional, Annotated
from zenml import step, pipeline
from zenml.integrations.comet.flavors.comet_experiment_tracker_flavor import CometExperimentTrackerSettings
from zenml.client import Client
from opik import track
from comet_ml import Experiment
import pymongo
from sentence_transformers import SentenceTransformer, InputExample, losses
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import uuid
import json
from datetime import datetime
from clm_system.config import settings  # Your settings module
from clm_system.core.utils.embeddings import get_embedding_model  # Assuming this exists in your codebase

logger = logging.getLogger(__name__)

# CometML experiment tracker name
COMET_TRACKER = "comet_tracker"

class QueryDataset(Dataset):
    """Custom dataset for query router fine-tuning."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

@step(experiment_tracker=COMET_TRACKER)
def collect_data(model_type: str) -> List[Dict[str, Any]]:
    """Collect training data from MongoDB based on model type."""
    mongo = pymongo.MongoClient(settings.mongodb_uri)
    db = mongo[settings.mongodb_database]
    
    try:
        if model_type == "embedding":
            documents = db["documents"].find({"metadata.document_type": "contract"})
            data = [{"text": clause["text"]} for doc in documents for clause in doc.get("clauses", [])]
            logger.info(f"Collected {len(data)} text samples for embedding model")
            return data
        elif model_type == "query_router":
            # Assume a 'query_logs' collection with query-label pairs
            queries = db["query_logs"].find()  # Replace with actual collection or data source
            data = [{"query": q["query"], "label": q["label"]} for q in queries]
            # Fallback: Sample data if query_logs is empty
            if not data:
                data = [
                    {"query": "When are payments due?", "label": "payment"},
                    {"query": "How long is the confidentiality period?", "label": "confidentiality"},
                    {"query": "What are the termination conditions?", "label": "termination"},
                ]
            logger.info(f"Collected {len(data)} query samples for query router")
            return data
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    finally:
        mongo.close()

@step(experiment_tracker=COMET_TRACKER)
def prepare_data(data: List[Dict[str, Any]], model_type: str) -> Any:
    """Prepare data for training."""
    if model_type == "embedding":
        # Prepare InputExamples for sentence-transformers
        examples = [InputExample(texts=[d["text"]]) for d in data]
        logger.info(f"Prepared {len(examples)} samples for embedding model")
        return examples
    elif model_type == "query_router":
        tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        label_map = {label: idx for idx, label in enumerate(set(d["label"] for d in data))}
        encodings = tokenizer(
            [d["query"] for d in data],
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        labels = [label_map[d["label"]] for d in data]
        dataset = QueryDataset(encodings, labels)
        logger.info(f"Prepared {len(dataset)} samples for query router")
        return {"dataset": dataset, "num_labels": len(label_map)}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

@step(experiment_tracker=COMET_TRACKER)
def fine_tune_model(prepared_data: Any, model_type: str) -> Any:
    """Fine-tune the model on prepared data."""
    experiment = Experiment(
        api_key=settings.comet_api_key,
        project_name=settings.comet_project_name,
        workspace=settings.comet_workspace
    )
    
    if model_type == "embedding":
        model = SentenceTransformer(settings.embedding_model)
        train_dataloader = torch.utils.data.DataLoader(prepared_data, batch_size=8, shuffle=True)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100
        )
        logger.info("Fine-tuned embedding model")
        experiment.log_metric("fine_tune_status", "success")
        return model
    elif model_type == "query_router":
        model = AutoModelForSequenceClassification.from_pretrained(
            settings.embedding_model,
            num_labels=prepared_data["num_labels"]
        )
        training_args = TrainingArguments(
            output_dir=f"./fine_tuned_{model_type}",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            logging_steps=10,
            save_strategy="epoch"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=prepared_data["dataset"]
        )
        trainer.train()
        logger.info("Fine-tuned query router model")
        experiment.log_metric("fine_tune_status", "success")
        return model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

@step(experiment_tracker=COMET_TRACKER)
def evaluate_model(fine_tuned_model: Any, model_type: str) -> bool:
    """Evaluate the fine-tuned model against the current version."""
    experiment = Experiment(
        api_key=settings.comet_api_key,
        project_name=settings.comet_project_name,
        workspace=settings.comet_workspace
    )
    
    if model_type == "embedding":
        # Placeholder: Evaluate embedding quality (e.g., cosine similarity on validation set)
        current_model = get_embedding_model()  # From your utils
        validation_texts = ["Sample validation text 1", "Sample validation text 2"]
        current_embeddings = current_model.encode(validation_texts)
        fine_tuned_embeddings = fine_tuned_model.encode(validation_texts)
        # Simplified metric: cosine similarity (replace with actual evaluation)
        current_score = 0.80  # Hypothetical
        fine_tuned_score = 0.85  # Hypothetical
    elif model_type == "query_router":
        # Placeholder: Evaluate classification accuracy
        tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        validation_data = [
            {"query": "What is the uptime guarantee?", "label": "sla"},
            {"query": "How are disputes resolved?", "label": "dispute"}
        ]
        label_map = {"payment": 0, "confidentiality": 1, "termination": 2, "sla": 3, "dispute": 4}
        inputs = tokenizer(
            [d["query"] for d in validation_data],
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        labels = [label_map[d["label"]] for d in validation_data]
        current_model = AutoModelForSequenceClassification.from_pretrained(settings.embedding_model, num_labels=len(label_map))
        with torch.no_grad():
            current_outputs = current_model(**inputs).logits.argmax(dim=1)
            fine_tuned_outputs = fine_tuned_model(**inputs).logits.argmax(dim=1)
        current_score = (current_outputs == torch.tensor(labels)).float().mean().item()
        fine_tuned_score = (fine_tuned_outputs == torch.tensor(labels)).float().mean().item()
    
    is_better = fine_tuned_score > current_score
    experiment.log_metrics({
        "current_score": current_score,
        "fine_tuned_score": fine_tuned_score,
        "is_better": int(is_better)
    })
    logger.info(f"Evaluation: fine-tuned {model_type} {'improved' if is_better else 'did not improve'}")
    return is_better

@step(experiment_tracker=COMET_TRACKER)
def deploy_model(fine_tuned_model: Any, model_type: str, is_better: bool) -> str:
    """Deploy the fine-tuned model if it performs better."""
    experiment = Experiment(
        api_key=settings.comet_api_key,
        project_name=settings.comet_project_name,
        workspace=settings.comet_workspace
    )
    
    if is_better:
        model_id = str(uuid.uuid4())
        output_path = f"./models/fine_tuned_{model_type}_{model_id}"
        fine_tuned_model.save_pretrained(output_path)
        logger.info(f"Saved fine-tuned {model_type} model to {output_path}")
        experiment.log_model(f"fine_tuned_{model_type}_{model_id}", output_path)
        
        # Model Access Options:
        # 1. FastAPI: Run a FastAPI server to serve the model (see below).
        #    Update your pipelines to query http://localhost:8000/predict/<model_type>
        # 2. LM Studio: Import 'output_path' into LM Studio for local serving.
        # 3. Direct Access: Update get_embedding_model() or QueryRouter to load from 'output_path'.
        # Update Process: Modify your pipelines to use the latest 'output_path' or API endpoint.
        
        experiment.log_metric("deploy_status", "success")
        return output_path
    else:
        logger.info(f"Skipping deployment for {model_type} as it did not improve")
        experiment.log_metric("deploy_status", "skipped")
        return ""

@step(experiment_tracker=COMET_TRACKER)
@track
def inference_evaluation(model_type: str, model_path: str, query: str) -> Dict[str, Any]:
    """Evaluate model performance during inference using Opik."""
    experiment = Experiment(
        api_key=settings.comet_api_key,
        project_name=settings.comet_project_name,
        workspace=settings.comet_workspace
    )
    
    if not model_path:
        logger.warning("No model path provided; skipping inference evaluation")
        return {"result": None, "score": 0.0}
    
    if model_type == "embedding":
        model = SentenceTransformer(model_path)
        embedding = model.encode([query])[0]
        # Placeholder: Evaluate embedding quality (e.g., similarity to expected)
        result = {"embedding": embedding.tolist()[:10]}  # Truncated for logging
        score = 0.85  # Hypothetical
    elif model_type == "query_router":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        inputs = tokenizer(query, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs).logits.argmax(dim=1).item()
        label_map = {0: "payment", 1: "confidentiality", 2: "termination", 3: "sla", 4: "dispute"}
        result = {"predicted_label": label_map.get(outputs, "unknown")}
        score = 0.90  # Hypothetical accuracy
        
        # Log to Opik
        opik.track_event(
            name="inference_evaluation",
            inputs={"query": query},
            outputs={"result": result},
            metadata={"model_type": model_type, "model_path": model_path}
        )
        opik.log_metric("inference_score", score)
    
    experiment.log_metrics({"inference_score": score})
    experiment.log_parameters({"query": query, "model_type": model_type})
    logger.info(f"Inference result for {model_type} on query '{query}': {result}")
    return {"result": result, "score": score}

@pipeline(
    name="fine_tuning_pipeline",
    settings={
        "experiment_tracker.comet": CometExperimentTrackerSettings(
            workspace=settings.comet_workspace,
            project_name=settings.comet_project_name
        )
    }
)
def fine_tuning_pipeline(model_type: str, query: Optional[str] = None):
    """ZenML pipeline for fine-tuning and managing LLMs with Opik integration."""
    data = collect_data(model_type)
    prepared_data = prepare_data(data, model_type)
    fine_tuned_model = fine_tune_model(prepared_data, model_type)
    is_better = evaluate_model(fine_tuned_model, model_type)
    model_path = deploy_model(fine_tuned_model, model_type, is_better)
    if query and model_path:
        inference_result = inference_evaluation(model_type, model_path, query)
    return model_path

if __name__ == "__main__":
    # Example usage
    fine_tuning_pipeline(model_type="embedding", query="Sample contract clause")
    fine_tuning_pipeline(model_type="query_router", query="When are payments due?")
    # Run with: python fine_tuning_pipeline.py
    # Or use ZenML CLI: zenml pipeline run fine_tuning_pipeline.py -c config.yaml

# Optional FastAPI deployment (save as separate file, e.g., api.py)
"""
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel

app = FastAPI()

class InferenceRequest(BaseModel):
    model_type: str
    query: str

# Load latest model (simplified; use a model registry or config in production)
embedding_model = None
query_router_model = None
tokenizer = None

@app.on_event("startup")
def load_models():
    global embedding_model, query_router_model, tokenizer
    # Replace with logic to load latest model path from ./models/
    embedding_model = SentenceTransformer("path_to_latest_embedding_model")
    query_router_model = AutoModelForSequenceClassification.from_pretrained("path_to_latest_query_router_model")
    tokenizer = AutoTokenizer.from_pretrained("path_to_latest_query_router_model")

@app.post("/predict")
async def predict(request: InferenceRequest):
    if request.model_type == "embedding":
        embedding = embedding_model.encode([request.query])[0].tolist()
        return {"result": embedding[:10]}  # Truncated for brevity
    elif request.model_type == "query_router":
        inputs = tokenizer(request.query, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = query_router_model(**inputs).logits.argmax(dim=1).item()
        label_map = {0: "payment", 1: "confidentiality", 2: "termination", 3: "sla", 4: "dispute"}
        return {"result": label_map.get(outputs, "unknown")}
    else:
        return {"error": "Invalid model_type"}

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""