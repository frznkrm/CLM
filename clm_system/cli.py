#!/usr/bin/env python
"""
Command-line interface for CLM Smart Search System.
Allows users to ingest contracts, search, and manage the system.
"""
import asyncio
import json
import logging
import os
import functools

import click
from dotenv import load_dotenv
from pydantic import ValidationError

from clm_system.schemas.schemas import ContractCreate
from clm_system.core.pipeline.orchestrator import PipelineService
from clm_system.core.queryEngine.search import QueryRouter
from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("clm-cli")

# Initialize services
pipeline = PipelineService()      # no args, uses its internal defaults
query_router = QueryRouter()

def async_command(func):
    """Decorator to run async commands."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

@click.group()
def cli():
    """CLM Smart Search System CLI"""
    pass

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@async_command
async def ingest(file_path):
    """Ingest a contract JSON file into the system."""
    # 1) Load raw JSON
    try:
        raw = json.load(open(file_path, 'r'))
    except Exception as e:
        click.echo(f"Failed to read file {file_path}: {e}", err=True)
        return

    # 2) Validate & coerce with Pydantic
    try:
        contract_obj = ContractCreate.parse_obj(raw)
    except ValidationError as ve:
        click.echo("Invalid contract JSON:", err=True)
        click.echo(ve.json(), err=True)
        return

    # 3) Run through your pipeline
    try:
        result = await pipeline.process_contract(contract_obj.dict())
    except Exception as e:
        click.echo(f"Error ingesting contract: {e}", err=True)
        return

    # 4) Success output
    click.echo(f"Contract ingested successfully: {result['id']}")
    click.echo(f"Title: {result['title']}")
    click.echo(f"Clauses: {result['clauses_count']}")
    click.echo(f"Status: {result['status']}")

@cli.command()
@click.argument('query')
@click.option('--filters', '-f', help='JSON string with filters')
@click.option('--top-k', '-k', type=int, default=5, help='Number of results to return')
@async_command
async def search(query, filters=None, top_k=5):
    """Search for contracts using the query router."""
    # Parse filters JSON if provided
    try:
        filter_dict = json.loads(filters) if filters else None
    except Exception as e:
        click.echo(f"Invalid filters JSON: {e}", err=True)
        return

    # Perform search
    try:
        results = await query_router.route_query(query, filter_dict, top_k)
    except Exception as e:
        click.echo(f"Error performing search: {e}", err=True)
        return

    # Display results
    click.echo(f"Query: {results['query']}")
    click.echo(f"Query type: {results['metadata']['query_type']}")
    click.echo(f"Total results: {results['total_results']}")
    click.echo(f"Execution time: {results['execution_time_ms']:.2f}ms")
    click.echo("\nResults:")
    for i, r in enumerate(results['results'], 1):
        click.echo(f"\n--- Result {i} ---")
        click.echo(f"Contract: {r['contract_title']} (ID: {r['contract_id']})")
        click.echo(f"Clause: {r.get('clause_title', r['clause_type'])}")
        click.echo(f"Relevance: {r['relevance_score']:.4f}")
        click.echo(f"Content: {r['content'][:100]}...")

@cli.command()
@click.option('--model-name', '-m', help='Override the default embedding model')
def test_embedding(model_name=None):
    """Test the embedding model with a sample text."""
    try:
        if model_name:
            os.environ["EMBEDDING_MODEL"] = model_name
            click.echo(f"Using model: {model_name}")
        else:
            click.echo(f"Using default model: {os.getenv('EMBEDDING_MODEL')}")

        model = get_embedding_model()
        sample_text = "This is a test sentence to verify the embedding model is working correctly."
        embedding = compute_embedding(sample_text, model)

        click.echo(f"Successfully computed embedding with dimensions: {len(embedding)}")
        click.echo(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        click.echo(f"Error testing embedding model: {e}", err=True)

if __name__ == "__main__":
    cli()
