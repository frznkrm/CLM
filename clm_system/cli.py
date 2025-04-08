#!/usr/bin/env python
"""
Command-line interface for CLM Smart Search System.
Allows users to ingest contracts, search, and manage the system.
"""
import asyncio
import json
import logging
import os
from typing import Optional

import click
from dotenv import load_dotenv
import functools 
from clm_system.core.ingestion import ContractIngestionService
from clm_system.core.search import QueryRouter
from clm_system.core.utils.embeddings import get_embedding_model

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("clm-cli")

# Initialize services
ingestion_service = ContractIngestionService()
query_router = QueryRouter()

@click.group()
def cli():
    """CLM Smart Search System CLI"""
    pass

def async_command(func):
    """Decorator to run async commands"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

# Changed decorator order and structure
@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@async_command
async def ingest(file_path):
    """Ingest a contract JSON file into the system."""
    try:
        with open(file_path, 'r') as f:
            contract_data = json.load(f)
        
        result = await ingestion_service.ingest_contract(contract_data)
        click.echo(f"Contract ingested successfully: {result['id']}")
        click.echo(f"Title: {result['title']}")
        click.echo(f"Clauses: {result['clauses_count']}")
        click.echo(f"Status: {result['status']}")
    except Exception as e:
        click.echo(f"Error ingesting contract: {str(e)}", err=True)

# Changed decorator order and structure
@cli.command()
@click.argument('query')
@click.option('--filters', '-f', help='JSON string with filters')
@click.option('--top-k', '-k', type=int, default=5, help='Number of results to return')
@async_command
async def search(query, filters=None, top_k=5):
    """Search for contracts using the query router."""
    try:
        filter_dict = json.loads(filters) if filters else None
        
        results = await query_router.route_query(query, filter_dict, top_k)
        
        click.echo(f"Query: {results['query']}")
        click.echo(f"Query type: {results['metadata']['query_type']}")
        click.echo(f"Total results: {results['total_results']}")
        click.echo(f"Execution time: {results['execution_time_ms']:.2f}ms")
        click.echo("\nResults:")
        
        for i, result in enumerate(results['results'], 1):
            click.echo(f"\n--- Result {i} ---")
            click.echo(f"Contract: {result['contract_title']} (ID: {result['contract_id']})")
            click.echo(f"Clause: {result.get('clause_title', result['clause_type'])}")
            click.echo(f"Relevance: {result['relevance_score']:.4f}")
            click.echo(f"Content: {result['content'][:100]}...")
    except Exception as e:
        click.echo(f"Error performing search: {str(e)}", err=True)

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
        from clm_system.core.utils.embeddings import compute_embedding
        
        embedding = compute_embedding(sample_text, model)
        
        click.echo(f"Successfully computed embedding with dimensions: {len(embedding)}")
        click.echo(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        click.echo(f"Error testing embedding model: {str(e)}", err=True)

if __name__ == "__main__":
    cli()