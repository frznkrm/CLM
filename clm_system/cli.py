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

from clm_system.schemas.schemas import ContractCreate, EmailCreate, DealCreate, RecapCreate
from clm_system.core.pipeline.orchestrator import PipelineService
from clm_system.core.query_engine.search import QueryRouter
from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding
from clm_system.core.pipeline.preprocessing.pdf_processor import PDFProcessor
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

def load_file(file_path):
    """Load content from a file."""
    try:
        return json.load(open(file_path, 'r'))
    except Exception as e:
        click.echo(f"Failed to read file {file_path}: {e}", err=True)
        return None

def detect_document_type(file_path):
    """Detect document type from file extension or content."""
    # Simple extension-based detection for demo
    if file_path.lower().endswith('.contract.json'):
        return 'contract'
    elif file_path.lower().endswith('.email.json'):
        return 'email'
    elif file_path.lower().endswith('.deal.json'):
        return 'deal'
    elif file_path.lower().endswith('.recap.json'):
        return 'recap'
    else:
        # Default to contract for now
        return 'contract'

def handle_validation_error(error):
    """Handle and display validation errors."""
    click.echo("Invalid document JSON:", err=True)
    click.echo(error.json(), err=True)

def show_ingestion_result(doc_type, result):
    """Display ingestion result based on document type."""
    click.echo(f"{doc_type.capitalize()} ingested successfully: {result['id']}")
    click.echo(f"Title: {result['title']}")
    
    if doc_type == 'contract' and 'clauses_count' in result:
        click.echo(f"Clauses: {result['clauses_count']}")
    elif doc_type == 'email' and 'has_attachments' in result:
        click.echo(f"Has attachments: {result['has_attachments']}")
    elif doc_type == 'deal' and 'deal_type' in result:
        click.echo(f"Deal type: {result['deal_type']}")
    elif doc_type == 'recap' and 'participants' in result:
        click.echo(f"Participants: {len(result.get('participants', []))}")
        
    click.echo(f"Status: {result['status']}")

@click.group()
def cli():
    """CLM Smart Search System CLI"""
    pass

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--type', '-t', type=click.Choice(['contract', 'email', 'recap', 'deal']),
              help='Explicit document type')
@async_command
async def ingest(file_path, type):
    """Ingest any supported document type into the system."""
    raw = load_file(file_path)
    if not raw:
        return
    
    # Map file extensions to types if not specified
    if not type:
        type = detect_document_type(file_path)
        
    try:
        # Validate against appropriate schema
        if type == 'contract':
            doc = ContractCreate.parse_obj(raw)
        elif type == 'email':
            doc = EmailCreate.parse_obj(raw)
        elif type == 'recap':
            doc = RecapCreate.parse_obj(raw)
        elif type == 'deal':
            doc = DealCreate.parse_obj(raw)
            
        result = await pipeline.process_document(doc.dict())
        show_ingestion_result(type, result)
        
    except ValidationError as e:
        handle_validation_error(e)

@cli.command()
@click.argument('query')
@click.option('--filters', '-f', help='JSON string with filters')
@click.option('--type', '-t', type=click.Choice(['contract', 'email', 'recap', 'deal']),
              help='Limit search to specific document type')
@click.option('--top-k', '-k', type=int, default=5, help='Number of results to return')
@async_command
async def search(query, filters=None, type=None, top_k=5):
    """Search for documents using the query router."""
    # Parse filters JSON if provided
    try:
        filter_dict = json.loads(filters) if filters else {}
    except Exception as e:
        click.echo(f"Invalid filters JSON: {e}", err=True)
        return

    # Add document_type filter if specified
    if type:
        filter_dict["metadata.document_type"] = type

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
        doc_type = r.get('metadata', {}).get('document_type', 'document')
        click.echo(f"{doc_type.capitalize()}: {r.get('document_title', 'Untitled')} (ID: {r.get('document_id', 'Unknown')})")
        if 'clause_title' in r or 'clause_type' in r:
            click.echo(f"Clause: {r.get('clause_title', r.get('clause_type', 'Unknown'))}")
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

@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--title', '-t', help='Optional title override')
@click.option('--type', '-d', type=click.Choice(['contract', 'email', 'recap', 'deal']), 
              default='contract', help='Document type')
@click.option('--document-type', '-dt', help='Specific document sub-type (e.g., NDA, lease)')
@click.option('--tags', help='Optional comma-separated tags')
@async_command
async def ingest_pdf(pdf_path, title=None, type='contract', document_type=None, tags=None):
    """Ingest a document from a PDF file."""
    try:
        if not pdf_path.lower().endswith('.pdf'):
            click.echo("File must be a PDF", err=True)
            return
            
        # Initialize PDF processor
        pdf_processor = PDFProcessor()
        
        # Process PDF
        click.echo(f"Processing PDF as {type}: {pdf_path}")
        document_data = await pdf_processor.process_pdf(pdf_path, doc_type=type)
        
        # Override with provided options if any
        if title:
            document_data["title"] = title
        if document_type:
            if type == 'contract':
                document_data["metadata"]["contract_type"] = document_type
            elif type == 'deal':
                document_data["metadata"]["deal_type"] = document_type
        if tags:
            document_data["metadata"]["tags"] = [tag.strip() for tag in tags.split(',')]
        
        # Add document_type to metadata if not already present
        if "metadata" in document_data and "document_type" not in document_data["metadata"]:
            document_data["metadata"]["document_type"] = type
        
        # Validate document structure based on type
        try:
            if type == 'contract':
                doc_obj = ContractCreate.parse_obj(document_data)
            elif type == 'email':
                doc_obj = EmailCreate.parse_obj(document_data)
            elif type == 'recap':
                doc_obj = RecapCreate.parse_obj(document_data)
            elif type == 'deal':
                doc_obj = DealCreate.parse_obj(document_data)
        except ValidationError as ve:
            click.echo(f"Invalid {type} structure:", err=True)
            click.echo(ve.json(), err=True)
            return
        
        # Process through pipeline
        result = await pipeline.process_document(doc_obj.dict())
        
        # Success output
        show_ingestion_result(type, result)
        
    except Exception as e:
        click.echo(f"Error processing PDF document: {e}", err=True)

if __name__ == "__main__":
    cli()