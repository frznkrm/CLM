# CLM Smart Search System

A Contract Lifecycle Management (CLM) system with advanced search capabilities, combining structured and semantic search to intelligently find relevant information across multiple document types: contracts, emails, deals, and meeting recaps.

## System Overview

The CLM Smart Search system is designed to ingest, process, and search various document types commonly found in legal and business contexts. It features:

- **Multi-Document Support**: Handles contracts, emails, oil industry deals, and meeting recaps
- **Dual-Mode Search**: Combines traditional structured search with semantic (meaning-based) vector search
- **Query Intelligence**: Automatically classifies queries and detects document types to route them to the appropriate search engine
- **Full Pipeline**: Manages document ingestion, normalization, cleaning, chunking, embedding, and indexing
- **Multiple Data Stores**: Uses MongoDB for document storage, Elasticsearch for structured search, and Qdrant for vector search
- **PDF Processing**: Converts PDF documents into structured JSON format for ingestion

## File and Folder Structure

```
clm_system/
├── api/
│   ├── __init__.py
│   └── routes.py           # FastAPI API endpoints
├── core/
│   ├── __init__.py
│   ├── database/           # Database clients
│   │   ├── elasticsearch_client.py
│   │   ├── mongodb_client.py
│   │   └── qdrant_client.py
│   ├── pipeline/           # Document processing pipeline
│   │   ├── base.py
│   │   ├── chunking/
│   │   │   ├── base.py
│   │   │   ├── contract.py
│   │   │   ├── deal.py
│   │   │   ├── email.py
│   │   │   └── recap.py
│   │   ├── cleaning/
│   │   │   ├── base.py
│   │   │   ├── contract.py
│   │   │   ├── deal.py
│   │   │   ├── email.py
│   │   │   └── recap.py
│   │   ├── ingestion/
│   │   │   ├── base.py
│   │   │   ├── contract.py
│   │   │   ├── deal.py
│   │   │   ├── email.py
│   │   │   └── recap.py
│   │   ├── orchestrator.py
│   │   └── preprocessing/
│   │       └── pdf_processor.py
│   ├── query_engine/       # Search functionality
│   │   ├── helpers.py
│   │   ├── query_classifier.py
│   │   └── search.py
│   └── utils/
│       ├── __init__.py
│       └── embeddings.py
├── schemas/
│   └── schemas.py         # Pydantic data models
├── cli.py                 # Command-line interface
├── config.py              # Application configuration
├── main.py                # FastAPI application entry point
├── test.py                # Unit tests
└── test_email_workflow.py # Email workflow tests
```

## Key Components

### 1. API Layer (`api/routes.py`)

REST endpoints for document management and search:
- **`POST /contracts/ingest`**: Ingests JSON-formatted documents
- **`POST /contracts/ingest-pdf`**: Ingests PDF documents
- **`GET /contracts/{contract_id}`**: Retrieves a document by ID
- **`GET /contracts`**: Lists documents with filtering
- **`POST /search`**: Searches across all document types

### 2. Data Models (`schemas/schemas.py`)

Pydantic models for validation:
- **Document Types**: `ContractCreate`, `EmailCreate`, `DealCreate`, `RecapCreate`
- **Responses**: `ContractResponse`, `EmailResponse`, `DealResponse`, `RecapResponse`
- **Search**: `QueryRequest`, `QueryResponse`, type-specific result schemas

### 3. Database Clients (`core/database/`)

#### MongoDB Client (`mongodb_client.py`)
- Stores complete documents of all types
- Handles CRUD operations with type-specific metadata

#### Elasticsearch Client (`elasticsearch_client.py`)
- Structured search across all document types
- Indexes with mappings for contracts, emails, deals, and recaps

#### Qdrant Client (`qdrant_client.py`)
- Vector database for semantic search
- Stores embedded chunks with type-specific metadata

### 4. Document Processing Pipeline (`core/pipeline/`)

#### Ingestion (`ingestion/[type].py`)
- Normalizes raw data for each document type
- Assigns IDs and timestamps

#### Cleaning (`cleaning/[type].py`)
- Type-specific cleaning (e.g., email subjects, deal volumes)
- Handles standardization and minimal PII scrubbing

#### Chunking (`chunking/[type].py`)
- Splits content into embeddable chunks
- Uses type-specific strategies (e.g., email parts, deal sections)

#### Orchestrator (`orchestrator.py`)
- Coordinates pipeline stages
- Routes documents to type-specific processors
- Persists to all databases

#### PDF Processor (`preprocessing/pdf_processor.py`)
- Extracts text from PDFs
- Structures content into JSON format

### 5. Query Engine (`core/query_engine/`)

#### Query Classifier (`query_classifier.py`)
- Uses OpenAI API (or local model) to classify queries
- Detects mentioned document types
- Types: Structured, Semantic, Hybrid

#### Query Router (`search.py`)
- Routes queries based on classification
- Executes searches per document type in parallel
- Merges results with Reciprocal Rank Fusion (RRF)

#### Search Helpers (`helpers.py`)
- Implements RRF algorithm for hybrid search result combination

### 6. Utils (`core/utils/`)

#### Embeddings (`embeddings.py`)
- Manages SentenceTransformer model
- Computes embeddings for semantic search

### 7. Configuration (`config.py`)
- Manages settings via Pydantic
- Loads from environment variables

### 8. CLI (`cli.py`)
- Commands: `ingest`, `search`, `test_embedding`, `ingest_pdf`
- Supports all document types

### 9. Application Entry Point (`main.py`)
- Configures FastAPI app with CORS
- Includes health check endpoint

## How It Works

### Document Ingestion Flow
1. Submit via API or CLI (JSON or PDF)
2. Pipeline processes through type-specific ingestion, cleaning, and chunking
3. Stores in MongoDB, indexes in Elasticsearch, embeds in Qdrant

### Search Flow
1. User submits query
2. Classifier determines query type and document types
3. Routes to Elasticsearch (structured), Qdrant (semantic), or both (hybrid)
4. Merges results across document types
5. Returns ranked results with type-specific metadata

## Implementation Highlights
- **Multi-Type Support**: Extensible framework for contracts, emails, deals, recaps
- **Async Operations**: Uses asyncio for concurrent processing
- **Smart Classification**: Leverages AI for query intent and type detection
- **Flexible Search**: Combines structured and semantic approaches
- **PDF Integration**: Converts PDFs to structured data
- **Robust Testing**: Includes unit and workflow tests

This system provides a powerful, type-aware search engine for business documents, delivering relevant results based on query nature and document context.