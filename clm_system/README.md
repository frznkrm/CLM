# CLM Smart Search System

A Contract Lifecycle Management (CLM) system with advanced search capabilities, combining structured and semantic search to intelligently find relevant contract clauses and information.

## System Overview

The CLM Smart Search system is designed to ingest, process, and search contract documents. It features:

- **Dual-Mode Search**: Combines traditional structured search with semantic (meaning-based) vector search
- **Query Intelligence**: Automatically classifies queries to route them to the appropriate search engine
- **Full Pipeline**: Handles contract ingestion, normalization, cleaning, chunking, embedding, and indexing
- **Multiple Data Stores**: Uses MongoDB for document storage, Elasticsearch for structured search, and Qdrant for vector search

## File and Folder Structure

```
clm_system/
├── api/
│   ├── __init__.py
│   └── routes.py           # FastAPI API endpoints
├── core/
│   ├── __init__.py
│   ├── database/           # Database clients
│   │   ├── __init__.py
│   │   ├── elasticsearch_client.py
│   │   ├── mongodb_client.py
│   │   └── qdrant_client.py
│   ├── pipeline/           # Contract processing pipeline
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── chunking/
│   │   │   ├── base.py
│   │   │   └── contract.py
│   │   ├── cleaning/
│   │   │   ├── base.py
│   │   │   └── contract.py
│   │   ├── ingestion/
│   │   │   ├── base.py
│   │   │   └── contract.py
│   │   ├── orchestrator.py
│   │   └── pipeline.py    # Missing in the files but referenced
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
└── main.py                # FastAPI application entry point
```

## Key Components

### 1. API Layer (`api/routes.py`)

The API layer provides REST endpoints for contract ingestion and search:

- **`POST /contracts`**: Ingests a new contract into the system
- **`POST /search`**: Searches for contracts using structured and/or semantic search

### 2. Data Models (`schemas/schemas.py`)

Defines Pydantic models to validate and structure data:

- **`ContractCreate`**: Schema for creating contracts
- **`ContractResponse`**: Response model for contract operations
- **`QueryRequest`**: Schema for search queries
- **`QueryResponse`**: Response model for search results

### 3. Database Clients (`core/database/`)

Three database clients handle different aspects of data storage:

#### MongoDB Client (`mongodb_client.py`)
- Stores complete contract documents
- Handles CRUD operations for contracts

#### Elasticsearch Client (`elasticsearch_client.py`)
- Provides structured search capabilities 
- Maintains an index with contract metadata and clauses
- Creates custom mappings for efficient filterable fields

#### Qdrant Client (`qdrant_client.py`)
- Vector database for semantic search
- Stores embedded clause chunks for similarity search
- Handles vector operations and collection management

### 4. Contract Processing Pipeline (`core/pipeline/`)

The pipeline processes incoming contracts through several stages:

#### Ingestion (`ingestion/contract.py`)
- Normalizes raw contract data
- Assigns IDs and timestamps

#### Cleaning (`cleaning/contract.py`)
- Applies standardization and cleaning operations
- Could handle PII scrubbing (minimal implementation in current code)

#### Chunking (`chunking/contract.py`)
- Splits long text into smaller chunks for embedding
- Uses NLP to maintain sentence boundaries

#### Orchestrator (`orchestrator.py`)
- Coordinates the entire pipeline process
- Manages the flow of data through each stage
- Persists data to all databases

### 5. Query Engine (`core/query_engine/`)

The search system intelligently handles different types of queries:

#### Query Classifier (`query_classifier.py`)
- Uses OpenAI APIs to classify queries as:
  - **Structured**: Filter-based queries
  - **Semantic**: Meaning-based queries
  - **Hybrid**: Combined approach
- Includes fallback heuristic classification 

#### Query Router (`search.py`)
- Routes queries to appropriate search engine
- Executes searches against Elasticsearch or Qdrant
- For hybrid searches, combines results using fusion algorithm

#### Search Helpers (`helpers.py`)
- Implements Reciprocal Rank Fusion (RRF) algorithm
- Combines and ranks results from multiple search engines

### 6. Utils (`core/utils/`)

#### Embeddings (`embeddings.py`)
- Manages embedding models
- Computes text embeddings for semantic search
- Caches model for reuse

### 7. Configuration (`config.py`)

- Uses Pydrant to manage application settings
- Loads configuration from environment variables
- Provides global settings for all components

### 8. CLI (`cli.py`)

Command-line interface for system operations:
- **`ingest`**: Ingests contracts from file
- **`search`**: Performs searches from the command line
- **`test_embedding`**: Tests the embedding functionality

### 9. Application Entry Point (`main.py`)

- Creates and configures the FastAPI application
- Sets up middleware, routes, and logging
- Provides a health check endpoint

## How It Works

### Contract Ingestion Flow

1. A contract is submitted via the API or CLI
2. The pipeline processes it through ingestion, cleaning, and chunking
3. The contract is stored in MongoDB for persistence
4. The contract is indexed in Elasticsearch for structured search
5. Clauses are embedded and stored in Qdrant for semantic search

### Search Flow

1. A user submits a search query
2. The query classifier determines if it's structured, semantic, or hybrid
3. For structured queries, Elasticsearch is used
4. For semantic queries, the query is embedded and searched in Qdrant
5. For hybrid queries, both engines are used and results are combined
6. Results are ranked and returned to the user

## Implementation Highlights

- **Async Programming**: Uses asyncio throughout for concurrent operations
- **Classification Intelligence**: Uses OpenAI to understand query intent
- **Result Fusion**: Combines results from different search systems
- **Extensible Design**: Uses abstract base classes and dependency injection
- **Error Handling**: Comprehensive error handling and logging

This system provides a powerful search engine for contracts, intelligently leveraging both traditional structured search and modern semantic search to deliver the most relevant results based on the nature of the user's query.