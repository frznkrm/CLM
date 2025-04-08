# File: clm_system/api/routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional

from clm_system.api.schemas import (
    ContractCreate,
    ContractResponse,
    QueryRequest,
    QueryResponse
)
from clm_system.core.ingestion import ContractIngestionService
from clm_system.core.search import QueryRouter

router = APIRouter()

@router.post("/contracts", response_model=ContractResponse, status_code=201)
async def create_contract(
    contract: ContractCreate,
    ingestion_service: ContractIngestionService = Depends(lambda: ContractIngestionService())
):
    """
    Ingest a new contract into the system.
    """
    try:
        result = await ingestion_service.ingest_contract(contract.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contract ingestion failed: {str(e)}")

@router.post("/search", response_model=QueryResponse)
async def search_contracts(
    request: QueryRequest,
    query_router: QueryRouter = Depends(lambda: QueryRouter())
):
    """
    Search for contracts using structured and/or semantic search.
    """
    try:
        result = await query_router.route_query(request.query, request.filters)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
