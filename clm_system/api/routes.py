# File: clm_system/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from clm_system.api.schemas import (
    ContractCreate,
    ContractResponse,
    QueryRequest,
    QueryResponse
)
from clm_system.core.pipeline.pipeline import ContractPipeline  # ✅ New pipeline
from clm_system.core.search import QueryRouter

router = APIRouter()

# Instantiate pipeline once
pipeline = ContractPipeline()

@router.post("/contracts", response_model=ContractResponse, status_code=201)
async def create_contract(
    contract: ContractCreate,
    pipeline: ContractPipeline = Depends(lambda: pipeline)
):
    """
    Ingest a new contract into the system.
    """
    try:
        result = await pipeline.process_contract(contract.dict())  # ✅ use new pipeline method
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
