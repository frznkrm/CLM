# clm_system/api/routes.py
import logging
import os
import tempfile
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from clm_system.core.pipeline.orchestrator import PipelineService
from clm_system.core.queryEngine.search import QueryRouter
from clm_system.core.preprocessing.pdf_processor import PDFProcessor
from clm_system.schemas.schemas import ContractCreate

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
pipeline = PipelineService()
query_router = QueryRouter()
pdf_processor = PDFProcessor()

@router.post("/contracts/ingest", response_model=Dict[str, Any])
async def ingest_contract(contract: ContractCreate):
    """
    Ingest a contract in JSON format.
    
    Args:
        contract: Contract data in JSON format
    
    Returns:
        Processed contract information
    """
    try:
        result = await pipeline.process_document(contract.dict())
        return result
    except Exception as e:
        logger.error(f"Error ingesting contract: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting contract: {str(e)}")

@router.post("/contracts/ingest-pdf", response_model=Dict[str, Any])
async def ingest_pdf_contract(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    contract_type: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """
    Ingest a contract from a PDF file.
    
    Args:
        file: PDF file
        title: Optional title override
        contract_type: Optional contract type
        tags: Optional comma-separated tags
    
    Returns:
        Processed contract information
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        # Write file content
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Process PDF
        contract_data = await pdf_processor.process_pdf(temp_file.name)
        
        # Override with provided form data if any
        if title:
            contract_data["title"] = title
        if contract_type:
            contract_data["metadata"]["contract_type"] = contract_type
        if tags:
            contract_data["metadata"]["tags"] = [tag.strip() for tag in tags.split(',')]
        
        # Validate contract structure
        try:
            contract_obj = ContractCreate.parse_obj(contract_data)
        except ValidationError as ve:
            logger.error(f"Invalid contract structure: {ve}")
            raise HTTPException(status_code=400, detail=f"Invalid contract structure: {ve}")
        
        # Process through pipeline
        result = await pipeline.process_document(contract.dict())
        return result
        
    except Exception as e:
        logger.error(f"Error processing PDF contract: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@router.get("/contracts/{contract_id}", response_model=Dict[str, Any])
async def get_contract(contract_id: str):
    """
    Get contract by ID.
    
    Args:
        contract_id: Contract ID
    
    Returns:
        Contract data
    """
    try:
        # Assuming pipeline or a repository has this method
        contract = await pipeline.get_contract(contract_id)
        if not contract:
            raise HTTPException(status_code=404, detail="Contract not found")
        return contract
    except Exception as e:
        logger.error(f"Error getting contract: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving contract: {str(e)}")

@router.get("/contracts", response_model=List[Dict[str, Any]])
async def list_contracts(
    limit: int = Query(100, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    contract_type: Optional[str] = None,
    status: Optional[str] = None
):
    """
    List contracts with optional filtering.
    
    Args:
        limit: Maximum number of contracts to return
        skip: Number of contracts to skip
        contract_type: Filter by contract type
        status: Filter by status
    
    Returns:
        List of contracts
    """
    try:
        # Build filters
        filters = {}
        if contract_type:
            filters["metadata.contract_type"] = contract_type
        if status:
            filters["metadata.status"] = status
            
        # Assuming pipeline or a repository has this method
        contracts = await pipeline.get_contracts(filters, limit, skip)
        return contracts
    except Exception as e:
        logger.error(f"Error listing contracts: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing contracts: {str(e)}")

@router.post("/search", response_model=Dict[str, Any])
async def search_contracts(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = Query(5, ge=1, le=100)
):
    """
    Search contracts using the query router.
    
    Args:
        query: Search query
        filters: Optional filters
        top_k: Number of results to return
    
    Returns:
        Search results
    """
    try:
        results = await query_router.route_query(query, filters, top_k)
        return results
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")