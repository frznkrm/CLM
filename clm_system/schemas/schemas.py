# File: clm_system/api/schemas.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import validator

class ClauseBase(BaseModel):
    """Base schema for contract clauses."""
    id: Optional[str] = None
    title: Optional[str] = None
    type: str
    text: str
    position: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ContractMetadata(BaseModel):
    contract_type: str
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    parties: List[Dict[str, str]] = Field(default_factory=list)
    status: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    @validator('effective_date', 'expiration_date', pre=True)
    def parse_dates(cls, value):
        if isinstance(value, str):
            try:
                return parser.parse(value)
            except:
                return None
        return value

class ContractCreate(BaseModel):
    """Schema for creating a new contract."""
    title: str
    metadata: ContractMetadata
    clauses: List[ClauseBase]


class ContractResponse(BaseModel):
    """Response schema for contract operations."""
    id: str
    title: str
    metadata: ContractMetadata
    created_at: datetime
    updated_at: datetime
    clauses_count: int
    status: str = "indexed"


class QueryRequest(BaseModel):
    """Schema for search queries."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = 5


class ClauseSearchResult(BaseModel):
    """Result schema for clause search."""
    clause_id: str
    contract_id: str
    contract_title: str
    clause_type: str
    clause_title: Optional[str] = None
    content: str
    relevance_score: float
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response schema for search queries."""
    query: str
    total_results: int
    results: List[ClauseSearchResult]
    metadata: Optional[Dict[str, Any]] = None
    execution_time_ms: float
