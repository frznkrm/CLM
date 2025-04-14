# File: clm_system/api/schemas.py
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
from dateutil import parser

class BaseMetadata(BaseModel):
    document_type: str
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ClauseBase(BaseModel):
    """Base schema for contract clauses."""
    id: Optional[str] = None
    title: Optional[str] = None
    type: str
    text: str
    position: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ContractMetadata(BaseMetadata):
    document_type: str = "contract"
    contract_type: str
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    parties: List[Dict[str, str]] = Field(default_factory=list)
    status: Optional[str] = None

    @validator('effective_date', 'expiration_date', pre=True)
    def parse_dates(cls, value):
        if isinstance(value, str):
            try:
                return parser.parse(value)
            except:
                return None
        return value

class EmailMetadata(BaseMetadata):
    document_type: str = "email"
    from_address: EmailStr
    to: List[EmailStr]
    cc: List[EmailStr] = []
    bcc: List[EmailStr] = []
    subject: str
    has_attachments: bool = False
    sent_date: datetime

class DealMetadata(BaseMetadata):
    document_type: str = "deal"
    deal_type: str  # (e.g., "oil_lease", "supply_contract")
    effective_date: datetime
    expiration_date: datetime
    parties: List[Dict[str, str]]
    volume: str
    price_per_unit: float

class RecapMetadata(BaseMetadata):
    document_type: str = "recap"
    meeting_date: datetime
    participants: List[str]
    decisions: List[str]
    action_items: List[str]

class ContractCreate(BaseModel):
    """Schema for creating a new contract."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    metadata: ContractMetadata
    clauses: List[ClauseBase]

class EmailCreate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: EmailMetadata
    content: str
    attachments: List[Dict] = []

class DealCreate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: DealMetadata
    clauses: List[ClauseBase]
    financial_terms: Dict[str, Any]

class RecapCreate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: RecapMetadata
    summary: str
    key_points: List[str]

class ContractResponse(BaseModel):
    """Response schema for contract operations."""
    id: str
    title: str
    metadata: ContractMetadata
    created_at: datetime
    updated_at: datetime
    clauses_count: int
    status: str = "indexed"

class EmailResponse(BaseModel):
    """Response schema for email operations."""
    id: str
    title: str  # Using subject as title
    metadata: EmailMetadata
    created_at: datetime
    updated_at: datetime
    has_attachments: bool
    status: str = "indexed"

class DealResponse(BaseModel):
    """Response schema for deal operations."""
    id: str
    title: str
    metadata: DealMetadata
    created_at: datetime
    updated_at: datetime
    clauses_count: int
    deal_type: str
    status: str = "indexed"

class RecapResponse(BaseModel):
    """Response schema for recap operations."""
    id: str
    title: str
    metadata: RecapMetadata
    created_at: datetime
    updated_at: datetime
    participants: List[str]
    status: str = "indexed"

class QueryRequest(BaseModel):
    """Schema for search queries."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = 5

class SearchResultBase(BaseModel):
    """Base result schema for search."""
    document_id: str
    document_title: str
    content: str
    relevance_score: float
    metadata: Optional[Dict[str, Any]] = None

class ClauseSearchResult(SearchResultBase):
    """Result schema for clause search."""
    clause_id: str
    contract_id: str = Field(alias="document_id")
    contract_title: str = Field(alias="document_title")
    clause_type: str
    clause_title: Optional[str] = None

class EmailSearchResult(SearchResultBase):
    """Result schema for email search."""
    email_id: str = Field(alias="document_id")
    email_subject: str = Field(alias="document_title")
    from_address: Optional[str] = None
    to_addresses: Optional[List[str]] = None

class DealSearchResult(SearchResultBase):
    """Result schema for deal search."""
    deal_id: str = Field(alias="document_id")
    deal_title: str = Field(alias="document_title")
    deal_type: Optional[str] = None
    price_per_unit: Optional[float] = None

class RecapSearchResult(SearchResultBase):
    """Result schema for recap search."""
    recap_id: str = Field(alias="document_id") 
    recap_title: str = Field(alias="document_title")
    meeting_date: Optional[datetime] = None
    participants: Optional[List[str]] = None

class QueryResponse(BaseModel):
    """Response schema for search queries."""
    query: str
    total_results: int
    results: List[Union[ClauseSearchResult, EmailSearchResult, DealSearchResult, RecapSearchResult]]
    metadata: Optional[Dict[str, Any]] = None
    execution_time_ms: float