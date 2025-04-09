import pytest
import asyncio
from unittest.mock import MagicMock, patch

from clm_system.core.queryEngine.search import QueryRouter

@pytest.fixture
def sample_contract():
    """Sample contract data for testing."""
    return {
        "id": "test-contract-123",
        "title": "Software License Agreement",
        "metadata": {
            "contract_type": "license",
            "effective_date": "2023-01-01T00:00:00Z",
            "expiration_date": "2024-01-01T00:00:00Z",
            "parties": [
                {"name": "ACME Corp", "id": "party-001"},
                {"name": "Supplier Inc", "id": "party-002"}
            ],
            "status": "active",
            "tags": ["software", "license", "annual"]
        },
        "clauses": [
            {
                "id": "clause-001",
                "title": "License Grant",
                "type": "grant",
                "text": "Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the Software.",
                "position": 1
            },
            {
                "id": "clause-002",
                "title": "Term and Termination",
                "type": "term",
                "text": "This Agreement shall commence on the Effective Date and continue for a period of one (1) year.",
                "position": 2
            }
        ]
    }

@pytest.mark.asyncio
async def test_query_router_classification():
    """Test query classification logic."""
    router = QueryRouter()
    
    # Test structured query classification
    structured_queries = [
        "contracts with effective date after 2023-01-01",
        "status: active type: license",
        "find contracts with party: ACME Corp",
        "contract type: license"
    ]
    
    for query in structured_queries:
        assert router._classify_query(query) in ["structured", "hybrid"], f"Failed for: {query}"
    
    # Test semantic query classification
    semantic_queries = [
        "what are the license terms",
        "termination conditions",
        "find software agreements"
    ]
    
    for query in semantic_queries:
        assert router._classify_query(query) == "semantic", f"Failed for: {query}"

@pytest.mark.asyncio
async def test_route_query():
    """Test query routing to appropriate search engines."""
    # Mock the Elasticsearch and Qdrant clients
    with patch('clm_system.core.search.ElasticsearchClient') as mock_es, \
         patch('clm_system.core.search.QdrantClient') as mock_qdrant, \
         patch('clm_system.core.search.compute_embedding') as mock_compute_embedding:
        
        # Setup mocks
        mock_es_instance = mock_es.return_value
        mock_es_instance.search.return_value = [{"clause_id": "test1"}]
        
        mock_qdrant_instance = mock_qdrant.return_value
        mock_qdrant_instance.search.return_value = [{"clause_id": "test2"}]
        
        mock_compute_embedding.return_value = [0.1] * 384  # Mock embedding
        
        router = QueryRouter()
        router.es_client = mock_es_instance
        router.qdrant_client = mock_qdrant_instance
        
        # Test structured query routing
        structured_result = await router.route_query("contract type: license", {"status": "active"})
        assert structured_result["query_type"] == "structured"
        assert mock_es_instance.search.called
        
        # Test semantic query routing
        semantic_result = await router.route_query("what are termination conditions")
        assert semantic_result["query_type"] == "semantic"
        assert mock_qdrant_instance.search.called
        
        # Test hybrid query routing
        mock_es_instance.search.reset_mock()
        mock_qdrant_instance.search.reset_mock()
        
        hybrid_result = await router.route_query("find active license contracts with termination clause")
        assert hybrid_result["query_type"] == "hybrid"
        assert mock_es_instance.search.called
        assert mock_qdrant_instance.search.called