import ipdb
import pytest
import asyncio
import logging
from datetime import datetime
from clm_system.core.pipeline.orchestrator import PipelineService
from clm_system.core.database.mongodb_client import MongoDBClient
from clm_system.core.database.elasticsearch_client import ElasticsearchClient
from clm_system.core.database.qdrant_client import QdrantClient
from clm_system.core.query_engine.search import QueryRouter
from qdrant_client import models
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Define your custom module filter
class ModuleFilter(logging.Filter):
    def __init__(self, module_name_prefix):
        super().__init__()
        self.module_name_prefix = module_name_prefix

    def filter(self, record):
        # Only allow log records that start with the specified module prefix.
        return record.name.startswith(self.module_name_prefix)

# Clear default handlers and set up a new stream handler
logging.getLogger().handlers.clear()
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Attach the filter so only logs from 'clm_system' are shown
handler.addFilter(ModuleFilter("clm_system"))
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)

# Increase log level for external libraries to reduce noise
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Create your own logger with the appropriate name
logger = logging.getLogger("clm_system")
logger.debug("This is a custom log message from clm_system")

@pytest.fixture
def sample_contract():
    return {
        "id": "contract_test_002",
        "title": "Master Service and Licensing Agreement",
        "metadata": {
            "document_type": "contract",
            "contract_type": "service_and_licensing",
            "effective_date": "2023-01-15T00:00:00Z",
            "expiration_date": "2025-01-15T00:00:00Z",
            "renewal_date": "2024-12-15T00:00:00Z",
            "parties": [
                {"name": "TechCorp Solutions", "id": "party_003", "role": "provider"},
                {"name": "Global Enterprises", "id": "party_004", "role": "client"}
            ],
            "status": "active",
            "jurisdiction": "State of New York",
            "tags": ["service", "licensing", "technology", "renewable"],
            "budget": 150000,
            "currency": "USD"
        },
        "clauses": [
            {
                "id": "clause_001",
                "title": "Payment Schedule",
                "type": "payment",
                "text": "Client agrees to pay TechCorp Solutions $50,000 upon signing, followed by $10,000 monthly for 10 months. Payments are due by the 5th of each month.",
                "position": 1
            },
            {
                "id": "clause_002",
                "title": "Software License Grant",
                "type": "licensing",
                "text": "TechCorp grants Global Enterprises a non-exclusive, non-transferable license to use the software for 2 years, renewable annually.",
                "position": 2
            },
            {
                "id": "clause_003",
                "title": "Confidentiality Agreement",
                "type": "confidentiality",
                "text": "Both parties shall maintain confidentiality of proprietary information for 7 years post-termination.",
                "position": 3
            },
            {
                "id": "clause_004",
                "title": "Termination Conditions",
                "type": "termination",
                "text": "This contract may be terminated by either party with 90 days written notice, or immediately upon material breach.",
                "position": 4
            },
            {
                "id": "clause_005",
                "title": "Service Level Agreement",
                "type": "sla",
                "text": "TechCorp guarantees 99.9% uptime for services, with penalties of $1,000 per 0.1% below this threshold.",
                "position": 5
            },
            {
                "id": "clause_006",
                "title": "Dispute Resolution",
                "type": "dispute",
                "text": "Disputes will be resolved through arbitration in New York under the American Arbitration Association rules.",
                "position": 6
            },
            {
                "id": "clause_007",
                "title": "Force Majeure",
                "type": "force_majeure",
                "text": "Neither party shall be liable for delays due to acts of God, war, or natural disasters.",
                "position": 7
            }
        ]
    }

@pytest.mark.asyncio
async def test_full_contract_workflow(sample_contract):
    mongo = MongoDBClient()
    es = ElasticsearchClient()
    qdrant = QdrantClient()
    pipeline = PipelineService()
    query_router = QueryRouter()

    try:
        # --- Phase 1: Ingestion ---
        logger.info("Starting contract ingestion process")
        processed_contract = await pipeline.process_document(sample_contract)
        logger.info(f"Contract processed: {processed_contract}")

        logger.info("Verifying MongoDB storage")
        db_contract = await mongo.get_document("contract_test_002")
        assert db_contract is not None, "Contract not found in MongoDB"
        assert db_contract["title"] == "Master Service and Licensing Agreement"
        assert len(db_contract["clauses"]) == 7
        logger.info("MongoDB storage verified")

        logger.info("Verifying Elasticsearch indexing")
        es_contract = await es.client.get(index="documents", id="contract_test_002")
        assert es_contract["found"], "Contract not found in Elasticsearch"
        assert es_contract["_source"]["title"] == "Master Service and Licensing Agreement"
        logger.debug(f"Elasticsearch document clauses: {[c['title'] for c in es_contract['_source'].get('clauses', [])]}")
        logger.info("Elasticsearch indexing verified")

        logger.info("Verifying Qdrant storage")
        qdrant_points = await qdrant.scroll("contract_test_002")
        assert len(qdrant_points) >= 7
        logger.debug(f"Qdrant points: {[p.get('clause_title') for p in qdrant_points]}")
        logger.info(f"Qdrant storage verified with {len(qdrant_points)} points")
        
        # --- Phase 2: Structured Queries ---
        structured_queries = [
            {
                "query": "clauses title of Payment Schedule",
                "description": "Search for the payment schedule clause title",
                "expected": "Payment Schedule",
                "field": "clause_title"
            },
            {
                "query": "TechCorp Solutions party",
                "description": "Search by provider party name",
                "expected": "TechCorp Solutions",
                "field": "metadata.parties.name"
            },
            {
                "query": "sla type",
                "description": "Search for clauses of type SLA",
                "expected": "Service Level Agreement",
                "field": "clause_title"
            },
            {
                "query": "clauses title of Confidentiality Agreement",
                "description": "Search for the confidentiality clause title",
                "expected": "Confidentiality Agreement",
                "field": "clause_title"
            }
        ]

        for test in structured_queries:
            logger.info(f"Testing structured query: '{test['query']}' - {test['description']}")
            result = await query_router.route_query(
                test["query"],
                filters={"metadata.document_type": "contract"},
                top_k=1
            )
            logger.info(f"Structured query result: {result}")
            if result["total_results"] == 0:
                logger.warning(f"No results for query: {test['query']}")
                assert False, f"No results for query: {test['query']}"
            if test["field"].startswith("metadata.parties"):
                party_names = [party["name"] for party in result["results"][0]["metadata"].get("parties", [])]
                assert test["expected"] in party_names, f"Incorrect party for query: {test['query']}"
            else:
                field_value = result["results"][0].get(test["field"])
                if field_value is None:
                    assert False, f"No matching clause found for query: {test['query']}"
                assert test["expected"] in field_value, f"Incorrect clause for query: {test['query']}"
            logger.info(f"Structured query test passed for: {test['query']}")

        # --- Phase 3: Semantic Queries ---
        # (Unchanged, assuming no issues here)
        semantic_queries = [
            {
                "query": "When are payments due?",
                "description": "Question about payment due dates",
                "expected": "5th of each month",
                "field": "content"
            },
            
            {
                "query": "How long is the confidentiality period?",
                "description": "Question about confidentiality duration",
                "expected": "7 years",
                "field": "content"
            },
            {
                "query": "What are the termination conditions?",
                "description": "Question about termination terms",
                "expected": "90 days written notice",
                "field": "content"
            },
            {
                "query": "What is the uptime guarantee?",
                "description": "Question about service level agreement",
                "expected": "99.9%",
                "field": "content"
            },
            {
                "query": "How are disputes resolved?",
                "description": "Question about dispute resolution",
                "expected": "arbitration in New York",
                "field": "content"
            }
        ]
        for test in semantic_queries:
            logger.info(f"Testing semantic query: '{test['query']}' - {test['description']}")
            result = await query_router.route_query(
                test["query"],
                filters={"metadata.document_type": "contract"},
                top_k=1
            )
            logger.info(f"Semantic query result: {result}")
            if result["total_results"] == 0:
                logger.warning(f"No results for query: {test['query']}")
                assert False, f"No results for query: {test['query']}"
            assert test["expected"] in result["results"][0][test["field"]], f"Expected content not found for query: {test['query']}"
            logger.info(f"Semantic query test passed for: {test['query']}")

    finally:
        logger.info("Cleaning up test data")
        await mongo.documents_collection.delete_one({"id": "contract_test_002"})
        await es.client.options(ignore_status=[404]).delete(index="documents", id="contract_test_002")
        qdrant.client.delete(
            collection_name="document_chunks",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value="contract_test_002")
                    )]
                )
            )
        )
        await es.close()
        logger.info("Cleanup completed")