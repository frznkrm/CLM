# clm_system/core/pipeline/orchestrator.py
import asyncio
import logging
from typing import Dict, Any
from clm_system.core.pipeline.ingestion.contract import ContractIngestor
from clm_system.core.pipeline.cleaning.contract import ContractCleaner
from clm_system.core.pipeline.chunking.contract import ContractChunker
from clm_system.core.database.mongodb_client import MongoDBClient
from clm_system.core.database.elasticsearch_client import ElasticsearchClient
from clm_system.core.database.qdrant_client import QdrantClient
from clm_system.core.utils.embeddings import get_embedding_model, compute_embedding

logger = logging.getLogger(__name__)

class PipelineService:
    def __init__(self):
        self.ingestor = ContractIngestor()
        self.cleaner  = ContractCleaner()
        self.chunker  = ContractChunker()
        self.mongo    = MongoDBClient()
        self.es       = ElasticsearchClient()
        self.qdrant   = QdrantClient()
        self.model    = get_embedding_model()

    async def process_contract(self, raw: Dict[str,Any]) -> Dict[str,Any]:
        # 1) Ingest
        normalized = self.ingestor.ingest(raw)
        # 2) Clean
        cleaned    = self.cleaner.clean(normalized)
        # 3) Chunk & embed
        chunks = []
        for clause in cleaned.get("clauses", []):
            for text in self.chunker.chunk(clause["text"]):
                emb = compute_embedding(text, self.model)
                chunks.append((clause, text, emb))
        # 4) Persist
        await self.mongo.insert_contract(cleaned)
        # prepare ES doc (convert dates → ISO, strip _id)
        es_doc = cleaned.copy()
        es_doc.pop("_id", None)
        await self.es.index_contract(es_doc)
        # 5) Store embeddings
        for clause, text, emb in chunks:
            await self.qdrant.store_embedding(
                contract_id   = cleaned["id"],
                contract_title= cleaned["title"],
                clause_id     = clause["id"],
                clause_type   = clause["type"],
                content       = text,
                metadata      = {**cleaned["metadata"], **clause.get("metadata",{})},
                embedding     = emb
            )
        logger.info(f"Pipeline complete for contract {cleaned['id']}")
        return {
            "id": cleaned["id"],
            "title": cleaned["title"],
            "clauses_count": len(cleaned.get("clauses",[])),
            "status": "indexed"
        }
