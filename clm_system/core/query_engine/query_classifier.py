import ipdb
import logging
import time
from typing import Optional
import asyncio
from openai import AsyncOpenAI, APIError, RateLimitError
from clm_system.config import settings
from typing import Dict, List, Any, Optional
logger = logging.getLogger(__name__)

class QueryClassifier:
    def __init__(self):
        #self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.client = AsyncOpenAI(
            base_url="http://192.168.10.1:1234/v1",
            api_key="qwen2.5-coder-14b-instruct",  # Required even if not used
            timeout=30.0  # Add timeout
        )
        self.cache = {}  # Simple cache for demo purposes
        self.cache_ttl = 3600  # 1 hour TTL
        self.cache_timestamps = {}

    # Update the query_classifier.py file

    async def classify(self, query: str) -> dict:
        """
        Classifies the query type and detects document types mentioned.
        
        Returns:
            Dict with query_type and detected document types
        """
        # Check cache and TTL
        if query in self.cache:
            timestamp = self.cache_timestamps.get(query, 0)
            if time.time() - timestamp < self.cache_ttl:
                return self.cache[query]
        
        # Basic query type classification
        query_type = self._heuristic_classify(query)
        
        # Default all document types
        doc_types = self._detect_document_types(query)
        
        try:
            response = await self.client.chat.completions.create(
            
                model="local-model",
                messages=[{
                    "role": "system",
                    "content": """Classify legal document search queries based on how their main elements align with the document mapping structure. Respond with JSON:
                    {
                        "type": "structured|semantic|hybrid",
                        "doc_types": ["contract", "email", "deal", "recap"]  # Include only relevant types
                    }

                    The document mapping includes:
                    - Document level: id, title, metadata (with subfields: document_type, status, jurisdiction, parties [nested: name, id, role], tags), clauses (nested: id, title, type, text, position)

                    Query types:
                    - Structured: All main elements explicitly reference fields using 'field:value' syntax and exist in the mapping, e.g., 'clauses.title:Payment Schedule', 'metadata.status:active'.
                    - Semantic: All main elements relate to content within the mapping’s fields, but the query uses natural language, e.g., 'When are payments due?' (relates to 'clauses.text'), 'What is the uptime guarantee?' (relates to 'clauses.text').
                    - Hybrid: Combines explicit field references with natural language, where all elements still align with the mapping, e.g., 'active contracts with arbitration clause' (ties to 'metadata.status:active' and 'clauses.text' or 'clauses.type').

                    Focus on whether the query’s main elements (e.g., topics, entities, or conditions) correspond to fields like 'metadata.status', 'clauses.text', or 'parties.name', even in natural language queries.

                    Detect document types based on keywords:
                    - Contract: 'contract', 'clause', 'nda', 'terms', 'provision', 'agreement', 'license', 'service level'
                    - Email: 'email', 'inbox', 'attachment', 'sent', 'received', 'message', 'correspondence', 'mail'
                    - Deal: 'deal', 'volume', 'price', 'barrel', 'lease', 'transaction', 'purchase', 'sale'
                    - Recap: 'meeting', 'minutes', 'action items', 'decisions', 'recap', 'summary', 'notes'

                    If any main element doesn’t align with the mapping, classify as 'semantic' and let the search system interpret it. Include all relevant doc_types based on keywords; if none are detected, include all types.

                    Respond only with the JSON object."""
                }, {
                    "role": "user",
                    "content": query
                }],
                temperature=0.1,
                max_tokens=100
            )
            #ipdb.set_trace()
            logging.info(f"LLM response: {response}")
            cleaned_content = response.choices[0].message.content.strip()
            if "<think>" in cleaned_content:
                cleaned_content = cleaned_content.split("</think>", 1)[-1].strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            
            try:
                import json
                llm_result = json.loads(cleaned_content)
                if isinstance(llm_result, dict):
                    query_type = llm_result.get("type", query_type)
                    doc_types = llm_result.get("doc_types", doc_types)
                    result = {"query_type": query_type, "doc_types": doc_types}
            # try:
            #     # Try to parse as JSON
            #     import json
            #     result = json.loads(response.choices[0].message.content)
                
            #     if isinstance(result, dict):
            #         if "type" in result:
            #             query_type = result["type"] 
            #         if "doc_types" in result and isinstance(result["doc_types"], list):
            #             doc_types = result["doc_types"]
            except:
                # If JSON parsing fails, use fallback
                pass
            #query_type = "structured"
            result = {
                "query_type": query_type,
                "doc_types": doc_types
            }
            
            # Update cache
            self.cache[query] = result
            self.cache_timestamps[query] = time.time()
            return result
                
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return {
                "query_type": query_type,
                "doc_types": doc_types
            }

    def _detect_document_types(self, query: str) -> List[str]:
        """Detect document types mentioned in the query"""
        query = query.lower()
        types = []
        
        # Email markers
        if any(term in query for term in ['email', 'inbox', 'message', 'sent', 'received']):
            types.append('email')
            
        # Deal markers
        if any(term in query for term in ['deal', 'volume', 'price', 'barrel', 'lease']):
            types.append('deal')
            
        # Recap markers
        if any(term in query for term in ['meeting', 'minutes', 'recap', 'action item']):
            types.append('recap')
            
        # Contract markers
        if any(term in query for term in ['contract', 'clause', 'nda', 'agreement', 'provision']):
            types.append('contract')
            
        # Return all types if none detected
        if not types:
            return ['contract', 'email', 'deal', 'recap']
            
        return types

    def _heuristic_classify(self, query: str) -> str:
        """Fallback classification using heuristics."""
        structured_keywords = [
            "clauses.title:", "clauses.type:", 
            "date:", "type:", "status:", "party:", "before:", "after:",
            "contract type", "effective date", "expiration date", "status is"
        ]
        
        has_structured = any(keyword in query.lower() for keyword in structured_keywords)
        
        if len(query.split()) <= 3 and not has_structured:
            return "semantic"
        
        if len(query.split()) > 3 and has_structured:
            return "hybrid"
        
        if has_structured:
            return "structured"
        
        return "semantic"