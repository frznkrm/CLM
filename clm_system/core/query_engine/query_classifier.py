import logging
import time
from typing import Optional
import asyncio
from openai import AsyncOpenAI, APIError, RateLimitError
from clm_system.config import settings

logger = logging.getLogger(__name__)

class QueryClassifier:
    def __init__(self):
        #self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.client = AsyncOpenAI(
            base_url="http://localhost:1234/v1",  # Default LM Studio port
            api_key="deepseek-r1-distill-qwen-7b"  # Dummy key required by client
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
                    "content": """Classify legal document search queries. Respond with JSON:
                    {
                        "type": "structured|semantic|hybrid",
                        "doc_types": ["contract", "email", "deal", "recap"]  # Include only relevant types
                    }
                    
                    Detect document types from these patterns:
                    - Email: 'email', 'inbox', 'attachment', 'sent', 'received', 'message'
                    - Deal: 'deal', 'volume', 'price', 'barrel', 'lease', 'agreement'
                    - Recap: 'meeting', 'minutes', 'action items', 'decisions', 'recap'  
                    - Contract: 'contract', 'clause', 'nda', 'terms', 'provision'
                    """
                }, {
                    "role": "user",
                    "content": query
                }],
                temperature=0.1,
                max_tokens=100
            )

            try:
                # Try to parse as JSON
                import json
                result = json.loads(response.choices[0].message.content)
                
                if isinstance(result, dict):
                    if "type" in result:
                        query_type = result["type"] 
                    if "doc_types" in result and isinstance(result["doc_types"], list):
                        doc_types = result["doc_types"]
            except:
                # If JSON parsing fails, use fallback
                pass
            
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