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

    async def classify(self, query: str) -> str:
        # Check cache and TTL
        if query in self.cache:
            timestamp = self.cache_timestamps.get(query, 0)
            if time.time() - timestamp < self.cache_ttl:
                return self.cache[query]
        
        # Fallback classification in case API fails
        fallback = self._heuristic_classify(query)
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    #model="gpt-3.5-turbo",
                    model="local-model",
                    messages=[{
                        "role": "system",
                        "content": """Classify legal contract search queries. Respond with ONE word:
                        - 'structured' for exact filters/terms (e.g., "contracts with effective date after 2023")
                        - 'semantic' for conceptual/meaning-based (e.g., "liability clauses protecting against data breaches")
                        - 'hybrid' for mixed queries (e.g., "confidentiality provisions in NDAs signed after January")"""
                    }, {
                        "role": "user",
                        "content": query
                    }],
                    temperature=0.1,
                    max_tokens=10
                )

                classification = response.choices[0].message.content.lower().strip()
                valid = {"structured", "semantic", "hybrid"}
                result = classification if classification in valid else fallback
                
                # Update cache
                self.cache[query] = result
                self.cache_timestamps[query] = time.time()
                return result
                
            except RateLimitError:
                logger.warning(f"Rate limit exceeded on attempt {attempt+1}, retrying in {retry_delay}s")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                
            except APIError as e:
                logger.error(f"OpenAI API error: {str(e)}")
                return fallback
                
            except Exception as e:
                logger.error(f"Classification failed: {str(e)}")
                return fallback
        
        # If all retries failed
        return fallback
    
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