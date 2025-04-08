# File: clm_system/config.py
import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    # MongoDB settings
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    mongodb_database: str = Field("clm_db", env="MONGODB_DATABASE")
    
    # Elasticsearch settings
    elasticsearch_uri: str = Field(..., env="ELASTICSEARCH_URI")
    
    # Qdrant settings
    qdrant_uri: str = Field(..., env="QDRANT_URI")
    
    # API settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # Embedding model
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )
    
    # Default chunk size for text splitting
    chunk_size: int = Field(500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    
    # Vector settings
    vector_dimension: int = Field(384, env="VECTOR_DIMENSION")  # Default for MiniLM-L6
    
    # Search settings
    default_top_k: int = Field(5, env="DEFAULT_TOP_K")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

