from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # API keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    comet_api_key: str = Field(..., alias="COMET_API_KEY")
    comet_workspace: str = Field(..., alias="COMET_WORKSPACE")
    comet_project_name: str = Field(..., alias="COMET_PROJECT_NAME")
    opik_api_key: str = Field(default="", alias="OPIK_API_KEY")
    opik_workspace: str = Field(default="", alias="OPIK_WORKSPACE")
    opik_project_name: str = Field(default="", alias="OPIK_PROJECT_NAME")

    # MongoDB settings
    mongodb_uri: str = Field(..., alias="MONGODB_URI")
    mongodb_database: str = Field(default="clm_db", alias="MONGODB_DATABASE")

    # Elasticsearch settings
    elasticsearch_uri: str = Field(..., alias="ELASTICSEARCH_URI")

    # Qdrant settings
    qdrant_uri: str = Field(..., alias="QDRANT_URI")

    # API settings
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Embedding model
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    vector_dimension: int = Field(default=384, alias="VECTOR_DIMENSION")  # Matches MiniLM-L6

    # Chunking settings
    chunk_size: int = Field(default=128, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=25, alias="CHUNK_OVERLAP")

    # Search settings
    default_top_k: int = Field(default=15, alias="DEFAULT_TOP_K")

    # Classifier cache
    classifier_cache_ttl: int = Field(default=3600, alias="CLASSIFIER_CACHE_TTL")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()