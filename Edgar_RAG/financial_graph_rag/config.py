"""
Configuration settings for Financial Graph RAG system
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings using Pydantic BaseSettings for validation
    """

    # LLM Configuration
    openai_api_key: Optional[str] = Field("ollama-local", env="OPENAI_API_KEY")
    openai_base_url: str = Field("http://localhost:11434/v1",
                                 env="OPENAI_BASE_URL")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    default_llm_provider: str = Field("openai", env="DEFAULT_LLM_PROVIDER")
    default_llm_model: str = Field("gemma3:27b", env="DEFAULT_LLM_MODEL")

    # Ollama Configuration
    ollama_base_url: str = Field("http://localhost:11434",
                                 env="OLLAMA_BASE_URL")
    ollama_model: str = Field("gemma3:27b", env="OLLAMA_MODEL")

    # Embedding Configuration
    embedding_model: str = Field("text-embedding-3-large",
                                 env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(3072, env="EMBEDDING_DIMENSION")

    # Knowledge Graph Database (Neo4j)
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field("password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", env="NEO4J_DATABASE")

    # Vector Database (ChromaDB)
    chroma_persist_directory: str = Field("./chroma_db",
                                          env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field("financial_documents",
                                        env="CHROMA_COLLECTION_NAME")

    # Data Storage
    data_directory: str = Field("./data", env="DATA_DIRECTORY")
    edgar_data_directory: str = Field("./data/edgar",
                                      env="EDGAR_DATA_DIRECTORY")
    cache_directory: str = Field("./cache", env="CACHE_DIRECTORY")

    # SEC EDGAR Settings
    edgar_user_agent: str = Field("Financial Graph RAG Research Tool",
                                  env="EDGAR_USER_AGENT")
    edgar_email: str = Field("research@example.com", env="EDGAR_EMAIL")

    # Processing Settings
    max_workers: int = Field(4, env="MAX_WORKERS")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")

    # Rate Limiting
    edgar_rate_limit: float = Field(
        0.1, env="EDGAR_RATE_LIMIT")  # seconds between requests
    llm_rate_limit: float = Field(1.0, env="LLM_RATE_LIMIT")

    # Knowledge Graph Extraction
    entity_extraction_prompt_template: str = Field(
        """Extract financial entities and relationships from this 10-K filing text.
        Focus on mergers, acquisitions, organizational changes, subsidiaries, and key personnel.
        
        Text: {text}
        
        Return a structured JSON with entities and relationships.""",
        env="ENTITY_EXTRACTION_PROMPT")

    # M&A Analysis
    ma_keywords: list = Field([
        "merger", "acquisition", "acquire", "divest", "subsidiary", "spin-off",
        "joint venture", "strategic alliance", "consolidation", "divestiture",
        "restructuring", "reorganization", "integration", "synergy"
    ],
                              env="MA_KEYWORDS")

    # Entity Extraction Method
    extraction_method: str = Field(
        "llm", env="EXTRACTION_METHOD")  # "llm" or "triplex"
    triplex_model_name: str = Field("sciphi/triplex", env="TRIPLEX_MODEL_NAME")
    triplex_device: str = Field(
        "auto", env="TRIPLEX_DEVICE")  # "cuda", "cpu", or "auto"

    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


# Global settings instance
settings = get_settings()


# Ensure required directories exist
def setup_directories():
    """Create necessary directories if they don't exist"""
    Path(settings.data_directory).mkdir(exist_ok=True)
    Path(settings.edgar_data_directory).mkdir(exist_ok=True)
    Path(settings.cache_directory).mkdir(exist_ok=True)
    Path(settings.chroma_persist_directory).mkdir(exist_ok=True)


setup_directories()
