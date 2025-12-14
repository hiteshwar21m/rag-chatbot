"""
Configuration Manager
Centralized configuration loading from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any


class Config:
    """Centralized configuration management"""
    
    _instance = None
    _loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not Config._loaded:
            # Load .env file from project root config folder
            env_path = Path(__file__).parent.parent.parent / "config" / ".env"
            load_dotenv(dotenv_path=env_path)
            Config._loaded = True
    
    # API Keys
    @property
    def openrouter_api_key(self) -> str:
        return os.getenv("OPENROUTER_API_KEY", "")
    
    # LLM Configuration
    @property
    def llm_model(self) -> str:
        return os.getenv("LLM_MODEL", "google/gemini-2.0-flash-lite-001")
    
    @property
    def llm_temperature(self) -> float:
        return float(os.getenv("LLM_TEMPERATURE", "0.3"))
    
    @property
    def llm_timeout(self) -> int:
        return int(os.getenv("LLM_TIMEOUT", "30"))
    
    # Embedding Models
    @property
    def embedding_model(self) -> str:
        return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    @property
    def embedding_dimension(self) -> int:
        return int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    @property
    def reranker_model(self) -> str:
        return os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Weaviate Configuration
    @property
    def weaviate_host(self) -> str:
        return os.getenv("WEAVIATE_HOST", "localhost")
    
    @property
    def weaviate_http_port(self) -> int:
        return int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    
    @property
    def weaviate_grpc_port(self) -> int:
        return int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    
    # Retrieval Configuration
    @property
    def top_k_summaries(self) -> int:
        return int(os.getenv("TOP_K_SUMMARIES", "10"))
    
    @property
    def top_k_chunks_direct(self) -> int:
        return int(os.getenv("TOP_K_CHUNKS_DIRECT", "20"))
    
    @property
    def top_k_final(self) -> int:
        return int(os.getenv("TOP_K_FINAL", "5"))
    
    @property
    def max_chunks_per_doc(self) -> int:
        return int(os.getenv("MAX_CHUNKS_PER_DOC", "2"))
    
    @property
    def doc_bonus_score(self) -> float:
        return float(os.getenv("DOC_BONUS_SCORE", "0.1"))
    
    # Processing Configuration
    @property
    def embedding_batch_size(self) -> int:
        return int(os.getenv("EMBEDDING_BATCH_SIZE", "256"))
    
    @property
    def summary_batch_size(self) -> int:
        return int(os.getenv("SUMMARY_BATCH_SIZE", "32"))
    
    # Paths
    @property
    def chunks_file(self) -> str:
        return os.getenv("CHUNKS_FILE", "data/processed/chunks_final.jsonl")
    
    @property
    def embeddings_file(self) -> str:
        return os.getenv("EMBEDDINGS_FILE", "data/processed/chunks_with_embeddings.jsonl")
    
    @property
    def query_log_file(self) -> str:
        return os.getenv("QUERY_LOG_FILE", "logs/query_log.jsonl")
    @property
    def summaries_output_file(self) -> str:
        return os.getenv("SUMMARIES_OUTPUT_FILE", "data/processed/chunks_with_summaries.jsonl")

    @property
    def summarization_error_log(self) -> str:
        return os.getenv("SUMMARIZATION_ERROR_LOG", "logs/summarization_errors.jsonl")
    @property
    def pdf_markdown_dir(self) -> str:
        return os.getenv("PDF_MARKDOWN_DIR", "data/extracted/pdf_markdown")

    @property
    def webpage_text_dir(self) -> str:
        return os.getenv("WEBPAGE_TEXT_DIR", "data/extracted/webpage_text")

    @property
    def chunks_output_file(self) -> str:
        return os.getenv("CHUNKS_OUTPUT_FILE", "data/processed/chunks_with_metadata.jsonl")
        
    def as_dict(self) -> Dict[str, Any]:
        """Return all config as dictionary"""
        return {
            # API
            "openrouter_api_key": self.openrouter_api_key,
            # LLM
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_timeout": self.llm_timeout,
            # Embeddings
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "reranker_model": self.reranker_model,
            # Weaviate
            "weaviate_host": self.weaviate_host,
            "weaviate_http_port": self.weaviate_http_port,
            "weaviate_grpc_port": self.weaviate_grpc_port,
            # Retrieval
            "top_k_summaries": self.top_k_summaries,
            "top_k_chunks_direct": self.top_k_chunks_direct,
            "top_k_final": self.top_k_final,
            "max_chunks_per_doc": self.max_chunks_per_doc,
            "doc_bonus_score": self.doc_bonus_score,
            # Processing
            "embedding_batch_size": self.embedding_batch_size,
            "summary_batch_size": self.summary_batch_size,
            # Paths
            "chunks_file": self.chunks_file,
            "embeddings_file": self.embeddings_file,
            "query_log_file": self.query_log_file,
        }


# Singleton instance
_config = None


def get_config() -> Config:
    """Get configuration instance (singleton pattern)"""
    global _config
    if _config is None:
        _config = Config()
    return _config
