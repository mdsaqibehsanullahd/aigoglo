"""
Configuration for RFQ Intelligence System
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://100.82.77.42:11434"
    OLLAMA_MODEL: str = "mistral-nemo:12b"
    OLLAMA_VISION_MODEL: str = "llava:13b"
    OLLAMA_AUDIO_MODEL: str = "karanchopda333/whisper:latest"
    OLLAMA_EMBEDDING_MODEL: str = "qwen3-embedding:8b"
    
    # API Configuration
    API_TITLE: str = "RFQ Intelligence System"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-powered RFQ normalization and enrichment platform"
    
    # Processing Configuration
    MAX_FILE_SIZE_MB: int = 50
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6
    
    # LLM Configuration
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4060
    LLM_TIMEOUT: int = 300
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
