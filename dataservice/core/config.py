"""Configuration management for data service"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
import yaml


class Settings(BaseSettings):
    """Service settings from environment and config file"""

    # Database
    database_url: str = "sqlite:///storage/dataservice.db"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # API Keys (optional)
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    serper_api_key: Optional[str] = None

    # Data paths
    data_path: str = "/app/data"
    storage_path: str = "/app/storage"

    # Cache
    cache_ttl: int = 300  # seconds

    class Config:
        env_file = ".env"
        extra = "ignore"

    @classmethod
    def load_yaml_config(cls, config_path: str = "config.yaml") -> dict:
        """Load additional config from YAML"""
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}
