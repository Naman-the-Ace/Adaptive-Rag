"""
Configuration settings for the Adaptive RAG application.
Loads environment variables from .env file using python-dotenv.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Load and manage configuration from prompts YAML file."""

    def __init__(self, config_file: str = None):
        """
        Initialize configuration from YAML file.

        Args:
            config_file: Optional path to config file. Defaults to prompts.yaml.
        """
        if config_file is None:
            base_dir = Path(__file__).parent
            config_file = base_dir / "prompts.yaml"

        with open(config_file, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def prompt(self, key: str) -> str:
        """
        Retrieve a prompt template by key.

        Args:
            key: The prompt key (e.g. 'classify_prompt').

        Returns:
            The prompt template string.
        """
        return self._config["prompts"][key]


class Settings:
    """Application settings loaded from environment variables."""

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "adaptive_rag")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
