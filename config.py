"""Configuration loader for the Confluence RAG Slack Bot."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""
    
    # OpenAI
    openai_api_key: str
    
    # Confluence
    confluence_url: str
    confluence_username: str
    confluence_api_token: str
    confluence_space_key: str
    
    # Slack
    slack_bot_token: str
    slack_app_token: str
    
    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"
    
    # Vector DB
    chroma_persist_dir: str = "./chroma_db"


def load_config() -> Config:
    """Load configuration from environment variables."""
    
    required_vars = [
        "OPENAI_API_KEY",
        "CONFLUENCE_URL", 
        "CONFLUENCE_USERNAME",
        "CONFLUENCE_API_TOKEN",
        "CONFLUENCE_SPACE_KEY",
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    return Config(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        confluence_url=os.getenv("CONFLUENCE_URL"),
        confluence_username=os.getenv("CONFLUENCE_USERNAME"),
        confluence_api_token=os.getenv("CONFLUENCE_API_TOKEN"),
        confluence_space_key=os.getenv("CONFLUENCE_SPACE_KEY"),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN"),
        slack_app_token=os.getenv("SLACK_APP_TOKEN"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k_results=int(os.getenv("TOP_K_RESULTS", "5")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
    )

