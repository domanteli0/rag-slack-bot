"""Configuration loader for the Confluence RAG Slack Bot."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

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
    confluence_space_keys: List[str]

    # Slack
    slack_bot_token: str
    slack_app_token: str

    # RAG settings
    chunk_size: int
    chunk_overlap: int
    top_k_results: int
    score_threshold: float  # Minimum relevance score (0-1) to include a document
    embedding_model: str
    llm_model: str

    # Vector DB
    chroma_persist_dir: str

    # Multi-collection routing
    faq_confidence_threshold: float  # Threshold for FAQ match (0-1)
    doc_weight: float  # Weight for documentation results
    code_weight: float  # Weight for code results
    qa_weight: float  # Weight for Q&A results

    # FAQ generation filtering
    faq_min_importance: float  # Minimum importance score to generate FAQs (0-1)
    faq_min_chunks: (
        int  # Minimum chunks per feature (default: 10 for substantial features)
    )

    # Recency boosting for documentation
    recency_boost_enabled: bool  # Enable recency-based score boosting
    recency_boost_factor: float  # Max boost factor (0-1, added to similarity score)
    recency_max_age_days: (
        float  # Days after which no boost is applied (linear decay to 0)
    )

    # Recency penalty for old documentation
    recency_penalty_enabled: bool  # Enable penalty for old documents
    recency_penalty_start_days: float  # Days after which penalty starts
    recency_penalty_factor: float  # Max penalty factor (0-1, multiplied with score)
    recency_penalty_max_days: float  # Days at which max penalty is reached

    # Repository configurations
    repositories: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "REDACTED",
                "path": "REDACTED",
                "language": "REDACTED",
                "exclude_patterns": ["node_modules/", "dist/", "build/", ".next/"],
            },
        ]
    )


def load_config() -> Config:
    """Load configuration from environment variables."""

    required_vars = [
        "OPENAI_API_KEY",
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    # Parse comma-separated space keys (already validated above)
    space_keys_str = os.environ["CONFLUENCE_SPACE_KEYS"]
    confluence_space_keys = [key.strip() for key in space_keys_str.split(",")]

    return Config(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        confluence_url=os.environ["CONFLUENCE_URL"],
        confluence_username=os.environ["CONFLUENCE_USERNAME"],
        confluence_api_token=os.environ["CONFLUENCE_API_TOKEN"],
        confluence_space_keys=confluence_space_keys,
        slack_bot_token=os.environ["SLACK_BOT_TOKEN"],
        slack_app_token=os.environ["SLACK_APP_TOKEN"],
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        top_k_results=int(os.getenv("TOP_K_RESULTS", 7)),
        score_threshold=float(os.getenv("SCORE_THRESHOLD", 0)),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_model=os.getenv("LLM_MODEL", "gpt-5.2-2025-12-11"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        faq_confidence_threshold=float(os.getenv("FAQ_CONFIDENCE_THRESHOLD", 0.9)),
        doc_weight=float(os.getenv("DOC_WEIGHT", 0.8)),
        code_weight=float(os.getenv("CODE_WEIGHT", 0.7)),
        qa_weight=float(os.getenv("QA_WEIGHT", 0.6)),
        faq_min_importance=float(os.getenv("FAQ_MIN_IMPORTANCE", 0.3)),
        faq_min_chunks=int(os.getenv("FAQ_MIN_CHUNKS", 10)),
        recency_boost_enabled=os.getenv("RECENCY_BOOST_ENABLED", "true").lower()
        == "true",
        recency_boost_factor=float(os.getenv("RECENCY_BOOST_FACTOR", 0.1)),
        recency_max_age_days=float(os.getenv("RECENCY_MAX_AGE_DAYS", 270.0)),
        recency_penalty_enabled=os.getenv("RECENCY_PENALTY_ENABLED", "true").lower()
        == "true",
        recency_penalty_start_days=float(
            os.getenv("RECENCY_PENALTY_START_DAYS", 365.0)
        ),
        recency_penalty_factor=float(os.getenv("RECENCY_PENALTY_FACTOR", 0.5)),
        recency_penalty_max_days=float(os.getenv("RECENCY_PENALTY_MAX_DAYS", 730.0)),
    )
