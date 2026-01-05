"""RAG Engine - retrieves context and generates answers using LLM."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Config
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from router import QueryRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG engine."""

    answer: str
    sources: List[dict]
    query: str
    metadata: Dict = field(default_factory=dict)
    """Additional metadata (route_taken, confidence, etc.)"""


class RAGEngine:
    """Retrieval-Augmented Generation engine for answering questions."""

    def __init__(self, config: Config, vectorstore: Optional[Chroma] = None):
        self.config = config

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.llm_model,
            api_key=config.openai_api_key,
            temperature=0.1,  # Low temperature for factual responses
        )

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model, api_key=config.openai_api_key
        )

        # Load or use provided vectorstore
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = Chroma(
                persist_directory=config.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="confluence_docs",
            )

        # Initialize multi-collection router if enabled
        logger.info("Initializing multi-collection router...")
        self.router = QueryRouter(config)
        logger.info("Multi-collection routing enabled")

    def query(
        self,
        question: str,
        thread_context: Optional[str] = None,
    ) -> RAGResponse:
        """Process a question and return an answer with sources.

        Args:
            question: The user's question
            thread_context: Previous conversation context from thread
            score_threshold: Minimum relevance score for documents (0-1), uses config default if not set
        """
        logger.info(f"Processing query: {question}")

        router_response = self.router.route_query(
            question, thread_context=thread_context
        )

        # Convert RouterResponse to RAGResponse
        return RAGResponse(
            answer=router_response.answer,
            sources=router_response.sources,
            query=question,
            metadata={
                "route_taken": router_response.route_taken,
                "confidence": router_response.confidence,
                "self_correction_grade": router_response.self_correction_grade,
                "audience_type": router_response.metadata.get("audience_type")
                if router_response.metadata
                else None,
            },
        )

    def format_slack_response(self, rag_response: RAGResponse) -> str:
        """Format RAG response for Slack display with traceability."""
        parts = [rag_response.answer]

        if rag_response.sources:
            parts.append("\n\n📚 *Sources:*")

            for source in rag_response.sources:
                source_type = source.get("type", "")

                if source_type == "faq":
                    # FAQ with traceability to code
                    question = source.get("question", "")
                    parts.append(f"• FAQ: {question}")

                    traceability = source.get("traceability", [])
                    if traceability:
                        parts.append("  *Code References:*")
                        for trace in traceability:
                            file_path = trace.get("file_path", "")
                            line_start = trace.get("line_start", "")
                            name = trace.get("name", "")
                            repo_name = trace.get("repo_name", "")

                            # GitHub link format (adjust to your GitHub org)
                            github_link = f"https://github.com/vinted/{repo_name}/blob/master/{file_path}#L{line_start}"
                            parts.append(
                                f"    - <{github_link}|{name} ({file_path}:{line_start})>"
                            )

                elif source_type == "code" or source.get("collection") == "source_code":
                    # Direct code reference
                    file_path = source.get("file_path", "")
                    line_start = source.get("line_start", "")
                    name = source.get("name", "Code")
                    repo_name = source.get("repo_name", "")

                    github_link = f"https://github.com/vinted/{repo_name}/blob/master/{file_path}#L{line_start}"
                    parts.append(f"• <{github_link}|{name} ({file_path}:{line_start})>")

                elif source_type == "slack_qa":
                    # Slack message reference
                    slack_url = source.get("source", "")
                    channel = source.get("channel", "unknown")
                    user = source.get("user", "unknown")
                    message_type = source.get("message_type", "message")

                    if message_type == "thread_parent":
                        parts.append(
                            f"• <{slack_url}|Slack: #{channel} thread> (by {user})"
                        )
                    else:
                        parts.append(f"• <{slack_url}|Slack: #{channel}> (by {user})")

                else:
                    # Documentation (existing format)
                    title = source.get("title", "Unknown")
                    url = source.get("url", "")
                    if url:
                        parts.append(f"• <{url}|{title}>")
                    else:
                        parts.append(f"• {title}")

        # Add routing metadata (optional, for debugging)
        if rag_response.metadata:
            metadata_parts = []
            route = rag_response.metadata.get("route_taken")
            if route and route not in ["legacy"]:
                metadata_parts.append(f"Route: {route}")

            audience = rag_response.metadata.get("audience_type")
            if audience:
                metadata_parts.append(f"Audience: {audience}")

            if metadata_parts:
                parts.append(f"\n_{' | '.join(metadata_parts)}_")

        return "\n".join(parts)
