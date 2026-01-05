"""
Query Router: Intelligent routing across multiple knowledge sources.

This module implements a multi-index routing strategy that performs hybrid search
across FAQs, documentation, source code, and Slack Q&A, combining results from all
sources to provide comprehensive answers with self-correction for relevance grading.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class RouterResponse:
    """Response from router with routing metadata."""

    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    route_taken: (
        str  # "faq_only", "hybrid", "code_only", "docs_only", "qa_only", "none"
    )
    query: str = ""
    self_correction_grade: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QueryRouter:
    """Route queries across multiple knowledge sources."""

    def __init__(self, config):
        self.config = config
        self.vectorstores: Dict[str, Chroma] = {}
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=0.1,  # Low temperature for factual routing
            api_key=config.openai_api_key,
        )
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model, api_key=config.openai_api_key
        )

        # Load all available collections
        self._load_all_collections()

    def _load_all_collections(self):
        """Load all available ChromaDB collections."""
        collection_names = [
            "synthetic_faqs",
            "confluence_docs",
            "source_code",
            "slack_qa",
        ]

        for name in collection_names:
            try:
                vectorstore: Chroma = Chroma(
                    persist_directory=self.config.chroma_persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=name,
                )

                # Verify collection exists by checking count
                count = vectorstore._collection.count()
                logger.info(f"Loaded collection '{name}' with {count} documents")
                self.vectorstores[name] = vectorstore

            except Exception as e:
                logger.warning(f"Collection '{name}' not available: {e}")

        if not self.vectorstores:
            logger.error("No collections available for routing!")

    def route_query(self, query: str, thread_context: Optional[str]) -> RouterResponse:
        """
        Main routing logic: Hybrid Search → Self-correction.

        Step 1: Hybrid Search across all sources (FAQs + Docs + Code + Q&A)
        Step 2: Self-Correction (Grade relevance)
        """

        logger.info(f"Routing query: {query[:100]}...")

        # Include thread context if available
        full_query = query
        if thread_context:
            full_query = f"{thread_context}\n\nCurrent question: {query}"

        # Hybrid Search across all sources (FAQs + Docs + Code + Q&A)
        hybrid_response = self._hybrid_search(full_query)

        if not hybrid_response:
            logger.warning("No results from hybrid search")
            return RouterResponse(
                answer="I couldn't find relevant information to answer your question. "
                "Please try rephrasing or provide more context.",
                sources=[],
                confidence=0.0,
                route_taken="none",
                query=query,
            )

        # Step 3: Self-Correction (Grade relevance)
        graded_response = self._grade_and_refine(query, hybrid_response)

        return graded_response

    def _hybrid_search(self, query: str) -> Optional[RouterResponse]:
        """Search across documentation and source code collections."""

        all_results = []

        # Query documentation (Confluence) if available
        if "confluence_docs" in self.vectorstores:
            try:
                doc_results = self.vectorstores[
                    "confluence_docs"
                ].similarity_search_with_score(query, k=self.config.top_k_results)

                # Apply recency boosting to documentation results
                if self.config.recency_boost_enabled:
                    doc_results = self._apply_recency_boost(doc_results)

                for doc, score in doc_results:
                    if score >= self.config.score_threshold:
                        doc.metadata["collection"] = "documentation"
                        # Apply documentation weight
                        weighted_score = score * self.config.doc_weight
                        all_results.append((doc, weighted_score, score))

                logger.info(f"Documentation search: {len(doc_results)} results")

            except Exception as e:
                logger.error(f"Error querying documentation: {e}, {e.__traceback__}")

        # Query source code if available
        if "source_code" in self.vectorstores:
            try:
                code_results = self.vectorstores[
                    "source_code"
                ].similarity_search_with_score(query, k=self.config.top_k_results)

                for doc, score in code_results:
                    if score >= self.config.score_threshold:
                        doc.metadata["collection"] = "source_code"
                        # Apply code weight
                        weighted_score = score * self.config.code_weight
                        all_results.append((doc, weighted_score, score))

                logger.info(f"Code search: {len(code_results)} results")

            except Exception as e:
                logger.error(f"Error querying source code: {e}")

        # Query Q&A if available
        if "slack_qa" in self.vectorstores:
            try:
                slack_results = self.vectorstores[
                    "slack_qa"
                ].similarity_search_with_score(query, k=self.config.top_k_results)

                for doc, score in slack_results:
                    if score >= self.config.score_threshold:
                        doc.metadata["collection"] = "slack_qa"
                        # Apply slack_qa_weight
                        weighted_score = score * self.config.qa_weight
                        all_results.append((doc, weighted_score, score))

                logger.info(f"Slack Q&A search: {len(slack_results)} results")

            except Exception as e:
                logger.error(f"Error querying Slack Q&A: {e}")

        # Query synthetic FAQs if available
        if "synthetic_faqs" in self.vectorstores:
            try:
                faq_results = self.vectorstores[
                    "synthetic_faqs"
                ].similarity_search_with_score(query, k=self.config.top_k_results)

                for doc, score in faq_results:
                    if score >= self.config.score_threshold:
                        doc.metadata["collection"] = "synthetic_faqs"
                        # Apply FAQ weight (default to same as docs if not configured)
                        faq_weight = getattr(
                            self.config, "faq_weight", self.config.doc_weight
                        )
                        weighted_score = score * faq_weight
                        all_results.append((doc, weighted_score, score))

                logger.info(f"FAQ search: {len(faq_results)} results")

            except Exception as e:
                logger.error(f"Error querying FAQs: {e}")

        if not all_results:
            return None

        # Sort by weighted score (highest first)
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Take top K results
        top_results = all_results[: self.config.top_k_results]

        # Format context for LLM
        context = self._format_context(top_results)

        # Generate answer with two-stage LLM (returns answer and audience type)
        answer, audience_type = self._generate_answer(query, context)

        # Extract sources
        sources = self._extract_sources(top_results)

        # Determine route taken
        has_faqs = any(
            doc.metadata.get("collection") == "synthetic_faqs"
            for doc, _, _ in top_results
        )
        has_docs = any(
            doc.metadata.get("collection") == "documentation"
            for doc, _, _ in top_results
        )
        has_code = any(
            doc.metadata.get("collection") == "source_code" for doc, _, _ in top_results
        )
        has_qa = any(
            doc.metadata.get("collection") == "slack_qa" for doc, _, _ in top_results
        )

        # Determine routing based on sources
        sources_count = sum([has_faqs, has_docs, has_code, has_qa])

        if sources_count > 1:
            route = "hybrid"
        elif has_faqs:
            route = "faq_only"
        elif has_qa:
            route = "qa_only"
        elif has_code:
            route = "code_only"
        elif has_docs:
            route = "docs_only"
        else:
            route = "unknown"

        return RouterResponse(
            answer=answer,
            sources=sources,
            confidence=top_results[0][1] if top_results else 0.0,
            route_taken=route,
            query=query,
            metadata={"audience_type": audience_type},
        )

    def _apply_recency_boost(
        self, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Apply time-based recency boost/penalty to documentation search results.

        Boost (for recent docs):
        - Documents updated within recency_max_age_days get a boost
        - Boost linearly decreases from max_boost (at 0 days) to 0 (at recency_max_age_days)

        Penalty (for old docs):
        - Documents older than recency_penalty_start_days get a penalty
        - Penalty linearly increases from 0 to max_penalty (at recency_penalty_max_days)

        Args:
            results: List of (document, score) tuples from similarity search

        Returns:
            List of (document, adjusted_score) tuples, re-sorted by score
        """
        current_time = datetime.now().timestamp()
        adjusted_results = []

        for doc, score in results:
            # Get last modified timestamp from metadata
            last_modified_timestamp = doc.metadata.get("last_modified_timestamp", None)

            if last_modified_timestamp is not None:
                # Calculate age in days
                age_seconds = current_time - last_modified_timestamp
                age_days = age_seconds / 86400.0  # Convert to days

                recency_boost = 0.0
                recency_penalty = 0.0

                # Apply boost for recent documents
                if age_days < self.config.recency_max_age_days:
                    decay_factor = 1.0 - (age_days / self.config.recency_max_age_days)
                    recency_boost = self.config.recency_boost_factor * decay_factor

                # Apply penalty for old documents
                if (
                    self.config.recency_penalty_enabled
                    and age_days > self.config.recency_penalty_start_days
                ):
                    # Calculate how far into the penalty zone we are
                    penalty_range = (
                        self.config.recency_penalty_max_days
                        - self.config.recency_penalty_start_days
                    )
                    days_into_penalty = (
                        age_days - self.config.recency_penalty_start_days
                    )

                    # Linear increase from 0 to max penalty
                    penalty_factor = min(1.0, days_into_penalty / penalty_range)
                    recency_penalty = (
                        self.config.recency_penalty_factor * penalty_factor
                    )

                # Calculate final score: boost increases, penalty decreases
                adjusted_score = score * (1 + recency_boost - recency_penalty)

                # Ensure score doesn't go negative
                adjusted_score = max(0.0, adjusted_score)

                # Log adjustments
                if recency_boost > 0:
                    logger.debug(
                        f"  Recency boost: {doc.metadata.get('title', 'Unknown')[:50]} "
                        f"({age_days:.0f} days old) - boost: +{recency_boost:.3f} "
                        f"(score: {score:.3f} → {adjusted_score:.3f})"
                    )
                elif recency_penalty > 0:
                    logger.debug(
                        f"  Recency penalty: {doc.metadata.get('title', 'Unknown')[:50]} "
                        f"({age_days:.0f} days old) - penalty: -{recency_penalty:.3f} "
                        f"(score: {score:.3f} → {adjusted_score:.3f})"
                    )

                doc.metadata["recency_boost"] = recency_boost
                doc.metadata["recency_penalty"] = recency_penalty
                doc.metadata["age_days"] = age_days
                adjusted_results.append((doc, adjusted_score))
            else:
                logger.debug(
                    f"  Skipped recency adjustment: {doc.metadata.get('title', 'Unknown')[:50]}"
                )
                # No timestamp available, keep original score
                doc.metadata["recency_boost"] = 0.0
                doc.metadata["recency_penalty"] = 0.0
                adjusted_results.append((doc, score))

        # Re-sort by adjusted score (highest first)
        adjusted_results.sort(key=lambda x: x[1], reverse=True)

        return adjusted_results

    def _format_context(self, results: List[tuple]) -> str:
        """Format retrieved documents into context string for LLM."""
        context_parts = []

        for i, (doc, weighted_score, original_score) in enumerate(results, 1):
            collection = doc.metadata.get("collection", "unknown")

            if collection == "synthetic_faqs":
                question = doc.metadata.get("question", "Unknown")
                answer = doc.metadata.get("answer", "")
                context_parts.append(
                    f"--- FAQ {i}: {question} (relevance: {original_score:.0%}) ---\\n"
                    f"Q: {question}\\n"
                    f"A: {answer}\\n"
                )

            elif collection == "documentation":
                title = doc.metadata.get("title", "Unknown")
                source = doc.metadata.get("source", "")
                age_days = doc.metadata.get("age_days")
                recency_boost = doc.metadata.get("recency_boost", 0.0)

                # Format age info
                age_info = ""
                if age_days is not None:
                    if age_days < 30:
                        age_info = f" [Updated {int(age_days)} days ago]"
                    elif age_days < 365:
                        age_info = f" [Updated {int(age_days / 30)} months ago]"
                    else:
                        age_info = f" [Updated {int(age_days / 365)} years ago]"

                    if recency_boost > 0.05:
                        age_info += " ⭐"  # Mark recently updated docs

                context_parts.append(
                    f"--- Document {i}: {title} (relevance: {original_score:.0%}){age_info} ---\n"
                    f"Source: {source}\n\n"
                    f"{doc.page_content}\n"
                )

            elif collection == "source_code":
                name = doc.metadata.get("name", "Unknown")
                file_path = doc.metadata.get("file_path", "")
                line_start = doc.metadata.get("line_start", "")
                repo_name = doc.metadata.get("repo_name", "")
                signature = doc.metadata.get("signature", "")

                context_parts.append(
                    f"--- Code {i}: {name} (relevance: {original_score:.0%}) ---\n"
                    f"Repository: {repo_name}\n"
                    f"File: {file_path}:{line_start}\n"
                    f"Signature: {signature}\n\n"
                    f"{doc.page_content}\n"
                )

            elif collection == "slack_qa":
                channel = doc.metadata.get("channel", "Unknown")
                timestamp = doc.metadata.get("timestamp", "")
                message_type = doc.metadata.get("message_type", "message")

                context_parts.append(
                    f"--- Slack Message {i}: #{channel} (relevance: {original_score:.0%}) ---\n"
                    f"Type: {message_type}\n"
                    f"Timestamp: {timestamp}\n\n"
                    f"{doc.page_content}\n"
                )

        return "\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> Tuple[str, str]:
        """Generate answer using three-stage LLM approach.

        Stage 1: Zero-shot CoT to extract comprehensive information from context
        Stage 2: Audience-adaptive summarization based on question type
        Stage 3: Slack formatting and final condensation

        Returns:
            Tuple of (answer, audience_type)
        """
        try:
            # Stage 1: Extract information with Chain-of-Thought reasoning
            logger.info("Stage 1: Extracting information with CoT reasoning...")
            extracted_info = self._extract_information_with_cot(query, context)
            logger.debug(f"CoT extraction complete ({len(extracted_info)} chars)")

            # Stage 2: Classify audience and generate appropriate response
            audience_type = self._classify_audience(query)
            logger.info(f"Stage 2: Generating response for '{audience_type}' audience")

            audience_response = self._summarize_for_audience(
                query, extracted_info, audience_type
            )
            logger.debug(f"Audience summary complete ({len(audience_response)} chars)")

            # Stage 3: Format for Slack and condense
            logger.info("Stage 3: Formatting for Slack and condensing...")
            final_answer = self._format_for_slack(query, audience_response)

            return final_answer, audience_type

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return (
                "I encountered an error generating the answer. Please try again.",
                "unknown",
            )

    def _extract_information_with_cot(self, query: str, context: str) -> str:
        """Stage 1: Use zero-shot Chain-of-Thought to extract comprehensive information."""

        cot_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at extracting and analyzing information from documentation and code.

Your task is to thoroughly analyze the provided context and extract ALL relevant information that could help answer the question.

Think step by step:
1. First, identify what the question is really asking
2. Scan through each piece of context and note relevant facts, details, and relationships
3. Identify any code patterns, configurations, or technical details mentioned
4. Note any caveats, limitations, or edge cases
5. Connect related pieces of information across different sources
6. Identify what information is missing or unclear

Be thorough and comprehensive. Extract everything potentially relevant, even if it seems minor.""",
                ),
                (
                    "human",
                    """Context from knowledge base:

{context}

---

Question: {question}

Let's think step by step. Analyze the context thoroughly and extract all relevant information that could help answer this question:""",
                ),
            ]
        )

        chain = cot_prompt | self.llm
        response = chain.invoke({"context": context, "question": query})
        return response.content

    def _classify_audience(self, query: str) -> str:
        """Classify the intended audience based on the question.

        Returns: "technical" or "general"
        """
        # Technical indicators in the question
        technical_indicators = [
            "ruby",
            "kotlin",
            "typescipt",
            "interactor",
            "code",
            "function",
            "class",
            "method",
            "api",
            "endpoint",
            "database",
            "query",
            "config",
            "deploy",
            "architecture",
            "debug",
            "exception",
            "stacktrace",
            "logs",
            "parameter",
            "variable",
            "return",
            "repository",
            "branch",
            "commit",
            "merge",
            "pr",
            "pull request",
            "ci",
            "cd",
            "pipeline",
            "docker",
            "kubernetes",
            "k8s",
            "pod",
            "service",
            "container",
            "yaml",
            "json",
            "xml",
            "graphql",
            "rest",
            "grpc",
            "kafka",
            "redis",
            "sql",
            "nosql",
            "async",
            "callback",
            "thread",
            "process",
            "memory",
            "cpu",
            "spec",
            "mock",
            "stub",
            "fixture",
            "jenkings",
            "github",
            "mysql",
        ]

        query_lower = query.lower()

        # Count technical indicators
        technical_count = sum(1 for ind in technical_indicators if ind in query_lower)

        # If 2+ technical terms or question pattern suggests technical
        if technical_count >= 2:
            return "technical"

        return "general"

    def _summarize_for_audience(
        self, query: str, extracted_info: str, audience_type: str
    ) -> str:
        """Stage 2: Summarize extracted information for the target audience."""

        if audience_type == "technical":
            system_prompt = """You are a helpful technical coworker answering questions about the product & the codebase, sometimes other related things.

Based on the analysis provided, give a clear, technical answer that includes:
- Specific file names, function names, and code references when available
- Technical implementation details
- Configuration options or parameters if relevant
- Code examples or patterns if helpful
- Any caveats or edge cases

Be precise and technical, but still conversational. Developers appreciate specificity."""

        else:  # general audience
            system_prompt = """You are a helpful coworker answering questions about our product.

Based on the analysis provided, give a clear, accessible answer that:
- Explains concepts in plain language
- Focuses on what things do, not how they're implemented
- Avoids unnecessary technical jargon (for example: backend)
- Doesn't delve into code or even refer to it.
- Provides practical, actionable information
- Keeps things concise, easy to understand and most importantly short

Be friendly and helpful. Not everyone needs the technical details."""

        summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    """Analysis of relevant information:

{extracted_info}

---

Original question: {question}

Based on the analysis above, provide a helpful answer to the question:""",
                ),
            ]
        )

        chain = summary_prompt | self.llm
        response = chain.invoke({"extracted_info": extracted_info, "question": query})
        return response.content

    def _format_for_slack(self, query: str, response: str) -> str:
        """Stage 3: Format and condense the response for Slack.

        - Condenses verbose responses
        - Formats with Slack-compatible markdown
        - Ensures appropriate length for chat context
        """
        slack_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are formatting a response for Slack chat. Your job is to make it concise and well-formatted.

Guidelines:
- Keep the response SHORT - aim for one paragprah for simple questions, 2-4 for complex ones
- Use Slack markdown formatting:
  - *bold* for emphasis
  - `code` for inline code, file names, function names
  - ```language
    code blocks
    ``` for multi-line code
  - • for bullet points (use sparingly)
- Remove redundant information and filler phrases
- Get straight to the point - no preamble like "Based on the documentation..."
- If there are multiple points, use bullet points but keep them brief
- Preserve important technical details (file names, function names, config values) if intended for technical audience
- Don't lose critical information while condensing
- Maintain a friendly, conversational tone""",
                ),
                (
                    "human",
                    """Original question: {question}

Draft response to condense and format for Slack:

{response}

---

Provide a concise, Slack-formatted version:""",
                ),
            ]
        )

        chain = slack_prompt | self.llm
        result = chain.invoke({"question": query, "response": response})
        return result.content

    def _extract_sources(self, results: List[tuple]) -> List[Dict[str, Any]]:
        """Extract unique source information from documents."""
        sources = []
        seen_sources = set()

        for doc, weighted_score, original_score in results:
            collection = doc.metadata.get("collection", "")

            if collection == "synthetic_faqs":
                source_key = doc.metadata.get("question", "")
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)

                # Extract traceability from metadata
                traceability_json = doc.metadata.get("traceability", "[]")
                try:
                    traceability = json.loads(traceability_json)
                except json.JSONDecodeError:
                    traceability = []

                sources.append(
                    {
                        "type": "faq",
                        "question": doc.metadata.get("question", ""),
                        "answer": doc.metadata.get("answer", ""),
                        "traceability": traceability,
                        "collection": collection,
                        "relevance": original_score,
                    }
                )

            elif collection == "documentation":
                page_id = doc.metadata.get("page_id", "")
                if page_id in seen_sources:
                    continue
                seen_sources.add(page_id)

                sources.append(
                    {
                        "type": "documentation",
                        "title": doc.metadata.get("title", "Unknown"),
                        "url": doc.metadata.get("source", ""),
                        "page_id": page_id,
                        "collection": collection,
                        "relevance": original_score,
                    }
                )

            elif collection == "source_code":
                source_key = doc.metadata.get("source", "")
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)

                sources.append(
                    {
                        "type": "code",
                        "name": doc.metadata.get("name", ""),
                        "file_path": doc.metadata.get("file_path", ""),
                        "line_start": doc.metadata.get("line_start", 0),
                        "line_end": doc.metadata.get("line_end", 0),
                        "repo_name": doc.metadata.get("repo_name", ""),
                        "signature": doc.metadata.get("signature", ""),
                        "collection": collection,
                        "relevance": original_score,
                    }
                )

            elif collection == "slack_qa":
                source_key = doc.metadata.get("source", "")
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)

                sources.append(
                    {
                        "type": "slack_qa",
                        "source": source_key,
                        "channel": doc.metadata.get("channel", ""),
                        "user": doc.metadata.get("user", ""),
                        "timestamp": doc.metadata.get("timestamp", ""),
                        "message_type": doc.metadata.get("message_type", ""),
                        "collection": collection,
                        "relevance": original_score,
                    }
                )

        return sources

    def _grade_and_refine(self, query: str, response: RouterResponse) -> RouterResponse:
        """Self-correction: Grade relevance and refine if needed."""

        # Use LLM to grade the response relevance
        grading_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are evaluating the relevance of an answer to a question. "
                    "Rate the relevance on a scale of 0.0 to 1.0, where 1.0 is perfectly relevant. "
                    "Return ONLY a number between 0.0 and 1.0, nothing else.",
                ),
                (
                    "human",
                    """Query: {query}

Answer: {answer}

How relevant is this answer to the query? Return only a number between 0.0 and 1.0.""",
                ),
            ]
        )

        try:
            chain = grading_prompt | self.llm
            grade_response = chain.invoke({"query": query, "answer": response.answer})

            grade_str = grade_response.content.strip()
            grade = float(grade_str)

            response.self_correction_grade = grade

            if grade < 0.5:
                logger.warning(
                    f"Low relevance grade ({grade:.2f}) for query: {query[:50]}..."
                )
                # Could implement re-query with refined parameters here
                # For now, just log and return original

            logger.info(f"Self-correction grade: {grade:.2f}")

        except Exception as e:
            logger.error(f"Error grading response: {e}")
            # Continue without grading

        return response


def create_router(config) -> QueryRouter:
    """Factory function to create a QueryRouter."""
    return QueryRouter(config)
