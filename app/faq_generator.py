"""
FAQ Generator: Generate FAQs and User Stories from indexed code.

This module provides functionality to generate user-facing FAQs and User Stories
from indexed source code using LLM, with traceability links back to the code.

Implements intelligent filtering to avoid generating FAQs for low-value components
like database migrations, tests, and trivial code.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class FAQ:
    """A generated FAQ entry."""

    question: str
    answer: str
    source_chunks: List[str]  # References to code chunk sources
    traceability: List[Dict[str, Any]]  # Links to source code
    confidence: float = 0.8
    tags: List[str] = field(default_factory=list)


class FAQGenerator:
    """Generate FAQs from code chunks using LLM with intelligent filtering."""

    # Exclusion patterns for files that don't need FAQs
    # Note: Patterns match anywhere in the file path, including inside engines/packages
    # Examples: db/migrate/, engines/lockers/db/migrate/, packages/foo/db/migrate/
    EXCLUDE_PATTERNS = [
        # Database migrations (main app, engines, packages)
        r"db/migrate/",  # Matches anywhere: db/migrate/, engines/*/db/migrate/
        r"_migration\.rb$",
        r"^\d{14}_",  # Timestamp-based migration files (e.g., 20240205072258_*)
        r"Migration$",  # Classes ending with Migration
        # Test files (main app, engines, packages)
        r"spec/",  # Matches anywhere: spec/, engines/*/spec/, packages/*/spec/
        r"test/",
        r"tests/",
        r"__tests__/",
        r"_spec\.rb$",
        r"_test\.(ts|js|kt)$",
        r"\.test\.(ts|js)$",
        r"\.spec\.(ts|js)$",
        # Configuration and build files
        r"config/",
        r"webpack",
        r"babel",
        r"eslint",
        r"\.config\.",
        # Generated code
        r"generated/",
        r"\.g\.dart$",
        r"\.generated\.",
        # Build artifacts
        r"build/",
        r"dist/",
        r"out/",
    ]

    # Patterns for important files (higher priority for FAQs)
    # Note: Patterns match anywhere in the file path, including inside engines/packages
    # Examples: app/domain/, engines/driver_routing/app/domain/, packages/*/app/domain/
    IMPORTANT_PATTERNS = [
        # Domain layer (main app, engines, packages)
        r"app/domain",  # Matches: app/domain/, engines/driver_routing/app/domain/
        r"domain",  # Broader match for domain directories
        # Services (business logic)
        r"app/services/",  # Matches: app/services/, engines/*/app/services/
        r"services/",
        r"Service\.rb$",
        r"Service\.(ts|js)$",
        # Controllers/API endpoints
        r"app/controllers/",  # Matches: app/controllers/, engines/*/app/controllers/
        r"controllers/",
        r"Controller\.rb$",
        r"api/",
        # Core models
        r"app/models/",  # Matches: app/models/, engines/*/app/models/
        r"models/",
        r"Model\.rb$",
        # React components
        r"components/",
        r"Component\.(tsx|jsx)$",
        # Use cases / interactors
        r"interactors/",  # Matches: app/interactors/, engines/*/app/interactors/
        r"use_cases/",
        r"useCases/",
    ]

    def __init__(
        self,
        config,
        code_vectorstore: Chroma,
        force_regen: bool = False,
    ):
        self.config = config
        self.code_vectorstore = code_vectorstore
        self.force_regen = force_regen  # Skip resume check if True
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=0.3,  # Slightly higher for creative FAQ generation
            api_key=config.openai_api_key,
        )
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model, api_key=config.openai_api_key
        )

        # Minimum importance score to generate FAQs (0-1 scale)
        self.min_importance_score = getattr(config, "faq_min_importance", 0.3)

        # Minimum chunks per component for FAQ generation
        self.min_chunks_per_component = getattr(config, "faq_min_chunks", 10)

    def generate_faqs_for_codebase(
        self, repo_name: Optional[str] = None, max_faqs_per_component: int = 3
    ) -> List[FAQ]:
        """Generate FAQs for entire codebase or specific repository."""
        logger.info(
            f"Generating FAQs for {'all repositories' if not repo_name else repo_name}"
        )

        if not self.code_vectorstore:
            logger.error("Code vectorstore not provided")
            return []

        # Retrieve all code chunks (or filter by repo_name)
        all_chunks = self._get_code_chunks(repo_name)

        if not all_chunks:
            logger.warning("No code chunks found for FAQ generation")
            return []

        logger.info(f"Found {len(all_chunks)} code chunks")

        # Group chunks by feature/file (less granular than class-level)
        grouped_chunks = self._group_chunks_by_feature(all_chunks)
        logger.info(f"Grouped into {len(grouped_chunks)} features")

        # Filter components by importance and minimum size
        filtered_components = self._filter_by_importance(grouped_chunks)
        logger.info(
            f"Filtered to {len(filtered_components)} important features "
            f"(excluded {len(grouped_chunks) - len(filtered_components)} low-value features)"
        )

        # Generate FAQs for each important component (with resume support)
        all_faqs = []
        skipped_count = 0

        for component_name, chunks in filtered_components.items():
            # Check if FAQs already exist for this component (resume support)
            if (not self.force_regen) and self._faqs_already_exist(
                component_name, chunks
            ):
                logger.info(f"⏭️  Skipping {component_name} (FAQs already exist)")
                skipped_count += 1
                continue

            logger.info(f"Generating FAQs for component: {component_name}")

            try:
                faqs = self._generate_faqs_for_component(
                    component_name, chunks, max_faqs=max_faqs_per_component
                )
                all_faqs.extend(faqs)
                logger.info(f"✓ Generated {len(faqs)} FAQs for {component_name}")

                # Save incrementally after each component (for resume support)
                if faqs:
                    self._save_faqs_incrementally(faqs)

            except Exception as e:
                logger.error(f"✗ Error generating FAQs for {component_name}: {e}")
                continue

        logger.info(f"Total FAQs generated: {len(all_faqs)} (skipped: {skipped_count})")
        return all_faqs

    def _get_code_chunks(self, repo_name: Optional[str] = None) -> List[Document]:
        """Retrieve code chunks from vectorstore."""
        try:
            # Get all documents from the collection
            collection = self.code_vectorstore._collection
            results = collection.get()

            # Convert to Documents
            documents = []
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                content = results["documents"][i] if results["documents"] else ""

                # Filter by repo_name if specified
                if repo_name and metadata.get("repo_name") != repo_name:
                    continue

                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error retrieving code chunks: {e}")
            return []

    def _group_chunks_by_feature(
        self, chunks: List[Document]
    ) -> Dict[str, List[Document]]:
        """
        Group code chunks by feature/module (file or directory level).

        This creates less granular groupings than class-level, avoiding
        FAQs for individual error classes, config classes, etc.

        Strategy:
        - Group by file for services, controllers (one FAQ set per file)
        - Group by directory for utilities, errors, models (one FAQ set per directory)
        """
        grouped = {}

        for chunk in chunks:
            file_path = chunk.metadata.get("file_path", "")
            chunk_type = chunk.metadata.get("chunk_type", "")
            name = chunk.metadata.get("name", "")

            # Determine feature/component name based on file location
            feature_name = self._extract_feature_name(file_path, chunk_type, name)

            if not feature_name:
                feature_name = "misc"

            if feature_name not in grouped:
                grouped[feature_name] = []

            grouped[feature_name].append(chunk)

        return grouped

    def _extract_feature_name(self, file_path: str, chunk_type: str, name: str) -> str:
        """
        Extract a meaningful feature name from file path using directory-based grouping.

        Strategy: Group by meaningful directory levels to create cohesive features
        - app/services/carriers/ → Carriers Service
        - app/domain/routing/ → Routing Domain
        - lib/shopify/ → Shopify Library
        - components/CarrierList/ → CarrierList Component
        """
        if not file_path:
            return name or "unknown"

        parts = file_path.split("/")

        # For services: use directory if multiple files, else file name
        if "/services/" in file_path:
            try:
                services_idx = [i for i, p in enumerate(parts) if p == "services"][0]
                # Check if there's a subdirectory after services
                if services_idx + 2 < len(parts):
                    # app/services/carriers/creator.rb → Carriers Service
                    subdir = parts[services_idx + 1]
                    return f"{subdir.capitalize()} Service"
                else:
                    # app/services/carrier_service.rb → CarrierService
                    file_name = (
                        parts[-1]
                        .replace(".rb", "")
                        .replace(".ts", "")
                        .replace(".kt", "")
                    )
                    return "".join(word.capitalize() for word in file_name.split("_"))
            except (IndexError, ValueError):
                pass

        # For controllers: use directory or file name
        if "/controllers/" in file_path:
            try:
                ctrl_idx = [i for i, p in enumerate(parts) if p == "controllers"][0]
                if ctrl_idx + 2 < len(parts):
                    # app/controllers/api/v1/carriers_controller.rb → API V1 Carriers
                    subdirs = parts[ctrl_idx + 1 : -1]
                    return " ".join(s.capitalize() for s in subdirs)
                else:
                    # app/controllers/carriers_controller.rb → Carriers Controller
                    file_name = (
                        parts[-1]
                        .replace("_controller.rb", "")
                        .replace("_controller.ts", "")
                    )
                    return f"{file_name.capitalize()} Controller"
            except (IndexError, ValueError):
                pass

        # For domain layer: group by domain subdirectory (use case level)
        if "/domain/" in file_path:
            try:
                domain_idx = [i for i, p in enumerate(parts) if p == "domain"][0]
                # Get all subdirectories after domain
                domain_subdirs = parts[domain_idx + 1 : -1]
                if len(domain_subdirs) >= 2:
                    # app/domain/routing/services/calculator.rb → Routing Domain
                    return f"{domain_subdirs[0].capitalize()} Domain"
                elif len(domain_subdirs) == 1:
                    # app/domain/routing/calculator.rb → Routing Domain
                    return f"{domain_subdirs[0].capitalize()} Domain"
            except (IndexError, ValueError):
                pass

        # For lib/ files: group by library (top-level directory)
        if file_path.startswith("lib/"):
            if len(parts) >= 2:
                # lib/shopify/... → Shopify Library
                lib_name = parts[1]
                return f"{lib_name.capitalize()} Library"

        # For models: use directory if present
        if "/models/" in file_path:
            try:
                models_idx = [i for i, p in enumerate(parts) if p == "models"][0]
                if models_idx + 2 < len(parts):
                    # app/models/carrier/shipment.rb → Carrier Models
                    subdir = parts[models_idx + 1]
                    return f"{subdir.capitalize()} Models"
            except (IndexError, ValueError):
                pass

        # For React components: use top-level component directory
        if "/components/" in file_path:
            try:
                comp_idx = [i for i, p in enumerate(parts) if p == "components"][0]
                if comp_idx + 1 < len(parts):
                    subdir = parts[comp_idx + 1]
                    if subdir in ["common", "shared", "ui", "layout"]:
                        return f"{subdir.capitalize()} Components"
                    # components/CarrierList/... → CarrierList Component
                    return f"{subdir} Component"
            except (IndexError, ValueError):
                pass

        # For Android: use feature directory
        if file_path.endswith(".kt"):
            # Look for feature directories like ui/, domain/, data/
            if "/ui/" in file_path:
                try:
                    ui_idx = [i for i, p in enumerate(parts) if p == "ui"][0]
                    if ui_idx + 1 < len(parts):
                        # app/ui/carriers/... → Carriers Screen
                        feature = parts[ui_idx + 1]
                        return f"{feature.capitalize()} Screen"
                except (IndexError, ValueError):
                    pass

        # Fallback: use file name (but this should be rare now)
        file_name = parts[-1].replace(".rb", "").replace(".ts", "").replace(".kt", "")
        return "".join(word.capitalize() for word in file_name.split("_"))

    def _filter_by_importance(
        self, grouped_chunks: Dict[str, List[Document]]
    ) -> Dict[str, List[Document]]:
        """Filter components by importance score, excluding low-value code."""
        filtered = {}

        for component_name, chunks in grouped_chunks.items():
            # Calculate importance score
            importance_score = self._calculate_importance_score(component_name, chunks)

            if importance_score >= self.min_importance_score:
                filtered[component_name] = chunks
                logger.info(f"✓ {component_name}: importance={importance_score:.2f}")
            else:
                logger.info(
                    f"✗ {component_name}: importance={importance_score:.2f} (below threshold {self.min_importance_score})"
                )

        return filtered

    def _calculate_importance_score(
        self, component_name: str, chunks: List[Document]
    ) -> float:
        """
        Calculate importance score for a component (0-1 scale).

        Higher scores indicate components that are more valuable for FAQ generation.
        Based on research in code documentation prioritization and importance metrics.
        """
        score = 0.5  # Base score

        if not chunks:
            return 0.0

        # Get file path from first chunk
        file_path = chunks[0].metadata.get("file_path", "")
        chunk_type = chunks[0].metadata.get("chunk_type", "")

        # === DRIVER_ROUTING FILTER ===
        # Only generate FAQs for driver_routing related files
        # if "shipment" not in file_path.lower():
        #     logger.info(f"✗ {component_name}: not shipment related (excluded)")
        #     return 0.0

        # === MINIMUM SIZE REQUIREMENT ===

        # Require minimum chunks for FAQ generation
        # Single classes are too granular for meaningful FAQs
        if len(chunks) < self.min_chunks_per_component:
            logger.info(
                f"✗ {component_name}: too few chunks ({len(chunks)}/{self.min_chunks_per_component}) for FAQ generation"
            )
            return 0.0

        # === EXCLUDE UTILITY CLASSES AND FILES ===

        # Error classes (too specific, not user-facing)
        if component_name.endswith("Error") or component_name.endswith("Exception"):
            return 0.0

        # Config/Configuration classes (too generic)
        if component_name in ["Config", "Configuration", "Settings"]:
            return 0.0

        # Base classes (too abstract)
        if component_name in ["Base", "BaseClass", "Abstract"]:
            return 0.0

        # Utility files and parsers (too low-level)
        utility_keywords = [
            "parser",
            "parsers",
            "serializer",
            "serializers",
            "validator",
            "validators",
            "formatter",
            "formatters",
            "helper",
            "helpers",
            "util",
            "utils",
            "utilities",
            "timezone",
            "timezones",
            "locale",
            "locales",
            "seed",
            "seeds",
            "fixture",
            "fixtures",
            "pagination",
            "paginator",
            "paginators",
            "decorator",
            "decorators",
            "concern",
            "concerns",
            "constant",
            "constants",
            "enum",
            "enums",
        ]

        component_lower = component_name.lower()
        if any(keyword in component_lower for keyword in utility_keywords):
            logger.info(f"✗ {component_name}: utility/helper file (excluded)")
            return 0.0

        # Generic names (too vague)
        if component_lower in ["errors", "models", "resources", "types", "interfaces"]:
            return 0.0

        # Lib/ directories that are utilities (not features)
        if file_path.startswith("lib/") and any(
            x in file_path
            for x in [
                "lib/tasks/",
                "lib/generators/",
                "lib/templates/",
            ]
        ):
            return 0.0

        # === EXCLUSION RULES (heavy penalties) ===

        # Database migrations (very low value for FAQs)
        if self._matches_any_pattern(
            file_path,
            [
                r"db/migrate/",
                r"_migration\.rb$",
                r"^\d{14}_",
            ],
        ) or self._matches_any_pattern(
            component_name,
            [
                r"^\d{14}_",  # Timestamp prefix
                r"Migration$",  # Ends with Migration
            ],
        ):
            return 0.0  # Migrations never need FAQs

        # Test files (tests document themselves)
        if self._matches_any_pattern(
            file_path,
            [
                r"spec/",
                r"test/",
                r"__tests__/",
                r"_spec\.",
                r"_test\.",
                r"\.test\.",
                r"\.spec\.",
            ],
        ):
            return 0.1  # Very low score for tests

        # Configuration files
        if self._matches_any_pattern(
            file_path,
            [
                r"config/",
                r"webpack",
                r"babel",
                r"eslint",
                r"\.config\.",
            ],
        ):
            return 0.1  # Config files rarely need FAQs

        # Generated code
        if self._matches_any_pattern(
            file_path,
            [
                r"generated/",
                r"\.g\.dart$",
                r"\.generated\.",
            ],
        ):
            return 0.0  # Generated code never needs FAQs

        # === BOOST RULES (increase importance) ===

        # Service classes (business logic)
        if self._matches_any_pattern(
            file_path,
            [
                r"app/services/",
                r"services/",
            ],
        ) or component_name.endswith("Service"):
            score += 0.3

        # Controllers/API endpoints (user-facing)
        if self._matches_any_pattern(
            file_path,
            [
                r"app/controllers/",
                r"controllers/",
                r"api/",
            ],
        ) or component_name.endswith("Controller"):
            score += 0.3

        # Models (core business entities)
        if self._matches_any_pattern(
            file_path,
            [
                r"app/models/",
                r"models/",
            ],
        ):
            score += 0.2

        # React components (user-facing UI)
        if (
            self._matches_any_pattern(
                file_path,
                [
                    r"components/",
                ],
            )
            or chunk_type == "interface"
        ):
            score += 0.2

        # Use cases / interactors (important business flows)
        if self._matches_any_pattern(
            file_path,
            [
                r"interactors/",
                r"use_cases/",
                r"useCases/",
            ],
        ):
            score += 0.3

        # === COMPLEXITY METRICS ===

        # Component size (more code = potentially more important)
        num_chunks = len(chunks)
        if num_chunks >= 5:
            score += 0.1  # Large component
        elif num_chunks >= 3:
            score += 0.05  # Medium component

        # Has documentation (suggests it's important)
        has_docstring = any(chunk.metadata.get("docstring") for chunk in chunks)
        if has_docstring:
            score += 0.1

        # Public API (class or module, not private functions)
        if chunk_type in ["class", "module", "object", "interface"]:
            score += 0.1

        # === PENALTY FOR TRIVIAL CODE ===

        # Very short components (likely trivial)
        total_code_length = sum(len(chunk.page_content) for chunk in chunks)
        if total_code_length < 100:
            score -= 0.2  # Too short to be important

        # Clamp score to [0, 1]
        return max(0.0, min(1.0, score))

    def _matches_any_pattern(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the regex patterns."""
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def _generate_faqs_for_component(
        self, component_name: str, chunks: List[Document], max_faqs: int = 3
    ) -> List[FAQ]:
        """Generate FAQs for a logical component."""

        # Build context from code chunks
        context = self._format_code_context(chunks)

        # Extract repo name from first chunk
        repo_name = chunks[0].metadata.get("repo_name", "") if chunks else ""

        # Create FAQ generation prompt
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a technical documentation expert helping developers understand a codebase.
Your task is to generate helpful FAQs that users might ask about the code.

Guidelines:
- Write questions from a developer's or user's perspective
- Focus on "how to" questions, common use cases, and key functionality
- Provide clear, concise answers with technical details
- Reference specific functions/classes when relevant
- Keep answers practical and actionable
- Generate {max_faqs} FAQs maximum""",
                ),
                (
                    "human",
                    """Component: {component_name}
Repository: {repo_name}

Code Context:
{context}

Generate {max_faqs} frequently asked questions (FAQs) about this code component.

Return your response as a JSON array with this structure:
[
  {{
    "question": "How do I...",
    "answer": "You can... by calling function X which...",
    "references": ["function_name", "class_name"]
  }}
]

Only return valid JSON, no other text.""",
                ),
            ]
        )

        # Invoke LLM
        try:
            chain = prompt_template | self.llm
            response = chain.invoke(
                {
                    "component_name": component_name,
                    "repo_name": repo_name,
                    "context": context,
                    "max_faqs": max_faqs,
                }
            )

            # Parse JSON response
            content = response.content.strip()

            # Try to extract JSON if wrapped in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            faqs_json = json.loads(content)

            # Convert to FAQ objects with traceability
            faqs = []
            for faq_data in faqs_json:
                traceability = self._build_traceability(
                    faq_data.get("references", []), chunks
                )

                faq = FAQ(
                    question=faq_data["question"],
                    answer=faq_data["answer"],
                    source_chunks=[chunk.metadata["source"] for chunk in chunks],
                    traceability=traceability,
                    confidence=0.8,
                    tags=[component_name, repo_name],
                )
                faqs.append(faq)

            return faqs

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.info(f"Response content: {response.content}")
            return []
        except Exception as e:
            logger.error(f"Error generating FAQs: {e}")
            return []

    def _format_code_context(self, chunks: List[Document]) -> str:
        """Format code chunks into context string for LLM."""
        context_parts = []

        for chunk in chunks[:10]:  # Limit to first 10 chunks to avoid token limits
            name = chunk.metadata.get("name", "Unknown")
            chunk_type = chunk.metadata.get("chunk_type", "code")
            file_path = chunk.metadata.get("file_path", "")
            signature = chunk.metadata.get("signature", "")
            docstring = chunk.metadata.get("docstring", "")

            context_parts.append(f"--- {chunk_type.title()}: {name} ---")
            context_parts.append(f"File: {file_path}")

            if signature:
                context_parts.append(f"Signature: {signature}")

            if docstring:
                context_parts.append(f"Documentation: {docstring}")

            # Include code content (truncate if too long)
            code = chunk.page_content
            if len(code) > 1000:
                code = code[:1000] + "\n... (truncated)"

            context_parts.append(f"\nCode:\n{code}\n")

        return "\n".join(context_parts)

    def _build_traceability(
        self, references: List[str], chunks: List[Document]
    ) -> List[Dict[str, Any]]:
        """Build traceability links from FAQ to source code."""
        traceability = []

        for ref in references:
            # Find matching chunk by function/class name
            for chunk in chunks:
                if chunk.metadata.get("name", "").lower() == ref.lower():
                    traceability.append(
                        {
                            "type": "code",
                            "source": chunk.metadata.get("source", ""),
                            "file_path": chunk.metadata.get("file_path", ""),
                            "line_start": chunk.metadata.get("line_start", 0),
                            "line_end": chunk.metadata.get("line_end", 0),
                            "name": chunk.metadata.get("name", ""),
                            "repo_name": chunk.metadata.get("repo_name", ""),
                        }
                    )
                    break  # Found match, move to next reference

        return traceability

    def _faqs_already_exist(self, component_name: str, chunks: List[Document]) -> bool:
        """
        Check if FAQs already exist for this component in the vectorstore.

        This enables resume functionality - if the process crashes or is interrupted,
        we can skip components that already have FAQs generated.

        Checks multiple indicators for backwards compatibility:
        - Component name in tags (new format)
        - File paths in source_chunks (old format)
        - Question content matching
        """
        try:
            # Try to load existing FAQ vectorstore
            existing_vectorstore = Chroma(
                persist_directory=self.config.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="synthetic_faqs",
            )

            # Query for FAQs with this component in tags
            collection = existing_vectorstore._collection
            results = collection.get()

            if not results or not results.get("metadatas"):
                return False

            # Extract file paths from current chunks for matching
            chunk_file_paths = set()
            for chunk in chunks:
                file_path = chunk.metadata.get("file_path", "")
                if file_path:
                    chunk_file_paths.add(file_path)

            # Check existing FAQs for matches
            for i, metadata in enumerate(results["metadatas"] or []):
                # Method 1: Check tags (new format)
                tags = metadata.get("tags", "").split(",")
                if component_name in tags:
                    logger.info(f"Found existing FAQ via tags: {component_name}")
                    return True

                # Method 2: Check traceability (contains file paths - works for all FAQs)
                traceability_raw = metadata.get("traceability", "")
                if traceability_raw:
                    try:
                        # Parse traceability JSON
                        traceability = json.loads(traceability_raw)

                        # Extract file paths from traceability
                        if isinstance(traceability, list):
                            for trace in traceability:
                                trace_file_path = trace.get("file_path", "")
                                # Check if any of our file paths match
                                if trace_file_path in chunk_file_paths:
                                    logger.info(
                                        f"Found existing FAQ via traceability file path: {trace_file_path}"
                                    )
                                    return True
                    except (json.JSONDecodeError, TypeError):
                        pass  # Traceability not parseable, skip

                # Method 3: Check question content for component name variations
                question = metadata.get("question", "").lower()
                # Check for component name variations (with/without spaces, underscores)
                component_variations = [
                    component_name.lower(),
                    component_name.lower().replace(" ", ""),
                    component_name.lower().replace(" ", "_"),
                    component_name.lower().replace("_", " "),
                ]
                if any(variant in question for variant in component_variations):
                    logger.info(
                        f"Found existing FAQ via question content: {component_name}"
                    )
                    return True

            return False

        except Exception as e:
            # Collection doesn't exist yet, or error accessing it
            logger.info(f"Could not check existing FAQs: {e}")
            return False

    def _save_faqs_incrementally(self, faqs: List[FAQ]):
        """
        Save FAQs incrementally (after each component).

        This allows resuming if the process is interrupted - already-saved
        FAQs will be skipped on the next run.
        """
        if not faqs:
            return

        try:
            # Try to load existing vectorstore
            try:
                existing_vectorstore = Chroma(
                    persist_directory=self.config.chroma_persist_dir,
                    embedding_function=self.embeddings,
                    collection_name="synthetic_faqs",
                )
                logger.info(f"Appending {len(faqs)} FAQs to existing collection")
            except Exception:
                # Collection doesn't exist yet, will be created
                logger.info(
                    f"Creating new synthetic_faqs collection with {len(faqs)} FAQs"
                )
                existing_vectorstore = None

            # Convert FAQs to documents
            documents = []
            for faq in faqs:
                doc = Document(
                    page_content=f"Q: {faq.question}\n\nA: {faq.answer}",
                    metadata={
                        "question": faq.question,
                        "answer": faq.answer,
                        "traceability": json.dumps(faq.traceability),
                        "confidence": faq.confidence,
                        "tags": ",".join(faq.tags),
                        "source_chunks": json.dumps(
                            faq.source_chunks
                        ),  # Save for resume detection
                        "collection": "synthetic_faqs",
                    },
                )
                documents.append(doc)

            # Add to collection (create or append)
            if existing_vectorstore:
                # Append to existing collection
                existing_vectorstore.add_documents(documents)
            else:
                # Create new collection
                Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.config.chroma_persist_dir,
                    collection_name="synthetic_faqs",
                )

            logger.info(f"✓ Saved {len(faqs)} FAQs incrementally")

        except Exception as e:
            logger.warning(f"Failed to save FAQs incrementally: {e}")
            # Don't fail the whole process if incremental save fails

    def create_faq_vectorstore(self, faqs: List[FAQ]) -> Optional[Chroma]:
        """Create ChromaDB collection for FAQs."""
        if not faqs:
            logger.warning("No FAQs to store")
            return None

        logger.info(f"Creating FAQ vectorstore with {len(faqs)} FAQs")

        documents = []

        for faq in faqs:
            # Format as Q&A for embedding
            doc = Document(
                page_content=f"Q: {faq.question}\n\nA: {faq.answer}",
                metadata={
                    "question": faq.question,
                    "answer": faq.answer,
                    "traceability": json.dumps(faq.traceability),
                    "confidence": faq.confidence,
                    "tags": ",".join(faq.tags),
                    "collection": "synthetic_faqs",
                },
            )
            documents.append(doc)

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.config.chroma_persist_dir,
            collection_name="synthetic_faqs",
        )

        logger.info(
            f"FAQ vectorstore created: {vectorstore._collection.count()} FAQs indexed"
        )

        return vectorstore


class UserStoryGenerator:
    """Generate User Stories from code (Agile format)."""

    def __init__(self, config, code_vectorstore: Optional[Chroma] = None):
        self.config = config
        self.code_vectorstore = code_vectorstore
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=0.3,
            api_key=config.openai_api_key,
        )

    def generate_user_stories(
        self, code_chunks: List[Document]
    ) -> List[Dict[str, Any]]:
        """Generate user stories in format: 'As a [user], I want [goal], so that [benefit]'."""

        if not code_chunks:
            return []

        # Build context from code
        context = self._format_code_context(code_chunks)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a product manager creating user stories from code.
Generate user stories in the Agile format:
- As a [user type]
- I want to [action/goal]
- So that [business value/benefit]

Include references to relevant code.""",
                ),
                (
                    "human",
                    """Code Context:
{context}

Generate 3-5 user stories for this code. Return as JSON:
[
  {{
    "as_a": "user type",
    "i_want": "action/goal",
    "so_that": "benefit",
    "references": ["function_name"]
  }}
]""",
                ),
            ]
        )

        try:
            chain = prompt_template | self.llm
            response = chain.invoke({"context": context})

            # Parse JSON
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            stories = json.loads(content)
            return stories

        except Exception as e:
            logger.error(f"Error generating user stories: {e}")
            return []

    def _format_code_context(self, chunks: List[Document]) -> str:
        """Format code chunks into context string."""
        context_parts = []

        for chunk in chunks[:5]:  # Limit to avoid token limits
            signature = chunk.metadata.get("signature", "")

            context_parts.append(f"{signature}")
            context_parts.append(chunk.page_content[:500])  # First 500 chars
            context_parts.append("")

        return "\n".join(context_parts)


def run_faq_generator(
    config, repo_configs: List[dict], force_regen: bool = False
) -> Optional[Chroma]:
    """
    Main entry point for FAQ generation.

    Args:
        config: Configuration object
        repo_configs: List of repository configurations to process. If None, processes all repos from config.
        force_regen: If True, regenerate all FAQs even if they exist

    Returns:
        Chroma vectorstore with all FAQs across repositories
    """
    logger.info("Loading code vectorstore...")

    # Load existing code vectorstore
    try:
        code_vectorstore = Chroma(
            persist_directory=config.chroma_persist_dir,
            embedding_function=OpenAIEmbeddings(
                model=config.embedding_model, api_key=config.openai_api_key
            ),
            collection_name="source_code",
        )

        logger.info(f"Loaded {code_vectorstore._collection.count()} code chunks")

    except Exception as e:
        logger.error(f"Failed to load code vectorstore: {e}")
        logger.info("Please run --index-code first to create the code index")
        return None

    # Initialize generator once (it handles incremental saving)
    generator = FAQGenerator(config, code_vectorstore, force_regen=force_regen)

    # Process each repository
    all_faqs = []
    for repo_config in repo_configs:
        repo_name = repo_config["name"]
        logger.info(f"Processing repository: {repo_name}")

        # Generate FAQs for this repo (incrementally saved)
        faqs = generator.generate_faqs_for_codebase(repo_name=repo_name)
        all_faqs.extend(faqs)

        logger.info(f"Generated {len(faqs)} FAQs for {repo_name}")

    logger.info(f"Total FAQs generated across all repositories: {len(all_faqs)}")

    if not all_faqs:
        logger.warning("No FAQs generated")
        return None

    # Load the final vectorstore (all repos are already saved incrementally)
    try:
        faq_vectorstore = Chroma(
            persist_directory=config.chroma_persist_dir,
            embedding_function=OpenAIEmbeddings(
                model=config.embedding_model, api_key=config.openai_api_key
            ),
            collection_name="synthetic_faqs",
        )
        logger.info(
            f"Final FAQ vectorstore: {faq_vectorstore._collection.count()} FAQs"
        )
        return faq_vectorstore
    except Exception as e:
        logger.error(f"Failed to load FAQ vectorstore: {e}")
        return None
