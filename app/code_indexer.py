"""
Code Indexer: Parse codebases using tree-sitter into semantic chunks.

This module provides functionality to index source code repositories by parsing
them with tree-sitter to extract functions, classes, and methods as semantic chunks.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tree_sitter_javascript as ts_javascript
import tree_sitter_kotlin as ts_kotlin

# Tree-sitter imports
import tree_sitter_ruby as ts_ruby
import tree_sitter_typescript as ts_typescript
from langchain_chroma import Chroma

# LangChain imports
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a semantic code chunk (function, class, method)."""

    content: str
    chunk_type: str  # "function", "class", "method", "module"
    name: str
    file_path: str
    line_start: int
    line_end: int
    language: str
    signature: str
    docstring: Optional[str] = None
    repo_name: str = ""


class LanguageParser:
    """Base class for language-specific AST parsing."""

    def __init__(self, language: Language, repo_name: str = ""):
        self.language = language
        self.parser = Parser(language)
        self.repo_name = repo_name
        self.current_file = ""
        self.source_code = b""

    def parse_file(self, file_path: str) -> List[CodeChunk]:
        """Parse a file and extract semantic chunks."""
        try:
            with open(file_path, "rb") as f:
                self.source_code = f.read()

            self.current_file = file_path
            tree = self.parser.parse(self.source_code)

            chunks = []
            chunks.extend(self.extract_functions(tree.root_node))
            chunks.extend(self.extract_classes(tree.root_node))

            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
            return chunks

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []

    def extract_functions(self, root_node: Node) -> List[CodeChunk]:
        """Extract function definitions from AST. Override in subclasses."""
        return []

    def extract_classes(self, root_node: Node) -> List[CodeChunk]:
        """Extract class definitions from AST. Override in subclasses."""
        return []

    def get_node_text(self, node: Node) -> str:
        """Get text content of a node."""
        return self.source_code[node.start_byte : node.end_byte].decode(
            "utf-8", errors="ignore"
        )

    def traverse_tree(self, node: Node, node_types: List[str]) -> List[Node]:
        """Traverse tree and collect nodes of specific types."""
        results = []

        def traverse(n: Node):
            if n.type in node_types:
                results.append(n)
            for child in n.children:
                traverse(child)

        traverse(node)
        return results


class RubyParser(LanguageParser):
    """Ruby-specific parser for Rails applications."""

    def __init__(self, repo_name: str = ""):
        super().__init__(Language(ts_ruby.language()), repo_name)

    def extract_functions(self, root_node: Node) -> List[CodeChunk]:
        """Extract Ruby method definitions."""
        chunks = []

        # Find all method nodes
        method_nodes = self.traverse_tree(root_node, ["method", "singleton_method"])

        for node in method_nodes:
            try:
                # Extract method name
                name_node = node.child_by_field_name("name")
                if not name_node:
                    continue

                name = self.get_node_text(name_node)

                # Extract parameters for signature
                params_node = node.child_by_field_name("parameters")
                params = self.get_node_text(params_node) if params_node else "()"

                signature = f"def {name}{params}"

                # Extract method body
                content = self.get_node_text(node)

                # Try to extract documentation (comment above method)
                docstring = self._extract_ruby_doc(node)

                chunks.append(
                    CodeChunk(
                        content=content,
                        chunk_type="method"
                        if node.type == "method"
                        else "singleton_method",
                        name=name,
                        file_path=self.current_file,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language="ruby",
                        signature=signature,
                        docstring=docstring,
                        repo_name=self.repo_name,
                    )
                )

            except Exception as e:
                logger.warning(f"Error extracting Ruby method: {e}")
                continue

        return chunks

    def extract_classes(self, root_node: Node) -> List[CodeChunk]:
        """Extract Ruby class and module definitions."""
        chunks = []

        # Find all class and module nodes
        class_nodes = self.traverse_tree(root_node, ["class", "module"])

        for node in class_nodes:
            try:
                # Extract class/module name
                name_node = node.child_by_field_name("name")
                if not name_node:
                    continue

                name = self.get_node_text(name_node)

                # Extract superclass if present
                superclass_node = node.child_by_field_name("superclass")
                superclass = (
                    f" < {self.get_node_text(superclass_node)}"
                    if superclass_node
                    else ""
                )

                chunk_type = "class" if node.type == "class" else "module"
                signature = f"{chunk_type} {name}{superclass}"

                # Extract full class/module body
                content = self.get_node_text(node)

                # Extract documentation
                docstring = self._extract_ruby_doc(node)

                chunks.append(
                    CodeChunk(
                        content=content,
                        chunk_type=chunk_type,
                        name=name,
                        file_path=self.current_file,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language="ruby",
                        signature=signature,
                        docstring=docstring,
                        repo_name=self.repo_name,
                    )
                )

            except Exception as e:
                logger.warning(f"Error extracting Ruby class: {e}")
                continue

        return chunks

    def _extract_ruby_doc(self, node: Node) -> Optional[str]:
        """Extract documentation comments above a node."""
        # Look for comments in previous siblings
        parent = node.parent
        if not parent:
            return None

        try:
            node_index = parent.children.index(node)
            if node_index > 0:
                prev_sibling = parent.children[node_index - 1]
                if prev_sibling.type == "comment":
                    return self.get_node_text(prev_sibling).strip("# ").strip()
        except Exception:
            pass

        return None


class TypeScriptParser(LanguageParser):
    """TypeScript/JavaScript parser for frontend applications."""

    def __init__(self, repo_name: str = "", language_variant: str = "typescript"):
        if language_variant == "typescript":
            lang = Language(ts_typescript.language_typescript())
        else:
            lang = Language(ts_javascript.language())
        super().__init__(lang, repo_name)

    def extract_functions(self, root_node: Node) -> List[CodeChunk]:
        """Extract TypeScript/JavaScript function definitions."""
        chunks = []

        # Find all function types
        function_nodes = self.traverse_tree(
            root_node,
            [
                "function_declaration",
                "arrow_function",
                "method_definition",
                "function_expression",
            ],
        )

        for node in function_nodes:
            try:
                name = self._extract_ts_function_name(node)
                if not name:
                    continue

                # Extract parameters
                params = self._extract_ts_params(node)
                signature = f"function {name}({params})"

                # Extract function body
                content = self.get_node_text(node)

                # Extract JSDoc
                docstring = self._extract_jsdoc(node)

                chunks.append(
                    CodeChunk(
                        content=content,
                        chunk_type=node.type,
                        name=name,
                        file_path=self.current_file,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language="typescript",
                        signature=signature,
                        docstring=docstring,
                        repo_name=self.repo_name,
                    )
                )

            except Exception as e:
                logger.warning(f"Error extracting TypeScript function: {e}")
                continue

        return chunks

    def extract_classes(self, root_node: Node) -> List[CodeChunk]:
        """Extract TypeScript/JavaScript class definitions."""
        chunks = []

        # Find all class and interface nodes
        class_nodes = self.traverse_tree(
            root_node, ["class_declaration", "interface_declaration"]
        )

        for node in class_nodes:
            try:
                name_node = node.child_by_field_name("name")
                if not name_node:
                    continue

                name = self.get_node_text(name_node)

                chunk_type = (
                    "class" if node.type == "class_declaration" else "interface"
                )
                signature = f"{chunk_type} {name}"

                # Extract full class/interface body
                content = self.get_node_text(node)

                # Extract JSDoc
                docstring = self._extract_jsdoc(node)

                chunks.append(
                    CodeChunk(
                        content=content,
                        chunk_type=chunk_type,
                        name=name,
                        file_path=self.current_file,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language="typescript",
                        signature=signature,
                        docstring=docstring,
                        repo_name=self.repo_name,
                    )
                )

            except Exception as e:
                logger.warning(f"Error extracting TypeScript class: {e}")
                continue

        return chunks

    def _extract_ts_function_name(self, node: Node) -> Optional[str]:
        """Extract function name from various TypeScript function types."""
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            return self.get_node_text(name_node) if name_node else None
        elif node.type == "method_definition":
            name_node = node.child_by_field_name("name")
            return self.get_node_text(name_node) if name_node else None
        elif node.type == "arrow_function":
            # Arrow functions might be in variable declarations
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                name_node = parent.child_by_field_name("name")
                return self.get_node_text(name_node) if name_node else None
        return None

    def _extract_ts_params(self, node: Node) -> str:
        """Extract function parameters as string."""
        params_node = node.child_by_field_name("parameters")
        if params_node:
            return self.get_node_text(params_node).strip("()").strip()
        return ""

    def _extract_jsdoc(self, node: Node) -> Optional[str]:
        """Extract JSDoc comment above a node."""
        parent = node.parent
        if not parent:
            return None

        try:
            node_index = parent.children.index(node)
            if node_index > 0:
                prev_sibling = parent.children[node_index - 1]
                if prev_sibling.type == "comment" and "/**" in self.get_node_text(
                    prev_sibling
                ):
                    return self.get_node_text(prev_sibling).strip()
        except Exception:
            pass

        return None


class KotlinParser(LanguageParser):
    """Kotlin parser for Android applications."""

    def __init__(self, repo_name: str = ""):
        super().__init__(Language(ts_kotlin.language()), repo_name)

    def extract_functions(self, root_node: Node) -> List[CodeChunk]:
        """Extract Kotlin function definitions."""
        chunks = []

        # Find all function nodes
        function_nodes = self.traverse_tree(root_node, ["function_declaration"])

        for node in function_nodes:
            try:
                # Extract function name by searching for 'identifier' child node
                name_node = None
                params_node = None

                for child in node.named_children:
                    if child.type == "identifier":
                        name_node = child
                    elif child.type == "function_value_parameters":
                        params_node = child

                if not name_node:
                    continue

                name = self.get_node_text(name_node)

                # Extract parameters
                params = self.get_node_text(params_node) if params_node else "()"

                signature = f"fun {name}{params}"

                # Extract function body
                content = self.get_node_text(node)

                # Extract KDoc
                docstring = self._extract_kdoc(node)

                chunks.append(
                    CodeChunk(
                        content=content,
                        chunk_type="function",
                        name=name,
                        file_path=self.current_file,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language="kotlin",
                        signature=signature,
                        docstring=docstring,
                        repo_name=self.repo_name,
                    )
                )

            except Exception as e:
                logger.warning(f"Error extracting Kotlin function: {e}")
                continue

        return chunks

    def extract_classes(self, root_node: Node) -> List[CodeChunk]:
        """Extract Kotlin class, object, and data class definitions."""
        chunks = []

        # Find all class-like nodes
        class_nodes = self.traverse_tree(
            root_node,
            [
                "class_declaration",
                "object_declaration",
            ],
        )

        for node in class_nodes:
            try:
                # Extract class name by searching for 'identifier' child node
                name_node = None
                for child in node.named_children:
                    if child.type == "identifier":
                        name_node = child
                        break

                if not name_node:
                    continue

                name = self.get_node_text(name_node)

                chunk_type = "object" if node.type == "object_declaration" else "class"
                signature = f"{chunk_type} {name}"

                # Extract full class/object body
                content = self.get_node_text(node)

                # Extract KDoc
                docstring = self._extract_kdoc(node)

                chunks.append(
                    CodeChunk(
                        content=content,
                        chunk_type=chunk_type,
                        name=name,
                        file_path=self.current_file,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language="kotlin",
                        signature=signature,
                        docstring=docstring,
                        repo_name=self.repo_name,
                    )
                )

            except Exception as e:
                logger.warning(f"Error extracting Kotlin class: {e}")
                continue

        return chunks

    def _extract_kdoc(self, node: Node) -> Optional[str]:
        """Extract KDoc comment above a node."""
        parent = node.parent
        if not parent:
            return None

        try:
            node_index = parent.children.index(node)
            if node_index > 0:
                prev_sibling = parent.children[node_index - 1]
                if prev_sibling.type == "comment" and "/**" in self.get_node_text(
                    prev_sibling
                ):
                    return self.get_node_text(prev_sibling).strip()
        except Exception:
            pass

        return None


class CodeIndexer:
    """Main indexer for code repositories."""

    def __init__(self, config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key,  # pyright: ignore[reportCallIssue]
        )

    def index_repositories(self, repo_configs: List[dict]) -> Optional[Chroma]:
        """Index multiple repositories."""
        logger.info(f"Indexing {len(repo_configs)} repositories...")

        all_documents = []

        for repo_config in repo_configs:
            repo_path = repo_config["path"]
            repo_name = repo_config["name"]
            language = repo_config.get("language", "")
            exclude_patterns = repo_config.get("exclude_patterns", [])

            logger.info(f"Indexing repository: {repo_name} ({repo_path})")

            documents = self.index_repository(
                repo_path, repo_name, language, exclude_patterns
            )
            all_documents.extend(documents)

            logger.info(f"Extracted {len(documents)} chunks from {repo_name}")

        logger.info(f"Total chunks extracted: {len(all_documents)}")

        # Create vector store
        return self.create_vector_store(all_documents, collection_name="source_code")

    def index_repository(
        self,
        repo_path: str,
        repo_name: str,
        language: str = "",
        exclude_patterns: List[str] = [],
    ) -> List[Document]:
        """Index a single repository."""
        if exclude_patterns is None:
            exclude_patterns = []

        # Initialize appropriate parser
        parser = self._get_parser(language, repo_name)
        if not parser:
            logger.warning(f"No parser available for language: {language}")
            return []

        # Discover source files
        source_files = self.discover_source_files(repo_path, language, exclude_patterns)
        logger.info(f"Found {len(source_files)} source files in {repo_name}")

        # Parse all files
        all_chunks = []
        for file_path in source_files:
            chunks = parser.parse_file(file_path)
            all_chunks.extend(chunks)

        # Convert to LangChain Documents
        documents = []
        for chunk in all_chunks:
            # Make file path relative to repo root
            relative_path = str(Path(chunk.file_path).relative_to(repo_path))

            doc = Document(
                page_content=chunk.content,
                metadata={
                    "source": f"{repo_name}:{relative_path}:{chunk.line_start}",
                    "chunk_type": chunk.chunk_type,
                    "name": chunk.name,
                    "file_path": relative_path,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "language": chunk.language,
                    "signature": chunk.signature,
                    "repo_name": repo_name,
                    "collection": "source_code",
                },
            )

            if chunk.docstring:
                doc.metadata["docstring"] = chunk.docstring

            documents.append(doc)

        return documents

    def _get_parser(self, language: str, repo_name: str) -> Optional[LanguageParser]:
        """Get appropriate parser for language."""
        parsers = {
            "ruby": RubyParser(repo_name),
            "typescript": TypeScriptParser(repo_name, "typescript"),
            "javascript": TypeScriptParser(repo_name, "javascript"),
            "kotlin": KotlinParser(repo_name),
        }
        return parsers.get(language.lower())

    def discover_source_files(
        self, repo_path: str, language: str, exclude_patterns: List[str]
    ) -> List[str]:
        """Discover source files, respecting exclude patterns."""
        extensions = {
            "ruby": [".rb"],
            "typescript": [".ts", ".tsx"],
            "javascript": [".js", ".jsx"],
            "kotlin": [".kt"],
        }

        file_extensions = extensions.get(language.lower(), [])
        if not file_extensions:
            return []

        repo_path_obj = Path(repo_path)
        source_files = []

        for ext in file_extensions:
            # Use glob to find all files with extension
            for file_path in repo_path_obj.rglob(f"*{ext}"):
                # Skip if matches exclude pattern
                relative_path = str(file_path.relative_to(repo_path_obj))

                if any(pattern in relative_path for pattern in exclude_patterns):
                    continue

                source_files.append(str(file_path))

        return source_files

    def create_vector_store(
        self, documents: List[Document], collection_name: str
    ) -> Optional[Chroma]:
        """Create ChromaDB vector store from documents."""
        if not documents:
            logger.warning("No documents to index")
            return None

        logger.info(
            f"Creating vector store '{collection_name}' with {len(documents)} documents"
        )

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.config.chroma_persist_dir,
            collection_name=collection_name,
        )

        logger.info(
            f"Vector store created: {vectorstore._collection.count()} documents indexed"
        )

        return vectorstore


def run_code_indexer(config, repo_configs: List[dict]) -> Optional[Chroma]:
    """Main entry point for code indexing."""
    return CodeIndexer(config).index_repositories(repo_configs)
