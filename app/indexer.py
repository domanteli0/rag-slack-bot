"""Confluence indexer - fetches pages and creates vector embeddings."""

import logging
from datetime import datetime
from typing import List, Optional

from atlassian import Confluence
from bs4 import BeautifulSoup
from config import Config
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfluenceIndexer:
    """Fetches Confluence pages and indexes them into a vector database."""

    def __init__(self, config: Config):
        self.config = config
        self.confluence = Confluence(
            url=config.confluence_url,
            username=config.confluence_username,
            password=config.confluence_api_token,
            cloud=True,
        )
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model, api_key=config.openai_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from Confluence HTML content."""
        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, "lxml")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()

        # Get text and clean up whitespace
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)

        return text

    def fetch_pages_from_space(self, space_key: str) -> List[Document]:
        """Fetch all pages from a single Confluence space."""
        documents = []

        logger.info(f"Fetching pages from Confluence space: {space_key}")

        # Get all pages in the space
        start = 0
        limit = 50

        while True:
            pages = self.confluence.get_all_pages_from_space(
                space=space_key,
                start=start,
                limit=limit,
                expand="history,space,version,body.storage,history.createdDate",
            )

            if not pages:
                break

            for page in pages:
                page_id = page["id"]
                title = page["title"]

                # Get page content
                html_content = page.get("body", {}).get("storage", {}).get("value", "")
                text_content = self._extract_text_from_html(html_content)

                if not text_content.strip():
                    logger.info(f"Skipping empty page: {title}")
                    continue

                # Extract date information
                version_info = page.get("version", {})
                history_info = page.get("history", {})

                # Last modified date
                last_modified = version_info.get("when", "")

                # Creation date
                created_date = history_info.get("createdDate", "")

                # Parse dates for age calculation
                last_modified_timestamp = self._parse_confluence_date(last_modified)
                created_timestamp = self._parse_confluence_date(created_date)

                # Build page URL
                page_url = f"{self.config.confluence_url}/wiki/spaces/{space_key}/pages/{page_id}"

                # Create document with metadata including dates
                doc = Document(
                    page_content=f"# {title}\n\n{text_content}",
                    metadata={
                        "source": page_url,
                        "title": title,
                        "page_id": page_id,
                        "space_key": space_key,
                        "created_date": created_date,
                        "created_timestamp": created_timestamp,
                        "last_modified": last_modified,
                        "last_modified_timestamp": last_modified_timestamp,
                        "version": version_info.get("number", 1),
                    },
                )
                documents.append(doc)
                logger.info(
                    f"Fetched: {title} (modified: {last_modified[:10] if last_modified else 'unknown'})"
                )

            start += limit

            if len(pages) < limit:
                break

        logger.info(f"Total pages fetched from {space_key}: {len(documents)}")
        return documents

    def _parse_confluence_date(self, date_str: str) -> float:
        """Parse Confluence date string to Unix timestamp.

        Returns 0 if parsing fails.
        """
        if not date_str:
            return 0.0

        try:
            # Confluence dates are typically in ISO 8601 format
            # Example: "2024-01-15T10:30:45.123Z"
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_str}': {e}")
            return 0.0

    def fetch_pages(self, space_keys: Optional[List[str]] = None) -> List[Document]:
        """Fetch all pages from multiple Confluence spaces."""
        space_keys = space_keys or self.config.confluence_space_keys
        all_documents = []

        logger.info(
            f"Fetching pages from {len(space_keys)} Confluence space(s): {', '.join(space_keys)}"
        )

        for space_key in space_keys:
            try:
                documents = self.fetch_pages_from_space(space_key)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error fetching pages from space {space_key}: {e}")
                continue

        logger.info(f"Total pages fetched across all spaces: {len(all_documents)}")
        return all_documents

    def create_index(
        self, documents: Optional[List[Document]] = None, force_replace: bool = False
    ) -> Chroma:
        """Create or update the vector index from documents.

        Args:
            documents: Documents to index. If None, fetches from Confluence.
            force_replace: If True, deletes existing collection before creating new one.
        """
        if documents is None:
            documents = self.fetch_pages()

        if not documents:
            raise ValueError("No documents to index. Check your Confluence space.")

        # If force_replace, delete existing collection first
        if force_replace:
            try:
                import chromadb

                client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)
                logger.info(
                    "Force replace enabled - deleting existing confluence_docs collection..."
                )
                client.delete_collection("confluence_docs")
                logger.info("Existing collection deleted successfully")
            except Exception as e:
                # Collection might not exist, which is fine
                logger.info(f"No existing collection to delete (this is fine): {e}")

        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Create vector store
        logger.info("Creating vector embeddings and storing in ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.config.chroma_persist_dir,
            collection_name="confluence_docs",
        )

        logger.info(f"Index created and persisted to {self.config.chroma_persist_dir}")
        return vectorstore

    def load_existing_index(self) -> Chroma:
        """Load an existing vector index from disk."""
        logger.info(f"Loading existing index from {self.config.chroma_persist_dir}")
        return Chroma(
            persist_directory=self.config.chroma_persist_dir,
            embedding_function=self.embeddings,
            collection_name="confluence_docs",
        )


def run_indexer(config: Config, force_reindex: bool = False) -> Chroma:
    """Run the indexer - creates new index or loads existing one.

    Args:
        config: Configuration object
        force_reindex: If True, deletes existing collection and rebuilds from scratch
    """
    import os

    indexer = ConfluenceIndexer(config)

    # Check if index already exists
    index_exists = os.path.exists(config.chroma_persist_dir) and os.listdir(
        config.chroma_persist_dir
    )

    if index_exists and not force_reindex:
        logger.info("Loading existing index...")
        return indexer.load_existing_index()
    else:
        if force_reindex:
            logger.info(
                "Force reindex enabled - will replace existing Confluence collection..."
            )
        else:
            logger.info("Creating new index from Confluence...")
        return indexer.create_index(force_replace=force_reindex)


if __name__ == "__main__":
    from config import load_config

    config = load_config()
    vectorstore = run_indexer(config, force_reindex=True)
    print(
        f"Indexing complete! Collection has {vectorstore._collection.count()} chunks."
    )
