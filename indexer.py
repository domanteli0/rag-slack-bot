"""Confluence indexer - fetches pages and creates vector embeddings."""

import logging
from typing import List, Optional

from atlassian import Confluence
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from config import Config

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
            cloud=True
        )
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
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
    
    def fetch_pages(self, space_key: Optional[str] = None) -> List[Document]:
        """Fetch all pages from a Confluence space."""
        space_key = space_key or self.config.confluence_space_key
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
                expand="body.storage,version,ancestors"
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
                    logger.debug(f"Skipping empty page: {title}")
                    continue
                
                # Build page URL
                page_url = f"{self.config.confluence_url}/wiki/spaces/{space_key}/pages/{page_id}"
                
                # Create document with metadata
                doc = Document(
                    page_content=f"# {title}\n\n{text_content}",
                    metadata={
                        "source": page_url,
                        "title": title,
                        "page_id": page_id,
                        "space_key": space_key,
                    }
                )
                documents.append(doc)
                logger.info(f"Fetched: {title}")
            
            start += limit
            
            if len(pages) < limit:
                break
        
        logger.info(f"Total pages fetched: {len(documents)}")
        return documents
    
    def create_index(self, documents: Optional[List[Document]] = None) -> Chroma:
        """Create or update the vector index from documents."""
        if documents is None:
            documents = self.fetch_pages()
        
        if not documents:
            raise ValueError("No documents to index. Check your Confluence space.")
        
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
            collection_name="confluence_docs"
        )
        
        logger.info(f"Index created and persisted to {self.config.chroma_persist_dir}")
        return vectorstore
    
    def load_existing_index(self) -> Chroma:
        """Load an existing vector index from disk."""
        logger.info(f"Loading existing index from {self.config.chroma_persist_dir}")
        return Chroma(
            persist_directory=self.config.chroma_persist_dir,
            embedding_function=self.embeddings,
            collection_name="confluence_docs"
        )


def run_indexer(config: Config, force_reindex: bool = False) -> Chroma:
    """Run the indexer - creates new index or loads existing one."""
    import os
    
    indexer = ConfluenceIndexer(config)
    
    # Check if index already exists
    index_exists = os.path.exists(config.chroma_persist_dir) and os.listdir(config.chroma_persist_dir)
    
    if index_exists and not force_reindex:
        logger.info("Loading existing index...")
        return indexer.load_existing_index()
    else:
        logger.info("Creating new index from Confluence...")
        return indexer.create_index()


if __name__ == "__main__":
    from config import load_config
    
    config = load_config()
    vectorstore = run_indexer(config, force_reindex=True)
    print(f"Indexing complete! Collection has {vectorstore._collection.count()} chunks.")

