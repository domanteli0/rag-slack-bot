"""RAG Engine - retrieves context and generates answers using LLM."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG engine."""
    answer: str
    sources: List[dict]
    query: str


class RAGEngine:
    """Retrieval-Augmented Generation engine for answering questions."""
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context from Confluence documentation.

Instructions:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information to answer, say so clearly
- Be concise but thorough
- If relevant, mention which document(s) the information comes from
- Use markdown formatting for better readability

Context from Confluence:
{context}
"""
    
    USER_PROMPT = """Question: {question}

Please provide a helpful answer based on the context above."""

    def __init__(self, config: Config, vectorstore: Optional[Chroma] = None):
        self.config = config
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.llm_model,
            openai_api_key=config.openai_api_key,
            temperature=0.1  # Low temperature for factual responses
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key
        )
        
        # Load or use provided vectorstore
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = Chroma(
                persist_directory=config.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="confluence_docs"
            )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.USER_PROMPT)
        ])
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve relevant documents for a query."""
        k = k or self.config.top_k_results
        
        logger.info(f"Retrieving top {k} documents for query: {query[:50]}...")
        docs = self.vectorstore.similarity_search(query, k=k)
        
        logger.info(f"Retrieved {len(docs)} documents")
        return docs
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", "Unknown")
            source = doc.metadata.get("source", "")
            content = doc.page_content
            
            context_parts.append(
                f"--- Document {i}: {title} ---\n"
                f"Source: {source}\n\n"
                f"{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, documents: List[Document]) -> List[dict]:
        """Extract source information from documents."""
        sources = []
        seen_pages = set()
        
        for doc in documents:
            page_id = doc.metadata.get("page_id")
            if page_id in seen_pages:
                continue
            seen_pages.add(page_id)
            
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "url": doc.metadata.get("source", ""),
                "page_id": page_id
            })
        
        return sources
    
    def query(self, question: str) -> RAGResponse:
        """Process a question and return an answer with sources."""
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant documents
        documents = self.retrieve(question)
        
        if not documents:
            return RAGResponse(
                answer="I couldn't find any relevant information in the Confluence documentation to answer your question.",
                sources=[],
                query=question
            )
        
        # Format context
        context = self._format_context(documents)
        
        # Generate answer
        logger.info("Generating answer with LLM...")
        messages = self.prompt.format_messages(
            context=context,
            question=question
        )
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Extract sources
        sources = self._extract_sources(documents)
        
        logger.info("Answer generated successfully")
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question
        )
    
    def format_slack_response(self, rag_response: RAGResponse) -> str:
        """Format RAG response for Slack display."""
        parts = [rag_response.answer]
        
        if rag_response.sources:
            parts.append("\n\n📚 *Sources:*")
            for source in rag_response.sources:
                title = source.get("title", "Unknown")
                url = source.get("url", "")
                if url:
                    parts.append(f"• <{url}|{title}>")
                else:
                    parts.append(f"• {title}")
        
        return "\n".join(parts)


if __name__ == "__main__":
    from config import load_config
    
    config = load_config()
    engine = RAGEngine(config)
    
    # Test query
    question = input("Ask a question: ")
    response = engine.query(question)
    
    print("\n" + "="*50)
    print("ANSWER:")
    print(response.answer)
    print("\nSOURCES:")
    for source in response.sources:
        print(f"  - {source['title']}: {source['url']}")

