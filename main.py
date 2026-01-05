#!/usr/bin/env python3
"""
Confluence RAG Slack Bot - Main Entry Point

A Slack chatbot that answers questions using knowledge from your Confluence space.

Usage:
    python main.py              # Start the bot (loads existing index)
    python main.py --reindex    # Rebuild the index from Confluence and start bot
    python main.py --index-only # Only rebuild the index, don't start bot
"""

import argparse
import logging
import sys

from config import load_config
from indexer import run_indexer
from rag_engine import RAGEngine
from slack_bot import ConfluenceSlackBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Confluence RAG Slack Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild the vector index from Confluence"
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only rebuild the index, don't start the bot"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Run indexer
        force_reindex = args.reindex or args.index_only
        logger.info(f"Initializing vector store (force_reindex={force_reindex})...")
        vectorstore = run_indexer(config, force_reindex=force_reindex)
        
        # Check if we have any documents
        doc_count = vectorstore._collection.count()
        logger.info(f"Vector store contains {doc_count} document chunks")
        
        if doc_count == 0:
            logger.warning("Vector store is empty! Run with --reindex to fetch documents from Confluence.")
        
        if args.index_only:
            logger.info("Index-only mode: exiting without starting bot")
            print(f"\n✅ Indexing complete! {doc_count} chunks indexed.")
            return 0
        
        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine(config, vectorstore)
        
        # Start Slack bot
        logger.info("Starting Slack bot...")
        print("\n" + "="*50)
        print("🤖 Confluence RAG Slack Bot is running!")
        print("="*50)
        print(f"📚 Knowledge base: {config.confluence_space_key}")
        print(f"📊 Indexed chunks: {doc_count}")
        print(f"🧠 LLM model: {config.llm_model}")
        print("="*50)
        print("\nThe bot is now listening for messages in Slack.")
        print("Press Ctrl+C to stop.\n")
        
        bot = ConfluenceSlackBot(config, rag_engine)
        bot.start()
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\n❌ Configuration error: {e}")
        print("\nMake sure you have created a .env file with all required variables.")
        print("See .env.example for the template.")
        return 1
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        print("\n\n👋 Bot stopped. Goodbye!")
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)

