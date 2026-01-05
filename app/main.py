#!/usr/bin/env python3
"""
Confluence RAG Slack Bot - Main Entry Point

A Slack chatbot that answers questions using knowledge from your Confluence space,
codebase, and generated FAQs.

Usage:
    python main.py --start-bot                  # Start bot
    python main.py --reindex-confluence         # Rebuild Confluence index
    python main.py --index-code                 # Index source code repositories
    python main.py --generate-faqs              # Generate FAQs from indexed code
"""

import argparse
import logging
import os
import sys

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from code_indexer import run_code_indexer
from config import load_config
from faq_generator import run_faq_generator
from indexer import run_indexer
from rag_engine import RAGEngine
from slack_bot import ConfluenceSlackBot

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Confluence RAG Slack Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-r",
        "--reindex-confluence",
        action="store_true",
        help="Force rebuild the vector index from Confluence",
    )
    parser.add_argument(
        "-b",
        "--start-bot",
        action="store_true",
        help="Start the bot",
    )
    parser.add_argument(
        "--index-code",
        action="store_true",
        help="Index source code from configured repositories using tree-sitter",
    )
    parser.add_argument(
        "--generate-faqs",
        action="store_true",
        help="Generate FAQs and User Stories from indexed code",
    )
    parser.add_argument(
        "--force-faq-regen",
        action="store_true",
        help="Force regeneration of FAQs (skip resume, regenerate all)",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Code indexing workflow
        if args.index_code:
            logger.info("=" * 60)
            logger.info("CODE INDEXING WORKFLOW")
            logger.info("=" * 60)

            # Get repository configurations
            repo_configs = config.repositories

            if not repo_configs:
                logger.error("No repositories found to index!")
                if args.repos:
                    logger.error(f"Requested repos: {args.repos}")
                    logger.error(
                        f"Available repos: {[r['name'] for r in config.repositories]}"
                    )
                return 1

            # Run code indexer
            code_vectorstore = run_code_indexer(config, repo_configs)

            if code_vectorstore:
                code_count = code_vectorstore._collection.count()
                logger.info(
                    f"✅ Code indexing complete! {code_count} code chunks indexed."
                )
                print(f"\n✅ Code indexing complete! {code_count} code chunks indexed.")
            else:
                logger.error("Code indexing failed!")
                return 1

        # FAQ generation workflow
        if args.generate_faqs:
            logger.info("=" * 60)
            logger.info("FAQ GENERATION WORKFLOW")
            logger.info("=" * 60)

            if args.force_faq_regen:
                logger.info("Force regeneration enabled - will regenerate all FAQs")

            # Generate FAQs for all repositories in one collection
            logger.info(f"Generating FAQs for {len(config.repositories)} repositories")

            faq_vectorstore = run_faq_generator(
                config,
                repo_configs=config.repositories,
                force_regen=args.force_faq_regen,
            )

            if faq_vectorstore:
                faq_count = faq_vectorstore._collection.count()
                logger.info(
                    f"✅ FAQ generation complete! {faq_count} FAQs created across all repositories."
                )
            else:
                logger.error("FAQ generation failed!")
                return 1

        # Confluence indexing
        logger.info("Initializing Confluence vector store")
        if args.reindex_confluence:
            vectorstore = run_indexer(config, force_reindex=True)
        else:
            vectorstore = run_indexer(config, force_reindex=False)

        # Check if we have any documents
        doc_count = vectorstore._collection.count()
        logger.info(f"Vector store contains {doc_count} document chunks")

        if doc_count == 0:
            logger.warning(
                "Vector store is empty! Run with --reindex to fetch documents from Confluence."
            )

        if args.start_bot:
            # Initialize RAG engine
            logger.info("Initializing RAG engine")
            rag_engine = RAGEngine(config, vectorstore)
            logger.info("Starting Slack bot...")

            print("\n" + "=" * 60)
            print("🤖 Confluence RAG Slack Bot is running!")
            print("=" * 60)
            print(f"📚 Confluence spaces: {', '.join(config.confluence_space_keys)}")
            print(f"📊 Confluence chunks: {doc_count}")
            print(f"🧠 LLM model: {config.llm_model}")
            print("=" * 60)
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
