"""Slack bot integration using Bolt framework."""

import logging
import re

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from config import Config
from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfluenceSlackBot:
    """Slack bot that answers questions using Confluence knowledge base."""
    
    def __init__(self, config: Config, rag_engine: RAGEngine):
        self.config = config
        self.rag_engine = rag_engine
        
        # Initialize Slack app
        self.app = App(token=config.slack_bot_token)
        
        # Register event handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register Slack event handlers."""
        
        # Handle app mentions (@bot question)
        @self.app.event("app_mention")
        def handle_mention(event, say):
            self._handle_question(event, say)
        
        # Handle direct messages
        @self.app.event("message")
        def handle_message(event, say):
            # Only respond to DMs (not channel messages without mention)
            if event.get("channel_type") == "im":
                self._handle_question(event, say)
        
        # Handle /ask slash command (optional)
        @self.app.command("/ask")
        def handle_ask_command(ack, command, say):
            ack()  # Acknowledge the command
            question = command.get("text", "").strip()
            
            if not question:
                say("Please provide a question. Usage: `/ask your question here`")
                return
            
            self._process_and_respond(question, say, command.get("user_id"))
    
    def _extract_question(self, text: str) -> str:
        """Extract the question from message text (remove bot mention)."""
        # Remove bot mention pattern
        cleaned = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
        return cleaned
    
    def _handle_question(self, event: dict, say):
        """Handle incoming question from Slack."""
        text = event.get("text", "")
        user_id = event.get("user")
        
        question = self._extract_question(text)
        
        if not question:
            say(f"Hi <@{user_id}>! Ask me anything about our Confluence documentation. Just mention me with your question!")
            return
        
        self._process_and_respond(question, say, user_id)
    
    def _process_and_respond(self, question: str, say, user_id: str):
        """Process question through RAG and send response."""
        logger.info(f"Question from {user_id}: {question}")
        
        # Send thinking indicator
        say(f"🔍 Searching Confluence for an answer...")
        
        try:
            # Get answer from RAG engine
            response = self.rag_engine.query(question)
            
            # Format and send response
            formatted_response = self.rag_engine.format_slack_response(response)
            say(formatted_response)
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            say(f"Sorry, I encountered an error while searching for an answer. Please try again later.\n\nError: {str(e)}")
    
    def start(self):
        """Start the Slack bot using Socket Mode."""
        logger.info("Starting Confluence RAG Slack Bot...")
        handler = SocketModeHandler(self.app, self.config.slack_app_token)
        handler.start()


def create_bot(config: Config, rag_engine: RAGEngine) -> ConfluenceSlackBot:
    """Factory function to create a configured bot instance."""
    return ConfluenceSlackBot(config, rag_engine)


if __name__ == "__main__":
    from config import load_config
    
    config = load_config()
    rag_engine = RAGEngine(config)
    bot = ConfluenceSlackBot(config, rag_engine)
    bot.start()

