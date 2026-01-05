"""Slack bot integration using Bolt framework."""

import logging
import re

from config import Config
from rag_engine import RAGEngine
from slack_bolt import App as SlackApp
from slack_bolt.adapter.socket_mode import SocketModeHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfluenceSlackBot:
    """Slack bot that answers questions using Confluence knowledge base."""

    def __init__(self, config: Config, rag_engine: RAGEngine):
        self.config = config
        self.rag_engine = rag_engine

        # Initialize Slack app
        self.slack_app = SlackApp(token=config.slack_bot_token, logger=logger)

        # Store bot's own user ID to filter out its messages
        self.bot_user_id = None

        # Register event handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register Slack event handlers."""

        # # Handle app mentions (@bot question)
        # @self.slack_app.event("app_mention")
        # def handle_mention(event, say):
        #     self._handle_question(event, say)

        # Handle direct messages
        @self.slack_app.event("message")
        def handle_message(event, say):
            # Ignore bot messages (prevent infinite loops)
            if event.get("bot_id") or event.get("subtype") == "bot_message":
                return

            self._handle_question(event, say)

        # # Handle /ask slash command (optional)
        # @self.slack_app.command("/ask")
        # def handle_ask_command(ack, command, say):
        #     ack()  # Acknowledge the command
        #     question = command.get("text", "").strip()

        #     if not question:
        #         say("Please provide a question. Usage: `/ask your question here`")
        #         return

        #     self._process_and_respond(question, say, command.get("user_id"))

    def _extract_question(self, text: str) -> str:
        """Extract the question from message text (remove bot mention)."""
        # Remove bot mention pattern
        cleaned = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
        return cleaned

    def _get_thread_ts(self, event: dict) -> str:
        """Get the thread timestamp for replying in threads.

        - If message is in a thread, reply to that thread
        - If message is top-level, start a new thread on that message
        """
        # If already in a thread, use the parent thread_ts
        # Otherwise, use the message's own ts to start a new thread
        return event.get("thread_ts") or event.get("ts")  # pyright: ignore[reportReturnType]

    def _get_thread_context(self, channel: str, thread_ts: str, current_ts: str) -> str:
        """Fetch all messages from a thread and format as conversation context.

        Args:
            channel: The channel ID
            thread_ts: The parent thread timestamp
            current_ts: The current message timestamp (to exclude it)

        Returns:
            Formatted conversation history string
        """
        try:
            # Fetch thread replies
            result = self.slack_app.client.conversations_replies(
                channel=channel,
                ts=thread_ts,
                limit=50,  # Get up to 50 messages from thread
            )

            messages = result.get("messages", [])

            if len(messages) <= 1:
                # No thread history (just the parent message or current message)
                return ""

            # Format conversation history
            conversation = []
            for msg in messages:
                # Skip the current message (we'll handle it separately)
                if msg.get("ts") == current_ts:
                    continue

                # Skip bot messages from this bot
                if msg.get("bot_id"):
                    # Include bot responses for context
                    text = msg.get("text", "")
                    conversation.append(f"Bot: {text[:500]}")  # Truncate long responses
                else:
                    # User message
                    # user = msg.get("user", "Unknown")
                    text = self._extract_question(msg.get("text", ""))
                    if text:
                        conversation.append(f"User: {text}")

            if not conversation:
                return ""

            return "Previous conversation in this thread:\n" + "\n".join(
                conversation[-10:]
            )  # Last 10 messages

        except Exception as e:
            logger.warning(f"Could not fetch thread history: {e}")
            return ""

    def _handle_question(self, event: dict, say):
        """Handle incoming question from Slack."""
        text = event.get("text", "")
        user_id: str = event.get("user")  # pyright: ignore[reportAssignmentType]
        channel: str = event.get("channel")  # pyright: ignore[reportAssignmentType]
        thread_ts = self._get_thread_ts(event)
        current_ts: str = event.get("ts")  # pyright: ignore[reportAssignmentType]

        question = self._extract_question(text)

        logger.info(f"Handling query from {user_id}: {question}")

        if not question:
            say(
                f"Hi <@{user_id}>! Ask me anything about our Confluence documentation. Just mention me with your question!",
                thread_ts=thread_ts,
            )
            return

        # If we're in a thread, get the conversation context
        thread_context = ""
        if event.get("thread_ts"):  # Only if actually in a thread
            thread_context = self._get_thread_context(channel, thread_ts, current_ts)
            if thread_context:
                logger.info(f"Thread context found: {len(thread_context)} chars")

        self._process_and_respond(question, say, user_id, thread_ts, thread_context)

    def _process_and_respond(
        self,
        question: str,
        say,
        user_id: str,
        thread_ts: str | None = None,
        thread_context: str = "",
    ):
        """Process question through RAG and send response."""
        logger.info(f"Question from {user_id}: {question}")

        # Send thinking indicator (in thread)
        say("🤔 Thinking...", thread_ts=thread_ts)

        try:
            # Get answer from RAG engine (with thread context if available)
            if thread_context:
                logger.info("Including thread context in query")

            response = self.rag_engine.query(
                question=question,
                thread_context=thread_context if thread_context else None,
            )

            # Format and send response (in thread)
            formatted_response = self.rag_engine.format_slack_response(response)
            say(formatted_response, thread_ts=thread_ts)

        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            say(
                "Sorry, I encountered an error while searching for an answer. Please try again later",
                thread_ts=thread_ts,
            )

    def start(self):
        """Start the Slack bot using Socket Mode."""
        logger.info("Starting Confluence RAG Slack Bot...")
        handler = SocketModeHandler(self.slack_app, self.config.slack_app_token)
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
