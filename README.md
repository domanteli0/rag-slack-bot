# Confluence RAG Slack Bot 🤖

A Slack chatbot that answers questions using knowledge from your Confluence space, powered by RAG (Retrieval-Augmented Generation).

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Slack      │────▶│  RAG Engine  │────▶│   OpenAI     │
│   Message    │     │  (retrieve)  │     │   (GPT-4o)   │
└──────────────┘     └──────┬───────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  ChromaDB    │◀──── Confluence Pages
                    │  (vectors)   │      (indexed)
                    └──────────────┘
```

## Setup

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create Environment File

Create a `.env` file in the project root with these variables:

```env
# OpenAI API Key (https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-...

# Confluence Configuration
CONFLUENCE_URL=https://yoursite.atlassian.net
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACE_KEY=YOUR_SPACE_KEY

# Slack Configuration
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

### 3. Get Your Credentials

#### Confluence API Token
1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Copy the token to `CONFLUENCE_API_TOKEN`

#### Confluence Space Key
- Found in your space URL: `https://yoursite.atlassian.net/wiki/spaces/YOURKEY/...`
- The space key is `YOURKEY`

#### Slack App Setup
1. Go to https://api.slack.com/apps and click "Create New App"
2. Choose "From scratch", name it (e.g., "Confluence Bot")
3. **Enable Socket Mode:**
   - Settings → Socket Mode → Enable
   - Create an app-level token with `connections:write` scope
   - Copy to `SLACK_APP_TOKEN` (starts with `xapp-`)
4. **Add Bot Scopes:**
   - OAuth & Permissions → Bot Token Scopes → Add:
     - `app_mentions:read`
     - `chat:write`
     - `im:history`
     - `im:read`
     - `im:write`
5. **Enable Events:**
   - Event Subscriptions → Enable Events
   - Subscribe to bot events:
     - `app_mention`
     - `message.im`
6. **Install the App:**
   - OAuth & Permissions → Install to Workspace
   - Copy Bot User OAuth Token to `SLACK_BOT_TOKEN` (starts with `xoxb-`)

### 4. Index Confluence Content

Build the vector index from your Confluence space:

```bash
python main.py --index-only
```

### 5. Run the Bot

```bash
python main.py
```

## Usage

### In Slack

**Mention the bot:**
```
@ConfluenceBot How do we deploy to production?
```

**Direct message:**
Just send a message directly to the bot.

**Slash command (if configured):**
```
/ask What is our refund policy?
```

## CLI Options

```bash
python main.py              # Start bot (uses existing index)
python main.py --reindex    # Rebuild index and start bot
python main.py --index-only # Only rebuild index, don't start bot
```

## Project Structure

```
bot/
├── .env                 # Environment variables (create this)
├── requirements.txt     # Python dependencies
├── config.py           # Configuration loader
├── indexer.py          # Confluence → Vector DB indexing
├── rag_engine.py       # RAG query processing
├── slack_bot.py        # Slack integration
├── main.py             # Entry point
├── chroma_db/          # Vector database (created automatically)
└── README.md           # This file
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Size of text chunks for indexing |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K_RESULTS` | 5 | Number of documents to retrieve |
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `LLM_MODEL` | gpt-4o | OpenAI chat model |

## Troubleshooting

### "Missing required environment variables"
Make sure all variables in your `.env` file are set correctly.

### "No documents to index"
- Check your `CONFLUENCE_SPACE_KEY` is correct
- Verify your API token has read access to the space
- Make sure the space has pages with content

### Bot not responding in Slack
- Check Socket Mode is enabled
- Verify the app is installed to your workspace
- Ensure bot events are subscribed (`app_mention`, `message.im`)

### Rate limits
If indexing large spaces, you may hit API rate limits. The indexer will continue after delays.

## Deployment

### Railway (Simplest)

1. Push code to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add environment variables in Railway dashboard
4. Done! Railway auto-deploys on push

```bash
# Or use Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```

### Fly.io (Free Tier)

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login and deploy
fly auth login
fly launch --no-deploy
fly secrets set OPENAI_API_KEY=sk-... CONFLUENCE_URL=... # etc
fly deploy
```

### Docker (Any VPS)

```bash
docker build -t confluence-bot .
docker run -d --env-file .env confluence-bot
```

## License

MIT

