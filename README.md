# RAG Slack Bot 🤖

A Slack chatbot that answers questions using knowledge from your Confluence space & source code, powered by RAG (Retrieval-Augmented Generation).

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Slack      │────▶│  RAG Engine  │────▶│   OpenAI     │
│   Message    │     │  (retrieve)  │     │              │
└──────────────┘     └──────┬───────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  ChromaDB    │◀──── Confluence Pages + Synthesized FAQs + Code
                    │  (vectors)   │      (indexed)
                    └──────────────┘
```

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Create Environment File

Create a `.env` file in the project root with these variables:

```env
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
     <!--- `app_mentions:read`-->
     - `chat:write`
     - `im:history`
5. **Enable Events:**
   - Event Subscriptions → Enable Events
   - Subscribe to bot events:
     <!--- `app_mention`-->
     - `message.im`
    - App Home > Messages Tab
      - Enable Messages Tab
      - Enable `Allow users to send Slash commands and messages from the messages tab`
6. **Install the App:**
   - OAuth & Permissions → Install to Workspace
   - Copy Bot User OAuth Token to `SLACK_BOT_TOKEN` (starts with `xoxb-`)

### 4. Run the Bot

```bash
python main.py -b
```
