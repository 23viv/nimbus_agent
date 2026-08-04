# 🌥️ Nimbus Support Agent

An AI-powered customer support chatbot for **Nimbus** (a fictional home-goods e-commerce company), built with:

- **Claude** (via Anthropic API) as the reasoning engine with tool use
- **RAG** (ChromaDB + sentence-transformers) for grounded policy Q&A
- **MCP** (FastMCP) for live user account lookups

> **v1 demo** — uses fake company docs and a mock user database. Designed to demonstrate the RAG + MCP + Claude agent pattern.

---

## Architecture

```
You (CLI)
    │
    ▼
Claude Agent (claude-sonnet-4-5, tool use)
    │
    ├──► search_knowledge_base ──► ChromaDB ──► docs/ (return policy, shipping, FAQ, care)
    │
    └──► get_user_by_email       ─┐
         get_user_account_status  ├──► FastMCP Server ──► data/users.json
```

**All tool calls are logged to `logs/tool_calls.jsonl`.**

---

## Project Structure

```
nimbus-support-agent/
├── docs/                    # Company policy documents (RAG source)
│   ├── return_policy.txt
│   ├── shipping_policy.txt
│   ├── product_care.txt
│   └── faq.txt
├── data/
│   └── users.json           # Fake user database (8 mock users)
├── vector_store/            # ChromaDB persistent store (auto-created)
├── mcp_server/
│   └── server.py            # FastMCP server (user lookup tools)
├── agent/
│   ├── agent.py             # Main agent loop (entry point)
│   ├── rag.py               # RAG pipeline (ingest + retrieve)
│   ├── mcp_client.py        # Async MCP client
│   ├── tools.py             # Anthropic tool definitions
│   └── prompts.py           # System prompt
├── scripts/
│   └── ingest.py            # One-shot doc ingestion script
├── logs/
│   └── tool_calls.jsonl     # Append-only tool call log (auto-created)
├── tests/
│   └── test_cases.py        # 18 hand-built test cases
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80 MB) on first run. This is a one-time download.

### 4. Configure your API key

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your key
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 5. Ingest company documents

```bash
python scripts/ingest.py
```

This reads `docs/*.txt`, chunks and embeds them, and stores the vectors in `vector_store/`. You only need to run this once (or when docs change). To force a rebuild:

```bash
python scripts/ingest.py --rebuild
```

---

## Running the Agent

```bash
python agent/agent.py
```

The agent will:
1. Check the API key
2. Load/verify the vector store
3. Connect to the MCP server
4. Start the interactive CLI

**Example session:**

```
============================================================
  🌥️  NIMBUS SUPPORT AI
  Your AI customer support assistant
============================================================
  Type your question and press Enter.
  Type 'quit' or press Ctrl+C to exit.
============================================================

You: What's your return policy?

Nimbus Support AI: According to our return policy, you can return most items within
30 days of delivery. Items must be unused, in original packaging, and purchased
directly from Nimbus...

You: What's the status of my account?

Nimbus Support AI: To look up your account, could you please provide the email
address associated with your Nimbus account?

You: sarah.chen@example.com

Nimbus Support AI: Hi Sarah! Your account is active on the Nimbus Premium plan...

You: I want a refund for order #4521

Nimbus Support AI: I understand your concern. I'm not able to handle refunds
directly, but I'd like to make sure you get the right help. Please contact our
human support team:
- Chat: nimbus.com/support
- Email: support@nimbus.com
...
```

---

## Running Tests

```bash
python tests/test_cases.py
```

Runs 18 hand-built test cases:

| Category | Cases | Tests |
|---|---|---|
| RAG | 11 | Return policy, shipping, product care, FAQ |
| MCP | 3 | Account lookup by email, last login, suspended account |
| MCP Edge | 2 | Email not found, account query without email |
| Escalation | 2 | Refund requests, complaints |
| Out of Scope | 1 | Off-topic question |

Each test validates the agent's response for required phrases and forbidden phrases (no hallucination, no data leakage).

---

## Mock Users (for testing MCP tools)

| Email | Name | Plan | Status |
|---|---|---|---|
| sarah.chen@example.com | Sarah Chen | Premium | Active |
| james.patel@example.com | James Patel | Free | Active |
| maria.gonzalez@example.com | Maria Gonzalez | Premium | Active |
| derek.kim@example.com | Derek Kim | Free | **Suspended** |
| linda.wu@example.com | Linda Wu | Premium | Active |
| tom.nguyen@example.com | Tom Nguyen | Free | Active |
| aisha.johnson@example.com | Aisha Johnson | Premium | Active |
| carlos.rivera@example.com | Carlos Rivera | Free | Active |

---

## Key Design Decisions

### No Hallucination Guarantee
The system prompt instructs Claude to **only** answer policy questions using retrieved knowledge base content. If retrieval returns no relevant chunks (distance above threshold), the tool returns an explicit "no results" message and Claude responds accordingly.

### MCP over Stdio
The MCP server runs as a subprocess spawned by the agent. This is the standard local pattern — no network ports needed, and the server lifecycle is tied to the agent session.

### Similarity Threshold
ChromaDB returns L2 distances. A threshold of `1.2` filters out chunks that are too semantically distant. This prevents the agent from using weakly-related content to answer unrelated questions.

### Tool Call Logging
Every tool call (RAG or MCP) is logged to `logs/tool_calls.jsonl` with timestamp, tool name, arguments, and a result preview. This is the foundation for future observability work.

---

## Extending This Project

- **Add more docs:** Drop `.txt` files in `docs/` and re-run `python scripts/ingest.py --rebuild`
- **Add more MCP tools:** Add `@mcp.tool` functions to `mcp_server/server.py`
- **Use a real database:** Replace `data/users.json` with a real DB query in the MCP server
- **Add Slack escalation:** In `agent/prompts.py` + `agent/agent.py`, detect escalation responses and POST to a Slack webhook
- **Add a web UI:** Wrap `agent/agent.py` with FastAPI + WebSockets for a browser chat interface

---

## Requirements

```
anthropic>=0.34.0
mcp>=1.0.0
fastmcp>=2.0.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
python-dotenv>=1.0.0
```

---

## License

MIT — free to use, modify, and extend.
