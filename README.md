# 🌥️ Nimbus Support Agent

An AI-powered customer support chatbot for **Nimbus** — a premium furniture e-commerce company.

Built with a modern open-source stack featuring **ChromaDB Vector RAG**, **MongoDB Atlas Session Storage**, **FastMCP Live Tools**, and **Langfuse Session Observability**.

> 🔗 This service depends on the **[nimbus_mcp](../nimbus_mcp/)** MCP server running separately. Start that first.

---

## Stack

| Layer | Technology |
|---|---|
| **LLM** | [OpenRouter](https://openrouter.ai/) → `google/gemma-4-26b-a4b-it:free` |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) StateGraph |
| **Vector DB (RAG)** | [ChromaDB](https://www.trychroma.com/) + [SentenceTransformers](https://sbert.net/) (`all-MiniLM-L6-v2`) |
| **Session DB** | [MongoDB Atlas](https://www.mongodb.com/atlas) (`motor` async driver + `pymongo`) |
| **Live Tools** | [FastMCP](https://github.com/jlowin/fastmcp) over HTTP — served by `nimbus_mcp` |
| **Web Server** | [FastAPI](https://fastapi.tiangolo.com/) + glassmorphic ChatGPT-style UI |
| **Observability** | [Langfuse](https://langfuse.com/) (`@observe` + `CallbackHandler` with `session_id` tracking) |
| **Guardrails** | Code-level safety layer (`agent/guardrails.py`) |

---

## Architecture

```
User (Browser UI / CLI)
        │
        ▼
FastAPI /chat endpoint
        │
        ├──► Input Guardrails ──────────────────────────────────► block + HTTP 400
        │         (injection, PII, length, repetition)
        │
        ├──► MongoDB Atlas (agent/db.py) ───────────────────────► load session history & save messages
        │
        ▼
LangGraph StateGraph
        │
        ├──► agent node (Gemma via OpenRouter)
        │         │
        │         ├──► search_knowledge_base ──► ChromaDB (vector_store/) ──► docs/*.txt
        │         │
        │         └──► get_user_by_email / get_user_account_status
        │                       │
        │                       └──► nimbus_mcp HTTP server ──► data/users.json
        │
        └──► tools node (ToolNode)
        │
        ▼
Output Guardrails ─────────────────────────────────────────► sanitise / truncate
        │
        ▼
Response to user + MongoDB Atlas message persistence
        │
        ▼
Langfuse Tracing: spans grouped by session_id
Logging:          tool_calls.jsonl  |  guardrail_violations.jsonl
```

---

## Project Structure

```
nimbus_agent/
├── docs/                          # RAG source documents (furniture company policies)
│   ├── about_nimbus.txt           # Company overview — who Nimbus is, products, membership
│   ├── faq.txt                    # Frequently asked questions
│   ├── return_policy.txt          # Return & refund policy
│   ├── shipping_policy.txt        # Shipping rates & timelines
│   └── product_care.txt           # Furniture care & maintenance guide
├── vector_store/                  # Persistent ChromaDB vector index (auto-generated)
├── data/
│   └── users.json                 # Local user database (8 mock customers)
├── agent/
│   ├── agent.py                   # LangGraph graph + system prompt + CLI entry point
│   ├── db.py                      # MongoDB Atlas async session manager
│   ├── guardrails.py              # Input & output safety guardrails
│   ├── rag.py                     # ChromaDB RAG pipeline (ingest + retrieve)
│   ├── tools.py                   # LangChain StructuredTool wrappers for MCP tools
│   └── mcp_client.py              # Async HTTP MCP client (connects to nimbus_mcp)
├── server.py                      # FastAPI web server (/chat, /sessions, /reset, /health)
├── ui/
│   └── index.html                 # ChatGPT-style web UI with session sidebar
├── scripts/
│   └── ingest.py                  # CLI script to chunk & embed docs into ChromaDB
├── logs/
│   ├── tool_calls.jsonl           # Append-only tool call log
│   └── guardrail_violations.jsonl # Guardrail violation log
├── tests/
│   └── test_cases.py              # Hand-built test suite
├── requirements.txt
├── .env                           # Environment variables (never committed)
└── README.md
```

---

## RAG Knowledge Base

The agent answers policy questions by searching **5 documents** embedded in ChromaDB:

| Document | What it covers |
|---|---|
| `about_nimbus.txt` | Company overview, product categories, materials, sustainability, Premium membership |
| `faq.txt` | Ordering, account, warranty, gift cards, promotions |
| `return_policy.txt` | 30-day returns, eligibility, refund timeline, damaged items |
| `shipping_policy.txt` | Rates, delivery windows, White Glove Delivery |
| `product_care.txt` | Furniture care instructions by material |

To rebuild the vector index after adding or editing documents:

```bash
python scripts/ingest.py --rebuild
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [OpenRouter](https://openrouter.ai/keys) API key
- [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) cluster (free tier works)
- [Langfuse](https://cloud.langfuse.com) account (optional — for observability)
- **`nimbus_mcp` running** on `http://127.0.0.1:8001` (see [nimbus_mcp README](../nimbus_mcp/README.md))

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

### 4. Configure environment variables

Create a `.env` file in this directory:

```env
# Required — OpenRouter API key (get one at https://openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-...

# Required — MongoDB Atlas
MONGODB_URI="mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?appName=Cluster0"
MONGODB_DB_NAME="nimbus_db"

# Required — URL of the running nimbus_mcp service
# Local:  http://127.0.0.1:8001/mcp
# Render: https://nimbus-mcp.onrender.com/mcp
MCP_SERVER_URL=http://127.0.0.1:8001/mcp
MCP_SERVER_PORT=8001

# Optional — Langfuse observability (https://cloud.langfuse.com)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 5. Ingest documents into ChromaDB

```bash
python scripts/ingest.py --rebuild
```

---

## Running Locally

> ⚠️ Start the `nimbus_mcp` server first, then start this server.

```bash
# Terminal 1 — MCP server (from nimbus_mcp/)
cd ../nimbus_mcp
python server.py

# Terminal 2 — Agent + Web UI
python server.py
# or: uvicorn server:app --reload
```

Open **`http://127.0.0.1:8000`** — the chat UI loads automatically.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send a message; returns AI reply + `session_id` |
| `GET` | `/sessions` | List all chat sessions stored in MongoDB Atlas |
| `GET` | `/sessions/{session_id}/messages` | Full message history for a session |
| `DELETE` | `/sessions/{session_id}` | Permanently delete a session |
| `POST` | `/reset` | Clear messages for a session |
| `GET` | `/health` | Returns server, ChromaDB, MCP, and MongoDB status |

### Example `/chat` request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your return policy?", "session_id": "user-abc"}'
```

```json
{
  "reply": "According to our return policy, you may return most items within 30 days of delivery...",
  "session_id": "user-abc"
}
```

---

## Observability

Every agent turn is traced in **Langfuse** and grouped by `session_id`:

```
run_agent_turn (session_id="abc-123")
├── guardrails.input
├── agent_node                    ← LLM call (Gemma via OpenRouter)
│   └── [LangGraph CallbackHandler spans]
├── search_knowledge_base
│   └── rag.retrieve              ← ChromaDB vector similarity search
├── get_user_by_email             ← MCP tool call → nimbus_mcp
└── guardrails.output
```

View full session traces in your [Langfuse dashboard](https://cloud.langfuse.com).

---

## Deploying to Render

Deploy as a **Web Service** from your GitHub repo:

| Setting | Value |
|---|---|
| **Root Directory** | `nimbus_agent` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn server:app --host 0.0.0.0 --port $PORT` |

**Environment variables to set in Render dashboard:**

```
OPENROUTER_API_KEY=sk-or-...
MONGODB_URI=mongodb+srv://...
MONGODB_DB_NAME=nimbus_db
MCP_SERVER_URL=https://nimbus-mcp.onrender.com/mcp
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

> ⚠️ Deploy `nimbus_mcp` **first** so its URL is available before the agent starts up.

---

## Running Tests

```bash
python tests/test_cases.py
```

---

## Key Design Decisions

### ChromaDB + SentenceTransformers for RAG
Dense vector embeddings with cosine similarity match questions by *meaning* — so "can I send something back?" correctly retrieves the return policy even without exact keyword overlap.

### MongoDB Atlas for Session Persistence
`motor` (AsyncIO driver) stores conversation turns per `session_id`, enabling multi-session support, sidebar history in the UI, and conversation continuity across browser restarts.

### MCP for Live Data
User account lookups are served by a **separate FastMCP service** (`nimbus_mcp`), keeping the agent stateless and the database layer independently deployable and scalable.

### Langfuse Session Observability
`session_id` is propagated into every Langfuse trace via `propagate_attributes`, so all turns of a conversation are grouped together in the Langfuse dashboard for easy debugging.

---

## Requirements

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
langchain>=0.2.0
langchain-openrouter>=0.1.0
langchain-core>=0.3.0
langgraph>=0.2.0
langfuse>=2.0.0
mcp>=1.0.0
fastmcp>=2.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
pymongo>=4.6.0
motor>=3.3.0
```

---

## License

MIT — free to use, modify, and extend.
