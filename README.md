# 🌥️ Nimbus Support Agent

An AI-powered customer support chatbot for **Nimbus** (a fictional home-goods e-commerce company).

Built with a modern open-source stack — featuring **ChromaDB Vector RAG**, **MongoDB Atlas Session Storage**, **FastMCP Live Tools**, and **Langfuse Session Observability**.

---

## Stack

| Layer | Technology |
|---|---|
| **LLM** | [OpenRouter](https://openrouter.ai/) → `google/gemma-4-31b-it:free` |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) StateGraph |
| **Vector DB (RAG)** | [ChromaDB](https://www.trychroma.com/) + [SentenceTransformers](https://sbert.net/) (`all-MiniLM-L6-v2`) |
| **Session DB** | [MongoDB Atlas](https://www.mongodb.com/atlas) (`motor` async driver + `pymongo`) |
| **Live Tools** | [FastMCP](https://github.com/jlowin/fastmcp) over stdio |
| **Web Server** | [FastAPI](https://fastapi.tiangolo.com/) + ChatGPT-style glassmorphic UI |
| **Observability** | [Langfuse](https://langfuse.com/) (`@observe` + `CallbackHandler` with `session_id` tracking) |
| **Guardrails** | Code-level safety layer (`agent/guardrails.py`) |

---

## Architecture

```
User (Browser ChatGPT UI / CLI)
        │
        ▼
FastAPI /chat endpoint
        │
        ├──► Input Guardrails ──────────────────────────────────► block + HTTP 400
        │         (injection, PII, length, repetition)
        │
        ├──► MongoDB Atlas (agent/db.py) ───────────────────────► Load session history & save messages
        │
        ▼
LangGraph StateGraph
        │
        ├──► agent node (Gemma via OpenRouter)
        │         │
        │         ├──► search_knowledge_base ──► ChromaDB Vector Store ──► vector_store/
        │         │
        │         └──► get_user_by_email / get_user_account_status
        │                       │
        │                       └──► FastMCP server ──► data/users.json
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
Langfuse Tracing: Spans grouped by session_id in Langfuse Dashboard
Logging: every tool call → logs/tool_calls.jsonl
         every guardrail violation → logs/guardrail_violations.jsonl
```

---

## Project Structure

```
nimbus-support-agent/
├── docs/                        # Company policy documents (RAG source)
│   ├── return_policy.txt
│   ├── shipping_policy.txt
│   ├── product_care.txt
│   └── faq.txt
├── vector_store/                # Persistent ChromaDB vector database index
├── data/
│   └── users.json               # Mock user database (8 users)
├── mcp_server/
│   └── server.py                # FastMCP server (user lookup tools)
├── agent/
│   ├── agent.py                 # LangGraph graph + CLI entry point
│   ├── db.py                    # MongoDB Atlas async session manager
│   ├── guardrails.py            # Input & output safety guardrails
│   ├── rag.py                   # ChromaDB vector RAG pipeline (ingest + retrieve)
│   ├── tools.py                 # LangChain StructuredTool wrappers
│   └── mcp_client.py            # Async MCP stdio client
├── server.py                    # FastAPI web server (/chat, /sessions, /reset, /health)
├── ui/                          # ChatGPT-style web UI with session sidebar (served by FastAPI)
│   └── index.html
├── scripts/
│   └── ingest.py                # Script to chunk & embed docs into ChromaDB (--rebuild)
├── logs/
│   ├── tool_calls.jsonl         # Append-only tool call log
│   └── guardrail_violations.jsonl  # Guardrail violation log
├── tests/
│   └── test_cases.py            # Hand-built test suite
├── requirements.txt
├── atlas-credentials.env        # MongoDB Atlas credentials
├── .env                         # Environment variables (not committed)
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/keys) API key
- A [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) cluster connection string
- A [Langfuse](https://cloud.langfuse.com) account for tracing

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

Create or update `.env`:

```env
# Required — OpenRouter API key
OPENROUTER_API_KEY=sk-or-...

# Required — MongoDB Atlas Connection
MONGODB_URI="mongodb+srv://<username>:<password>@cluster0.v3dfdhj.mongodb.net/?appName=Cluster0"
MONGODB_DB_NAME="nimbus_db"

# Optional — enables Langfuse session observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 5. Ingest Documents into ChromaDB Vector Store

Build the persistent local vector store index:

```bash
python scripts/ingest.py --rebuild
```

---

## Running

### Web Server (Recommended)

```bash
python server.py
# or: uvicorn server:app --reload
```

Open `http://127.0.0.1:8000` — the ChatGPT-style interface loads automatically with session sidebar.

**API Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send a message with `session_id`, get a reply & store turn in MongoDB Atlas |
| `GET` | `/sessions` | List all past user chat sessions stored in MongoDB Atlas |
| `GET` | `/sessions/{session_id}/messages` | Retrieve full message history for a specific session |
| `POST` | `/reset` | Clear messages for a specific session |
| `GET` | `/health` | Check server, ChromaDB, MCP, and MongoDB Atlas connection status |

---

## Observability & Session Tracking

Every agent turn is traced in **Langfuse** and grouped by `session_id`:

```
run_agent_turn (session_id="session-123")  ← @observe trace tagged with session_id
├── guardrails.input
├── agent_node
├── [LangGraph CallbackHandler spans]
│   ├── ChatOpenRouter call
│   └── ToolNode execution
├── search_knowledge_base
│   └── rag.retrieve (ChromaDB vector similarity search)
└── guardrails.output
```

---

## Running Tests

```bash
python tests/test_cases.py
```

---

## Key Design Decisions

### ChromaDB Vector Search over BM25
The RAG pipeline uses **ChromaDB** with dense **SentenceTransformers** (`all-MiniLM-L6-v2`) embeddings stored on disk (`vector_store/`). Semantic similarity search matches questions based on *meaning* rather than exact keyword matches.

### MongoDB Atlas Session Persistence
Session message history is stored in **MongoDB Atlas** using `motor` (AsyncIO MongoDB driver). Users can resume past chat sessions, switch conversations in the UI sidebar, and maintain context across browser restarts.

### Langfuse Session Observability
`session_id` is passed into `run_agent_turn` and attached to Langfuse trace contexts (`CallbackHandler`), allowing full session-level trajectory inspection in the Langfuse dashboard.

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

