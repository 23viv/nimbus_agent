"""
Nimbus Support Agent — FastAPI Web Server
Exposes /chat, /reset, /health endpoints and serves the UI.

MCP is started as a background asyncio task so the stdio_client context
manager always lives in the same task that created it (avoids the anyio
cancel-scope cross-task error). If MCP fails, the agent still works with
just the RAG (search_knowledge_base) tool.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── project path ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from agent import rag
from agent.agent import build_graph, run_agent_turn
from agent.guardrails import input_guardrails, reset_repetition_tracker
from agent.tools import build_langchain_tools

load_dotenv()

# ── Globals ────────────────────────────────────────────────────────────────────
_graph = None
_conversation_history: list = []
_mcp_ready = False


async def _boot_mcp_and_build_graph():
    """
    Runs entirely inside one asyncio task so the anyio stdio_client context
    manager is entered and exited in the same task.
    """
    global _graph, _mcp_ready
    from agent.mcp_client import NimbusMCPClient
    from langchain_openrouter import ChatOpenRouter

    chat_model = ChatOpenRouter(
        model="google/gemma-4-26b-a4b-it:free",
        max_tokens=400,
    )

    try:
        async with NimbusMCPClient() as mcp_client:
            mcp_tool_defs = await mcp_client.list_tools()
            langchain_tools = build_langchain_tools(mcp_tool_defs, mcp_client)
            print(f"[startup] MCP ready — {len(mcp_tool_defs)} tools.")
            _mcp_ready = True
            _graph = build_graph(chat_model, langchain_tools)
            print("[startup] Agent graph compiled (with MCP).")
            # Keep running until the server shuts down
            await asyncio.get_event_loop().create_future()
    except Exception as exc:
        print(f"[startup] MCP unavailable ({exc}). Running RAG-only.")
        langchain_tools = build_langchain_tools([], _MockMCPClient())
        _graph = build_graph(chat_model, langchain_tools)
        print("[startup] Agent graph compiled (RAG-only).")


class _MockMCPClient:
    """Placeholder so build_langchain_tools works with zero MCP tools."""
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph

    # 1. Ingest documents (sync-safe, pure Python)
    count = rag.ingest_documents()
    print(f"[startup] Knowledge base ready — {count} chunks.")

    # 2. Boot MCP + build graph in a dedicated background task
    mcp_task = asyncio.create_task(_boot_mcp_and_build_graph())

    # Wait up to 15 s for the graph to be ready
    for _ in range(30):
        if _graph is not None:
            break
        await asyncio.sleep(0.5)

    if _graph is None:
        print("[startup] WARNING: graph not ready after 15 s — continuing anyway.")

    yield  # ── server is live ──

    mcp_task.cancel()
    try:
        await mcp_task
    except (asyncio.CancelledError, Exception):
        pass
    print("[shutdown] Done.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Nimbus Support AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_DIR = Path(__file__).parent / "ui"
if _UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(_UI_DIR / "index.html"))


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not _graph:
        raise HTTPException(status_code=503, detail="Agent is still starting up. Please wait a moment.")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # ── Input guardrails (fast, synchronous — runs before the LLM) ────────────────
    guard = input_guardrails(req.message)
    if guard.blocked:
        raise HTTPException(
            status_code=400,
            detail={"blocked": True, "reason": guard.rule_triggered, "message": guard.sanitized_text},
        )

    reply = await run_agent_turn(
        user_message=req.message,
        conversation_history=_conversation_history,
        graph=_graph,
    )
    return ChatResponse(reply=reply)


@app.post("/reset")
async def reset():
    _conversation_history.clear()
    reset_repetition_tracker()   # also clear the guardrail repetition buffer
    return {"status": "conversation reset"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ready": _graph is not None,
        "mcp": _mcp_ready,
        "chunks": len(rag._chunks),
    }
