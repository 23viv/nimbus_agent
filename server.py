"""
Nimbus Furniture Support Agent — FastAPI Web Server
Exposes /chat, /reset, /health endpoints and serves the UI.

The MCP server is a SEPARATE service in nimbus_mcp/.
Start locally:  cd ../nimbus_mcp && python server.py
Then start this: python server.py
"""

import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

# ── project path ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from agent import db, rag
from agent.agent import MODEL, MAX_TOKENS, build_graph, run_agent_turn
from agent.guardrails import input_guardrails, reset_repetition_tracker
from agent.tools import build_langchain_tools

load_dotenv()

# ── Globals ────────────────────────────────────────────────────────────────────
_graph = None
_mcp_ready = False


async def _connect_mcp_and_build_graph():
    """
    Connect to the standalone MCP HTTP server and build the agent graph.
    Falls back to RAG-only if the MCP server is not reachable.
    """
    global _graph, _mcp_ready
    from agent.mcp_client import NimbusMCPClient
    from langchain_openrouter import ChatOpenRouter

    chat_model = ChatOpenRouter(
        model=MODEL,
        max_tokens=MAX_TOKENS,
    )

    try:
        async with NimbusMCPClient() as mcp_client:
            mcp_tool_defs = await mcp_client.list_tools()
            langchain_tools = build_langchain_tools(mcp_tool_defs, mcp_client)
            print(f"[startup] MCP ready — {len(mcp_tool_defs)} tools.")
            _mcp_ready = True
            _graph = build_graph(chat_model, langchain_tools)
            print(f"[startup] Agent graph compiled ({MODEL} with MCP).")
    except Exception as exc:
        print(f"[startup] MCP unavailable ({exc}). Running RAG-only.")
        print("[startup] Make sure the MCP server is running: cd ../nimbus_mcp && python server.py")
        langchain_tools = build_langchain_tools([], _MockMCPClient())
        _graph = build_graph(chat_model, langchain_tools)
        print(f"[startup] Agent graph compiled ({MODEL} RAG-only).")


class _MockMCPClient:
    """Placeholder so build_langchain_tools works with zero MCP tools."""
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph

    # 1. Initialize MongoDB connection
    await db.init_db()

    # 2. Ingest documents (sync-safe, pure Python / ChromaDB)
    count = rag.ingest_documents()
    print(f"[startup] Knowledge base ready — {count} chunks.")

    # 3. Connect to the standalone MCP server and build the agent graph
    await _connect_mcp_and_build_graph()

    if _graph is None:
        print("[startup] WARNING: graph not ready — continuing anyway.")

    yield  # ── server is live ──

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
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not _graph:
        raise HTTPException(status_code=503, detail="Agent is still starting up. Please wait a moment.")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Assign or reuse session_id
    session_id = req.session_id.strip() if req.session_id and req.session_id.strip() else str(uuid.uuid4())

    # ── Input guardrails (fast, synchronous — runs before the LLM) ────────────────
    guard = input_guardrails(req.message)
    if guard.blocked:
        raise HTTPException(
            status_code=400,
            detail={"blocked": True, "reason": guard.rule_triggered, "message": guard.sanitized_text},
        )

    # 1. Fetch past conversation history for this session from MongoDB Atlas
    stored_messages = await db.get_session_messages(session_id)
    conversation_history: list = []
    for msg in stored_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            conversation_history.append(HumanMessage(content=content))
        elif role == "assistant":
            conversation_history.append(AIMessage(content=content))

    # 2. Run Agent turn (passes session_id for Langfuse session observability)
    reply = await run_agent_turn(
        user_message=req.message,
        conversation_history=conversation_history,
        graph=_graph,
        session_id=session_id,
    )

    # 3. Save User message & Assistant reply to MongoDB Atlas
    await db.save_message(session_id=session_id, role="user", content=req.message)
    await db.save_message(session_id=session_id, role="assistant", content=reply)

    return ChatResponse(reply=reply, session_id=session_id)


@app.get("/sessions")
async def list_sessions():
    """Retrieve list of all chat sessions stored in MongoDB Atlas."""
    sessions = await db.get_all_sessions()
    return {"sessions": sessions}


@app.get("/sessions/{session_id}/messages")
async def get_session_history(session_id: str):
    """Retrieve all messages for a specific session_id."""
    messages = await db.get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Permanently delete a session and all its messages."""
    await db.clear_session(session_id)
    reset_repetition_tracker()
    return {"status": "deleted", "session_id": session_id}


class ResetRequest(BaseModel):
    session_id: Optional[str] = None


@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    session_id = req.session_id if req else None
    if session_id:
        await db.clear_session(session_id)
    reset_repetition_tracker()   # also clear the guardrail repetition buffer
    return {"status": "conversation reset", "session_id": session_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ready": _graph is not None,
        "mcp": _mcp_ready,
        "mongo": db._mongo_ready,
        "model": MODEL,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)


