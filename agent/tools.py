"""
Nimbus Support Agent — LangChain Tool Definitions
Creates LangChain StructuredTool objects for use with LangGraph create_react_agent.
Each tool wraps either the local RAG pipeline or a live MCP tool call.
Tool calls are logged to logs/tool_calls.jsonl.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from langsmith import traceable
from pydantic import BaseModel, create_model

from agent import rag

# ── JSONL tool call logger ────────────────────────────────────────────────────
_LOG_FILE = Path(__file__).parent.parent / "logs" / "tool_calls.jsonl"
_LOG_FILE.parent.mkdir(exist_ok=True)


def _log_tool_call(tool_name: str, arguments: dict, result: str) -> None:
    """Append one tool call record to logs/tool_calls.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "arguments": arguments,
        "result_preview": result[:300] + ("…" if len(result) > 300 else ""),
    }
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── RAG tool ──────────────────────────────────────────────────────────────────
class _SearchInput(BaseModel):
    query: str


@traceable(name="search_knowledge_base")
def _search_knowledge_base(query: str) -> str:
    chunks = rag.retrieve(query)
    if not chunks:
        result = (
            "No relevant information found in the Nimbus knowledge base for that query. "
            "Do not guess — tell the user you don't have that information and suggest "
            "they contact support@nimbus.com."
        )
    else:
        context = rag.format_context(chunks)
        result = f"Retrieved {len(chunks)} relevant chunks from the knowledge base:\n\n{context}"

    _log_tool_call("search_knowledge_base", {"query": query}, result)
    return result


def _make_rag_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_search_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Search Nimbus's internal knowledge base for information about company policies, "
            "shipping, returns, product care instructions, warranties, FAQ, and general "
            "information about Nimbus products and services. "
            "Always use this tool before answering any policy or product question."
        ),
        args_schema=_SearchInput,
    )


# ── MCP tools ─────────────────────────────────────────────────────────────────
def _make_mcp_tool(tool_def: dict, mcp_client: Any) -> StructuredTool:
    """
    Convert one MCP tool definition into a LangChain StructuredTool.

    Uses a factory inner function to avoid the late-binding closure bug
    when building multiple tools in a loop (each closure captures its own
    `tool_name` and `client` rather than the loop variable).
    """
    name = tool_def["name"]
    description = tool_def.get("description", "")
    schema = tool_def.get("input_schema") or {}
    properties = schema.get("properties", {})

    # Dynamically build a Pydantic model matching the MCP tool's input schema.
    # All MCP args in this project are strings; extend here if needed.
    fields: dict[str, Any] = {field_name: (str, ...) for field_name in properties}
    ArgsModel = create_model(f"{name}_Args", **fields)

    def _factory(tool_name: str, client: Any):
        async def _tool_coroutine(**kwargs: Any) -> str:
            result = await client.call_tool(tool_name, kwargs)
            _log_tool_call(tool_name, kwargs, result)
            return result

        _tool_coroutine.__name__ = tool_name
        return _tool_coroutine

    return StructuredTool.from_function(
        coroutine=_factory(name, mcp_client),
        name=name,
        description=description,
        args_schema=ArgsModel,
    )


def build_langchain_tools(mcp_tool_defs: list[dict], mcp_client: Any) -> list[StructuredTool]:
    """
    Build the complete list of LangChain StructuredTool objects:
      - search_knowledge_base  (RAG, runs locally)
      - get_user_by_email      (MCP)
      - get_user_account_status (MCP)
    """
    tool_list: list[StructuredTool] = [_make_rag_tool()]
    for tool_def in mcp_tool_defs:
        tool_list.append(_make_mcp_tool(tool_def, mcp_client))
    return tool_list
