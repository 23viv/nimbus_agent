"""
Nimbus Support Agent — MCP Client (HTTP)
Async context manager that connects to the standalone MCP HTTP server and
exposes list_tools() / call_tool() for the agent loop.

The MCP server lives in the separate nimbus_mcp/ project.
Run it locally with:  cd ../nimbus_mcp && python server.py
On Render it is deployed as its own web service.

Set MCP_SERVER_URL in .env (defaults to http://127.0.0.1:8001/mcp).
On Render, set MCP_SERVER_URL to your nimbus_mcp Render URL, e.g.:
  MCP_SERVER_URL=https://nimbus-mcp.onrender.com/mcp
"""

import os
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

_MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")


class NimbusMCPClient:
    """
    Async context manager wrapping the Nimbus MCP HTTP server.

    Usage:
        async with NimbusMCPClient() as client:
            tools = await client.list_tools()
            result = await client.call_tool("get_user_by_email", {"email": "..."})
    """

    def __init__(self):
        self._session: ClientSession | None = None
        self._http_ctx = None
        self._session_ctx = None

    async def __aenter__(self) -> "NimbusMCPClient":
        self._http_ctx = streamablehttp_client(_MCP_SERVER_URL)
        read, write, _ = await self._http_ctx.__aenter__()

        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()

        await self._session.initialize()
        return self

    async def __aexit__(self, *args):
        if self._session_ctx:
            await self._session_ctx.__aexit__(*args)
        if self._http_ctx:
            await self._http_ctx.__aexit__(*args)

    async def list_tools(self) -> list[dict]:
        """Return tool definitions as plain dicts (name, description, input_schema)."""
        assert self._session, "Client not initialised — use as async context manager."
        response = await self._session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool and return its result as a string."""
        assert self._session, "Client not initialised — use as async context manager."
        result = await self._session.call_tool(name, arguments)

        if result.content:
            parts = [
                block.text if hasattr(block, "text") else str(block)
                for block in result.content
            ]
            return "\n".join(parts)
        return "(no result)"
