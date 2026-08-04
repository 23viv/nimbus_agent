"""
Nimbus Support Agent — MCP Client
Async context manager that spawns the MCP server subprocess and
exposes list_tools() / call_tool() for the agent loop.
"""

import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_SERVER_SCRIPT = str(Path(__file__).parent.parent / "mcp_server" / "server.py")


class NimbusMCPClient:
    """
    Async context manager wrapping the Nimbus MCP server.

    Usage:
        async with NimbusMCPClient() as client:
            tools = await client.list_tools()
            result = await client.call_tool("get_user_by_email", {"email": "..."})
    """

    def __init__(self):
        self._session: ClientSession | None = None
        self._stdio_ctx = None
        self._session_ctx = None

    async def __aenter__(self) -> "NimbusMCPClient":
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[_SERVER_SCRIPT],
            env=None,
        )
        self._stdio_ctx = stdio_client(server_params)
        read, write = await self._stdio_ctx.__aenter__()

        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()

        await self._session.initialize()
        return self

    async def __aexit__(self, *args):
        if self._session_ctx:
            await self._session_ctx.__aexit__(*args)
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(*args)

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
