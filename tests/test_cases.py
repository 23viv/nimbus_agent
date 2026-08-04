"""
tests/test_cases.py
Hand-built test suite for the Nimbus Support Agent.
Covers ~18 cases: RAG, MCP, escalation, out-of-scope, and edge cases.

Usage:
    python tests/test_cases.py

Requires OPENROUTER_API_KEY in environment / .env file.
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_openrouter import ChatOpenRouter

from agent import rag
from agent.agent import MODEL, MAX_TOKENS, build_graph, run_agent_turn
from agent.tools import build_langchain_tools
from agent.mcp_client import NimbusMCPClient
from agent.prompts import SYSTEM_PROMPT


# ── Test case schema ───────────────────────────────────────────────────────────
@dataclass
class TestCase:
    id: str
    category: str
    description: str
    messages: list[str]
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    check_fn: Callable[[str], bool] | None = None


# ── Test case definitions ──────────────────────────────────────────────────────
TEST_CASES: list[TestCase] = [
    # ── RAG: Return Policy ──────────────────────────────────────────────────
    TestCase(
        id="RAG-01",
        category="RAG",
        description="Basic return window query",
        messages=["What is your return policy?"],
        must_contain=["30 days", "30-day"],
    ),
    TestCase(
        id="RAG-02",
        category="RAG",
        description="Holiday return window",
        messages=["Do you have an extended return window for holiday purchases?"],
        must_contain=["January 31", "holiday"],
    ),
    TestCase(
        id="RAG-03",
        category="RAG",
        description="Non-returnable items",
        messages=["Are there any items I cannot return?"],
        must_contain=["final sale", "custom", "personalized"],
    ),
    TestCase(
        id="RAG-04",
        category="RAG",
        description="Refund timeline (policy only — not processing request)",
        messages=["How long does it take to get a refund after I send back an item?"],
        must_contain=["5", "7", "business day"],
    ),

    # ── RAG: Shipping Policy ────────────────────────────────────────────────
    TestCase(
        id="RAG-05",
        category="RAG",
        description="International shipping availability",
        messages=["Do you ship internationally?"],
        must_contain=["Canada", "United Kingdom", "Australia"],
    ),
    TestCase(
        id="RAG-06",
        category="RAG",
        description="Free shipping threshold",
        messages=["How do I get free shipping?"],
        must_contain=["$75", "75"],
    ),
    TestCase(
        id="RAG-07",
        category="RAG",
        description="Shipping to Australia duty info",
        messages=["If I order from Australia, do I pay customs fees?"],
        must_contain=["Australia", "customs", "responsible"],
    ),

    # ── RAG: Product Care ───────────────────────────────────────────────────
    TestCase(
        id="RAG-08",
        category="RAG",
        description="Candle care first burn",
        messages=["How should I care for my Nimbus candle?"],
        must_contain=["wick", "burn"],
    ),
    TestCase(
        id="RAG-09",
        category="RAG",
        description="Towel softener advice",
        messages=["Can I use fabric softener on Nimbus towels?"],
        must_contain=["fabric softener", "absorbency"],
    ),

    # ── RAG: FAQ ────────────────────────────────────────────────────────────
    TestCase(
        id="RAG-10",
        category="RAG",
        description="Nimbus Premium membership details",
        messages=["What is Nimbus Premium and what does it cost?"],
        must_contain=["$9.99", "premium", "free standard shipping"],
    ),
    TestCase(
        id="RAG-11",
        category="RAG",
        description="Gift card expiry",
        messages=["Do Nimbus gift cards expire?"],
        must_contain=["never expire", "never"],
    ),

    # ── MCP: Account Queries ────────────────────────────────────────────────
    TestCase(
        id="MCP-01",
        category="MCP",
        description="Account lookup by email (active premium user)",
        messages=["What's my account status? My email is sarah.chen@example.com"],
        must_contain=["Sarah", "premium", "active"],
        must_not_contain=["James", "Derek", "suspended"],
    ),
    TestCase(
        id="MCP-02",
        category="MCP",
        description="Last login lookup",
        messages=["When did I last log in? Email: linda.wu@example.com"],
        must_contain=["Linda", "2026"],
    ),
    TestCase(
        id="MCP-03",
        category="MCP",
        description="Suspended account status",
        messages=["Can you check my account? My email is derek.kim@example.com"],
        must_contain=["Derek", "suspended"],
    ),

    # ── MCP: Edge Cases ─────────────────────────────────────────────────────
    TestCase(
        id="MCP-EDGE-01",
        category="MCP Edge",
        description="Email not found in database",
        messages=["Check my account: notexist@example.com"],
        must_contain=["not found", "no account", "couldn't find"],
    ),
    TestCase(
        id="MCP-EDGE-02",
        category="MCP Edge",
        description="Account query without providing email — agent must ask",
        messages=["What's the status of my account?"],
        must_contain=["email", "email address"],
        must_not_contain=["Sarah", "James", "active", "premium"],
    ),

    # ── Escalation ──────────────────────────────────────────────────────────
    TestCase(
        id="ESC-01",
        category="Escalation",
        description="Refund request should escalate",
        messages=["I want a refund for my order #4521"],
        must_contain=["support", "human", "team"],
    ),
    TestCase(
        id="ESC-02",
        category="Escalation",
        description="Complaint should escalate",
        messages=["I want to file a complaint. My order was completely wrong and I'm very upset."],
        must_contain=["support", "team", "assist"],
    ),

    # ── Out of Scope ────────────────────────────────────────────────────────
    TestCase(
        id="OOS-01",
        category="Out of Scope",
        description="Completely off-topic question",
        messages=["What's the weather like in New York today?"],
        must_not_contain=["sunny", "rainy", "temperature", "°F"],
        must_contain=["Nimbus"],
    ),
]


# ── Test runner ────────────────────────────────────────────────────────────────
async def run_test(
    tc: TestCase,
    graph,
) -> tuple[bool, str, str]:
    """Run a single test case. Returns (passed, final_response, failure_reason)."""
    conversation_history: list = []
    final_response = ""

    for message in tc.messages:
        final_response = await run_agent_turn(
            user_message=message,
            conversation_history=conversation_history,
            graph=graph,
        )

    response_lower = final_response.lower()

    for phrase in tc.must_contain:
        if phrase.lower() not in response_lower:
            return False, final_response, f"Missing required phrase: '{phrase}'"

    for phrase in tc.must_not_contain:
        if phrase.lower() in response_lower:
            return False, final_response, f"Found forbidden phrase: '{phrase}'"

    if tc.check_fn and not tc.check_fn(final_response):
        return False, final_response, "Custom check function failed"

    return True, final_response, ""


async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Create a .env file.")
        sys.exit(1)

    print("Initialising knowledge base…", end=" ", flush=True)
    chunk_count = rag.ingest_documents()
    print(f"ready ({chunk_count} chunks).\n")

    async with NimbusMCPClient() as mcp_client:
        mcp_tool_defs = await mcp_client.list_tools()
        langchain_tools = build_langchain_tools(mcp_tool_defs, mcp_client)

        chat_model = ChatOpenRouter(model=MODEL, max_tokens=MAX_TOKENS)
        graph = build_graph(chat_model, langchain_tools)

        total = len(TEST_CASES)
        results: list[tuple[TestCase, bool, str, str]] = []

        print(f"Running {total} test cases…\n")
        print(f"{'ID':<14} {'Category':<14} {'Description':<45} {'Result'}")
        print("-" * 95)

        for tc in TEST_CASES:
            try:
                passed, response, reason = await run_test(tc=tc, graph=graph)
            except Exception as e:
                passed, response, reason = False, "", f"Exception: {e}"

            status = "✅ PASS" if passed else "❌ FAIL"
            desc_truncated = tc.description[:43] + ("…" if len(tc.description) > 43 else "")
            print(f"{tc.id:<14} {tc.category:<14} {desc_truncated:<45} {status}")
            if not passed:
                print(f"  ↳ Reason: {reason}")
                print(f"  ↳ Response snippet: {response[:200]!r}")

            results.append((tc, passed, response, reason))

        passed_count = sum(1 for _, p, _, _ in results if p)
        failed_count = total - passed_count

        print("\n" + "=" * 95)
        print(f"  Results: {passed_count}/{total} passed | {failed_count} failed")
        print("=" * 95)

        if failed_count == 0:
            print("  🎉 All test cases passed!")
        else:
            print("  ⚠️  Some tests failed. Review the output above.")

        return failed_count == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
