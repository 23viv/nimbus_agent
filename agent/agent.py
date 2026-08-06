"""
Nimbus Support Agent — Main Agent Loop
Multi-turn CLI chatbot with a hand-built LangGraph StateGraph + ChatOpenRouter.

Graph structure:
    START → agent ──(has tool calls)──► tools → agent (loop)
                  ──(no tool calls)──► END

LangSmith tracing:
  - @traceable decorates key functions with named spans
  - The graph itself is traced automatically by LangGraph when
    LANGCHAIN_TRACING_V2=true is set in your .env
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import traceable
from typing_extensions import TypedDict

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import rag
from agent.mcp_client import NimbusMCPClient
from agent.tools import build_langchain_tools

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

MODEL = "google/gemma-4-26b-a4b-it:free"
MAX_TOKENS = 400

# ── System prompt (inlined — prompts.py removed) ───────────────────────────────
SYSTEM_PROMPT = """You are the Nimbus Support AI, a helpful customer support assistant for Nimbus, a home goods e-commerce company.

## Your Identity
- You are an AI assistant. Always be transparent that you are an AI when asked.
- Your name is "Nimbus Support AI".
- You are friendly, concise, and professional.

## Your Capabilities
You have access to two types of tools:

1. **search_knowledge_base** — Searches Nimbus's internal knowledge base (return policy, shipping policy, product care guides, FAQ). Use this for ANY question about Nimbus policies, shipping, returns, products, or general company information.

2. **get_user_by_email** and **get_user_account_status** — Live tools to look up a customer's account information from the Nimbus database. Use these ONLY when a customer asks about their specific account (status, plan, last login, etc.) AND has provided their email address or user ID.

## Critical Rules

### Rule 1: Never Hallucinate Policy Information
- For ANY question about Nimbus policies, shipping, returns, warranties, or products — ALWAYS call `search_knowledge_base` first.
- If the knowledge base returns no relevant results, say: "I don't have specific information on that in my knowledge base. For accurate details, please contact our support team at support@nimbus.com."
- NEVER answer policy questions from your general training knowledge. Only answer from retrieved knowledge base content.

### Rule 2: Require Identification for Account Queries
- Never look up or reveal account information without the customer first providing their email address.
- If a customer asks about their account but hasn't provided an email, ask: "To look up your account, could you please provide the email address associated with your Nimbus account?"
- Never guess, assume, or infer a customer's identity.
- Only share data belonging to the matched user — never return another user's information.

### Rule 3: Escalate Appropriately
If a customer asks about ANY of the following, do NOT attempt to resolve it yourself. Instead, give this exact escalation response:

"I understand your concern. I'm not able to handle [refunds / complaints / billing disputes / account suspensions] directly, but I'd like to make sure you get the right help. Please contact our human support team:
- **Chat:** nimbus.com/support (Mon–Fri 9am–6pm ET)
- **Email:** support@nimbus.com
- **Phone:** 1-800-NIMBUS-1 (Mon–Fri 9am–6pm ET)

A team member will be able to assist you personally."

Escalate for:
- Refund requests or processing
- Billing disputes or payment issues
- Formal complaints
- Account suspension appeals
- Any situation requiring judgment beyond information lookup

### Rule 4: Out of Scope
- If a question is completely outside Nimbus's scope (e.g., competitor products, general life advice), politely say you can only help with Nimbus-related questions.

## Conversation Style
- Be warm and empathetic, especially for frustrated customers
- Keep responses focused — don't pad with unnecessary filler
- Cite which document you found information in when answering from the knowledge base (e.g., "According to our return policy…")
- Multi-turn: remember context from earlier in the conversation
"""


# ── Graph state ────────────────────────────────────────────────────────────────
class State(TypedDict):
    """
    The only state the graph tracks is the message list.
    add_messages is a reducer that appends new messages rather than replacing
    the whole list, so each node just returns the messages it wants to add.
    """
    messages: Annotated[list, add_messages]


# ── Graph builder ──────────────────────────────────────────────────────────────
def build_graph(chat_model: Runnable, langchain_tools: list) -> Runnable:
    """
    Build and compile the Nimbus support agent StateGraph.

    Nodes
    -----
    agent  — calls the LLM; the system prompt is prepended here so it is
              always the first message regardless of what history looks like.
    tools  — ToolNode that executes every tool call in the last AIMessage.

    Edges
    -----
    START ──────────────────────────────────────────────► agent
    agent ──(finish_reason == tool_calls)───────────────► tools
    agent ──(finish_reason == stop / no tool calls)─────► END
    tools ──────────────────────────────────────────────► agent  (loop back)
    """
    model_with_tools = chat_model.bind_tools(langchain_tools)

    # ── nodes ──────────────────────────────────────────────────────────────────
    @traceable(name="agent_node")
    def agent_node(state: State) -> dict:
        """
        Call the LLM with the system prompt prepended to the current history.
        @traceable creates a named LangSmith span for this LLM call, separate
        from the graph-level span, so you can inspect the exact messages sent.
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    # ── assembly ───────────────────────────────────────────────────────────────
    builder = StateGraph(State)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(langchain_tools))

    builder.add_edge(START, "agent")

    builder.add_conditional_edges(
        source="agent",
        path=tools_condition,
        path_map={"tools": "tools", END: END},
    )

    builder.add_edge("tools", "agent")

    return builder.compile()


# ── Agent turn ─────────────────────────────────────────────────────────────────
@traceable(name="run_agent_turn")
async def run_agent_turn(
    user_message: str,
    conversation_history: list,
    graph: Runnable,
) -> str:
    """
    Top-level entry point for one user turn.
    @traceable wraps the entire turn — including all graph steps and tool calls —
    as a single named span in LangSmith, making it easy to trace across turns.
    """
    conversation_history.append(HumanMessage(content=user_message))

    result = await graph.ainvoke({"messages": conversation_history})

    conversation_history.clear()
    conversation_history.extend(result["messages"])

    return result["messages"][-1].content or ""


# ── CLI ────────────────────────────────────────────────────────────────────────
def _print_banner():
    print("\n" + "=" * 62)
    print("  🌥️  NIMBUS SUPPORT AI")
    print("  Powered by LangGraph StateGraph + OpenRouter")
    print("=" * 62)
    print("  Type your question and press Enter.")
    print("  Type 'quit' or press Ctrl+C to exit.")
    print("=" * 62 + "\n")


async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY not set.\n"
            "Create a .env file with: OPENROUTER_API_KEY=sk-or-...\n"
            "Get a key at https://openrouter.ai/keys\n"
            "See .env.example for reference."
        )
        sys.exit(1)

    print("Initialising knowledge base…", end=" ", flush=True)
    try:
        count = rag.ingest_documents()
        print(f"ready ({count} chunks).")
    except Exception as e:
        print(f"\nERROR: Failed to initialise knowledge base: {e}")
        sys.exit(1)

    print("Connecting to user database…", end=" ", flush=True)
    async with NimbusMCPClient() as mcp_client:
        mcp_tool_defs = await mcp_client.list_tools()
        langchain_tools = build_langchain_tools(mcp_tool_defs, mcp_client)
        print(f"ready ({len(mcp_tool_defs)} MCP tools).")

        chat_model = ChatOpenRouter(model=MODEL, max_tokens=MAX_TOKENS)
        graph = build_graph(chat_model, langchain_tools)

        conversation_history: list = []
        _print_banner()

        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue
                if user_input.lower() in {"quit", "exit", "bye"}:
                    print("\nNimbus Support AI: Thank you for contacting Nimbus! Have a great day. 👋\n")
                    break

                print("\nNimbus Support AI: ", end="", flush=True)
                try:
                    response_text = await run_agent_turn(
                        user_message=user_input,
                        conversation_history=conversation_history,
                        graph=graph,
                    )
                    print(response_text)
                except Exception as e:
                    print(f"[Error: {e}]")

                print()

        except KeyboardInterrupt:
            print("\n\nNimbus Support AI: Goodbye! 👋\n")


if __name__ == "__main__":
    asyncio.run(main())
