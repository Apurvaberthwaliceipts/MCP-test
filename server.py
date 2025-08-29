# client_async_stream_tools_with_inventory.py
import os
import asyncio
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# MCP client over HTTP (streamable)
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# MCP -> LangChain tools
from langchain_mcp_adapters.tools import load_mcp_tools

# LangGraph agent + errors
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

# LangChain tool helper (decorator) for custom tools
from langchain_core.tools import tool

def system_instructions() -> str:
    return (
        "You are a helpful assistant with access to MCP tools.\n"
        "- Use tools when they add value and avoid repeating the same call with identical inputs.\n"
        "- Produce a concise final answer when the goal is met.\n"
        "- You may use 'mcp_list_tools' or 'mcp_count_tools' to answer questions about available tools.\n"
        "- At most one hello and one add call per user turn unless new info appears.\n"
    )

async def chat_loop():
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Missing GITHUB_TOKEN for GitHub Models.")
    mcp_url = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")

    # OpenAI-compatible LLM via GitHub Models
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=512,
        api_key=token,
        base_url="https://models.inference.ai.azure.com",
        streaming=True,
    )

    # Maintain one MCP session for the full chat
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Load server tools as LangChain tools
            mcp_tools = await load_mcp_tools(session)

            # Add introspection tools that query list_tools on demand
            @tool("mcp_list_tools")
            async def mcp_list_tools_tool() -> str:
                "List available MCP tools with their descriptions."
                tl = await session.list_tools()
                lines = [f"- {t.name}: {getattr(t, 'description', '') or ''}".rstrip() for t in tl.tools]
                return "\n".join(lines) if lines else "No tools available."

            @tool("mcp_count_tools")
            async def mcp_count_tools_tool() -> str:
                "Return the number of available MCP tools."
                tl = await session.list_tools()
                return str(len(tl.tools))

            tools = [mcp_list_tools_tool, mcp_count_tools_tool] + mcp_tools

            # Build the LangGraph ReAct agent
            agent = create_react_agent(llm, tools, prompt=system_instructions())

            # Conversation history
            history: List[dict] = []
            print("MCP Chatbot ready. Type 'exit' to quit. Type '/tools' to show locally.\n")

            # Allow up to ~5 tool uses per turn (each use consumes multiple steps)
            RECURSION_LIMIT = 2 * 5 + 1

            while True:
                try:
                    user = input("User: ").strip()
                except EOFError:
                    break
                if user.lower() in {"exit", "quit"}:
                    break

                # Optional: fast local command (no agent) to show tools immediately
                if user.strip() == "/tools":
                    tl = await session.list_tools()
                    print(f"\nTools ({len(tl.tools)}):")
                    for t in tl.tools:
                        print(f"- {t.name}: {getattr(t, 'description', '') or ''}")
                    print()
                    continue

                history.append({"role": "user", "content": user})
                assistant_text_parts: List[str] = []

                try:
                    # Async streaming to support async tools and show ToolMessages as they arrive
                    async for msg_chunk, meta in agent.astream(
                        {"messages": history},
                        {"recursion_limit": RECURSION_LIMIT},
                        stream_mode="messages",
                    ):
                        # Assistant token chunks
                        if getattr(msg_chunk, "type", None) == "ai":
                            # If provider streams tool-call args, show them
                            tcc = getattr(msg_chunk, "tool_call_chunks", None)
                            if tcc:
                                for tc in tcc:
                                    name = tc.get("name") or ""
                                    args = tc.get("args") or ""
                                    if name or args:
                                        print(f"\n→ tool_call: {name} {args}", end="", flush=True)

                            text = msg_chunk.content or ""
                            if isinstance(text, str) and text:
                                print(text, end="", flush=True)
                                assistant_text_parts.append(text)

                        # Tool results streamed as ToolMessages
                        elif getattr(msg_chunk, "type", None) == "tool":
                            tool_name = getattr(msg_chunk, "name", "tool")
                            tool_content = msg_chunk.content
                            if not isinstance(tool_content, str):
                                tool_content = str(tool_content)
                            print(f"\n[{tool_name}] {tool_content}", flush=True)

                    print()
                    final_text = "".join(assistant_text_parts) or "(no content)"
                    history.append({"role": "assistant", "content": final_text})

                except GraphRecursionError:
                    print("\nAssistant: Stopped due to max steps. Please refine the request.")
                    history.append({"role": "assistant", "content": "Stopped due to max steps."})

if __name__ == "__main__":
    asyncio.run(chat_loop())
