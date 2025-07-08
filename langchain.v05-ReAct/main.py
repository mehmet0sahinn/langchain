"""ReAct Chat CLI – LangChain + Tavily demo
========================================

Production‑ready command‑line chat interface built on
LangChain’s ReAct agent pattern. It supports streaming output, tool
invocation (Tavily web search), and persistent conversation memory.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver


def build_agent(verbose: bool = True) -> AgentExecutor:
    """Instantiate a ReAct agent with Tavily search."""

    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
        streaming=True,
        temperature=float(os.getenv("TEMPERATURE", "0")),
    )

    search_tool = TavilySearchResults(max_results=2)
    tools = [search_tool]

    prompt = hub.pull("hwchase17/react-chat")

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
    return executor


def interactive_loop(
    executor: AgentExecutor, thread_id: str, memory_path: str | None
) -> None:
    """Simple REPL that streams the agent response chunk by chunk."""

    memory = SqliteSaver.from_conn_string(memory_path or ":memory:")
    config = {"configurable": {"thread_id": thread_id}}

    chat_history: List[str] = []

    print("ReAct agent ready (Ctrl‑C to quit)\n")
    try:
        while True:
            user_input = input("> ").strip()
            if not user_input:
                continue

            chat_history.append(f"Human: {user_input}")
            response_chunks: List[str] = []

            for chunk in executor.stream(
                {"input": user_input, "chat_history": "\n".join(chat_history)}, config
            ):
                if "text" in chunk:
                    print(chunk["text"], end="", flush=True)
                    response_chunks.append(chunk["text"])

            ai_response = "".join(response_chunks)
            chat_history.append(f"AI: {ai_response}")
            print("\n-------")
    except KeyboardInterrupt:
        print("\n Bye!")
        sys.exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReAct chat CLI")
    parser.add_argument("-t", "--thread", default="default", help="Thread ID for checkpointing")
    parser.add_argument(
        "-m", "--memory", default=":memory:", help="SQLite DB path for conversation memory"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Turn off verbose logging")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    executor = build_agent(verbose=not args.quiet)
    interactive_loop(executor, thread_id=args.thread, memory_path=args.memory)


if __name__ == "__main__":
    main()