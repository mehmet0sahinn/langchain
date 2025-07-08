"""
Simple Agent – LangGraph ReAct Chat CLI
===================================
Interactive command‑line assistant built with **LangGraph** that combines an
OpenAI chat model and the **Tavily** web‑search tool.  Minimal dependences, no
framework magic.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def last_ai_content(messages: List[BaseMessage]) -> Optional[str]:
    """Return the most recent AIMessage content, or None if absent."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return None


def build_agent(thread_id: str):
    """Construct LangGraph agent plus config."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", 0))

    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    search_tool = TavilySearchResults(max_results=2)
    tools = [search_tool]

    agent = create_react_agent(llm, tools)

    db_path = os.getenv("CHAT_DB", "chat_history.sqlite")
    memory = SqliteSaver.from_conn_string(db_path)

    config = {"configurable": {"thread_id": thread_id, "memory": memory}}
    return agent, config


def ask(agent, config, prompt: str) -> str:
    """Run a single prompt through the agent and return the assistant reply."""
    state = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config)
    answer = last_ai_content(state["messages"])
    return answer or "(no assistant response)"


def run_interactive(agent, config):
    """REPL loop – Ctrl‑C / 'exit' / 'quit' to stop."""
    print("LangGraph Chat – type 'exit' to quit\n")
    try:
        while True:
            user_input = input("> ").strip()
            if user_input.lower() in {"exit", "quit", "q"}:
                break
            if not user_input:
                continue

            answer = ask(agent, config, user_input)
            print(answer)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


def run_once(agent, config, question: str):
    """Non‑interactive, prints answer and exits."""
    print(ask(agent, config, question))


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="LangGraph ReAct Chat CLI")
    parser.add_argument("question", nargs="?", help="Optional single‑shot question")
    parser.add_argument("--thread", default="default", help="Thread / session id (default: %(default)s)")
    args = parser.parse_args()

    agent, config = build_agent(args.thread)

    if args.question:
        run_once(agent, config, args.question)
    else:
        run_interactive(agent, config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: broad-except
        print(f"Unhandled error: {exc}", file=sys.stderr)
        sys.exit(1)
