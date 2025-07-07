"""
main.py
-----------
A minimal LangChain + OpenAI chat CLI that keeps per‑session conversation history in memory.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# Environment & model initialisation
# ---------------------------------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    sys.stderr.write(
        "OPENAI_API_KEY not found in environment. "
        "Create a .env file or export the variable and retry.\n"
    )
    sys.exit(1)


def build_model(model_name: str = "gpt-3.5-turbo", temperature: float = 0.2) -> ChatOpenAI:
    """Return an initialised ChatOpenAI model instance."""
    return ChatOpenAI(model=model_name, temperature=temperature, streaming=True)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return chat history for a given session, creating one if necessary."""
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def build_chain(chat_model: ChatOpenAI):
    """Build a LangChain Runnable with in‑memory message history."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful, concise assistant."),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | chat_model
    return RunnableWithMessageHistory(chain, get_session_history)


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------

def chat_loop(chain, session_id: str) -> None:
    """Simple synchronous REPL loop."""
    cfg = {"configurable": {"session_id": session_id}}
    print(f"Session: {session_id}. Type 'exit' or Ctrl‑C to quit.\n")
    try:
        while True:
            user_in = input("> ").strip()
            if not user_in or user_in.lower() in {"exit", "quit"}:
                print("Bye!")
                break

            response = chain.invoke([HumanMessage(content=user_in)], config=cfg)
            print(response.content)
    except KeyboardInterrupt:
        print("\nBye!")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session‑aware LLM chat CLI")
    parser.add_argument("--session", default="default", help="Chat session id")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(args.model, args.temperature)
    chain = build_chain(model)
    chat_loop(chain, args.session)


if __name__ == "__main__":
    main()