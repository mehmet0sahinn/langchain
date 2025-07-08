# Simple Agent – LangGraph ReAct Chat CLI

Interactive command‑line assistant that combines LLMs with the Tavily search tool via LangGraph's ReAct agent template.

## Features

- **ReAct agent** built with `create_react_agent` – handles tool calls & reasoning.
- **TavilySearchResults** tool (max 2 hits) for web search.
- **SQLite memory** – conversation state is persisted between runs (file path defaults to _chat_history.sqlite_, overridable with the `CHAT_DB` env var).
- **Model/temperature tweaks** via env vars: `OPENAI_MODEL`, `OPENAI_TEMPERATURE`.
- Two modes:
  1. _REPL_ → `python main.py --thread mysession`
  2. _One‑shot_ → `python main.py "What is the weather in Cekmekoy?"`
- Graceful exit with `Ctrl‑C` / `exit` / `quit`.

## Prerequisites

```bash
pip install -r requirements.txt
```

Add an `.env` file with:

```bash
OPENAI_API_KEY=sk‑...
TAVILY_API_KEY=tvly‑...
```

## Usage Examples

```bash
# Interactive session (default thread)
python main.py

# Interactive session with named thread id (persists separately)
python main.py --thread pasha00

# Fire‑and‑forget single question
python ai_chat.py "Summarise today's top AI news in Turkish" --thread daily‑brief
```
