# RAG Demo – Botpress “AI Agent” Article

======================================

Fetches a blog post, splits it into chunks, indexes the chunks in a Chroma vector database, and answers questions with a simple RAG pipeline.

## Create a virtual environment and install dependencies

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Add your API key to the environment

cp .env.example .env
vim .env # OPENAI_API_KEY=...

## Ask a question

python main.py "What is the difference between an AI agent and a traditional chatbot?"
