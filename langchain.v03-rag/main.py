"""
RAG Demo – Botpress “AI Agent” Article
======================================

Fetches a blog post, splits it into chunks, indexes the chunks
in a Chroma vector database, and answers questions with a simple RAG
pipeline.

Usage
-----
$ python main.py "What is the difference between an AI agent and a traditional chatbot?"
"""

from __future__ import annotations

import argparse
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------- #
# 1) Environment & LLM                                                        #
# --------------------------------------------------------------------------- #
load_dotenv()                    # reads OPENAI_API_KEY from .env
llm = ChatOpenAI()               # default temperature = 0.7

# --------------------------------------------------------------------------- #
# 2) Load the article – strip navigation noise                                #
# --------------------------------------------------------------------------- #
loader = WebBaseLoader(
    web_paths=("https://botpress.com/tr/blog/what-is-an-ai-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(           # keep only these tags
            name=["p", "li", "h1", "h2", "h3", "h4"]
        )
    ),
)
docs = loader.load()

# --------------------------------------------------------------------------- #
# 3) Chunk + Embed + Retriever                                                #
# --------------------------------------------------------------------------- #
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1_000,
    chunk_overlap=200,
)
splits = text_splitter.split_documents(docs)

# Keep Chroma in-memory; add persist_directory if you want disk storage
vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# --------------------------------------------------------------------------- #
# 4) RAG chain                                                                #
# --------------------------------------------------------------------------- #
prompt = hub.pull("rlm/rag-prompt")  # generic RAG prompt template

def format_docs(documents):
    """Concatenate retrieved docs into a single string."""
    return "\n\n".join(d.page_content for d in documents)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------------------------------- #
# 5) Minimal CLI                                                              #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RAG query")
    parser.add_argument("question", nargs="+", help="Question to ask")
    args = parser.parse_args()
    question = " ".join(args.question)

    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print()  # final newline

if __name__ == "__main__":
    main()
