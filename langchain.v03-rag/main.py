"""
RAG Demo
---
Usage

$ python main.py "What is the difference between an AI agent and a traditional chatbot?"
"""
from dotenv import load_dotenv
load_dotenv()  

import bs4

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import argparse

# 1 . Load the Model
model = ChatOpenAI() 

# 2. Load the Content
loader = WebBaseLoader(
    web_paths=("https://botpress.com/tr/blog/what-is-an-ai-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            name=["p", "li", "h1", "h2", "h3", "h4"]
        )
    ),
)
docs = loader.load()

# 3. Chunk and Split the Content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1_000,
    chunk_overlap=200,
)
splits = text_splitter.split_documents(docs)

# 4. Embed the Content
vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())

# 5. Retriever
retriever = vectorstore.as_retriever()

# 6. RAG Chat Prompt Template
prompt = hub.pull("rlm/rag-prompt") 

# 7. Parser
def format_docs(documents):
    """
    Concatenate retrieved docs into a single string.
    """
    return "\n\n".join(d.page_content for d in documents)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 8. main
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RAG query")
    parser.add_argument("question", nargs="+", help="Question to ask")
    args = parser.parse_args()
    question = " ".join(args.question)

    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print()

if __name__ == "__main__":
    main()
