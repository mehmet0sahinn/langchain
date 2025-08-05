"""
Machine Translator – FastAPI + LangChain Demo
"""

from dotenv import load_dotenv
load_dotenv()

import logging
logger = logging.getLogger(__name__)


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI
from langserve import add_routes


# 1. Load the Model
model = ChatOpenAI()

# 2. Chat Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following text into {language}."),
        ("user", "{text}"),
    ]
)

# 3. Parser
chain = prompt | model | StrOutputParser()

# 4. FastAPI application
app = FastAPI(
    title="Machine Translator",
    version="1.0.0",
    description="LangChain‑powered translation microservice",
)

# 5. Adding Chain Route
add_routes(
    app,
    chain,
    path="/chain",
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting local dev server at http://localhost:8000 …")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
