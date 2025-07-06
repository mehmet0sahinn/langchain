"""
Machine Translator – FastAPI + LangChain Demo
===========================================
Run with:  `uvicorn main:app --reload`
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

load_dotenv()

logger = logging.getLogger(__name__)

# 1) Load the Model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,    # low temperature for consistent translations
)

# 2) Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following text into {language}."),
        ("user", "{text}"),
    ]
)

# 3) Chain = prompt » model » output parser
chain = prompt | model | StrOutputParser()

# 4) FastAPI application
app = FastAPI(
    title="Machine Translator",
    version="1.0.0",
    description="LangChain‑powered translation microservice",
)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5) Expose the chain at /chain
add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting local dev server at http://localhost:8000 …")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
