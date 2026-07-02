"""
Main FastAPI application entry point for the Adaptive-RAG backend.
Uses Google Gemini (via langchain-google-genai) instead of OpenAI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="Adaptive RAG API",
    description=(
        "Adaptive Retrieval-Augmented Generation system powered by LangGraph. "
        "Dynamically routes queries between vector search (FAISS), general LLM "
        "knowledge, and real-time web search (Tavily)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "adaptive-rag-api"}
