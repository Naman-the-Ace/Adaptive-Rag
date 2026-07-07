"""
FastAPI request model for RAG query endpoint.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /rag/query."""

    query: str = Field(..., min_length=1, examples=["What is Adaptive RAG?"])
    session_id: str = Field(
        default="default",
        examples=["user_alice"],
        description="Unique identifier for chat history session (username from auth service).",
    )
