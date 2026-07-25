"""
Agent proxy router: forwards authenticated chat requests to the Adaptive RAG service.
Migrated from Rust controllers/chat_controller.rs.

Endpoints:
    POST /agent/chat  – Proxy a RAG query (JWT required)
"""

import logging
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_current_user
from app.models import User
from app.schemas import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Agent Proxy"])

# URL of the Adaptive-RAG FastAPI backend
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")


@router.post("/chat", summary="Send a chat query to the RAG service (JWT required)")
async def chat(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Authenticated proxy endpoint:
    1. Validates the Bearer JWT token.
    2. Forwards the query to the Adaptive-RAG FastAPI service.
    3. Returns the RAG-generated answer.

    The `session_id` is derived from the authenticated username for per-user
    chat history persistence in MongoDB.
    """
    request_body = {
        "query": payload.query,
        "session_id": current_user.username,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{RAG_SERVICE_URL}/rag/query",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        logger.error("RAG service timeout for user: %s", current_user.username)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="RAG service timed out. Please try again.",
        )
    except httpx.HTTPStatusError as exc:
        logger.error("RAG service error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RAG service returned error: {exc.response.status_code}",
        )

    logger.info("Chat query from user '%s' completed.", current_user.username)
    return data
