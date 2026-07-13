"""
API client for communicating with the Auth Service and Adaptive-RAG backend.
Updated to use JWT Bearer token authentication (no X-API-Token).
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

# Service URLs (override via environment variables for deployment)
AUTH_BASE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8080")
RAG_BASE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")


# ── Auth Service calls ────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> dict:
    """
    Register a new user via POST /auth/register.

    Returns:
        {"success": True} or {"success": False, "error": "..."}.
    """
    try:
        resp = requests.post(
            f"{AUTH_BASE_URL}/auth/register",
            json={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 201:
            return {"success": True}
        return {"success": False, "error": resp.json().get("detail", "Registration failed.")}
    except requests.RequestException as exc:
        logger.error("Register error: %s", exc)
        return {"success": False, "error": str(exc)}


def login_user(username: str, password: str) -> dict:
    """
    Login via POST /auth/login and retrieve JWT token.

    Returns:
        {"success": True, "token": "..."} or {"success": False, "error": "..."}.
    """
    try:
        resp = requests.post(
            f"{AUTH_BASE_URL}/auth/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"success": True, "token": data["access_token"]}
        return {"success": False, "error": resp.json().get("detail", "Login failed.")}
    except requests.RequestException as exc:
        logger.error("Login error: %s", exc)
        return {"success": False, "error": str(exc)}


def get_user_profile(token: str) -> dict | None:
    """
    Fetch the current user profile via GET /user/me.

    Returns:
        User dict or None on failure.
    """
    try:
        resp = requests.get(
            f"{AUTH_BASE_URL}/user/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.RequestException as exc:
        logger.error("Profile fetch error: %s", exc)
        return None


# ── RAG Service calls ─────────────────────────────────────────────────────────

def query_backend(query: str, token: str, session_id: str) -> str:
    """
    Send a chat query via the Auth Service proxy (POST /agent/chat).
    The auth service forwards it to the RAG backend with session context.

    Returns:
        The AI-generated answer string.
    """
    try:
        resp = requests.post(
            f"{AUTH_BASE_URL}/agent/chat",
            json={"query": query},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("answer", "No answer received.")
        return f"Error {resp.status_code}: {resp.text}"
    except requests.RequestException as exc:
        logger.error("Query error: %s", exc)
        return f"Connection error: {exc}"


def document_upload_rag(file_bytes: bytes, filename: str, token: str) -> dict:
    """
    Upload a document directly to the RAG backend for indexing.

    Returns:
        Dict with indexing status.
    """
    try:
        resp = requests.post(
            f"{RAG_BASE_URL}/rag/documents/upload",
            files={"file": (filename, file_bytes)},
            headers={"Authorization": f"Bearer {token}"},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "message": resp.text}
    except requests.RequestException as exc:
        logger.error("Upload error: %s", exc)
        return {"status": "error", "message": str(exc)}
