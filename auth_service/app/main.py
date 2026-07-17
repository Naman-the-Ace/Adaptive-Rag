"""
Auth Service — FastAPI application entry point.
Python migration of the Rust Agentic-BFF (actix-web).

Services:
    - User registration and login (JWT-based)
    - Protected /user/me profile endpoint
    - Authenticated /agent/chat proxy to Adaptive-RAG backend
"""

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import auth, user, agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB tables on startup."""
    logger.info("Initializing database…")
    await init_db()
    logger.info("Auth service ready.")
    yield
    logger.info("Auth service shutting down.")


app = FastAPI(
    title="Adaptive-RAG Auth Service",
    description=(
        "Backend-For-Frontend authentication layer for the Adaptive-RAG system. "
        "Handles user registration, JWT login, and proxies authenticated requests "
        "to the RAG pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(agent.router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health probe for load balancers and container orchestrators."""
    return {"status": "ok", "service": "auth-service"}
