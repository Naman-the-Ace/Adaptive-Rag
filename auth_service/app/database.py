"""
SQLAlchemy async database setup using SQLite (aiosqlite).
Migrated from Rust rusqlite + AppState to Python SQLAlchemy 2.0 async.
"""

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv("AUTH_DATABASE_URL", "sqlite+aiosqlite:///./users.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """FastAPI dependency: yield an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
