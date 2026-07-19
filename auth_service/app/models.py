"""
SQLAlchemy ORM models for the auth service.
Migrated from Rust models.rs (CreateUserRequest / LoginRequest) + rusqlite schema.
"""

from sqlalchemy import Column, DateTime, Integer, String, func

from app.database import Base


class User(Base):
    """Users table: stores username and Argon2id-hashed password."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
