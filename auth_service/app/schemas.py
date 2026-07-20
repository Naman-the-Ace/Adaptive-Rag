"""
Pydantic v2 request/response schemas for the auth service.
Migrated from Rust models.rs structs (CreateUserRequest, LoginRequest, TokenResponse).
"""

from pydantic import BaseModel, Field


class CreateUserRequest(BaseModel):
    """Schema for POST /auth/register."""

    username: str = Field(..., min_length=3, max_length=64, examples=["alice"])
    password: str = Field(..., min_length=6, max_length=128, examples=["s3cr3t!"])


class LoginRequest(BaseModel):
    """Schema for POST /auth/login."""

    username: str = Field(..., examples=["alice"])
    password: str = Field(..., examples=["s3cr3t!"])


class TokenResponse(BaseModel):
    """JWT token returned on successful login."""

    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Public user info returned from GET /user/me."""

    id: int
    username: str

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    """Schema for POST /agent/chat — proxied to the RAG service."""

    query: str = Field(..., min_length=1, examples=["What is Adaptive RAG?"])
