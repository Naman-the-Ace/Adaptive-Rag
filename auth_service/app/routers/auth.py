"""
Auth router: user registration and login endpoints.
Migrated from Rust controllers/user_controller.rs.

Endpoints:
    POST /auth/register  – Create a new user account
    POST /auth/login     – Authenticate and receive JWT
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.auth import create_access_token, hash_password, verify_password
from app.database import get_db
from app.models import User
from app.schemas import CreateUserRequest, LoginRequest, TokenResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", status_code=status.HTTP_201_CREATED, summary="Register a new user")
async def register(payload: CreateUserRequest, db: AsyncSession = Depends(get_db)):
    """
    Create a new user account with an Argon2id-hashed password.

    - Usernames must be unique (returns 409 if taken).
    - Passwords are never stored in plain-text.
    """
    hashed = hash_password(payload.password)
    user = User(username=payload.username, password_hash=hashed)

    db.add(user)
    try:
        await db.commit()
        await db.refresh(user)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{payload.username}' is already taken.",
        )

    logger.info("New user registered: %s", payload.username)
    return {"status": "created", "username": user.username}


@router.post("/login", response_model=TokenResponse, summary="Login and receive JWT")
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate a user and return a signed JWT access token.

    The token must be sent as `Authorization: Bearer <token>` on protected routes.
    """
    result = await db.execute(select(User).where(User.username == payload.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user.username})
    logger.info("User logged in: %s", user.username)
    return TokenResponse(access_token=token)
