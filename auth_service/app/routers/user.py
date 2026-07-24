"""
User router: profile endpoints for authenticated users.
Migrated from Rust controllers/user_controller.rs (get current user info).

Endpoints:
    GET /user/me  – Return the current authenticated user's profile
"""

from fastapi import APIRouter, Depends

from app.dependencies import get_current_user
from app.models import User
from app.schemas import UserResponse

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/me", response_model=UserResponse, summary="Get current user profile")
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Return the profile of the currently authenticated user.

    Requires a valid `Authorization: Bearer <token>` header.
    """
    return current_user
