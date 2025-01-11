from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from .auth import AuthResponse, User
from .client import ClientDep

router = APIRouter(prefix="/auth", tags=["auth"])


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    phone: Optional[str] = None


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class ResetPasswordRequest(BaseModel):
    email: EmailStr


@router.post("/signup", response_model=AuthResponse)
async def sign_up(request: SignUpRequest, client=ClientDep) -> AuthResponse:
    try:
        return await client.auth.sign_up(
            {
                "email": request.email,
                "password": request.password,
                "phone": request.phone,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/signin", response_model=AuthResponse)
async def sign_in(request: SignInRequest, client=ClientDep) -> AuthResponse:
    try:
        return await client.auth.sign_in_with_password(
            {"email": request.email, "password": request.password}
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/signout")
async def sign_out(client=ClientDep):
    try:
        await client.auth.sign_out()
        return {"message": "Successfully signed out"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest, client=ClientDep):
    try:
        await client.auth.reset_password_for_email(request.email)
        return {"message": "Password reset email sent"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/user", response_model=User)
async def get_user(client=ClientDep) -> User:
    try:
        user = await client.auth.get_user()
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Not authenticated")
