# auth_service.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from fastapi import HTTPException, status
from pydantic import BaseModel


class AuthService(ABC):
    """
    Abstract base class defining the interface for authentication services.
    All concrete authentication implementations must inherit from this class.
    """
    
    @abstractmethod
    async def sign_up(self, email: str, password: str, **kwargs) -> Dict[str, Any]:
        """Register a new user"""
        pass

    @abstractmethod
    async def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate a user"""
        pass

    @abstractmethod
    async def sign_out(self, access_token: str) -> Dict[str, Any]:
        """Invalidate user session"""
        pass

    @abstractmethod
    async def get_current_user(self, access_token: str) -> Dict[str, Any]:
        """Get current authenticated user"""
        pass

    @abstractmethod
    async def update_user(
        self, 
        access_token: str,
        attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update user attributes"""
        pass

    @abstractmethod
    async def delete_user(self, access_token: str) -> Dict[str, Any]:
        """Permanently delete user"""
        pass

    @abstractmethod
    async def reset_password(self, email: str) -> Dict[str, Any]:
        """Initiate password reset"""
        pass

    @abstractmethod
    async def verify_email(self, access_token: str) -> Dict[str, Any]:
        """Verify user email address"""
        pass

    @abstractmethod
    async def refresh_session(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token"""
        pass


class AuthError(HTTPException):
    """Base exception for authentication operations"""
    def __init__(
        self,
        status_code: int = status.HTTP_401_UNAUTHORIZED,
        detail: str = "Authentication failed"
    ):
        super().__init__(status_code=status_code, detail=detail)
