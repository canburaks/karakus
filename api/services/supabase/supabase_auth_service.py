from typing import Any, Dict, Optional, cast

from fastapi import Depends, status
from gotrue.errors import AuthError as SupabaseAuthError
from gotrue.types import (
    AdminUserAttributes,
    AuthResponse,
    User,
    UserAttributes,
    UserResponse,
    VerifyOtpParams,
)
from pydantic import EmailStr

from api.core.logger import log
from api.interfaces.auth_service import AuthError, AuthService
from api.services.supabase.supabase_client import SupabaseClient, get_supabase_client


class SupabaseAuthService(AuthService):
    """
    Service for handling Supabase authentication operations.
    Uses the singleton SupabaseClient for all operations.
    """

    def __init__(self, client: SupabaseClient):
        self.client = client

    async def sign_up(self, email: str, password: str, **kwargs) -> Dict[str, Any]:
        """
        Register a new user.

        Args:
            email (str): User's email
            password (str): User's password
            **kwargs: Additional user metadata

        Returns:
            Dict[str, Any]: Authentication response with user and session
        """
        try:
            response = self.client.auth.sign_up(
                {
                    "email": email,
                    "password": password,
                    "options": {"data": kwargs} if kwargs else {},
                }
            )
            session = cast(AuthResponse, response).session
            return {
                "access_token": session.access_token if session else None,
                "refresh_token": session.refresh_token if session else None,
                "expires_in": session.expires_in if session else None,
            }
        except SupabaseAuthError as e:
            log.error(f"Failed to sign up user {email}: {str(e)}")
            raise AuthError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Registration failed: {str(e)}",
            )

    async def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """
        Sign in an existing user.

        Args:
            email (str): User's email
            password (str): User's password

        Returns:
            Dict[str, Any]: Authentication response with user and session
        """
        try:
            response = self.client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            session = cast(AuthResponse, response).session
            return {
                "access_token": session.access_token if session else None,
                "refresh_token": session.refresh_token if session else None,
                "expires_in": session.expires_in if session else None,
            }
        except SupabaseAuthError as e:
            log.error(f"Failed to sign in user {email}: {str(e)}")
            raise AuthError(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
            )

    async def sign_out(self, access_token: str) -> Dict[str, Any]:
        """Sign out the current user."""
        try:
            self.client.auth.sign_out()
            return {"message": "Successfully signed out"}
        except SupabaseAuthError as e:
            log.error(f"Failed to sign out: {str(e)}")
            raise AuthError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Sign out failed: {str(e)}",
            )

    async def get_current_user(self, access_token: str) -> Dict[str, Any]:
        """Get current user information."""
        try:
            session = self.client.auth.get_session()
            user = cast(AuthResponse, session).user
            return (
                {
                    "id": user.id,
                    "email": user.email,
                    "phone": user.phone,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at,
                }
                if user
                else {}
            )
        except SupabaseAuthError as e:
            log.error(f"Failed to get current user: {str(e)}")
            raise AuthError(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session",
            )

    async def update_user(
        self, access_token: str, attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user data.

        Args:
            access_token (str): User's access token
            attributes (Dict[str, Any]): Data to update

        Returns:
            Dict[str, Any]: Updated user information
        """
        try:
            user_attrs = UserAttributes(**attributes)
            response = self.client.auth.update_user(user_attrs)
            user = cast(AuthResponse, response).user
            return (
                {
                    "id": user.id,
                    "email": user.email,
                    "phone": user.phone,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at,
                }
                if user
                else {}
            )
        except SupabaseAuthError as e:
            log.error(f"Failed to update user: {str(e)}")
            raise AuthError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to update user: {str(e)}",
            )

    async def delete_user(self, access_token: str) -> Dict[str, Any]:
        """
        Delete a user.

        Args:
            access_token (str): User's access token

        Returns:
            Dict[str, Any]: Response indicating user deletion
        """
        try:
            session = self.client.auth.get_session()
            user = cast(AuthResponse, session).user
            if user and user.id:
                self.client.auth.admin.delete_user(user.id)
                return {"message": "User successfully deleted"}
            raise AuthError(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )
        except SupabaseAuthError as e:
            log.error(f"Failed to delete user: {str(e)}")
            raise AuthError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to delete user: {str(e)}",
            )

    async def reset_password(self, email: str) -> Dict[str, Any]:
        """
        Send a password reset email to the user's email.

        Args:
            email (str): User's email

        Returns:
            Dict[str, Any]: Response indicating password reset email sent
        """
        try:
            self.client.auth.reset_password_email(email)
            return {"message": "Password reset email sent"}
        except SupabaseAuthError as e:
            log.error(f"Failed to send reset password email: {str(e)}")
            raise AuthError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to send reset password email: {str(e)}",
            )

    async def verify_email(self, access_token: str) -> Dict[str, Any]:
        """
        Verify the user's email.

        Args:
            access_token (str): User's access token

        Returns:
            Dict[str, Any]: Authentication response with verified session
        """
        try:
            params = cast(VerifyOtpParams, {"token": access_token, "type": "email"})
            response = self.client.auth.verify_otp(params)
            session = cast(AuthResponse, response).session
            return {
                "access_token": session.access_token if session else None,
                "refresh_token": session.refresh_token if session else None,
                "expires_in": session.expires_in if session else None,
            }
        except SupabaseAuthError as e:
            log.error(f"Failed to verify email: {str(e)}")
            raise AuthError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to verify email: {str(e)}",
            )

    async def refresh_session(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh the user's session.

        Args:
            refresh_token (str): The refresh token

        Returns:
            Dict[str, Any]: New authentication response
        """
        try:
            response = self.client.auth.refresh_session(refresh_token)
            session = cast(AuthResponse, response).session
            return {
                "access_token": session.access_token if session else None,
                "refresh_token": session.refresh_token if session else None,
                "expires_in": session.expires_in if session else None,
            }
        except SupabaseAuthError as e:
            log.error(f"Failed to refresh session: {str(e)}")
            raise AuthError(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )


# Global instance
_auth_instance = None


async def get_auth_service(
    client: SupabaseClient = Depends(get_supabase_client),
) -> SupabaseAuthService:
    """
    Get or create the auth service instance.
    This is a FastAPI dependency that ensures only one service exists.
    """
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = SupabaseAuthService(client)
    return _auth_instance
