from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from gotrue.types import AuthResponse as GoTrueAuthResponse
from gotrue.types import SignUpWithEmailAndPasswordCredentials
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

from api.core.logger import log
from api.models.auth import SignInRequest, SignUpRequest, TokenResponse
from api.services.supabase.config import SupabaseSettings, get_supabase_settings


class SupabaseService:
    """
    Service for interacting with Supabase, handling authentication and vector operations.

    Provides methods for user management, token handling, and vector similarity search
    using Supabase's vector extension.

    Attributes:
        client (Client): Supabase client instance
    """

    def __init__(self, settings: SupabaseSettings):
        self.client: Client = create_client(
            supabase_url=settings.SUPABASE_URL,
            supabase_key=settings.SUPABASE_KEY,
            options=SyncClientOptions(persist_session=True, auto_refresh_token=True),
        )

    async def sign_up(self, user_data: SignUpRequest) -> GoTrueAuthResponse:
        """
        Register a new user with Supabase.

        Args:
            user_data (SignUpRequest): User registration data

        Returns:
            GoTrueAuthResponse: Authentication response with user and session

        Raises:
            HTTPException: If registration fails
        """
        log.info(f"Supabase: Signing up user: {user_data}")
        try:
            credentials: SignUpWithEmailAndPasswordCredentials = {
                "email": str(user_data.email),
                "password": user_data.password,
                "options": {"data": user_data.metadata if user_data.metadata else {}},
            }
            auth_response = self.client.auth.sign_up(credentials)
            log.info(f"Supabase: Auth response: {auth_response}")
            if not auth_response.user or not auth_response.session:
                log.error(f"Supabase: Auth response: {auth_response}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to create user: {auth_response}",
                )
            return auth_response
        except Exception as e:
            log.error(f"Supabase: Auth response: {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def sign_in(self, credentials: SignInRequest) -> GoTrueAuthResponse:
        """
        Authenticate user with email and password.

        Args:
            credentials (SignInRequest): User login credentials

        Returns:
            GoTrueAuthResponse: Authentication response with user and session

        Raises:
            HTTPException: If authentication fails
        """
        try:
            auth_response = self.client.auth.sign_in_with_password(
                {
                    "email": str(credentials.email),
                    "password": credentials.password,
                    "options": {
                        "data": credentials.metadata if credentials.metadata else {}
                    },
                }
            )

            if not auth_response.user or not auth_response.session:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                )

            return auth_response
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
            )

    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        try:
            auth_response = self.client.auth.refresh_session(refresh_token)

            if not auth_response.session:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token",
                )

            return TokenResponse(
                access_token=str(auth_response.session.access_token),
                refresh_token=str(auth_response.session.refresh_token),
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

    async def sign_out(self) -> Dict[str, Any]:
        try:
            self.client.auth.sign_out()
            return {"message": "Successfully signed out"}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    def get_embeddings_collection(self, collection_name: str):
        """
        Retrieve embeddings from a named collection.

        Args:
            collection_name (str): Name of the collection to query

        Returns:
            Any: Collection data from Supabase
        """
        response = self.client.rpc(
            "get_embeddings_collection", {"collection_name": collection_name}
        ).execute()
        return response.data

    def upsert_embeddings(
        self,
        collection_name: str,
        embeddings: list[float],
        metadata: dict,
        id: Optional[str] = None,
    ):
        """
        Insert or update embeddings in a collection.

        Args:
            collection_name (str): Target collection name
            embeddings (list[float]): Embedding vector
            metadata (dict): Additional metadata
            id (Optional[str]): Unique identifier for the embedding

        Returns:
            Any: Response from Supabase
        """
        response = self.client.rpc(
            "upsert_embeddings",
            {
                "p_collection_name": collection_name,
                "p_embedding": embeddings,
                "p_metadata": metadata,
                "p_id": id,
            },
        ).execute()
        return response.data

    def search_embeddings(
        self,
        collection_name: str,
        query_embedding: list[float],
        similarity_threshold: float = 0.8,
        match_count: int = 10,
    ):
        """
        Search for similar embeddings in a collection.

        Args:
            collection_name (str): Collection to search in
            query_embedding (list[float]): Query vector
            similarity_threshold (float): Minimum similarity score
            match_count (int): Maximum number of matches to return

        Returns:
            Any: Matching embeddings from Supabase
        """
        response = self.client.rpc(
            "match_embeddings",
            {
                "p_collection_name": collection_name,
                "p_query_embedding": query_embedding,
                "p_similarity_threshold": similarity_threshold,
                "p_match_count": match_count,
            },
        ).execute()
        return response.data


# Global instance
_supabase_instance = None


async def get_supabase_service(
    settings: SupabaseSettings = Depends(get_supabase_settings),
) -> SupabaseService:
    global _supabase_instance
    if _supabase_instance is None:
        _supabase_instance = SupabaseService(settings)
    return _supabase_instance
