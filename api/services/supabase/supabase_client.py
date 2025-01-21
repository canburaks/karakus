from functools import lru_cache
from typing import Optional

from fastapi import Depends
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

from api.core.logger import log
from api.services.supabase.config import SupabaseSettings, get_supabase_settings


class SupabaseClient:
    """
    Main Supabase client that handles the connection and provides
    access to Supabase services. This is a singleton class that
    ensures only one client instance exists.
    """
    _instance = None
    client: Client

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, settings: SupabaseSettings):
        if not hasattr(self, 'client'):
            try:
                self.client = create_client(
                    supabase_url=settings.SUPABASE_URL,
                    supabase_key=settings.SUPABASE_KEY,
                    options=SyncClientOptions(
                        schema="public",
                        headers={},
                        auto_refresh_token=True,
                        persist_session=True
                    )
                )
                log.info("Supabase client initialized successfully")
            except Exception as e:
                log.error(f"Failed to initialize Supabase client: {str(e)}")
                raise e

    @property
    def auth(self):
        """Get the auth client"""
        return self.client.auth

    @property
    def db(self):
        """Get the database client"""
        return self.client.table

    @property
    def storage(self):
        """Get the storage client"""
        return self.client.storage

    @property
    def functions(self):
        """Get the edge functions client"""
        return self.client.functions


@lru_cache()
def get_supabase_client(
    settings: SupabaseSettings = Depends(get_supabase_settings)
) -> SupabaseClient:
    """
    Get or create the Supabase client instance.
    This is a FastAPI dependency that ensures only one client exists.
    """
    return SupabaseClient(settings) 
