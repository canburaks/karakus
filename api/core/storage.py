from enum import Enum
from typing import Optional, Type

from supabase import Client, create_client

from api.interfaces.storage_service import StorageService
from api.services.supabase.config import get_supabase_settings
from api.services.supabase.supabase_client import SupabaseClient
from api.services.supabase.supabase_storage_service import SupabaseStorage


class StorageProvider(str, Enum):
    """Supported storage providers"""

    SUPABASE = "supabase"
    # Add more providers here as needed
    # AWS_S3 = "aws_s3"
    # GOOGLE_CLOUD = "google_cloud"


class StorageClient:
    """
    Factory class for creating storage service instances.
    Supports multiple storage providers while maintaining a single interface.
    """

    _instance: Optional[StorageService] = None
    _provider: Optional[StorageProvider] = None
    _supabase_client: Optional[Client] = None

    @classmethod
    def initialize(cls, provider: StorageProvider = StorageProvider.SUPABASE) -> None:
        """
        Initialize the storage client with a specific provider.

        Args:
            provider (StorageProvider): The storage provider to use
        """
        cls._provider = provider
        cls._instance = None
        cls._supabase_client = None

    @classmethod
    def get_instance(cls) -> StorageService:
        """
        Get or create a storage service instance.

        Returns:
            StorageService: Storage service instance
        """
        if not cls._instance:
            if not cls._provider:
                cls.initialize()

            if cls._provider == StorageProvider.SUPABASE:
                # Initialize Supabase client if not exists
                if not cls._supabase_client:
                    settings = get_supabase_settings()
                    cls._supabase_client = create_client(
                        settings.SUPABASE_URL, settings.SUPABASE_KEY
                    )
                cls._instance = SupabaseStorage(cls._supabase_client)
            # Add more provider initializations here
            # elif cls._provider == StorageProvider.AWS_S3:
            #     cls._instance = AWSS3Storage()
            else:
                raise ValueError(f"Unsupported storage provider: {cls._provider}")

        if not cls._instance:
            raise ValueError("Failed to initialize storage service")

        return cls._instance


# Initialize with default provider
StorageClient.initialize()
