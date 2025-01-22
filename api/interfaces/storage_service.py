# storage_service.py
from abc import ABC, abstractmethod
from io import BufferedReader, FileIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException, status


class StorageService(ABC):
    """
    Abstract base class defining the interface for cloud storage services.
    All concrete storage implementations must inherit from this class.
    """

    @abstractmethod
    async def create_bucket(self, bucket_name: str, **kwargs) -> Dict[str, Any]:
        """Create a new storage bucket"""
        pass

    @abstractmethod
    async def list_buckets(self) -> List[Dict[str, Any]]:
        """List all available buckets"""
        pass

    @abstractmethod
    async def delete_bucket(self, bucket_name: str) -> Dict[str, Any]:
        """Delete a storage bucket"""
        pass

    @abstractmethod
    async def upload_file(
        self,
        bucket_name: str,
        file_path: str,
        file: Union[BufferedReader, bytes, FileIO, str, Path],
        **kwargs,
    ) -> Dict[str, Any]:
        """Upload a file to storage"""
        pass

    @abstractmethod
    async def download_file(self, bucket_name: str, file_path: str) -> bytes:
        """Download a file from storage"""
        pass

    @abstractmethod
    async def list_files(
        self, bucket_name: str, path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List files in a bucket"""
        pass

    @abstractmethod
    async def delete_file(
        self, bucket_name: str, file_paths: List[str]
    ) -> Dict[str, Any]:
        """Delete files from storage"""
        pass

    @abstractmethod
    async def move_file(
        self, bucket_name: str, source_path: str, destination_path: str
    ) -> Dict[str, Any]:
        """Move/rename a file within storage"""
        pass

    @abstractmethod
    async def create_signed_url(
        self, bucket_name: str, file_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Generate a signed URL for temporary access"""
        pass


class StorageError(HTTPException):
    """Base exception for storage operations"""

    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "Storage operation failed",
    ):
        super().__init__(status_code=status_code, detail=detail)
