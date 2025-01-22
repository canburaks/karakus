from io import BufferedReader, FileIO
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, cast
from fastapi import status
from supabase import Client, StorageException, create_client
from supabase.lib.client_options import SyncClientOptions
from concurrent.futures import ThreadPoolExecutor

from api.core.logger import log
from api.interfaces.storage_service import StorageService, StorageError
from api.services.supabase.config import get_supabase_settings
from api.utils.slugify import slugify
from api.utils.threadpool import run_in_threadpool

class FileOptions(TypedDict, total=False):
    contentType: str
    # Add other storage options as needed
    upsert: Optional[bool]


class SupabaseStorage(StorageService):
    """
    Service for interacting with Supabase Storage, handling file operations.
    
    Provides methods for uploading, downloading, listing, and managing files
    in Supabase Storage buckets.
    
    Attributes:
        base_url (str): Base URL of the Supabase instance
        key (str): API key for authentication
        headers (dict): Headers for authentication
    """

    client: Client

    def __init__(self, client: Optional[Client] = None) -> None:
        settings = get_supabase_settings()
        if client is not None:
            self.client = client
        else:
            self.client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY,
                options=SyncClientOptions(schema="public")
            )
        log.info(f"Initialized Supabase storage with URL: {settings.SUPABASE_URL}")

    async def create_bucket(self, bucket_name: str, **kwargs) -> Dict[str, Any]:
        """Create a storage bucket"""
        try:
            is_public = bool(kwargs.get('is_public', False))
            result = await run_in_threadpool(
                self.client.storage.create_bucket,
                id=slugify(bucket_name),
                name=bucket_name,
                options={"public": is_public}
            )
            return cast(Dict[str, Any], result)
        except StorageException as e:
            log.error(f"Failed to create bucket: {str(e)}")
            raise StorageError(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def list_buckets(self) -> List[Dict[str, Any]]:
        """List all storage buckets"""
        try:
            result = await run_in_threadpool(self.client.storage.list_buckets)
            return cast(List[Dict[str, Any]], result)
        except StorageException as e:
            raise StorageError(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def delete_bucket(self, bucket_name: str) -> Dict[str, Any]:
        """Delete a storage bucket"""
        try:
            result = await run_in_threadpool(
                self.client.storage.delete_bucket,
                bucket_name
            )
            return cast(Dict[str, Any], result)
        except StorageException as e:
            raise StorageError(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def upload_file(
        self,
        bucket_name: str,
        file_path: str,
        file: Union[BufferedReader, bytes, FileIO, str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """Upload a file to storage"""
        try:
            log.info(f"Attempting to upload file to bucket: {bucket_name}, path: {file_path}")
            
            file_options: Dict[str, Any] = {}
            if content_type := kwargs.get('content_type'):
                file_options['contentType'] = content_type
            
            bucket = self.client.storage.from_(bucket_name)
            result = await run_in_threadpool(
                bucket.upload,
                path=file_path,
                file=file,
                file_options=file_options
            )
            log.info(f"Upload successful: {result}")
            return cast(Dict[str, Any], result)
        except StorageException as e:
            log.error(f"Failed to upload file: {str(e)}")
            raise StorageError(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def download_file(self, bucket_name: str, file_path: str) -> bytes:
        """Download a file from storage"""
        try:
            result = await run_in_threadpool(
                self.client.storage.from_(bucket_name).download,
                file_path
            )
            return cast(bytes, result)
        except StorageException as e:
            raise StorageError(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def list_files(self, bucket_name: str, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files in a bucket"""
        try:
            bucket = self.client.storage.from_(bucket_name)
            result = await run_in_threadpool(bucket.list, path=path) if path else await run_in_threadpool(bucket.list)
            return cast(List[Dict[str, Any]], result)
        except StorageException as e:
            raise StorageError(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def delete_file(self, bucket_name: str, file_paths: List[str]) -> Dict[str, Any]:
        """Delete files from storage"""
        try:
            result = await run_in_threadpool(
                self.client.storage.from_(bucket_name).remove,
                file_paths
            )
            return cast(Dict[str, Any], result)
        except StorageException as e:
            raise StorageError(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def move_file(
        self,
        bucket_name: str,
        source_path: str,
        destination_path: str
    ) -> Dict[str, Any]:
        """Move/rename a file within storage"""
        try:
            result = await run_in_threadpool(
                self.client.storage.from_(bucket_name).move,
                source_path,
                destination_path
            )
            return cast(Dict[str, Any], result)
        except StorageException as e:
            raise StorageError(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def create_signed_url(self, bucket_name: str, file_path: str, **kwargs) -> Dict[str, Any]:
        """Generate a signed URL for temporary access"""
        try:
            expires_in = int(kwargs.get('expires_in', 3600))
            result = await run_in_threadpool(
                self.client.storage.from_(bucket_name).create_signed_url,
                file_path,
                expires_in
            )
            return cast(Dict[str, Any], result)
        except StorageException as e:
            log.error(f"Failed to create signed URL: {str(e)}")
            raise StorageError(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# Global instance
_storage_instance: Optional[SupabaseStorage] = None

def get_storage_service() -> SupabaseStorage:
    """
    Get or create a storage service instance.
    This is a FastAPI dependency that ensures only one service exists.
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = SupabaseStorage()
    return _storage_instance 
