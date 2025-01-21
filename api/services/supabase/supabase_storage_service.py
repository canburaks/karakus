from io import BufferedReader, FileIO
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, cast
from fastapi import Depends, status
from supabase import StorageException

from api.core.logger import log
from api.interfaces.storage_service import StorageService, StorageError
from api.services.supabase.supabase_client import SupabaseClient, get_supabase_client


class FileOptions(TypedDict, total=False):
    contentType: str


class SupabaseStorage(StorageService):
    """
    Service for interacting with Supabase Storage, handling file operations.
    
    Provides methods for uploading, downloading, listing, and managing files
    in Supabase Storage buckets.
    
    Attributes:
        client (SupabaseClient): Supabase client instance
    """

    def __init__(self, client: SupabaseClient):
        self.client = client

    async def create_bucket(self, bucket_name: str, **kwargs) -> Dict[str, Any]:
        """
        Create a new storage bucket.

        Args:
            bucket_name (str): Name of the bucket to create
            is_public (bool): Whether the bucket should be public

        Returns:
            Dict[str, Any]: Created bucket information
        """
        try:
            is_public = kwargs.get('is_public', False)
            result = self.client.storage.create_bucket(bucket_name, options={"public": is_public})
            return cast(Dict[str, Any], result)
        except StorageException as e:
            log.error(f"Bucket creation failed: {str(e)}")
            raise StorageError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Bucket creation failed: {str(e)}"
            )

    async def list_buckets(self) -> List[Dict[str, Any]]:
        """
        List all storage buckets.

        Returns:
            List[Dict[str, Any]]: List of buckets
        """
        try:
            result = self.client.storage.list_buckets()
            return cast(List[Dict[str, Any]], result)
        except StorageException as e:
            log.error(f"Failed to list buckets: {str(e)}")
            raise StorageError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to list buckets: {str(e)}"
            )

    async def delete_bucket(self, bucket_name: str) -> Dict[str, Any]:
        """
        Delete a storage bucket.

        Args:
            bucket_name (str): Name of the bucket to delete

        Returns:
            Dict[str, Any]: Response data
        """
        try:
            result = self.client.storage.delete_bucket(bucket_name)
            return cast(Dict[str, Any], result)
        except StorageException as e:
            log.error(f"Failed to delete bucket {bucket_name}: {str(e)}")
            raise StorageError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to delete bucket: {str(e)}"
            )

    async def upload_file(
        self,
        bucket_name: str,
        file_path: str,
        file: Union[BufferedReader, bytes, FileIO, str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Upload a file to a storage bucket.

        Args:
            bucket_name (str): Target bucket name
            file_path (str): Path where the file will be stored
            file (Union[BufferedReader, bytes, FileIO, str, Path]): File to upload
            content_type (Optional[str]): Content type of the file

        Returns:
            Dict[str, Any]: Upload response data
        """
        try:
            file_options: Any = {"content-type": kwargs.get('content_type')} if kwargs.get('content_type') else None
            result = self.client.storage.from_(bucket_name).upload(
                path=file_path,
                file=file,
                file_options=file_options
            )
            return cast(Dict[str, Any], result)
        except StorageException as e:
            log.error(f"Failed to upload file to {bucket_name}/{file_path}: {str(e)}")
            raise StorageError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to upload file: {str(e)}"
            )

    async def download_file(self, bucket_name: str, file_path: str) -> bytes:
        """
        Download a file from a storage bucket.

        Args:
            bucket_name (str): Source bucket name
            file_path (str): Path to the file

        Returns:
            bytes: File content
        """
        try:
            return self.client.storage.from_(bucket_name).download(file_path)
        except StorageException as e:
            log.error(f"Failed to download file from {bucket_name}/{file_path}: {str(e)}")
            raise StorageError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download file: {str(e)}"
            )

    async def list_files(self, bucket_name: str, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files in a storage bucket.

        Args:
            bucket_name (str): Target bucket name
            path (Optional[str]): Path prefix to filter files

        Returns:
            List[Dict[str, Any]]: List of files
        """
        try:
            result = self.client.storage.from_(bucket_name).list(path)
            return cast(List[Dict[str, Any]], result)
        except StorageException as e:
            log.error(f"Failed to list files in {bucket_name}: {str(e)}")
            raise StorageError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to list files: {str(e)}"
            )

    async def delete_file(self, bucket_name: str, file_paths: List[str]) -> Dict[str, Any]:
        """
        Delete files from a storage bucket.

        Args:
            bucket_name (str): Target bucket name
            file_paths (List[str]): Paths of files to delete

        Returns:
            Dict[str, Any]: Deletion response data
        """
        try:
            result = self.client.storage.from_(bucket_name).remove(file_paths)
            return cast(Dict[str, Any], result)
        except StorageException as e:
            log.error(f"Failed to delete files from {bucket_name}: {str(e)}")
            raise StorageError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to delete files: {str(e)}"
            )

    async def move_file(
        self,
        bucket_name: str,
        source_path: str,
        destination_path: str
    ) -> Dict[str, Any]:
        """
        Move/rename a file within a storage bucket.

        Args:
            bucket_name (str): Target bucket name
            source_path (str): Current file path
            destination_path (str): New file path

        Returns:
            Dict[str, Any]: Move response data
        """
        try:
            result = self.client.storage.from_(bucket_name).move(
                source_path,
                destination_path
            )
            return cast(Dict[str, Any], result)
        except StorageException as e:
            log.error(f"Failed to move file in {bucket_name}: {str(e)}")
            raise StorageError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to move file: {str(e)}"
            )

    async def create_signed_url(
        self,
        bucket_name: str,
        file_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a signed URL for temporary file access.

        Args:
            bucket_name (str): Target bucket name
            file_path (str): Path to the file
            expires_in (int): URL expiration time in seconds

        Returns:
            Dict[str, Any]: Signed URL data
        """
        try:
            expires_in = kwargs.get('expires_in', 3600)
            result = self.client.storage.from_(bucket_name).create_signed_url(
                file_path,
                expires_in
            )
            return cast(Dict[str, Any], result)
        except StorageException as e:
            log.error(f"Failed to create signed URL for {bucket_name}/{file_path}: {str(e)}")
            raise StorageError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create signed URL: {str(e)}"
            )


# Global instance
_storage_instance = None


async def get_storage_service(
    client: SupabaseClient = Depends(get_supabase_client)
) -> SupabaseStorage:
    """
    Get or create a SupabaseStorage service instance.

    Args:
        client (SupabaseClient): Supabase client instance

    Returns:
        SupabaseStorage: Storage service instance
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = SupabaseStorage(client)
    return _storage_instance 
