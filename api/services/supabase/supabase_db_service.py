from typing import Any, Dict, List, Optional, TypeVar, Generic, cast
from fastapi import Depends, status
from postgrest.exceptions import APIError
from pydantic import BaseModel

from api.core.logger import log
from api.interfaces.database_service import DatabaseService, DatabaseError
from api.services.supabase.supabase_client import SupabaseClient, get_supabase_client


T = TypeVar('T', bound=BaseModel)


class SupabaseDBService(DatabaseService, Generic[T]):
    """
    Generic service for handling Supabase database operations.
    Uses the singleton SupabaseClient for all operations.
    """

    def __init__(self, client: SupabaseClient, table_name: str):
        self.client = client
        self.table_name = table_name

    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new record.

        Args:
            data (Dict[str, Any]): Data to insert

        Returns:
            Dict[str, Any]: Created record
        """
        try:
            response = self.client.db(self.table_name).insert(data).execute()
            return cast(Dict[str, Any], response.data[0])
        except APIError as e:
            log.error(f"Failed to create record in {self.table_name}: {str(e)}")
            raise DatabaseError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Create operation failed: {str(e)}"
            )

    async def get(self, id: str) -> Dict[str, Any]:
        """
        Get a record by ID.

        Args:
            id (str): Record ID

        Returns:
            Dict[str, Any]: Retrieved record
        """
        try:
            response = self.client.db(self.table_name).select("*").eq("id", id).execute()
            if not response.data:
                raise DatabaseError(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Record not found in {self.table_name}"
                )
            return cast(Dict[str, Any], response.data[0])
        except APIError as e:
            log.error(f"Failed to get record from {self.table_name}: {str(e)}")
            raise DatabaseError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        select: str = "*"
    ) -> List[Dict[str, Any]]:
        """
        List records with optional filters.

        Args:
            filters (Optional[Dict[str, Any]]): Filter conditions
            select (str): Fields to select

        Returns:
            List[Dict[str, Any]]: List of records
        """
        try:
            query = self.client.db(self.table_name).select(select)
            if filters:
                for field, value in filters.items():
                    query = query.eq(field, value)
            response = query.execute()
            return cast(List[Dict[str, Any]], response.data)
        except APIError as e:
            log.error(f"Failed to list records from {self.table_name}: {str(e)}")
            raise DatabaseError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

    async def update(self, id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a record by ID.

        Args:
            id (str): Record ID
            data (Dict[str, Any]): Data to update

        Returns:
            Dict[str, Any]: Updated record
        """
        try:
            response = self.client.db(self.table_name).update(data).eq("id", id).execute()
            if not response.data:
                raise DatabaseError(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Record not found in {self.table_name}"
                )
            return cast(Dict[str, Any], response.data[0])
        except APIError as e:
            log.error(f"Failed to update record in {self.table_name}: {str(e)}")
            raise DatabaseError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

    async def delete(self, id: str) -> Dict[str, Any]:
        """
        Delete a record by ID.

        Args:
            id (str): Record ID

        Returns:
            Dict[str, Any]: Deleted record
        """
        try:
            response = self.client.db(self.table_name).delete().eq("id", id).execute()
            if not response.data:
                raise DatabaseError(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Record not found in {self.table_name}"
                )
            return cast(Dict[str, Any], response.data[0])
        except APIError as e:
            log.error(f"Failed to delete record from {self.table_name}: {str(e)}")
            raise DatabaseError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

    async def upsert(self, data: Dict[str, Any], unique_columns: List[str]) -> Dict[str, Any]:
        """
        Insert or update a record based on unique columns.

        Args:
            data (Dict[str, Any]): Data to upsert
            unique_columns (List[str]): Columns that determine uniqueness

        Returns:
            Dict[str, Any]: Upserted record
        """
        try:
            response = self.client.db(self.table_name).upsert(data, on_conflict=",".join(unique_columns)).execute()
            return cast(Dict[str, Any], response.data[0])
        except APIError as e:
            log.error(f"Failed to upsert record in {self.table_name}: {str(e)}")
            raise DatabaseError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )


def get_db_service(
    table_name: str,
    client: SupabaseClient = Depends(get_supabase_client)
) -> SupabaseDBService:
    """
    Create a database service instance for a specific table.
    This is a FastAPI dependency factory.
    """
    return SupabaseDBService(client, table_name) 
