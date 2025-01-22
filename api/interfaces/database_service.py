# database_service.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status


class DatabaseService(ABC):
    """
    Abstract base class defining the interface for database services.
    All concrete database implementations must inherit from this class.
    """

    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new record in the database"""
        pass

    @abstractmethod
    async def get(self, id: str) -> Dict[str, Any]:
        """Retrieve a record by its ID"""
        pass

    @abstractmethod
    async def list(
        self, filters: Optional[Dict[str, Any]] = None, select: str = "*"
    ) -> List[Dict[str, Any]]:
        """List records with optional filtering"""
        pass

    @abstractmethod
    async def update(self, id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing record"""
        pass

    @abstractmethod
    async def delete(self, id: str) -> Dict[str, Any]:
        """Delete a record by its ID"""
        pass

    @abstractmethod
    async def upsert(
        self, data: Dict[str, Any], unique_columns: List[str]
    ) -> Dict[str, Any]:
        """Upsert (insert or update) a record"""
        pass


class DatabaseError(HTTPException):
    """Base exception for database operations"""

    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "Database operation failed",
    ):
        super().__init__(status_code=status_code, detail=detail)
