import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
from pydantic import BaseModel, Field
from unstructured.documents.elements import Element


class DestinationConfig(BaseModel):
    """Base configuration for destination connectors."""

    pass


class FileDestinationConfig(DestinationConfig):
    """Configuration for file destination connector."""

    output_dir: Path = Field(..., description="Output directory path")
    file_format: str = Field(default="txt", description="Output file format")
    encoding: str = Field(default="utf-8", description="File encoding")
    create_dirs: bool = Field(default=True, description="Create directories if needed")


class DatabaseDestinationConfig(DestinationConfig):
    """Configuration for database destination connector."""

    connection_string: str = Field(..., description="Database connection string")
    table_name: str = Field(..., description="Target table name")
    batch_size: int = Field(default=100, description="Batch size for inserts")
    create_table: bool = Field(default=True, description="Create table if not exists")


class DestinationConnector(ABC):
    """Abstract base class for destination connectors."""

    def __init__(self, config: DestinationConfig):
        self.config = config

    @abstractmethod
    async def write(self, elements: List[Element]) -> None:
        """Write elements to destination."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass


class FileDestinationConnector(DestinationConnector):
    """Connector for writing content to files."""

    def __init__(self, config: FileDestinationConfig):
        super().__init__(config)
        self.config: FileDestinationConfig = config
        if self.config.create_dirs:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

    async def write(self, elements: List[Element]) -> None:
        """
        Write elements to files.

        Args:
            elements (List[Element]): Elements to write
        """
        output_path = self.config.output_dir / f"output.{self.config.file_format}"

        async with aiofiles.open(
            output_path, mode="w", encoding=self.config.encoding
        ) as f:
            if self.config.file_format == "json":
                content = [
                    {
                        "text": str(element),
                        "type": element.__class__.__name__,
                        "metadata": getattr(element, "metadata", {}),
                    }
                    for element in elements
                ]
                await f.write(json.dumps(content, indent=2))
            else:
                for element in elements:
                    await f.write(str(element) + "\n")

    async def close(self) -> None:
        """No cleanup needed for file connector."""
        pass
