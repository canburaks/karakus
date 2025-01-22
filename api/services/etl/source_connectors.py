import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union

import aiofiles
import aiohttp
from pydantic import BaseModel, Field
from unstructured.documents.elements import Element

from .extract import DocumentExtractor, ExtractionOptions


class SourceConfig(BaseModel):
    """Base configuration for source connectors."""

    extraction_options: Optional[ExtractionOptions] = None


class FileSourceConfig(SourceConfig):
    """Configuration for file source connector."""

    base_path: Path = Field(..., description="Base path for file operations")
    recursive: bool = Field(default=True, description="Recursively scan directories")
    file_patterns: List[str] = Field(
        default=["*"], description="File patterns to match"
    )
    exclude_patterns: List[str] = Field(default=[], description="Patterns to exclude")


class WebSourceConfig(SourceConfig):
    """Configuration for web source connector."""

    base_url: str = Field(..., description="Base URL for web operations")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")


class SourceConnector(ABC):
    """Abstract base class for source connectors."""

    def __init__(self, config: SourceConfig):
        self.config = config
        self.extractor = DocumentExtractor(options=config.extraction_options)

    @abstractmethod
    async def extract(self) -> AsyncGenerator[Element, None]:
        """Extract content from the source."""
        pass


class FileSourceConnector(SourceConnector):
    """Connector for extracting content from files."""

    def __init__(self, config: FileSourceConfig):
        super().__init__(config)
        self.config: FileSourceConfig = config

    async def _get_files(self) -> List[Path]:
        """Get list of files matching patterns."""
        files = []
        base_path = self.config.base_path

        if self.config.recursive:
            for pattern in self.config.file_patterns:
                files.extend(base_path.rglob(pattern))
        else:
            for pattern in self.config.file_patterns:
                files.extend(base_path.glob(pattern))

        # Apply exclusion patterns
        if self.config.exclude_patterns:
            excluded = set()
            for pattern in self.config.exclude_patterns:
                if self.config.recursive:
                    excluded.update(base_path.rglob(pattern))
                else:
                    excluded.update(base_path.glob(pattern))
            files = [f for f in files if f not in excluded]

        return sorted(files)

    async def extract(self) -> AsyncGenerator[Element, None]:  # type: ignore
        """
        Extract content from files.

        Yields:
            Element: Document elements
        """
        files = await self._get_files()

        for file_path in files:
            try:
                document = self.extractor.extract_from_file(file_path)
                for element in document.elements:
                    yield element
            except Exception as e:
                # Log error and continue with next file
                print(f"Error processing {file_path}: {e}")
                continue


class WebSourceConnector(SourceConnector):
    """Connector for extracting content from web sources."""

    def __init__(self, config: WebSourceConfig):
        super().__init__(config)
        self.config: WebSourceConfig = config
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers=self.config.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            )
        return self.session

    async def _fetch_url(self, url: str) -> str:
        """Fetch content from URL with retries."""
        session = await self._get_session()
        retries = self.config.max_retries

        while retries > 0:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                retries -= 1
                await asyncio.sleep(1)
                if retries == 0:
                    raise e
        raise Exception("Failed to fetch URL after maximum retries")

    async def extract(self) -> AsyncGenerator[Element, None]:  # type: ignore
        """
        Extract content from web sources.

        Yields:
            Element: Document elements
        """
        try:
            content = await self._fetch_url(self.config.base_url)
            document = self.extractor.extract_from_text(content, content_type="html")
            for element in document.elements:
                yield element
        finally:
            if self.session:
                await self.session.close()
