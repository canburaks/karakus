import inspect
import os
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar, Union, cast
from urllib.parse import urlparse

import aiofiles
import httpx
from fastapi import HTTPException, status

from api.core.logger import log

T = TypeVar("T")


async def download_and_process(
    url: str,
    callback: Union[Callable[[Path], T], Callable[[Path], Awaitable[T]]],
    chunk_size: int = 8192,
) -> T:
    """
    Download a file to a temporary location and process it with a callback.
    Automatically handles cleanup after processing.

    Args:
        url (str): URL of the file to download
        callback (Union[Callable[[Path], T], Callable[[Path], Awaitable[T]]]): Sync or async function to process the file
        chunk_size (int, optional): Size of chunks for streaming. Defaults to 8192.

    Returns:
        T: Result from the callback function

    Raises:
        HTTPException: If download fails
    """
    # Validate URL
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL")
    except Exception as e:
        log.error(f"Invalid URL {url}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid URL provided"
        )

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = Path(temp_file.name)
    log.info(f"Downloading file to {temp_path}")

    try:
        # Download file
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            log.info(f"Sending GET request to {url}")
            response = await client.get(url)
            log.info(f"Response status: {response.status_code}")
            response.raise_for_status()

            # Write content to file
            content = response.content
            log.info(f"Downloaded content size: {len(content)} bytes")
            with open(temp_path, "wb") as f:
                f.write(content)
            log.info(f"Content written to {temp_path}")

        # Process file
        log.info(f"Processing file {temp_path}")
        result = callback(temp_path)
        if inspect.iscoroutine(result):
            result = await result
        log.info(f"Processed file {temp_path}")
        return cast(T, result)

    except httpx.HTTPError as e:
        log.error(f"Failed to download file from {url}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download file: {str(e)}",
        )
    except Exception as e:
        log.error(f"Error processing file from {url}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}",
        )
    finally:
        # Cleanup
        try:
            log.info(f"Cleaning up temporary file {temp_path}")
            if temp_path.exists():
                os.unlink(temp_path)
            log.info(f"Cleaned up temporary file {temp_path}")
        except Exception as e:
            log.error(f"Failed to cleanup temporary file {temp_path}: {str(e)}")


# Example usage:
# async def process_pdf(file_path: Path) -> str:
#     # Process PDF file
#     return "Processed content"
#
# content = await download_and_process(
#     "https://example.com/file.pdf",
#     process_pdf
# )
