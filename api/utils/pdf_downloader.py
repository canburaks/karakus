# pdf_downloader.py
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable

import httpx

from api.core.logger import log


async def download_pdf(
    url: str, callback: Callable[[Path], Awaitable[Any]], chunk_size: int = 1024 * 8
) -> AsyncGenerator[Any, None]:
    """
    Asynchronously downloads a PDF file and executes a callback with the temp file path.
    Ensures cleanup after callback execution.

    Args:
        url (str): URL of the PDF to download
        callback (Callable[[Path], Awaitable[Any]]): Async callback function that processes the PDF file
        chunk_size (int): Download chunk size in bytes

    Yields:
        dict: Progress updates and final result/error

    Usage:
        async for update in download_pdf(url, process_pdf):
            print(update)
    """
    log.info(f"Starting PDF download from {url}")

    # Configure client with longer timeouts and redirects
    timeout = httpx.Timeout(
        connect=600.0,  # connection timeout
        read=300.0,  # read timeout
        write=600.0,  # write timeout
        pool=600.0,  # pool timeout
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        verify=False,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as client:
        try:
            log.info("Sending GET request")
            response = await client.get(url)
            log.info(f"Response status: {response.status_code}")
            log.info(f"Response headers: {response.headers}")
            response.raise_for_status()

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_path = Path(temp_file.name)
                log.info(f"Saving to {temp_path}")
                temp_file.write(response.content)
                log.info(f"Saved {len(response.content)} bytes")

            try:
                log.info("Processing file")
                result = await callback(temp_path)
                log.info("Processing complete")
                yield {"status": "completed", "result": result}
            finally:
                if temp_path.exists():
                    os.unlink(temp_path)
                    log.info("Temporary file deleted")

        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error {e.response.status_code}: {str(e)}")
            yield {
                "status": "error",
                "message": f"HTTP error {e.response.status_code}: {str(e)}",
            }
        except Exception as e:
            log.error(f"Error: {str(e)}")
            yield {"status": "error", "message": str(e)}


# To run: asyncio.run(main())
