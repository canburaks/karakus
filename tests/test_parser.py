import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import httpx
import numpy as np
import pytest
from fastapi import FastAPI, File, UploadFile

sys.path.append("./static/assets/test_files")
from pathlib import Path

from api.core.parser import parse_pdf

PDF_TEST_FILE = "./static/assets/test_files/software-development-lifecycle.pdf"


def file_exists(file_path: str) -> bool:
    """Check if a file exists in the specified path."""
    return os.path.exists(file_path)


@pytest.fixture
def pdf_file():
    # Get the absolute path to the test file
    file_path = Path(PDF_TEST_FILE).absolute()

    # Open the file in binary read mode
    with open(file_path, "rb") as f:
        # Create SpooledTemporaryFile to simulate file upload
        file = UploadFile(file=f, filename="software-development-lifecycle.pdf")
        yield file
        # Cleanup
        file.file.close()


@pytest.mark.asyncio
async def test_parse_pdf(pdf_file: UploadFile):
    result = await parse_pdf(pdf_file)
    assert result is not None
    assert len(result) > 0
