from fastapi import Request
from pathlib import Path
from typing import Dict, Any
from api.services.etl.partitioning import (
    partition_auto,
    PartitionAutoConfig,
)
from api.core.logger import log
from api.utils.pdf_downloader import download_pdf

SAMPLE_PDF_URL = "https://www.resmigazete.gov.tr/eskiler/2023/01/20230102.pdf"
SAMPLE_HTML_URL = "https://www.meb.gov.tr/mevzuat/liste.php?ara=6"

async def process_pdf(file_path: Path) -> Dict[str, Any]:
    try:
        elements = partition_auto(PartitionAutoConfig(filename=str(file_path), ssl_verify=False))
        return {"text": "\n\n".join(element.text for element in elements)}
    except Exception as e:
        log.error(f"Error processing PDF: {e}")
        return {"error": str(e)}

async def method_test(request: Request) -> Dict[str, Any]:
    async for update in download_pdf(url=SAMPLE_PDF_URL, callback=process_pdf):
        if update["status"] == "completed":
            return update["result"]
        elif update["status"] == "error":
            return {"error": update["message"]}
    return {"error": "Download failed"}
