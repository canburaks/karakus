from fastapi import Request, Depends
from pathlib import Path
from typing import Dict, Any, Callable, Awaitable
from api.services.etl.partitioning import (
    partition_auto,
    PartitionAutoConfig,
)
from api.core.logger import log
from api.utils.pdf_downloader import download_pdf
from api.core.storage import StorageClient
from api.interfaces.storage_service import StorageService
from api.services.llama_index.llama_index_service import get_llama_index_service
from api.models.documents import YasalBelge
from api.core.constants import PDF_DOWNLOAD_PATH
import tempfile
import os


SAMPLE_PDF_URL = "https://www.resmigazete.gov.tr/eskiler/2023/01/20230102.pdf"
SAMPLE_HTML_URL = "https://www.meb.gov.tr/mevzuat/liste.php?ara=6"
STORAGE_BUCKET = "karakus-static"  # Default bucket for PDFs
STORED_PDF_PATH = "tmp3awwuhkr.pdf"


async def method_test(request: Request) -> Dict[str, Any]:
    await method_llama_index_s3_load(request)
    return {"message": "Hello, World!"}


async def method_llama_index_s3_load(request: Request) -> Dict[str, Any]:
    storage = StorageClient.get_instance()
    
    # Download file bytes from Supabase
    file_bytes = await storage.download_file(
        bucket_name=STORAGE_BUCKET, 
        file_path=STORED_PDF_PATH
    )
    
    # Create a temporary file with .pdf extension
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
    try:
        llama_index_service = get_llama_index_service()
        pdf_loader_documents = llama_index_service.read_files(
            input_files=[temp_path],
            recursive=False,
            encoding="utf-8",
            filename_as_id=True,
            required_exts=None,
            file_metadata=None,
            num_workers=1  # Reduce worker count to prevent timeouts
        )
        
        doc_text = "\n\n".join([doc.text for doc in pdf_loader_documents])
        log.info(f"Processing document text of length: {len(doc_text)}")

        sllm = llama_index_service.get_sllm(
            provider="ollama",
            model="qwen2.5:7b",
            schema=YasalBelge
        )
        
        try:
            log.info("Attempting to extract structured information...")
            structured_response = sllm(doc_text)
            log.info("Successfully extracted structured information")
            log.debug(f"Structured response: {structured_response.model_dump_json(indent=2)}")
            
            # Clean up the temporary file
            os.unlink(temp_path)
            log.info("Cleaned up temporary file")
            
            return {
                "message": "Success", 
                "documents": pdf_loader_documents, 
                "structured_response": structured_response.model_dump()
            }
        except Exception as e:
            log.error(f"Error in structured output: {str(e)}")
            # Clean up on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                log.info("Cleaned up temporary file after error")
            raise ValueError(f"Failed to generate structured output: {str(e)}")
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        log.error(f"Error loading PDF: {str(e)}")
        raise ValueError(f"Failed to load PDF: {str(e)}")


async def process_and_upload_pdf(
    file_path: Path, storage_service: StorageService
) -> Dict[str, Any]:
    """Process PDF content and upload to storage"""
    try:
        # Process PDF
        elements = partition_auto(
            PartitionAutoConfig(filename=str(file_path), ssl_verify=False)
        )
        processed_text = "\n\n".join(element.text for element in elements)

        # Upload to storage
        file_name = file_path.name
        with open(file_path, "rb") as pdf_file:
            upload_result = await storage_service.upload_file(
                bucket_name=STORAGE_BUCKET,
                file_path=file_name,
                file=pdf_file,
                content_type="application/pdf",
            )

        return {
            "text": processed_text,
            "storage": {
                "bucket": STORAGE_BUCKET,
                "file_path": file_name,
                "upload_result": upload_result,
            },
        }
    except Exception as e:
        log.error(f"Error processing/uploading PDF: {e}")
        return {"error": str(e)}


async def uploadtest(request: Request) -> Dict[str, Any]:
    """
    Downloads a PDF, processes its content, and uploads to storage.
    Returns both the processed content and storage information.
    """
    try:
        # Get storage instance
        storage = StorageClient.get_instance()

        # Ensure storage bucket exists
        try:
            await storage.create_bucket(STORAGE_BUCKET, is_public=False)
        except Exception as e:
            log.info(f"Bucket might already exist: {e}")

        # Create callback with storage instance
        callback: Callable[[Path], Awaitable[Dict[str, Any]]] = (
            lambda path: process_and_upload_pdf(path, storage)
        )

        # Process with storage
        async for update in download_pdf(url=SAMPLE_PDF_URL, callback=callback):
            if update["status"] == "completed":
                return update["result"]
            elif update["status"] == "error":
                return {"error": update["message"]}

        return {"error": "Download failed"}
    except Exception as e:
        log.error(f"Error in method_test: {e}")
        return {"error": str(e)}
