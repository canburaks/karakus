from fastapi import HTTPException, UploadFile
from markitdown import MarkItDown
import tempfile
import os

md = MarkItDown()

async def parse_pdf(file: UploadFile) -> str:
    try:
        content = await file.read()
        
        # Create a temporary file to save the PDF content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        try:
            # Convert using the file path
            result = md.convert(temp_path)
            if not result or not result.text_content:
                raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
            return result.text_content
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")


async def read_text_file(file: UploadFile) -> str:
    content = await file.read()
    return content.decode('utf-8')
