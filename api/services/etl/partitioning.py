from typing import IO, Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from unstructured.documents.elements import CompositeElement, Element, Text, Title
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy


class PartitionPDFConfig(BaseModel):
    filename: Optional[str] = Field(default=None, description="Target filename path")
    file: Optional[IO[bytes]] = Field(
        default=None, description="File-like object in 'rb' mode"
    )
    include_page_breaks: bool = Field(
        default=False, description="Include PageBreak elements between pages"
    )
    strategy: str = Field(
        default=PartitionStrategy.HI_RES,
        description="PDF partitioning strategy (hi_res, ocr_only, fast, or auto)",
    )
    infer_table_structure: bool = Field(
        default=True,
        description="Extract table structure as HTML when using hi_res strategy",
    )
    languages: Optional[List[str]] = Field(
        default=["tur"],
        description="Languages for partitioning/OCR, requires Tesseract language packs",
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata in output"
    )
    metadata_filename: Optional[str] = Field(
        default=None, description="Custom filename for metadata"
    )
    metadata_last_modified: Optional[str] = Field(
        default=None, description="Last modified date"
    )
    chunking_strategy: Optional[str] = Field(
        default=None, description="Strategy for chunking text"
    )
    hi_res_model_name: Optional[str] = Field(
        default=None, description="Layout detection model for hi_res strategy"
    )
    extract_image_block_types: Optional[List[str]] = Field(
        default=["Image", "Table"],
        description="Element types to extract as images (e.g. ['Image', 'Table'])",
    )
    extract_image_block_output_dir: Optional[str] = Field(
        default="static/assets/junk/pdf_images/",
        description="Directory to save extracted images",
    )
    extract_image_block_to_payload: bool = Field(
        default=False, description="Store extracted images as base64 in metadata"
    )
    date_from_file_object: bool = Field(
        default=False, description="Infer last_modified from file bytes"
    )
    starting_page_number: int = Field(default=1, description="Starting page number")
    extract_forms: bool = Field(
        default=False, description="Extract form fields as FormKeysValues elements"
    )
    form_extraction_skip_tables: bool = Field(
        default=True, description="Skip form extraction in table regions"
    )

    class Config:
        arbitrary_types_allowed = True


class PartitionHTMLConfig(BaseModel):
    filename: Optional[str] = Field(default=None, description="Target filename path")
    file: Optional[IO[bytes]] = Field(
        default=None, description="File-like object in 'r' mode"
    )
    text: Optional[str] = Field(
        default=None, description="String representation of HTML document"
    )
    encoding: Optional[str] = Field(
        default=None, description="Encoding method for text input, defaults to utf-8"
    )
    url: Optional[str] = Field(
        default=None, description="URL of webpage to parse (must return HTML)"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict, description="HTTP headers for URL requests"
    )
    ssl_verify: bool = Field(
        default=True, description="Whether to verify SSL in HTTP requests"
    )
    date_from_file_object: bool = Field(
        default=False, description="Infer last_modified from file bytes"
    )
    detect_language_per_element: bool = Field(
        default=False, description="Detect language for each element individually"
    )
    languages: Optional[List[str]] = Field(
        default=["auto"], description="Languages for text detection"
    )
    metadata_last_modified: Optional[str] = Field(
        default=None, description="Last modified date for the document"
    )
    skip_headers_and_footers: bool = Field(
        default=False, description="Ignore content within <header> or <footer> tags"
    )
    detection_origin: Optional[str] = Field(
        default=None, description="Origin of detection for debugging"
    )

    class Config:
        arbitrary_types_allowed = True


def partition_pdf_file(config: PartitionPDFConfig) -> List[Element]:
    return partition_pdf(**config.model_dump())


def partition_html_file(config: PartitionHTMLConfig) -> List[Element]:
    return partition_html(**config.model_dump())
