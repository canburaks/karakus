from typing import IO, Any, Dict, List, Optional, Union, Callable, Literal

from pydantic import BaseModel, Field
from unstructured.documents.elements import CompositeElement, Element, Text, Title, DataSourceMetadata
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
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
        default=PartitionStrategy.FAST,
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


class PartitionAutoConfig(BaseModel):
    filename: Optional[str] = Field(default=None, description="Target filename path")
    content_type: Optional[str] = Field(default=None, description="File content MIME type")
    file: Optional[IO[bytes]] = Field(default=None, description="File-like object in 'rb' mode")
    file_filename: Optional[str] = Field(default=None, description="Filename for file object (deprecated)")
    url: Optional[str] = Field(default=None, description="URL of document to parse")
    include_page_breaks: bool = Field(default=False, description="Include page break elements")
    strategy: str = Field(
        default=PartitionStrategy.FAST,
        description="Partitioning strategy (auto, hi_res, ocr_only, fast)"
    )
    encoding: Optional[str] = Field(default=None, description="Text encoding method")
    paragraph_grouper: Optional[Callable[[str], str]] | Literal[False] = Field(
        default=None,
        description="Function to group paragraphs or False to disable"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers for URL requests"
    )
    skip_infer_table_types: List[str] = Field(
        default=["pdf", "jpg", "png", "heic"],
        description="Document types to skip table extraction"
    )
    ssl_verify: bool = Field(default=True, description="Verify SSL in HTTP requests")
    ocr_languages: Optional[str] = Field(default=None, description="OCR languages (deprecated)")
    languages: Optional[List[str]] = Field(default=None, description="Languages for text detection")
    detect_language_per_element: bool = Field(
        default=False,
        description="Detect language per element"
    )
    pdf_infer_table_structure: bool = Field(
        default=False,
        description="Extract table structure (deprecated)"
    )
    extract_images_in_pdf: bool = Field(
        default=False,
        description="Extract images from PDF (deprecated)"
    )
    extract_image_block_types: Optional[List[str]] = Field(
        default=None,
        description="Element types to extract as images"
    )
    extract_image_block_output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save extracted images"
    )
    extract_image_block_to_payload: bool = Field(
        default=False,
        description="Store images as base64 in metadata"
    )
    xml_keep_tags: bool = Field(
        default=False,
        description="Retain XML tags in output"
    )
    data_source_metadata: Optional[DataSourceMetadata] = Field(
        default=None,
        description="Additional metadata about data source"
    )
    metadata_filename: Optional[str] = Field(
        default=None,
        description="Custom filename for metadata"
    )
    request_timeout: Optional[int] = Field(
        default=None,
        description="Timeout for HTTP requests"
    )
    hi_res_model_name: Optional[str] = Field(
        default=None,
        description="Layout detection model for hi_res strategy"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Layout detection model name (deprecated)"
    )
    date_from_file_object: bool = Field(
        default=False,
        description="Infer last_modified from file bytes"
    )
    starting_page_number: int = Field(
        default=1,
        description="Starting page number for document"
    )

    class Config:
        arbitrary_types_allowed = True


def partition_pdf_file(config: PartitionPDFConfig) -> List[Element]:
    return partition_pdf(**config.model_dump())


def partition_html_from_url(config: PartitionHTMLConfig) -> List[Element]:
    return partition_html(**config.model_dump())


def partition_auto(config: PartitionAutoConfig) -> List[Element]:
    return partition(**config.model_dump())
