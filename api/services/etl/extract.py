from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from unstructured.cleaners.core import group_broken_paragraphs
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition
from unstructured.partition.common import convert_office_doc
from unstructured.partition.docx import partition_docx
from unstructured.partition.epub import partition_epub
from unstructured.partition.html import partition_html
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text


class ElementType(str, Enum):
    """Enum for different types of document elements"""

    FORMULA = "Formula"
    FIGURE_CAPTION = "FigureCaption"
    NARRATIVE_TEXT = "NarrativeText"
    LIST_ITEM = "ListItem"
    TITLE = "Title"
    ADDRESS = "Address"
    EMAIL_ADDRESS = "EmailAddress"
    IMAGE = "Image"
    PAGE_BREAK = "PageBreak"
    TABLE = "Table"
    HEADER = "Header"
    FOOTER = "Footer"
    CODE_SNIPPET = "CodeSnippet"
    PAGE_NUMBER = "PageNumber"
    UNCATEGORIZED_TEXT = "UncategorizedText"


class DocumentElement(BaseModel):
    """Model representing a document element with its type and content"""

    element_type: ElementType = Field(..., description="Type of the document element")
    content: str = Field(..., description="Content of the element")
    page_number: Optional[int] = Field(
        None, description="Page number where the element appears"
    )
    coordinates: Optional[tuple[float, float, float, float]] = Field(
        None, description="Coordinates of the element in the document (x1, y1, x2, y2)"
    )

    @classmethod
    def from_unstructured_element(cls, element: Element) -> "DocumentElement":
        """Convert unstructured Element to DocumentElement."""
        element_type = ElementType(element.__class__.__name__)
        return cls(
            element_type=element_type,
            content=str(element),
            page_number=getattr(element, "page_number", None),
            coordinates=getattr(element, "coordinates", None),
        )

    class Config:
        arbitrary_types_allowed = True


class EmphasisTag(str, Enum):
    """Types of text emphasis"""

    BOLD = "bold"
    ITALIC = "italic"


class MetadataFields(BaseModel):
    """Metadata fields for document elements"""

    filename: str = Field(..., description="Filename of the document")

    file_directory: str = Field(..., description="Directory path of the file")

    last_modified: datetime = Field(
        ..., description="Last modified timestamp of the file"
    )

    filetype: str = Field(..., description="Type/extension of the file")

    coordinates: Optional[Tuple[float, float, float, float]] = Field(
        None, description="XY Bounding Box Coordinates (x1, y1, x2, y2)"
    )

    parent_id: Optional[str] = Field(
        None, description="ID of the parent element in document hierarchy"
    )

    category_depth: Optional[int] = Field(
        None, description="Depth of element relative to others of same category"
    )

    text_as_html: Optional[str] = Field(
        None, description="HTML representation of extracted tables"
    )

    languages: List[str] = Field(
        default_factory=list,
        description="Ordered list of detected languages by probability",
    )

    emphasized_text_contents: List[str] = Field(
        default_factory=list,
        description="Text content that is emphasized in the document",
    )

    emphasized_text_tags: List[EmphasisTag] = Field(
        default_factory=list, description="Tags indicating type of text emphasis"
    )

    is_continuation: bool = Field(
        False, description="Indicates if element is a continuation of previous element"
    )

    detection_class_prob: Optional[Dict[str, float]] = Field(
        None,
        description="Detection model class probabilities from unstructured-inference",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "document.pdf",
                "file_directory": "/path/to/documents",
                "last_modified": "2024-01-20T10:30:00",
                "filetype": "pdf",
                "coordinates": (100.0, 200.0, 300.0, 400.0),
                "parent_id": "title_1",
                "category_depth": 2,
                "text_as_html": "<table><tr><td>Data</td></tr></table>",
                "languages": ["en", "es"],
                "emphasized_text_contents": ["important text"],
                "emphasized_text_tags": ["bold"],
                "is_continuation": False,
                "detection_class_prob": {"title": 0.95, "text": 0.05},
            }
        }


class ExtractedDocument(BaseModel):
    """Represents an extracted document with metadata."""

    elements: List[Element]
    metadata: Union[MetadataFields, Dict[str, Any]]
    file_type: str
    extraction_strategy: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {Element: lambda e: str(e)}


class ExtractionOptions(BaseModel):
    """Options for document extraction."""

    strategy: str = Field(default="hi_res", description="Extraction strategy to use")
    include_metadata: bool = Field(
        default=True, description="Include metadata in extraction"
    )
    include_page_breaks: bool = Field(default=True, description="Include page breaks")
    include_images: bool = Field(default=False, description="Include images")
    languages: Optional[List[str]] = Field(
        default=["tur"], description="Languages for OCR"
    )
    pdf_infer_table_structure: bool = Field(
        default=False, description="Infer table structure in PDFs"
    )
    encoding: str = Field(default="utf-8", description="Text encoding")
    group_broken_paragraphs: bool = Field(
        default=True, description="Group broken paragraphs"
    )


class DocumentExtractor:
    """
    Service for extracting content from various document types.

    This class provides a unified interface for extracting content from different
    document formats. It handles file type detection, appropriate parser selection,
    and extraction configuration.

    Supported formats:
    - PDF (with optional OCR and table structure inference)
    - Word documents (DOCX, DOC with automatic conversion)
    - HTML (with encoding support)
    - Markdown
    - EPUB
    - Plain text

    Example:
        extractor = DocumentExtractor(
            options=ExtractionOptions(include_metadata=True)
        )
        document = extractor.extract_from_file("document.pdf")
        elements = document.elements
    """

    def __init__(self, options: Optional[ExtractionOptions] = None):
        self.options = options or ExtractionOptions()

    def extract_from_file(
        self, file_path: Union[str, Path], file_type: Optional[str] = None
    ) -> ExtractedDocument:
        """
        Extract content from a file.

        Args:
            file_path (Union[str, Path]): Path to the file
            file_type (Optional[str]): File type override

        Returns:
            ExtractedDocument: Extracted document content

        Raises:
            ValueError: If file type is not supported
        """
        file_path = Path(file_path)
        if not file_type:
            file_type = file_path.suffix.lower().lstrip(".")

        extraction_kwargs = {
            "include_metadata": self.options.include_metadata,
            "include_page_breaks": self.options.include_page_breaks,
            "languages": self.options.languages,
            "group_broken_paragraphs": self.options.group_broken_paragraphs,
            "strategy": self.options.strategy,
        }

        if self.options.languages:
            extraction_kwargs["languages"] = self.options.languages

        if file_type == "pdf":
            elements = partition_pdf(
                str(file_path),
                infer_table_structure=self.options.pdf_infer_table_structure,
                **extraction_kwargs,
            )
        elif file_type in ["docx", "doc"]:
            if file_type == "doc":
                # Convert doc to docx
                docx_path = convert_office_doc(
                    str(file_path), output_directory=str(file_path.parent)
                )
                elements = partition_docx(docx_path, **extraction_kwargs)
            else:
                elements = partition_docx(str(file_path), **extraction_kwargs)
        elif file_type == "html":
            elements = partition_html(
                str(file_path),
                encoding=self.options.encoding,
                **extraction_kwargs,
            )
        elif file_type == "md":
            elements = partition_md(str(file_path), **extraction_kwargs)
        elif file_type == "epub":
            elements = partition_epub(str(file_path), **extraction_kwargs)
        elif file_type == "txt":
            elements = partition_text(
                str(file_path),
                encoding=self.options.encoding,
                **extraction_kwargs,
            )
        elif self.options.strategy == "auto":
            elements = partition(str(file_path), **extraction_kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return ExtractedDocument(
            elements=elements,
            file_type=file_type,
            extraction_strategy=self.options.strategy,
            metadata={
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "extraction_options": self.options.model_dump(),
            },
        )

    def extract_from_text(
        self, text: str, content_type: str = "text"
    ) -> ExtractedDocument:
        """Extract content from text."""
        if content_type == "text":
            elements = partition_text(text=text)
        elif content_type == "html":
            elements = partition_html(text=text)
        elif content_type == "md":
            elements = partition_md(text=text)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

        return ExtractedDocument(
            elements=elements,
            file_type=content_type,
            extraction_strategy=self.options.strategy,
            metadata={
                "content_type": content_type,
                "extraction_options": self.options.model_dump(),
            },
        )
