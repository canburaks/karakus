from typing import List, Optional, Union

from pydantic import BaseModel, Field
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import CompositeElement, Element, Text, Title


class ChunkingOptions(BaseModel):
    """Options for text chunking."""

    max_characters: int = Field(
        default=10000, description="Maximum characters per chunk"
    )
    overlap: int = Field(default=0, description="Overlap between chunks")
    min_chunk_size: int = Field(default=500, description="Minimum chunk size")
    combine_under_n_chars: int = Field(
        default=8000, description="Combine chunks under n chars"
    )
    new_after_n_chars: int = Field(
        default=10000, description="Start new chunk after n chars"
    )
    chunking_strategy: str = Field(
        default="by_title", description="Chunking strategy to use (by_title or basic)"
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata in chunks"
    )
    include_page_breaks: bool = Field(
        default=True, description="Include page breaks in chunks"
    )
    max_title_level: int = Field(
        default=3, description="Maximum header level for markdown chunking"
    )


class TextChunk(BaseModel):
    """Represents a chunk of text with metadata."""

    text: str
    metadata: dict = Field(default_factory=dict)
    chunk_type: str = "text"
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    orig_elements: Optional[List[Element]] = None

    class Config:
        arbitrary_types_allowed = True


def chunk_by_characters(text: str, options: ChunkingOptions) -> List[TextChunk]:
    """
    Chunk text by character count with overlap.

    This function splits text into chunks of specified size while attempting to break
    at natural boundaries like sentence endings or paragraph breaks. It includes
    overlap between chunks to maintain context.

    Args:
        text (str): Text to chunk
        options (ChunkingOptions): Configuration for chunking behavior

    Returns:
        List[TextChunk]: List of text chunks with metadata

    Example:
        options = ChunkingOptions(max_characters=1000, overlap=100)
        chunks = chunk_by_characters("long text...", options)
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + options.max_characters

        if end > len(text):
            end = len(text)
        else:
            # Try to end at a sentence or paragraph break
            for i in range(min(end + 100, len(text)), max(end - 100, start), -1):
                if i >= len(text):
                    continue
                if text[i] in ".!?\n" and (i + 1 >= len(text) or text[i + 1].isspace()):
                    end = i + 1
                    break

        chunk_text = text[start:end].strip()
        if len(chunk_text) >= options.min_chunk_size:
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    metadata={"chunk_method": "character"},
                )
            )

        start = end - options.overlap

    return chunks


def chunk_by_markdown_headers(text: str, options: ChunkingOptions) -> List[TextChunk]:
    """
    Chunk text by markdown headers.

    This function splits text at markdown headers, creating chunks based on document
    structure. Headers up to the specified level (max_title_level) are used as
    chunk boundaries.

    Args:
        text (str): Markdown text to chunk
        options (ChunkingOptions): Configuration for chunking behavior

    Returns:
        List[TextChunk]: List of text chunks with metadata including titles

    Example:
        options = ChunkingOptions(max_title_level=3)
        chunks = chunk_by_markdown_headers("# Title\\n## Section\\nContent...", options)
    """
    chunks = []
    lines = text.split("\n")
    current_chunk = []
    current_title = None

    for line in lines:
        if line.startswith("#"):
            # Process previous chunk if it exists
            if current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if len(chunk_text) >= options.min_chunk_size:
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            metadata={
                                "chunk_method": "markdown",
                                "title": current_title or "Untitled",
                            },
                        )
                    )

            # Start new chunk
            header_level = len(line.split()[0])
            if header_level <= options.max_title_level:
                current_chunk = [line]
                current_title = line.lstrip("#").strip()
            else:
                current_chunk.append(line)
        else:
            current_chunk.append(line)

    # Process final chunk
    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if len(chunk_text) >= options.min_chunk_size:
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    metadata={
                        "chunk_method": "markdown",
                        "title": current_title or "Untitled",
                    },
                )
            )

    return chunks


def chunk_by_semantic_sections(
    elements: List[Element], options: ChunkingOptions
) -> List[TextChunk]:
    """
    Chunk document elements by semantic sections using unstructured's chunking strategies.
    """
    if not elements:
        return []

    try:
        # Use unstructured's chunking function with error handling
        chunks = chunk_by_title(
            elements,
            max_characters=options.max_characters,
            combine_text_under_n_chars=options.combine_under_n_chars,
            new_after_n_chars=options.new_after_n_chars,
        )
    except Exception as e:
        # Fallback to basic chunking if title-based chunking fails
        chunks = []
        current_chunk = []
        current_length = 0

        for element in elements:
            element_text = str(element)
            if (
                current_length + len(element_text) > options.max_characters
                and current_chunk
            ):
                # Create a Text element from the combined text
                combined_text = "\n".join(str(e) for e in current_chunk)
                chunks.append(Text(text=combined_text))
                current_chunk = []
                current_length = 0
            current_chunk.append(element)
            current_length += len(element_text)

        if current_chunk:
            # Create a Text element from the remaining text
            combined_text = "\n".join(str(e) for e in current_chunk)
            chunks.append(Text(text=combined_text))

    # Convert chunks to our TextChunk format while preserving metadata
    text_chunks = []
    for chunk in chunks:
        try:
            # Get original elements if available
            orig_elements = getattr(chunk, "metadata", {}).get(
                "orig_elements", None
            ) or [chunk]

            # Get title from metadata or first title element
            title = None
            if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "title"):
                title = str(chunk.metadata.title)
            else:
                # Try to find first title element
                for element in orig_elements or []:
                    if isinstance(element, Title):
                        title = str(element)
                        break

            # Create TextChunk with metadata
            text_chunk = TextChunk(
                text=str(chunk),
                metadata={
                    "chunk_method": "semantic",
                    "title": title,
                    "page_numbers": list(
                        {
                            getattr(e, "page_number", None)
                            for e in (orig_elements or [chunk])
                            if getattr(e, "page_number", None) is not None
                        }
                    ),
                    "coordinates": [
                        getattr(e, "coordinates", None)
                        for e in (orig_elements or [chunk])
                        if getattr(e, "coordinates", None) is not None
                    ],
                },
                orig_elements=orig_elements,
            )
            text_chunks.append(text_chunk)
        except Exception as e:
            # If chunk conversion fails, create a simple text chunk
            text_chunks.append(
                TextChunk(text=str(chunk), metadata={"chunk_method": "semantic"})
            )

    return text_chunks


def chunk_document(
    content: Union[str, List[Element]],
    method: str = "character",
    options: Optional[ChunkingOptions] = None,
) -> List[TextChunk]:
    """
    Chunk document content using specified method.

    Args:
        content (Union[str, List[Element]]): Document content
        method (str): Chunking method ('character', 'markdown', or 'semantic')
        options (Optional[ChunkingOptions]): Chunking options

    Returns:
        List[TextChunk]: List of text chunks

    Raises:
        ValueError: If invalid method specified
    """
    if options is None:
        options = ChunkingOptions()

    if method == "character":
        if not isinstance(content, str):
            content = "\n".join(str(e) for e in content)
        return chunk_by_characters(content, options)
    elif method == "markdown":
        if not isinstance(content, str):
            content = "\n".join(str(e) for e in content)
        return chunk_by_markdown_headers(content, options)
    elif method == "semantic":
        if isinstance(content, str):
            raise ValueError("Semantic chunking requires document elements")
        return chunk_by_semantic_sections(content, options)
    else:
        raise ValueError(f"Invalid chunking method: {method}")
