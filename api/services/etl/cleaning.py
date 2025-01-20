import re
from typing import List, Optional, Union

from pydantic import BaseModel, Field
from unstructured.cleaners.core import (
    clean,
    clean_bullets,
    clean_extra_whitespace,
    clean_ordered_bullets,
    clean_postfix,
    clean_prefix,
    group_broken_paragraphs,
    replace_unicode_quotes,
)
from unstructured.documents.elements import Element, Text


class CleaningOptions(BaseModel):
    """Options for text cleaning."""

    remove_extra_whitespace: bool = Field(
        default=True, description="Remove extra whitespace"
    )
    remove_empty_lines: bool = Field(default=False, description="Remove empty lines")
    remove_urls: bool = Field(default=False, description="Remove URLs")
    remove_email_addresses: bool = Field(
        default=False, description="Remove email addresses"
    )
    remove_phone_numbers: bool = Field(default=True, description="Remove phone numbers")
    normalize_unicode: bool = Field(
        default=True, description="Normalize Unicode characters"
    )
    normalize_bullets: bool = Field(default=True, description="Normalize bullet points")
    normalize_quotes: bool = Field(default=True, description="Normalize quotes")
    fix_broken_paragraphs: bool = Field(
        default=True, description="Fix broken paragraphs"
    )
    custom_patterns: Optional[List[str]] = Field(
        default=None, description="Custom regex patterns to remove"
    )
    replace_unicode_quotes: bool = Field(
        default=True, description="Replace unicode characters"
    )


class TextCleaner:
    """
    Service for cleaning and normalizing text content.

    Provides various text cleaning operations using Unstructured library and custom rules.
    """

    def __init__(self, options: Optional[CleaningOptions] = None):
        self.options = options or CleaningOptions()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for cleaning."""
        self.patterns = {
            "url": re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            ),
            "email": re.compile(r"\S+@\S+\.\S+"),
            "phone": re.compile(
                r"\+?1?\d{9,15}|(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4})"
            ),
        }

        if self.options.custom_patterns:
            self.patterns.update(
                {
                    f"custom_{i}": re.compile(pattern)
                    for i, pattern in enumerate(self.options.custom_patterns)
                }
            )

    def clean_text(self, text: str) -> str:
        """
        Clean text using configured options.

        Args:
            text (str): Text to clean

        Returns:
            str: Cleaned text
        """
        if not text:
            return text

        # Apply Unstructured cleaners
        cleaned = text
        if self.options.normalize_bullets:
            cleaned = clean_bullets(cleaned)
            cleaned = clean_ordered_bullets(cleaned)

        if self.options.normalize_quotes:
            cleaned = replace_unicode_quotes(cleaned)

        if self.options.fix_broken_paragraphs:
            cleaned = group_broken_paragraphs(cleaned)

        # Apply custom cleaners
        if self.options.remove_urls:
            cleaned = self.patterns["url"].sub("", cleaned)

        if self.options.remove_email_addresses:
            cleaned = self.patterns["email"].sub("", cleaned)

        if self.options.remove_phone_numbers:
            cleaned = self.patterns["phone"].sub("", cleaned)

        if self.options.custom_patterns:
            for pattern in self.patterns.values():
                if str(pattern.pattern) in self.options.custom_patterns:
                    cleaned = pattern.sub("", cleaned)

        if self.options.remove_extra_whitespace:
            cleaned = clean_extra_whitespace(cleaned)

        if self.options.remove_empty_lines:
            cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip())

        if self.options.replace_unicode_quotes:
            cleaned = replace_unicode_quotes(text=cleaned)

        return cleaned.strip()

    def clean_element(self, element: Element) -> Element:
        """
        Clean a document element.

        Args:
            element (Element): Element to clean

        Returns:
            Element: Cleaned element
        """
        if isinstance(element, Text):
            cleaned_text = self.clean_text(str(element))
            return Text(text=cleaned_text, metadata=element.metadata)
        return element

    def clean_elements(self, elements: List[Element]) -> List[Element]:
        """
        Clean a list of document elements.

        Args:
            elements (List[Element]): Elements to clean

        Returns:
            List[Element]: Cleaned elements
        """
        cleaned = []
        for element in elements:
            cleaned_element = self.clean_element(element)
            if str(cleaned_element).strip():  # Skip empty elements
                cleaned.append(cleaned_element)
        return cleaned
