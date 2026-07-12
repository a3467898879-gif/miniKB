"""
Text splitter — chunks long documents into semantically coherent pieces.

Uses LangChain's RecursiveCharacterTextSplitter which:
1. Tries to split on paragraph breaks (\n\n) first
2. Falls back to line breaks (\n), then sentences (.), then spaces
3. Keeps a configurable overlap between chunks for context continuity

This mirrors MaxKB's chunking strategy and is the industry standard for RAG.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def get_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create a configured text splitter instance."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """
    Split a long text into overlapping chunks.

    Args:
        text: The full document text.
        chunk_size: Max characters per chunk (default from settings).
        chunk_overlap: Overlap characters between adjacent chunks.

    Returns:
        List of text chunks, each ≤ chunk_size characters.
    """
    splitter = get_splitter(chunk_size, chunk_overlap)
    return splitter.split_text(text)
