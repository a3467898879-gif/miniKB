"""
Document loader — extracts plain text from various file formats.

Strategy: dispatch by file extension to a dedicated parser.
Each parser returns clean UTF-8 text, stripping binary noise.

Supported: PDF, DOCX, TXT, MD, HTML
"""

from pathlib import Path


def load_pdf(file_path: Path) -> str:
    """Extract text from PDF using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_docx(file_path: Path) -> str:
    """Extract text from DOCX using python-docx."""
    from docx import Document as DocxDocument
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_html(file_path: Path) -> str:
    """Extract text from HTML using BeautifulSoup."""
    from bs4 import BeautifulSoup
    html = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def load_text(file_path: Path) -> str:
    """Load plain text / markdown files."""
    return file_path.read_text(encoding="utf-8", errors="ignore").strip()


# Registry: file extension → loader function
LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
    ".html": load_html,
    ".htm": load_html,
}

SUPPORTED_EXTENSIONS = set(LOADERS.keys())


def load_document(file_path: Path) -> tuple[str, str]:
    """
    Load a document and return (text, file_type).

    Raises ValueError for unsupported file types.
    """
    ext = file_path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    loader = LOADERS[ext]
    text = loader(file_path)
    if not text:
        raise ValueError(f"No text content extracted from {file_path.name}")
    return text, ext.lstrip(".")
