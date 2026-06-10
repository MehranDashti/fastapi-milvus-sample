import pdfplumber
from logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(filepath: str) -> str:
    """
    Extract all text from a PDF file page by page.
    Returns a single string with all pages joined.

    pdfplumber is better than PyPDF2 for:
    - tables (tries to preserve structure)
    - multi-column layouts
    - special characters and unicode

    Args:
        filepath: absolute or relative path to the PDF file

    Returns:
        full text content as a single string
    """
    full_text = []

    with pdfplumber.open(filepath) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"[PDF] Extracting {total_pages} pages from '{filepath}'")

        for i, page in enumerate(pdf.pages):
            # extract_text() returns None for image-only pages
            text = page.extract_text()

            if text and text.strip():
                full_text.append(text.strip())
                logger.info(f"[PDF] Page {i+1}/{total_pages}: {len(text)} chars")
            else:
                logger.warning(f"[PDF] Page {i+1}/{total_pages}: no text found (image-only page?)")

    if not full_text:
        raise ValueError(f"No text could be extracted from '{filepath}'. "
                         f"The PDF may be scanned/image-based.")

    # Join pages with double newline so paragraph splitter works correctly
    result = "\n\n".join(full_text)
    logger.info(f"[PDF] Total extracted: {len(result)} characters")

    return result


def extract_text_from_pdf_bytes(content: bytes, filename: str) -> str:
    """
    Extract text from PDF bytes (for uploaded files via FastAPI).
    Writes to a temp file, extracts, then cleans up.

    Args:
        content:  raw PDF bytes from UploadFile.read()
        filename: original filename for logging
    """
    import tempfile
    import os

    # Write bytes to a temp file — pdfplumber needs a file path
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)   # always delete temp file, even on error