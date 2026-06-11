import os
import tempfile

import pdfplumber

from app.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(filepath: str) -> str:
    full_text = []

    with pdfplumber.open(filepath) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"[PDF] Extracting {total_pages} pages from '{filepath}'")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                full_text.append(text.strip())
                logger.info(f"[PDF] Page {i + 1}/{total_pages}: {len(text)} chars")
            else:
                logger.warning(
                    f"[PDF] Page {i + 1}/{total_pages}: no text found (image-only page?)"
                )

    if not full_text:
        raise ValueError(
            f"No text could be extracted from '{filepath}'. "
            "The PDF may be scanned/image-based."
        )

    result = "\n\n".join(full_text)
    logger.info(f"[PDF] Total extracted: {len(result)} characters")
    return result


def extract_text_from_pdf_bytes(content: bytes, filename: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)
