# src/document_detector.py
# Classifies documents and individual PDF pages by content type.
#
# Why classify before processing?
# Different content types need different extraction strategies.
# Sending a scanned PDF through PyPDFLoader returns empty strings.
# Sending a text PDF through VLM wastes money — OCR is cheaper.
# Classification lets us route each document to the right extractor.

import pymupdf as fitz  # PyMuPDF — better than pypdf for page analysis
from pathlib import Path
from enum import Enum

# Install PyMuPDF if not already installed:
# pip install pymupdf

class DocumentType(Enum):
    """
    Classification of document content types.
    Each type requires a different extraction approach.
    """
    TEXT_PDF    = "text_pdf"     # native text, use PyPDFLoader
    MIXED_PDF   = "mixed_pdf"    # text + embedded images, use both
    SCANNED_PDF = "scanned_pdf"  # images of pages, use VLM/OCR
    IMAGE       = "image"        # single image file, use VLM
    UNKNOWN     = "unknown"      # fallback

class PageType(Enum):
    """Classification of individual PDF pages."""
    TEXT_ONLY   = "text_only"    # pure text, no significant images
    IMAGE_HEAVY = "image_heavy"  # mostly images, little text
    MIXED       = "mixed"        # both text and images
    EMPTY       = "empty"        # no content detected

def classify_pdf_page(page) -> tuple[PageType, dict]:
    """
    Classifies a single PDF page and returns metadata about its content.

    Args:
        page: a fitz.Page object

    Returns:
        Tuple of (PageType, metadata_dict)

    How it works:
    PyMuPDF lets us inspect the page's raw content streams —
    we can count text blocks and image blocks separately.
    This is like examining a document's layer structure in Photoshop.
    """
    # Extract text — empty or very short means little/no text content
    text = page.get_text().strip()
    text_length = len(text)

    # Get list of images embedded in this page
    # Each entry is (xref, smask, width, height, bpc, colorspace, ...)
    images = page.get_images()
    image_count = len(images)

    # Calculate image area as fraction of page area
    page_area = page.rect.width * page.rect.height
    image_area = 0

    for img in images:
        # Get image dimensions from the xref
        try:
            xref = img[0]
            img_info = page.parent.extract_image(xref)
            img_area = img_info["width"] * img_info["height"]
            image_area += img_area
        except Exception:
            pass

    image_coverage = image_area / page_area if page_area > 0 else 0

    metadata = {
        "text_length": text_length,
        "image_count": image_count,
        "image_coverage": round(image_coverage, 3),
        "text_preview": text[:100] if text else ""
    }

    # Classification rules:
    # These thresholds were determined empirically
    # Adjust based on your document types

    if text_length < 50 and image_count == 0:
        return PageType.EMPTY, metadata

    if text_length < 100 and image_count > 0:
        # Almost no text, has images → image heavy
        return PageType.IMAGE_HEAVY, metadata

    if image_count > 0 and text_length > 100:
        # Both text and images → mixed
        return PageType.MIXED, metadata

    # Significant text, no meaningful images → text only
    return PageType.TEXT_ONLY, metadata

def classify_document(file_path: str) -> tuple[DocumentType, dict]:
    """
    Classifies an entire document and returns its type with metadata.

    Args:
        file_path: path to any supported file

    Returns:
        Tuple of (DocumentType, metadata_dict)
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    # Image files are always processed with VLM
    if extension in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
        return DocumentType.IMAGE, {"file_type": extension}

    if extension != ".pdf":
        return DocumentType.UNKNOWN, {"file_type": extension}

    # Analyze PDF structure
    doc = fitz.open(file_path)
    total_pages = len(doc)

    page_classifications = []
    for page_num in range(min(total_pages, 10)):
        # Sample up to 10 pages for efficiency
        # Analyzing all pages of a 200-page PDF is slow
        page = doc[page_num]
        page_type, meta = classify_pdf_page(page)
        page_classifications.append(page_type)

    doc.close()

    # Count page types
    text_pages = sum(1 for p in page_classifications if p == PageType.TEXT_ONLY)
    image_pages = sum(1 for p in page_classifications if p == PageType.IMAGE_HEAVY)
    mixed_pages = sum(1 for p in page_classifications if p == PageType.MIXED)
    empty_pages = sum(1 for p in page_classifications if p == PageType.EMPTY)

    sampled = len(page_classifications)

    metadata = {
        "total_pages": total_pages,
        "sampled_pages": sampled,
        "text_pages": text_pages,
        "image_pages": image_pages,
        "mixed_pages": mixed_pages,
        "empty_pages": empty_pages,
    }

    # Classification rules:
    # If most pages are image-heavy → scanned PDF
    # If most pages are text-only → text PDF
    # Otherwise → mixed PDF

    if image_pages + empty_pages > sampled * 0.7:
        # More than 70% of pages are images or empty → scanned
        return DocumentType.SCANNED_PDF, metadata

    if text_pages > sampled * 0.8:
        # More than 80% pure text pages → text PDF
        return DocumentType.TEXT_PDF, metadata

    # Mix of content types
    return DocumentType.MIXED_PDF, metadata