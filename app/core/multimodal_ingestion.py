# src/multimodal_ingestion.py
# Unified ingestion pipeline for any document type.
# Routes each document and page to the correct extractor
# and produces unified Document chunks for the vector store.

# app/core/multimodal_ingestion.py
# Make system-dependent imports optional
# so the app starts even without Tesseract/Poppler installed

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available — OCR disabled")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not available — scanned PDF support disabled")
    

import hashlib
import tempfile
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pymupdf as fitz  # PyMuPDF
from app.config import settings
from PIL import Image
import io

from app.core.document_detector import (
    classify_document,
    classify_pdf_page,
    DocumentType,
    PageType
)
from app.core.visual_extractor import describe_image_for_rag
from dotenv import load_dotenv
load_dotenv()

CHROMA_PATH = "chroma_db"

# ── Text splitter ─────────────────────────────────────────────────────────────
# Same settings as LangChain Copilot — proven to work well
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", " "]
)

def get_file_hash(file_path: str) -> str:
    """MD5 hash of file contents for duplicate detection."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def make_chunk(
    content: str,
    source: str,
    page: int,
    content_type: str,
    file_hash: str,
    original_filename: str = None,   # ← add this parameter
    extra_metadata: dict = None
) -> Document:
    """
    Creates a standardized Document chunk with consistent metadata.

    original_filename: the real filename from the user's upload.
    If not provided, falls back to the basename of source path.
    This ensures temp file paths never leak into stored metadata.
    """
    metadata = {
        "source": source,
        "page": page,
        "content_type": content_type,
        "file_hash": file_hash,
        # Use original_filename if provided, otherwise fall back to source basename
        "file_name": original_filename or Path(source).name,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return Document(page_content=content, metadata=metadata)

def extract_text_pdf(
    file_path: str,
    file_hash: str,
    original_filename: str = None
) -> list[Document]:
    """Extracts text from a native text PDF."""
    print(f"  Extracting text from PDF...")
    doc = fitz.open(file_path)
    chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if not text:
            continue

        page_doc = Document(
            page_content=text,
            metadata={"source": file_path, "page": page_num}
        )
        page_chunks = TEXT_SPLITTER.split_documents([page_doc])

        for chunk in page_chunks:
            chunks.append(make_chunk(
                content=chunk.page_content,
                source=file_path,
                page=page_num,
                content_type="text",
                file_hash=file_hash,
                original_filename=original_filename  # ← pass through
            ))

    doc.close()
    print(f"  Extracted {len(chunks)} text chunks")
    return chunks

def pdf_page_to_image(pdf_path: str, page_num: int) -> Image.Image:
    """
    Renders a PDF page as a PIL Image for VLM processing.

    Why render to image?
    For scanned PDFs, the page IS an image. For mixed PDFs,
    we need to see the visual layout including embedded charts.
    PyMuPDF renders pages at any DPI — we use 150 DPI as a
    balance between readability and file size.

    150 DPI: text is clearly readable, charts are legible
    300 DPI: print quality, unnecessary for VLM, large files
    72 DPI: screen resolution, too low for small text in charts
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # mat is a transformation matrix — zoom factor controls DPI
    # zoom = DPI / 72 (72 is PDF's native resolution)
    zoom = 150 / 72  # 150 DPI
    mat = fitz.Matrix(zoom, zoom)

    # Render page to pixmap (raster image)
    pixmap = page.get_pixmap(matrix=mat)
    doc.close()

    # Convert pixmap to PIL Image
    img_bytes = pixmap.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))

def save_image_to_temp(image: Image.Image) -> str:
    """
    Saves a PIL Image to a temporary file and returns the path.
    Used when we need a file path for visual_extractor functions.
    """
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as tmp:
        image.save(tmp, format="PNG")
        return tmp.name

def extract_mixed_pdf(
    file_path: str,
    file_hash: str,
    original_filename: str = None
) -> list[Document]:
    """Extracts content from a mixed PDF."""
    print(f"  Processing mixed PDF page by page...")
    doc = fitz.open(file_path)
    chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_type, meta = classify_pdf_page(page)

        print(f"  Page {page_num + 1}/{len(doc)}: {page_type.value}")

        if page_type == PageType.EMPTY:
            continue

        elif page_type == PageType.TEXT_ONLY:
            text = page.get_text().strip()
            if text:
                page_doc = Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num}
                )
                page_chunks = TEXT_SPLITTER.split_documents([page_doc])
                for chunk in page_chunks:
                    chunks.append(make_chunk(
                        content=chunk.page_content,
                        source=file_path,
                        page=page_num,
                        content_type="text",
                        file_hash=file_hash,
                        original_filename=original_filename  # ← pass through
                    ))

        elif page_type == PageType.IMAGE_HEAVY:
            page_image = pdf_page_to_image(file_path, page_num)
            tmp_path = save_image_to_temp(page_image)
            try:
                description = describe_image_for_rag(tmp_path)
                chunks.append(make_chunk(
                    content=description,
                    source=file_path,
                    page=page_num,
                    content_type="chart",
                    file_hash=file_hash,
                    original_filename=original_filename  # ← pass through
                ))
            finally:
                os.unlink(tmp_path)

        elif page_type == PageType.MIXED:
            text = page.get_text().strip()
            if text:
                chunks.append(make_chunk(
                    content=text,
                    source=file_path,
                    page=page_num,
                    content_type="text",
                    file_hash=file_hash,
                    original_filename=original_filename,  # ← pass through
                    extra_metadata={"has_visual_content": True}
                ))

            page_image = pdf_page_to_image(file_path, page_num)
            tmp_path = save_image_to_temp(page_image)
            try:
                description = describe_image_for_rag(tmp_path)
                chunks.append(make_chunk(
                    content=f"[Visual content on page {page_num + 1}] {description}",
                    source=file_path,
                    page=page_num,
                    content_type="mixed",
                    file_hash=file_hash,
                    original_filename=original_filename  # ← pass through
                ))
            finally:
                os.unlink(tmp_path)

    doc.close()
    print(f"  Extracted {len(chunks)} total chunks")
    return chunks

def extract_scanned_pdf(
    file_path: str,
    file_hash: str,
    original_filename: str = None
) -> list[Document]:
    """
    Extracts content from scanned PDF using VLM.
    Falls back gracefully if pdf2image is not available.
    """
    print(f"  Processing scanned PDF with VLM...")
    doc = fitz.open(file_path)
    chunks = []

    for page_num in range(len(doc)):
        print(f"  Processing page {page_num + 1}/{len(doc)}...")

        try:
            page_image = pdf_page_to_image(file_path, page_num)
            tmp_path = save_image_to_temp(page_image)

            try:
                description = describe_image_for_rag(tmp_path)
                chunks.append(make_chunk(
                    content=description,
                    source=file_path,
                    page=page_num,
                    content_type="scanned",
                    file_hash=file_hash,
                    original_filename=original_filename
                ))
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            # If image rendering fails, fall back to text extraction
            print(f"  Warning: Could not render page {page_num} as image: {e}")
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                chunks.append(make_chunk(
                    content=text,
                    source=file_path,
                    page=page_num,
                    content_type="text",
                    file_hash=file_hash,
                    original_filename=original_filename
                ))

    doc.close()
    return chunks

def extract_image_file(
    file_path: str,
    file_hash: str,
    original_filename: str = None
) -> list[Document]:
    """Extracts content from a single image file using VLM."""
    print(f"  Processing image with VLM...")
    description = describe_image_for_rag(file_path)

    return [make_chunk(
        content=description,
        source=file_path,
        page=0,
        content_type="image",
        file_hash=file_hash,
        original_filename=original_filename  # ← pass through
    )]

def ingest_document(
    file_path: str,
    file_hash: str,
    original_filename: str = None   # ← add this parameter
) -> list[Document]:
    """
    Routes document to correct extractor.
    original_filename is stored in chunk metadata instead of temp path.
    """
    file_name = original_filename or Path(file_path).name

    print(f"\nAnalyzing: {file_name}")

    doc_type, metadata = classify_document(file_path)
    print(f"  Type: {doc_type.value}")
    print(f"  Metadata: {metadata}")

    # Pass original_filename to each extractor
    if doc_type == DocumentType.TEXT_PDF:
        chunks = extract_text_pdf(file_path, file_hash, original_filename)
    elif doc_type == DocumentType.MIXED_PDF:
        chunks = extract_mixed_pdf(file_path, file_hash, original_filename)
    elif doc_type == DocumentType.SCANNED_PDF:
        chunks = extract_scanned_pdf(file_path, file_hash, original_filename)
    elif doc_type == DocumentType.IMAGE:
        chunks = extract_image_file(file_path, file_hash, original_filename)
    else:
        print(f"  Unsupported document type: {doc_type.value}")
        return []

    print(f"  Total chunks produced: {len(chunks)}")
    return chunks

def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Creates a new vector store from chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"\nStored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}")
    return vectorstore

def load_vectorstore() -> Chroma:
    """Loads existing vector store from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

def is_already_ingested(vectorstore: Chroma, file_hash: str) -> bool:
    """Checks if document already exists in vector store."""
    results = vectorstore.get(
        where={"file_hash": {"$eq": file_hash}},
        limit=1
    )
    return len(results["ids"]) > 0

def ingest_to_vectorstore(
    file_path: str,
    original_filename: str = None  # ← add this parameter
) -> Chroma:
    """
    Complete idempotent ingestion pipeline.

    original_filename: real filename to store in metadata.
    If None, uses the basename of file_path.
    Always pass this when ingesting uploaded files so temp
    paths never appear in the vector store.
    """
    file_hash = get_file_hash(file_path)
    file_name = original_filename or Path(file_path).name

    if Path(settings.chroma_path).exists():
        vectorstore = load_vectorstore()

        if is_already_ingested(vectorstore, file_hash):
            print(f"'{file_name}' already ingested. Skipping.")
            return vectorstore

        # New document — extract and add
        chunks = ingest_document(file_path, file_hash, original_filename)
        if chunks:
            vectorstore.add_documents(chunks)
            print(f"Added {len(chunks)} chunks to existing vector store.")
        return vectorstore

    else:
        chunks = ingest_document(file_path, file_hash, original_filename)
        if not chunks:
            raise ValueError(f"No content extracted from {file_name}")
        return build_vectorstore(chunks)