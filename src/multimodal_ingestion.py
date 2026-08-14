# src/multimodal_ingestion.py
# Unified ingestion pipeline for any document type.
# Routes each document and page to the correct extractor
# and produces unified Document chunks for the vector store.

import hashlib
import tempfile
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pymupdf as fitz  # PyMuPDF
from PIL import Image
import io

from document_detector import (
    classify_document,
    classify_pdf_page,
    DocumentType,
    PageType
)
from visual_extractor import describe_image_for_rag
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
    extra_metadata: dict = None
) -> Document:
    """
    Creates a standardized Document chunk with consistent metadata.

    Why standardize metadata?
    Every chunk — whether from text, charts, or scanned pages —
    gets the same metadata structure. This makes filtering,
    citation display, and debugging consistent across all content types.

    content_type values:
    - "text"    → extracted from PDF text layer
    - "image"   → VLM description of an image file
    - "chart"   → VLM description of a chart on a PDF page
    - "scanned" → VLM description of a scanned page
    - "mixed"   → VLM description of a mixed content page
    """
    metadata = {
        "source": source,
        "page": page,
        "content_type": content_type,
        "file_hash": file_hash,
        "file_name": Path(source).name,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return Document(page_content=content, metadata=metadata)

def extract_text_pdf(file_path: str, file_hash: str) -> list[Document]:
    """
    Extracts text from a native text PDF using PyMuPDF.
    Splits into chunks and returns Document list.
    """
    print(f"  Extracting text from PDF...")
    doc = fitz.open(file_path)
    chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if not text:
            continue

        # Split page text into chunks
        page_doc = Document(
            page_content=text,
            metadata={"source": file_path, "page": page_num}
        )
        page_chunks = TEXT_SPLITTER.split_documents([page_doc])

        # Add standardized metadata to each chunk
        for chunk in page_chunks:
            chunks.append(make_chunk(
                content=chunk.page_content,
                source=file_path,
                page=page_num,
                content_type="text",
                file_hash=file_hash
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

def extract_mixed_pdf(file_path: str, file_hash: str) -> list[Document]:
    """
    Extracts content from a mixed PDF (text + images).

    Strategy per page:
    - TEXT_ONLY pages → text extraction (fast, cheap)
    - IMAGE_HEAVY pages → VLM description (accurate for visuals)
    - MIXED pages → text extraction + VLM for visual context
    - EMPTY pages → skip
    """
    print(f"  Processing mixed PDF page by page...")
    doc = fitz.open(file_path)
    chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_type, meta = classify_pdf_page(page)

        print(f"  Page {page_num + 1}/{len(doc)}: {page_type.value} "
              f"(text: {meta['text_length']} chars, "
              f"images: {meta['image_count']})")

        if page_type == PageType.EMPTY:
            continue

        elif page_type == PageType.TEXT_ONLY:
            # Pure text — extract directly, no VLM needed
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
                        file_hash=file_hash
                    ))

        elif page_type == PageType.IMAGE_HEAVY:
            # Render page as image and describe with VLM
            page_image = pdf_page_to_image(file_path, page_num)
            tmp_path = save_image_to_temp(page_image)

            try:
                description = describe_image_for_rag(tmp_path)
                chunks.append(make_chunk(
                    content=description,
                    source=file_path,
                    page=page_num,
                    content_type="chart",
                    file_hash=file_hash
                ))
            finally:
                os.unlink(tmp_path)

        elif page_type == PageType.MIXED:
            # Extract text AND generate VLM description
            # Both go into the vector store as separate chunks
            # Text chunk for precise text retrieval
            text = page.get_text().strip()
            if text:
                chunks.append(make_chunk(
                    content=text,
                    source=file_path,
                    page=page_num,
                    content_type="text",
                    file_hash=file_hash,
                    extra_metadata={"has_visual_content": True}
                ))

            # VLM chunk for visual content retrieval
            page_image = pdf_page_to_image(file_path, page_num)
            tmp_path = save_image_to_temp(page_image)

            try:
                description = describe_image_for_rag(tmp_path)
                chunks.append(make_chunk(
                    content=f"[Visual content on page {page_num + 1}] {description}",
                    source=file_path,
                    page=page_num,
                    content_type="mixed",
                    file_hash=file_hash
                ))
            finally:
                os.unlink(tmp_path)

    doc.close()
    print(f"  Extracted {len(chunks)} total chunks")
    return chunks

def extract_scanned_pdf(file_path: str, file_hash: str) -> list[Document]:
    """
    Extracts content from a scanned PDF using VLM for each page.

    Every page is rendered as an image and described by GPT-4o.
    This is slower and more expensive than text extraction but
    is the only reliable approach for scanned documents.
    """
    print(f"  Processing scanned PDF with VLM...")
    doc = fitz.open(file_path)
    chunks = []

    for page_num in range(len(doc)):
        print(f"  Processing page {page_num + 1}/{len(doc)}...")

        page_image = pdf_page_to_image(file_path, page_num)
        tmp_path = save_image_to_temp(page_image)

        try:
            description = describe_image_for_rag(tmp_path)
            chunks.append(make_chunk(
                content=description,
                source=file_path,
                page=page_num,
                content_type="scanned",
                file_hash=file_hash
            ))
        finally:
            os.unlink(tmp_path)

    doc.close()
    print(f"  Extracted {len(chunks)} chunks from scanned PDF")
    return chunks

def extract_image_file(file_path: str, file_hash: str) -> list[Document]:
    """
    Extracts content from a single image file using VLM.
    Returns a single Document chunk with the VLM description.
    """
    print(f"  Processing image with VLM...")
    description = describe_image_for_rag(file_path)

    return [make_chunk(
        content=description,
        source=file_path,
        page=0,
        content_type="image",
        file_hash=file_hash
    )]

def ingest_document(file_path: str) -> list[Document]:
    """
    Main ingestion function — routes any document to the correct extractor.

    This is the single entry point for all document types.
    Callers do not need to know which extractor to use —
    the pipeline detects and handles everything automatically.

    Args:
        file_path: path to any supported document or image

    Returns:
        List of Document chunks ready for embedding and storage
    """
    file_hash = get_file_hash(file_path)
    file_name = Path(file_path).name

    print(f"\nAnalyzing: {file_name}")

    # Classify document type
    doc_type, metadata = classify_document(file_path)
    print(f"  Type: {doc_type.value}")
    print(f"  Metadata: {metadata}")

    # Route to correct extractor
    if doc_type == DocumentType.TEXT_PDF:
        chunks = extract_text_pdf(file_path, file_hash)

    elif doc_type == DocumentType.MIXED_PDF:
        chunks = extract_mixed_pdf(file_path, file_hash)

    elif doc_type == DocumentType.SCANNED_PDF:
        chunks = extract_scanned_pdf(file_path, file_hash)

    elif doc_type == DocumentType.IMAGE:
        chunks = extract_image_file(file_path, file_hash)

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

def ingest_to_vectorstore(file_path: str) -> Chroma:
    """
    Complete idempotent ingestion pipeline.
    Creates or updates the vector store.
    Skips documents already ingested.
    """
    file_hash = get_file_hash(file_path)
    file_name = Path(file_path).name

    # Load or create vector store
    if Path(CHROMA_PATH).exists():
        vectorstore = load_vectorstore()

        if is_already_ingested(vectorstore, file_hash):
            print(f"'{file_name}' already ingested. Skipping.")
            return vectorstore
    else:
        vectorstore = None

    # Extract chunks
    chunks = ingest_document(file_path)

    if not chunks:
        print(f"No chunks extracted from {file_name}")
        return vectorstore

    # Store chunks
    if vectorstore is None:
        vectorstore = build_vectorstore(chunks)
    else:
        vectorstore.add_documents(chunks)
        print(f"Added {len(chunks)} chunks to existing vector store")

    return vectorstore