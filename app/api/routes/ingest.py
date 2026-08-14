# app/api/routes/ingest.py
# Handles upload of PDFs and images.
# Unlike LangChain Copilot, this endpoint accepts both PDFs and images.

import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

from app.core.multimodal_ingestion import ingest_to_vectorstore
from app.core.multimodal_chain import build_multimodal_rag_chain

from app.core.multimodal_agent import build_multimodal_agent

router = APIRouter()

# Supported file types
SUPPORTED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"
}

class IngestResponse(BaseModel):
    message: str
    file_name: str
    file_type: str
    chunks_added: int
    content_types: dict

@router.post("", response_model=IngestResponse)
async def ingest_document(request: Request, file: UploadFile = File(...)):
    file_ext = "." + file.filename.split(".")[-1].lower()

    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        vectorstore = ingest_to_vectorstore(
            file_path=tmp_path,
            original_filename=file.filename  # ← pass real filename here
        )

        request.app.state.vectorstore = vectorstore
        request.app.state.agent = build_multimodal_agent(vectorstore)

        all_docs = vectorstore.get()
        total_chunks = len(all_docs["ids"])

        content_types = {}
        for meta in all_docs["metadatas"]:
            ct = meta.get("content_type", "unknown")
            content_types[ct] = content_types.get(ct, 0) + 1

    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        message=f"Successfully processed {file.filename}",
        file_name=file.filename,
        file_type=file_ext,
        chunks_added=total_chunks,
        content_types=content_types
    )