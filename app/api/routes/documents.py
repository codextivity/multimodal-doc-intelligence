# app/api/routes/documents.py

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

class ContentTypeSummary(BaseModel):
    content_type: str
    chunk_count: int
    description: str

class DocumentInfo(BaseModel):
    file_name: str
    chunk_count: int
    content_types: list[str]

class DocumentsResponse(BaseModel):
    total_chunks: int
    documents: list[DocumentInfo]
    content_breakdown: list[ContentTypeSummary]

CONTENT_DESCRIPTIONS = {
    "text":    "Native text extracted from PDF",
    "image":   "Image file processed with VLM",
    "chart":   "Chart or graph described by VLM",
    "scanned": "Scanned page processed with VLM",
    "mixed":   "Mixed content page with text and visuals",
}

@router.get("", response_model=DocumentsResponse)
async def list_documents(request: Request):
    """
    Lists all ingested documents and their content breakdown.
    Shows how many chunks of each type were extracted.
    """
    vectorstore = request.app.state.vectorstore

    if vectorstore is None:
        return DocumentsResponse(
            total_chunks=0,
            documents=[],
            content_breakdown=[]
        )

    all_docs = vectorstore.get()
    total_chunks = len(all_docs["ids"])

    # Group by file name
    file_data: dict[str, dict] = {}
    content_type_totals: dict[str, int] = {}

    for meta in all_docs["metadatas"]:
        name = meta.get("file_name", "unknown")
        ct = meta.get("content_type", "unknown")

        if name not in file_data:
            file_data[name] = {"count": 0, "types": set()}
        file_data[name]["count"] += 1
        file_data[name]["types"].add(ct)

        content_type_totals[ct] = content_type_totals.get(ct, 0) + 1

    documents = [
        DocumentInfo(
            file_name=name,
            chunk_count=data["count"],
            content_types=list(data["types"])
        )
        for name, data in file_data.items()
    ]

    content_breakdown = [
        ContentTypeSummary(
            content_type=ct,
            chunk_count=count,
            description=CONTENT_DESCRIPTIONS.get(ct, "Unknown content type")
        )
        for ct, count in content_type_totals.items()
    ]

    return DocumentsResponse(
        total_chunks=total_chunks,
        documents=documents,
        content_breakdown=content_breakdown
    )