# app/api/routes/extract.py
# Structured extraction endpoint for visual content.

import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.core.visual_extractor import (
    extract_chart_data,
    extract_table_data,
    extract_document_page
)

router = APIRouter()

SCHEMA_MAP = {
    "chart":    extract_chart_data,
    "table":    extract_table_data,
    "document": extract_document_page,
}

class ExtractResponse(BaseModel):
    extraction_type: str
    file_name: str
    data: dict

@router.post("", response_model=ExtractResponse)
async def extract_from_image(
    file: UploadFile = File(...),
    extraction_type: str = "chart"
):
    """
    Extract structured data from an uploaded image.

    extraction_type options:
    - "chart"    → extracts chart type, axes, data points, trend
    - "table"    → extracts headers, rows, key insight
    - "document" → extracts text, topics, content classification

    Returns typed, validated structured data.
    """
    if extraction_type not in SCHEMA_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid extraction_type. Choose from: {list(SCHEMA_MAP.keys())}"
        )

    file_ext = "." + file.filename.split(".")[-1].lower()
    if file_ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail="Only image files supported for extraction"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        extractor_fn = SCHEMA_MAP[extraction_type]
        result = extractor_fn(tmp_path)
    finally:
        os.unlink(tmp_path)

    return ExtractResponse(
        extraction_type=extraction_type,
        file_name=file.filename,
        data=result.model_dump()
    )