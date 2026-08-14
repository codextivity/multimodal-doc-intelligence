from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    capabilities: list[str]

@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        capabilities=[
            "text_pdf_ingestion",
            "mixed_pdf_ingestion",
            "scanned_pdf_ingestion",
            "image_ingestion",
            "multimodal_qa",
            "structured_extraction"
        ]
    )