# app/main.py

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from pathlib import Path
from app.api.routes import health, ingest, chat, documents, extract
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Multimodal Document Intelligence API...")

    from app.core.multimodal_ingestion import load_vectorstore, ingest_to_vectorstore
    from app.core.multimodal_agent import build_multimodal_agent

    chroma_path = Path(settings.chroma_path)

    if chroma_path.exists() and any(chroma_path.iterdir()):
        print(f"Loading vector store from {settings.chroma_path}")
        app.state.vectorstore = load_vectorstore()
        app.state.agent = build_multimodal_agent(app.state.vectorstore)
        print("Agent ready.")
    else:
        sample_dir = Path("samples")
        sample_files = []

        if sample_dir.exists():
            sample_files = (
                list(sample_dir.glob("*.pdf")) +
                list(sample_dir.glob("*.png")) +
                list(sample_dir.glob("*.jpg")) +
                list(sample_dir.glob("*.jpeg"))
            )

        if sample_files:
            print(f"Auto-ingesting {len(sample_files)} sample files...")
            vectorstore = None
            for f in sample_files:
                vectorstore = ingest_to_vectorstore(
                    str(f),
                    original_filename=f.name
                )
            app.state.vectorstore = vectorstore
            app.state.agent = build_multimodal_agent(vectorstore)
            print("Sample files ingested. Agent ready.")
        else:
            print("No vector store found. Use POST /ingest to add documents.")
            app.state.vectorstore = None
            app.state.agent = None

    yield
    print("Shutting down...")

app = FastAPI(
    title="Multimodal Document Intelligence Copilot",
    description="AI assistant for text, charts, tables, and scanned documents",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(extract.router, prefix="/extract", tags=["Extraction"])