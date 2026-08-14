# Dockerfile
FROM python:3.11-slim

# Create non-root user — required for Hugging Face, good practice everywhere
RUN useradd -m -u 1000 appuser

WORKDIR /app

# System dependencies
# tesseract-ocr: OCR engine for scanned documents
# poppler-utils: required by pdf2image for PDF rendering
# libgl1: required by OpenCV/PIL for image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY samples/ ./samples/

# Create directories with correct permissions
RUN mkdir -p /data/chroma_db && \
    chown -R appuser:appuser /app /data

USER appuser

# Port — ${PORT:-7860} works for both Render ($PORT) and local (7860)
EXPOSE 7860 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]