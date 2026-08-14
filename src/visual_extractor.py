# src/visual_extractor.py
# Extracts structured data from visual content using GPT-4o.
# Combines image preprocessing with Pydantic structured output.

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
import json
from image_processor import prepare_image_for_vlm

client = OpenAI()

# ── Pydantic schemas for visual content ──────────────────────────────────────

class ChartData(BaseModel):
    """Structured data extracted from a chart or graph."""

    chart_type: str = Field(
        description="Type of chart: bar, line, pie, scatter, etc."
    )
    title: Optional[str] = Field(
        default=None,
        description="Chart title if visible"
    )
    x_axis_label: Optional[str] = Field(
        default=None,
        description="Label on the x axis"
    )
    y_axis_label: Optional[str] = Field(
        default=None,
        description="Label on the y axis"
    )
    data_points: list[dict] = Field(
        description="List of data points, each as a dict with x and y keys"
    )
    trend: str = Field(
        description="Overall trend description: increasing, decreasing, stable, cyclical"
    )
    key_insight: str = Field(
        description="Most important insight from this chart in one sentence"
    )

class TableData(BaseModel):
    """Structured data extracted from a table."""

    headers: list[str] = Field(
        description="Column headers in order from left to right"
    )
    rows: list[list[str]] = Field(
        description="Table rows, each row is a list of cell values"
    )
    row_count: int = Field(
        description="Total number of data rows excluding header"
    )
    key_insight: str = Field(
        description="Most important observation from this table"
    )

class DocumentPage(BaseModel):
    """Structured content extracted from a scanned document page."""

    page_type: str = Field(
        description="Type of content: text_heavy, chart_heavy, table_heavy, mixed"
    )
    extracted_text: str = Field(
        description="All text content extracted from the page, preserving structure"
    )
    contains_charts: bool = Field(
        description="Whether the page contains any charts or graphs"
    )
    contains_tables: bool = Field(
        description="Whether the page contains any tables"
    )
    key_topics: list[str] = Field(
        description="Main topics or subjects covered on this page"
    )

def safe_parse(schema_class, raw_json: dict):
    """
    Tries to parse raw_json into schema_class.
    If validation fails, prints the raw JSON so you can
    see exactly what the LLM returned and why it failed.

    This is the most useful debugging tool for structured output —
    always show the raw response before raising the error.
    """
    try:
        return schema_class(**raw_json)
    except Exception as e:
        print(f"\n[DEBUG] Pydantic validation failed for {schema_class.__name__}")
        print(f"[DEBUG] Raw JSON received from LLM:")
        print(json.dumps(raw_json, indent=2))
        print(f"[DEBUG] Error: {e}\n")
        raise

def extract_chart_data(image_path: str) -> ChartData:
    """
    Extracts structured data from a chart image.
    """
    image_block = prepare_image_for_vlm(image_path, detail="high")

    prompt = """Extract all data from this chart and return a JSON object
with EXACTLY these field names:

{
    "chart_type": "type of chart (bar, line, pie, scatter, etc.)",
    "title": "chart title if visible, or null if not present",
    "x_axis_label": "x axis label if visible, or null",
    "y_axis_label": "y axis label if visible, or null",
    "data_points": [
        {"x": "label or value", "y": "numeric value"},
        {"x": "label or value", "y": "numeric value"}
    ],
    "trend": "overall trend: increasing, decreasing, stable, or cyclical",
    "key_insight": "most important insight from this chart in one sentence"
}

Rules:
- Use exactly these field names, no others
- Extract every visible data point
- Be exact with numbers — do not round unless the chart rounds
- If a value is unclear, use null"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_block
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=2000
    )

    raw_json = json.loads(response.choices[0].message.content)
    return safe_parse(ChartData, raw_json)

def extract_table_data(image_path: str) -> TableData:
    """Extracts structured data from a table image."""

    image_block = prepare_image_for_vlm(image_path, detail="high")

    prompt = """Extract all data from this table and return a JSON object
with EXACTLY these field names:

{
    "headers": ["column1", "column2", "column3"],
    "rows": [
        ["row1col1", "row1col2", "row1col3"],
        ["row2col1", "row2col2", "row2col3"]
    ],
    "row_count": 5,
    "key_insight": "most important observation from this table"
}

Rules:
- Use exactly these field names, no others
- Extract every row and column exactly as shown
- Preserve exact values — do not reformat numbers or dates
- row_count is the number of data rows, not including the header"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_block
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=2000
    )

    raw_json = json.loads(response.choices[0].message.content)
    return safe_parse(TableData, raw_json)

def extract_document_page(image_path: str) -> DocumentPage:
    """Extracts structured content from a scanned document page."""

    image_block = prepare_image_for_vlm(image_path, detail="high")

    prompt = """Analyze this document page and return a JSON object
with EXACTLY these field names:

{
    "page_type": "one of: text_heavy, chart_heavy, table_heavy, mixed",
    "extracted_text": "all text content preserving structure with markdown",
    "contains_charts": true or false,
    "contains_tables": true or false,
    "key_topics": ["topic1", "topic2", "topic3"]
}

Rules:
- Use exactly these field names, no others
- For extracted_text: use markdown to show headings, bullets, numbered lists
- Be comprehensive — extract every piece of text visible"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_block
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=3000
    )

    raw_json = json.loads(response.choices[0].message.content)
    return safe_parse(DocumentPage, raw_json)

def describe_image_for_rag(image_path: str) -> str:
    """
    Generates a rich text description of an image for RAG ingestion.

    This is the key function for the multimodal RAG pipeline.
    Instead of storing raw images in the vector store (not possible),
    we store rich text descriptions that capture the semantic content.

    The description is designed to be:
    1. Searchable — uses natural language that matches likely queries
    2. Complete — captures all data visible in the image
    3. Structured — preserves relationships between data points

    This description becomes a Document chunk in the vector store,
    exactly like a text chunk from a PDF.
    """
    image_block = prepare_image_for_vlm(image_path, detail="high")

    prompt = """Generate a comprehensive text description of this image
for use in a search system.

The description must:
1. Identify what type of visual content this is
2. Extract ALL specific data, numbers, labels, and text visible
3. Describe relationships and trends
4. Use natural language that would match search queries about this content

Format the description as flowing text that captures everything
a reader would need to answer questions about this image
without seeing it.

Be specific with numbers — include every data point visible."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_block
            ]
        }],
        max_tokens=1000
    )

    return response.choices[0].message.content