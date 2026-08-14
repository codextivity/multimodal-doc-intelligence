# src/test_visual_extraction.py

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from visual_extractor import (
    extract_chart_data,
    extract_table_data,
    extract_document_page,
    describe_image_for_rag
)
import json

def test_extraction():
    test_dir = Path("../test_images")
    images = list(test_dir.glob("*.jpg")) + \
             list(test_dir.glob("*.jpeg")) + \
             list(test_dir.glob("*.png"))

    if not images:
        print("No images found in test_images/")
        return

    # Test chart extraction
    print("=" * 60)
    print("TEST 1: Chart Structured Extraction")
    print("=" * 60)
    chart_result = extract_chart_data(str(images[0]))
    print(f"Chart type:    {chart_result.chart_type}")
    print(f"Title:         {chart_result.title}")
    print(f"X axis:        {chart_result.x_axis_label}")
    print(f"Y axis:        {chart_result.y_axis_label}")
    print(f"Data points:   {json.dumps(chart_result.data_points, indent=2)}")
    print(f"Trend:         {chart_result.trend}")
    print(f"Key insight:   {chart_result.key_insight}")

    if len(images) > 1:
        print("\n" + "=" * 60)
        print("TEST 2: RAG Description Generation")
        print("=" * 60)
        description = describe_image_for_rag(str(images[1]))
        print(description)
        print(f"\nDescription length: {len(description)} characters")
        print("This text would be stored as a chunk in the vector store.")

if __name__ == "__main__":
    test_extraction()