# src/test_vlm_basic.py
# Week 1, Day 1 — understand how GPT-4o processes images.
# We test three types of visual content before building any pipeline.

import base64
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()

def encode_image_to_base64(image_path: str) -> str:
    """
    Converts an image file to base64 string.

    Why base64?
    The OpenAI API accepts images as either URLs or base64-encoded strings.
    Base64 lets us send local files directly without hosting them anywhere.
    The tradeoff is message size — a 1MB image becomes ~1.37MB of text.

    This is exactly like how you would encode binary data for transmission
    in any network protocol — the same concept from your CV work with
    image serialization.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ask_vlm_about_image(image_path: str, question: str) -> str:
    """
    Sends an image and question to GPT-4o and returns the answer.

    Args:
        image_path: path to local image file
        question:   what to ask about the image

    Returns:
        GPT-4o's response as a string
    """
    # Determine image format from extension
    # The API needs to know the MIME type to decode the image correctly
    extension = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    mime_type = mime_types.get(extension, "image/jpeg")

    # Encode image to base64
    image_data = encode_image_to_base64(image_path)

    # Build the multimodal message
    # Notice the content is a LIST — it can contain multiple items
    # mixing text and images in any order
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                            # detail controls how many tokens the image costs:
                            # "low"  → 85 tokens, low resolution, fast
                            # "high" → up to 1105 tokens, high resolution, slow
                            # "auto" → OpenAI decides based on image size
                            # For charts with small text, use "high"
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content

def test_with_sample_images():
    """
    Tests VLM on different image types.
    Put test images in a test_images/ folder before running.
    """

    test_cases = [
        {
            "description": "Chart understanding",
            "question": """Analyze this chart and extract all data:
1. What type of chart is this?
2. What are the axis labels?
3. List every data point you can see (x value, y value)
4. What is the overall trend?"""
        },
        {
            "description": "Table extraction",
            "question": """Extract all data from this table:
1. List all column headers
2. List all row data exactly as shown
3. Format the output as: Column1 | Column2 | Column3"""
        },
        {
            "description": "Scanned document",
            "question": """Extract all text from this scanned document.
Preserve the structure — headings, paragraphs, bullet points.
If there are numbers or dates, extract them exactly."""
        }
    ]

    # Check for test images
    test_dir = Path("test_images")
    if not test_dir.exists():
        print("Create a test_images/ folder and add some images to test.")
        print("Good test images to use:")
        print("  - A screenshot of a bar chart or line chart")
        print("  - A screenshot of a table with data")
        print("  - A photo or scan of a printed document")
        return

    images = list(test_dir.glob("*.jpg")) + \
             list(test_dir.glob("*.jpeg")) + \
             list(test_dir.glob("*.png"))

    if not images:
        print("No images found in test_images/")
        return

    for i, image_path in enumerate(images[:3]):
        test_case = test_cases[i % len(test_cases)]
        print(f"\n{'='*60}")
        print(f"Image: {image_path.name}")
        print(f"Test:  {test_case['description']}")
        print(f"{'='*60}")

        answer = ask_vlm_about_image(str(image_path), test_case["question"])
        print(answer)

if __name__ == "__main__":
    test_with_sample_images()