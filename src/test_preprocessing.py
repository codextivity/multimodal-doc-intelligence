# src/test_preprocessing.py

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from PIL import Image
from image_processor import (
    load_image,
    resize_for_vlm,
    detect_image_type,
    estimate_token_cost,
    prepare_image_for_vlm
)
from openai import OpenAI

client = OpenAI()

def test_preprocessing():
    """
    Tests image preprocessing on your three test images.
    Shows token cost before and after resizing.
    """
    test_dir = Path("../test_images")
    images = list(test_dir.glob("*.jpg")) + \
             list(test_dir.glob("*.jpeg")) + \
             list(test_dir.glob("*.png"))

    print("IMAGE PREPROCESSING ANALYSIS")
    print("=" * 60)

    for image_path in images:
        print(f"\nFile: {image_path.name}")

        # Load original
        original = load_image(str(image_path))
        original_tokens = estimate_token_cost(original, "high")

        # Resize
        resized = resize_for_vlm(original)
        resized_tokens = estimate_token_cost(resized, "high")

        # Detect format
        img_format = detect_image_type(resized)

        print(f"  Original:  {original.size} → {original_tokens} tokens")
        print(f"  Resized:   {resized.size} → {resized_tokens} tokens")
        print(f"  Format:    {img_format}")
        print(f"  Savings:   {original_tokens - resized_tokens} tokens "
              f"({(1 - resized_tokens/original_tokens)*100:.1f}% reduction)")

    print("\n" + "=" * 60)
    print("VISUAL QUALITY TEST")
    print("=" * 60)

    # Test that preprocessing does not destroy readability
    # by asking VLM the same question before and after resize
    if images:
        test_image = str(images[0])
        question = "List every number you can see in this image."

        print(f"\nTesting on: {Path(test_image).name}")
        print("Question: List every number visible in the image")

        # Prepare preprocessed image
        content_block = prepare_image_for_vlm(test_image, detail="high")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    content_block
                ]
            }],
            max_tokens=500
        )

        print(f"\nVLM response:\n{response.choices[0].message.content}")

if __name__ == "__main__":
    test_preprocessing()