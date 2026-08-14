# src/image_processor.py
# Handles image loading, preprocessing, and preparation for VLM input.
#
# Core principle from computer vision:
# Use the minimum resolution that preserves the semantic content.
# For charts: need enough resolution to read axis labels and data points
# For tables: need enough resolution to read cell contents
# For scanned text: need enough resolution to distinguish characters

import base64
import io
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

# Maximum dimensions we send to the VLM.
# These values balance readability with token cost.
# GPT-4o processes images at 512px tiles — sending images larger than
# 2048px on the longest side gives diminishing returns.
MAX_WIDTH = 2048
MAX_HEIGHT = 2048

# Minimum dimensions — below this, text becomes unreadable
MIN_WIDTH = 512
MIN_HEIGHT = 512

def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from disk and converts to RGB.

    Why convert to RGB?
    Some images are RGBA (with transparency), grayscale (L mode),
    or CMYK (print format). The VLM API expects RGB.
    Converting ensures consistent behavior regardless of source format.
    This is the same normalization step you would do before feeding
    an image to a CNN — standardize the input format first.
    """
    image = Image.open(image_path)

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

def resize_for_vlm(image: Image.Image) -> Image.Image:
    """
    Resizes image to optimal dimensions for VLM processing.

    Strategy:
    - If image fits within MAX bounds → keep as-is (no upscaling)
    - If image exceeds MAX bounds → downscale proportionally

    Why no upscaling?
    Upscaling a small image does not add information — it just
    increases token cost without improving what the VLM can see.
    The VLM sees the same content whether the image is 601px or 780px
    wide, but the larger image costs more tokens.
    """
    width, height = image.size

    # If image already fits within maximum bounds, keep it as-is
    # We only resize if the image is TOO LARGE, not too small
    if width <= MAX_WIDTH and height <= MAX_HEIGHT:
        print(f"  Size OK: {width}x{height} (no resize needed)")
        return image

    # Image exceeds bounds — downscale proportionally
    scale_w = MAX_WIDTH / width
    scale_h = MAX_HEIGHT / height
    scale = min(scale_w, scale_h)  # take smaller to fit both dimensions

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = image.resize((new_width, new_height), Image.LANCZOS)
    print(f"  Resized: {width}x{height} → {new_width}x{new_height} "
          f"(scale: {scale:.2f})")

    return resized

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Converts a PIL Image to base64 string for API transmission.

    Why JPEG as default?
    JPEG compression significantly reduces file size for photos
    and scanned documents with minimal quality loss.
    PNG is better for charts and diagrams with sharp edges and
    solid colors — JPEG artifacts can obscure thin axis lines.

    We auto-detect the best format based on content type.
    """
    buffer = io.BytesIO()

    if format == "JPEG":
        # quality=85 is a good balance — visually lossless for text
        # but significantly smaller than quality=100
        image.save(buffer, format="JPEG", quality=85, optimize=True)
    else:
        image.save(buffer, format="PNG", optimize=True)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def detect_image_type(image: Image.Image) -> str:
    """
    Heuristically determines the best encoding format for an image.

    Charts and diagrams:
    - Have large areas of solid color
    - Sharp edges between colors
    - PNG preserves these better than JPEG

    Photos and scanned documents:
    - Have gradual color transitions
    - JPEG compression is efficient and visually lossless

    This is a simplified version of the content-aware compression
    decision your CV intuition already understands.
    """
    # Sample pixel variance as a proxy for image complexity
    # High variance = lots of color variation = photo/scan
    # Low variance = few colors = chart/diagram
    import numpy as np

    img_array = np.array(image)
    variance = np.var(img_array)

    # Threshold determined empirically
    # Charts typically have variance < 1000
    # Photos typically have variance > 3000
    if variance < 2000:
        return "PNG"   # chart or diagram — use lossless
    else:
        return "JPEG"  # photo or scan — use lossy compression

def prepare_image_for_vlm(
    image_path: str,
    detail: str = "high"
) -> dict:
    """
    Complete image preparation pipeline for VLM API calls.

    Steps:
    1. Load and normalize to RGB
    2. Resize to optimal dimensions
    3. Detect best encoding format
    4. Encode to base64

    Args:
        image_path: path to any supported image format
        detail:     "low" (fast, cheap) or "high" (accurate, expensive)
                   Use "high" for charts and tables with small text.
                   Use "low" for quick document classification.

    Returns:
        Dict ready to be included in an OpenAI API message content list
    """
    print(f"Preparing image: {Path(image_path).name}")

    # Step 1: Load
    image = load_image(image_path)
    print(f"  Original size: {image.size[0]}x{image.size[1]}")

    # Step 2: Resize
    image = resize_for_vlm(image)

    # Step 3: Detect format
    img_format = detect_image_type(image)
    print(f"  Encoding as: {img_format}")

    # Step 4: Encode
    extension = Path(image_path).suffix.lower()
    if extension == ".png":
        img_format = "PNG"  # always use PNG for PNG sources

    base64_image = image_to_base64(image, format=img_format)
    mime_type = "image/png" if img_format == "PNG" else "image/jpeg"

    # Return the content block ready for API insertion
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{base64_image}",
            "detail": detail
        }
    }

def estimate_token_cost(image: Image.Image, detail: str = "high") -> int:
    """
    Estimates token cost for an image before sending it.

    OpenAI's pricing model:
    - "low" detail:  always 85 tokens regardless of size
    - "high" detail: 85 base tokens + 170 tokens per 512x512 tile

    This lets us calculate cost before making API calls —
    useful for deciding whether to downscale further.
    """
    if detail == "low":
        return 85

    width, height = image.size

    # Number of 512x512 tiles needed to cover the image
    tiles_w = -(-width // 512)   # ceiling division
    tiles_h = -(-height // 512)
    total_tiles = tiles_w * tiles_h

    return 85 + (170 * total_tiles)