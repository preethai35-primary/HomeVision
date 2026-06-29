"""
vision/depth_estimator.py
══════════════════════════
Phase 1 — STEP 2 of 2

What this does:
  Run Depth-Anything v2 (local model, no API cost) on your room photo.
  Output: a greyscale depth map PNG where bright = close, dark = far.
  This depth map is later used to constrain ControlNet image generation.

First run: downloads ~400MB model weights from HuggingFace (cached after).
Subsequent runs: ~8–15s on CPU.

Run directly to test:
  python vision/depth_estimator.py data/my_room.jpg
  # check data/outputs/depth_map.png
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image


def estimate_depth(
    image_path: str,
    model_size: str = "small",
    output_path: str = "data/outputs/depth_map.png",
) -> dict:
    """
    Estimate per-pixel depth from a room photo using Depth-Anything v2.

    Args:
        image_path:  path to room photo
        model_size:  "small" (fast, CPU ok) | "base" | "large" (GPU recommended)
        output_path: where to save the depth map PNG

    Returns:
        dict with:
            depth_array:  numpy array (H, W) float32, normalised 0–1
                          1.0 = closest point, 0.0 = furthest point
            depth_image:  PIL Image (grayscale) ready to save / display
            output_path:  path where depth map was saved
            stats:        min / max / mean depth values
    """
    from transformers import pipeline

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # HuggingFace pipeline abstracts all the model loading complexity.
    # On first run it downloads weights to ~/.cache/huggingface/
    # Model sizes:
    #   small  → 24M params, ~400MB download, ~8s on CPU  ← use this
    #   base   → 97M params, ~800MB download, ~20s on CPU
    #   large  → 335M params, needs GPU
    model_id = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
    print(f"Loading depth model: {model_id}")
    print("(First run downloads ~400MB — subsequent runs are instant)")

    pipe = pipeline(
        task="depth-estimation",
        model=model_id,
        device="cpu",   # change to "cuda" if you have a GPU
    )

    # Load and run
    image = Image.open(path).convert("RGB")
    print(f"Running depth estimation on {image.size[0]}×{image.size[1]} image...")
    result = pipe(image)

    # result["depth"] is a PIL Image from the model
    depth_pil = result["depth"]
    depth_arr = np.array(depth_pil).astype(np.float32)

    # Normalise to 0–1 range
    # Raw model output is in arbitrary units — normalise so we can save as 8-bit PNG
    d_min = depth_arr.min()
    d_max = depth_arr.max()
    depth_norm = (depth_arr - d_min) / (d_max - d_min + 1e-8)
    # 1e-8 prevents division by zero on flat/uniform images

    # Resize to match original image dimensions
    # The model may output at a different resolution than the input
    orig_w, orig_h = image.size
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_resized = np.array(
        Image.fromarray(depth_uint8).resize((orig_w, orig_h), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Build PIL image for saving
    depth_image = Image.fromarray((depth_resized * 255).astype(np.uint8))

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    depth_image.save(out)
    print(f"Depth map saved: {out}")

    return {
        "depth_array": depth_resized,
        "depth_image": depth_image,
        "output_path": str(out),
        "stats": {
            "min": float(depth_resized.min()),
            "max": float(depth_resized.max()),
            "mean": float(depth_resized.mean()),
        },
        "original_size": (orig_w, orig_h),
    }


# ── Run directly for testing ──────────────────────────────────────────────────

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/my_room.jpg"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/outputs/depth_map.png"

    try:
        result = estimate_depth(image_path, output_path=output_path)
        print("\n" + "─" * 50)
        print("DEPTH ESTIMATION COMPLETE:")
        print(f"  Output:       {result['output_path']}")
        print(f"  Image size:   {result['original_size']}")
        print(f"  Depth stats:  {result['stats']}")
        print("\nOpen data/outputs/depth_map.png to inspect.")
        print("Bright pixels = close to camera, dark = far away.")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure data/my_room.jpg exists first.")
    except Exception as e:
        print(f"\nError: {e}")
