"""
scripts/download_models.py
══════════════════════════
Download all model weights HomeVision needs.

Run once before starting the app:
  python scripts/download_models.py

Models downloaded (~10 GB total, cached in ~/.cache/huggingface/hub):
  1. stabilityai/stable-diffusion-xl-base-1.0   (~6.5 GB fp16)
  2. diffusers/controlnet-depth-sdxl-1.0         (~2.4 GB fp16)
  3. madebyollin/sdxl-vae-fp16-fix               (~160 MB)
  4. depth-anything/Depth-Anything-V2-Small-hf   (~100 MB)
  5. open_clip ViT-B-32 (openai)                 (~350 MB)
  6. sentence-transformers all-MiniLM-L6-v2       (~90 MB)

After this completes, build the IKEA catalogue index:
  python rag/ikea_search.py --download
  python rag/ikea_search.py --build
"""

import os
import sys
import time

# avoid OMP crash on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _banner(msg: str):
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}")


def _ok(msg: str):
    print(f"  ✓  {msg}")


def _check_imports():
    missing = []
    for pkg in ("torch", "diffusers", "transformers", "open_clip", "sentence_transformers"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run:  pip install -r requirements.txt")
        sys.exit(1)


# ── 1. SDXL base ─────────────────────────────────────────────────────────────

def download_sdxl():
    _banner("1 / 6  SDXL base  (stabilityai/stable-diffusion-xl-base-1.0, ~6.5 GB)")
    from diffusers import StableDiffusionXLControlNetPipeline
    import torch

    t0 = time.time()
    StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    _ok(f"done in {time.time() - t0:.0f}s")


# ── 2. ControlNet depth ───────────────────────────────────────────────────────

def download_controlnet():
    _banner("2 / 6  ControlNet depth  (diffusers/controlnet-depth-sdxl-1.0, ~2.4 GB)")
    from diffusers import ControlNetModel
    import torch

    t0 = time.time()
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    _ok(f"done in {time.time() - t0:.0f}s")


# ── 3. VAE ────────────────────────────────────────────────────────────────────

def download_vae():
    _banner("3 / 6  VAE  (madebyollin/sdxl-vae-fp16-fix, ~160 MB)")
    from diffusers import AutoencoderKL
    import torch

    t0 = time.time()
    AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    _ok(f"done in {time.time() - t0:.0f}s")


# ── 4. Depth-Anything ─────────────────────────────────────────────────────────

def download_depth():
    _banner("4 / 6  Depth-Anything v2 Small  (~100 MB)")
    from transformers import pipeline as hf_pipeline

    t0 = time.time()
    hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu",
    )
    _ok(f"done in {time.time() - t0:.0f}s")


# ── 5. CLIP (open_clip) ───────────────────────────────────────────────────────

def download_clip():
    _banner("5 / 6  CLIP ViT-B-32  (open_clip, ~350 MB)")
    import open_clip
    import warnings

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    _ok(f"done in {time.time() - t0:.0f}s")


# ── 6. sentence-transformers ─────────────────────────────────────────────────

def download_embedder():
    _banner("6 / 6  sentence-transformers all-MiniLM-L6-v2  (~90 MB)")
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    _ok(f"done in {time.time() - t0:.0f}s")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nHomeVision model downloader")
    print("All weights cached in ~/.cache/huggingface/hub — no re-download on next run.\n")

    _check_imports()

    steps = [
        download_sdxl,
        download_controlnet,
        download_vae,
        download_depth,
        download_clip,
        download_embedder,
    ]

    wall = time.time()
    for fn in steps:
        try:
            fn()
        except Exception as e:
            print(f"\n  ERROR: {e}")
            print("  Check your internet connection and try again.")
            sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"  All models downloaded in {(time.time() - wall) / 60:.1f} min")
    print(f"{'═' * 60}")
    print("\nNext steps:")
    print("  1.  python rag/ikea_search.py --download   # ~200 MB IKEA catalogue")
    print("  2.  python rag/ikea_search.py --build      # builds FAISS index (~5 min)")
    print("  3.  cp .env.example .env && nano .env      # add your OPENAI_API_KEY")
    print("  4.  python ui/gradio_app.py                # launch the UI\n")
