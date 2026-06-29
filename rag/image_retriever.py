"""
rag/image_retriever.py
══════════════════════
Phase 2A — CLIP + FAISS interior design image retrieval

Commands:
  Download images (pick one):
    python rag/image_retriever.py --unsplash YOUR_KEY   ← recommended, free
    python rag/image_retriever.py --hf                  ← HuggingFace, tries multiple datasets

  Build CLIP index (run once after downloading):
    python rag/image_retriever.py --build data/interior_images/

  Test retrieval:
    python rag/image_retriever.py --query "scandinavian warm bedroom"

  Check what's downloaded:
    python rag/image_retriever.py --check

Get a free Unsplash key: https://unsplash.com/developers → New Application
"""

import json
import sys
import argparse
import time
from pathlib import Path

INDEX_PATH  = Path("data/clip_index.faiss")
META_PATH   = Path("data/clip_meta.json")
BATCH_SIZE  = 32   # images per embedding batch — reduce if RAM is tight


# ── model loading ─────────────────────────────────────────────────────────────

def get_device():
    """Auto-detect best available device: CUDA GPU > MPS (Apple) > CPU."""
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {name}")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple MPS GPU detected")
        return "mps"
    else:
        print("No GPU — using CPU")
        return "cpu"


def load_clip():
    """
    Load CLIP model (ViT-B/32). Auto-uses GPU if available.
    First run downloads ~350MB weights to ~/.cache/
    Returns (model, preprocess_fn, tokenizer, device)
    """
    import open_clip
    device = get_device()
    print(f"Loading CLIP (ViT-B-32) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print(f"CLIP loaded on {device}")
    return model, preprocess, tokenizer, device


# ── index building ────────────────────────────────────────────────────────────

def _load_existing_index() -> tuple[object, list, set]:
    """
    Load existing FAISS index + metadata if they exist.
    Returns (index_or_None, metadata_list, set_of_already_indexed_paths)
    """
    import faiss

    if INDEX_PATH.exists() and META_PATH.exists():
        index    = faiss.read_index(str(INDEX_PATH))
        metadata = json.loads(META_PATH.read_text())
        # build a set of paths already in the index for fast lookup
        indexed_paths = {m["path"] for m in metadata}
        print(f"Existing index: {index.ntotal} vectors, {len(metadata)} images")
        return index, metadata, indexed_paths
    else:
        return None, [], set()


def build_index(image_dir: str, rebuild: bool = False):
    """
    Embed images in image_dir with CLIP and save/update FAISS index.

    INCREMENTAL BY DEFAULT:
      - Loads existing index if present
      - Only embeds images not already in the index
      - Appends new vectors + metadata to existing index
      - Skips the whole run if nothing new

    Set rebuild=True (or --rebuild flag) to re-embed everything from scratch.

    Example:
      First run:  10,000 images → embeds all 10,000
      Add 500:    10,500 images → embeds only 500 new ones
      No change:  10,500 images → nothing to do, exits immediately
    """
    import faiss
    import torch
    import numpy as np
    from PIL import Image

    image_dir = Path(image_dir)

    # collect all image paths in the directory
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    all_paths  = []
    for ext in extensions:
        all_paths.extend(image_dir.rglob(ext))
    all_paths = sorted(str(p) for p in all_paths)

    if not all_paths:
        print(f"No images found in {image_dir}")
        print("Run: python rag/image_retriever.py --unsplash YOUR_KEY")
        return

    # load existing index (if any) unless rebuild requested
    if rebuild:
        print("Rebuild mode — re-embedding all images from scratch")
        index, metadata, indexed_paths = None, [], set()
    else:
        index, metadata, indexed_paths = _load_existing_index()

    # filter to only new images
    new_paths = [p for p in all_paths if p not in indexed_paths]

    if not new_paths:
        print(f"Nothing to do — all {len(all_paths)} images already indexed.")
        print(f"Use --rebuild to force re-embed everything.")
        return

    print(f"Total images in folder: {len(all_paths)}")
    print(f"Already indexed:        {len(indexed_paths)}")
    print(f"New to embed:           {len(new_paths)}")

    model, preprocess, _, device = load_clip()

    new_embeddings = []
    new_metadata   = []
    errors         = 0

    for batch_start in range(0, len(new_paths), BATCH_SIZE):
        batch_paths   = new_paths[batch_start : batch_start + BATCH_SIZE]
        batch_tensors = []
        batch_meta    = []

        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                batch_tensors.append(img)
                batch_meta.append({"path": p, "filename": Path(p).name})
            except Exception:
                errors += 1
                continue

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        new_embeddings.append(emb.cpu().numpy().astype("float32"))
        new_metadata.extend(batch_meta)

        done = min(batch_start + BATCH_SIZE, len(new_paths))
        if (batch_start // BATCH_SIZE) % 5 == 0:
            print(f"  Embedded {done}/{len(new_paths)} new images...")

    if not new_embeddings:
        print("No new images could be embedded.")
        return

    new_matrix = np.vstack(new_embeddings)

    # add to existing index or create new one
    if index is None:
        index = faiss.IndexFlatIP(new_matrix.shape[1])
        print(f"Created new index (dim={new_matrix.shape[1]})")
    else:
        print(f"Extending existing index: {index.ntotal} → {index.ntotal + len(new_matrix)}")

    index.add(new_matrix)
    metadata.extend(new_metadata)

    # ── critical: save index and metadata atomically ──────────────────────────
    # write to temp files first, then rename — prevents corrupt state if
    # the process is interrupted mid-write
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_index = INDEX_PATH.with_suffix(".tmp")
    tmp_meta  = META_PATH.with_suffix(".tmp")

    faiss.write_index(index, str(tmp_index))
    tmp_meta.write_text(json.dumps(metadata, indent=2))

    tmp_index.replace(INDEX_PATH)
    tmp_meta.replace(META_PATH)
    # ──────────────────────────────────────────────────────────────────────────

    print(f"\nIndex updated:")
    print(f"  New images added:  {len(new_metadata)}")
    print(f"  Total in index:    {index.ntotal}")
    print(f"  Errors skipped:    {errors}")
    print(f"  Index file:        {INDEX_PATH} ({INDEX_PATH.stat().st_size/1024/1024:.1f} MB)")
    print(f"\nTest: python rag/image_retriever.py --query 'scandinavian bedroom'")


# ── dataset download ──────────────────────────────────────────────────────────

IMG_DIR = Path("data/interior_images")

# Unsplash queries — covers the main interior styles the project supports
_UNSPLASH_QUERIES = [
    "scandinavian interior bedroom",
    "minimalist living room design",
    "japandi interior bedroom",
    "mid century modern living room",
    "bohemian bedroom interior",
    "industrial apartment interior",
    "contemporary bedroom white",
    "cozy bedroom warm light",
    "modern kitchen interior",
    "scandinavian living room natural wood",
    "indian vintage haveli interior",
    "moroccan interior design colourful",
    "wabi sabi japanese interior",
    "luxury bedroom marble",
    "small bedroom interior design",
]


def download_unsplash(access_key: str, per_query: int = 30):
    """
    Download interior design photos from Unsplash API.
    Free tier: 50 requests/hour, 30 photos per request.
    Legal, high quality, no scraping needed.

    Get a free key: https://unsplash.com/developers → New Application
    """
    import urllib.request
    import urllib.parse
    import json as _json
    import time as _time

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    print(f"Downloading from Unsplash — {len(_UNSPLASH_QUERIES)} queries × {per_query} photos")
    print(f"Estimated total: ~{len(_UNSPLASH_QUERIES) * per_query} images\n")

    for query in _UNSPLASH_QUERIES:
        url = (
            "https://api.unsplash.com/search/photos"
            f"?query={urllib.parse.quote(query)}"
            f"&per_page={per_query}"
            "&orientation=landscape"
        )
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": f"Client-ID {access_key}"}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = _json.loads(r.read())
        except Exception as e:
            print(f"  '{query}' failed: {e}")
            continue

        photos = data.get("results", [])
        saved = 0
        for photo in photos:
            img_url  = photo["urls"]["small"]   # 400px wide — enough for CLIP
            photo_id = photo["id"]
            out_path = IMG_DIR / f"unsplash_{photo_id}.jpg"
            if out_path.exists():
                saved += 1
                continue
            try:
                urllib.request.urlretrieve(img_url, out_path)
                saved += 1
            except Exception:
                continue

        total_saved += saved
        print(f"  '{query}' → {saved} photos  (total so far: {total_saved})")
        _time.sleep(2)   # stay within Unsplash rate limits

    print(f"\nDone. {total_saved} images saved to {IMG_DIR}")
    print(f"Next: python rag/image_retriever.py --build data/interior_images/")


def download_huggingface():
    """
    Download from verified HuggingFace interior design datasets.

    Datasets confirmed working (checked May 2026):
      rrustom/architecture2022          — 3,290 real room photos + captions
      victorzarzu/interior-design-...  — 4,260 before/after interior pairs
      MohamedAli77/interior-rooms       — 100 rows with depth maps

    Combined: ~7,600 interior images. Enough for a solid CLIP index.
    """
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage
    except ImportError:
        print("Run: pip install datasets Pillow")
        return

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # (dataset_name, image_keys_to_try, max_rows, subfolder_prefix)
    datasets_to_download = [
        (
            "rrustom/architecture2022",
            ["image"],
            3300,
            "arch",
        ),
        (
            "victorzarzu/interior-design-prompt-editing-dataset-train",
            ["image", "input_image", "original_image"],
            4300,
            "victor",
        ),
        (
            "MohamedAli77/interior-rooms",
            ["full_room"],   # also has empty_room, depth, mask — use full_room for CLIP
            100,
            "rooms",
        ),
        # fallbacks — try if above fail
        ("ellljoy/interior-design",           ["image"], 2000, "elll"),
        ("razor7x/Interior_Design_Dataset",   ["image"], 2000, "razor"),
    ]

    total_saved = 0

    for ds_name, img_keys, max_rows, prefix in datasets_to_download:
        print(f"\nDownloading {ds_name}...")
        try:
            ds = load_dataset(ds_name, split="train", streaming=True)
            sample = next(iter(ds))

            # find the right image key
            img_key = next((k for k in img_keys if k in sample), None)
            if img_key is None:
                print(f"  No image key found. Keys: {list(sample.keys())}")
                continue

            print(f"  Using key: '{img_key}'")
            saved = 0

            for i, item in enumerate(ds):
                if i >= max_rows:
                    break
                try:
                    img = item[img_key]
                    if not isinstance(img, PILImage.Image):
                        continue
                    out_path = IMG_DIR / f"{prefix}_{i:05d}.jpg"
                    img.convert("RGB").save(out_path, quality=85)
                    saved += 1
                    if saved % 200 == 0:
                        print(f"  Saved {saved}...")
                except Exception:
                    continue

            print(f"  Saved {saved} images from {ds_name}")
            total_saved += saved

        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")
            continue

    print(f"\nTotal downloaded: {total_saved} images → {IMG_DIR}")
    if total_saved > 0:
        print(f"Next: python rag/image_retriever.py --build data/interior_images/")
    else:
        print("All datasets failed. Try Unsplash instead:")
        print("  python rag/image_retriever.py --unsplash YOUR_KEY")
        print("  (free key at https://unsplash.com/developers)")


def check_images():
    """Show what's already downloaded."""
    images = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png"))
    if images:
        print(f"{len(images)} images in {IMG_DIR}")
        if INDEX_PATH.exists():
            print(f"CLIP index exists: {INDEX_PATH}")
            print("Ready to query: python rag/image_retriever.py --query 'scandinavian bedroom'")
        else:
            print("Index not built yet.")
            print(f"Run: python rag/image_retriever.py --build {IMG_DIR}")
    else:
        print(f"No images in {IMG_DIR}")
        print("\nDownload options:")
        print("  python rag/image_retriever.py --unsplash YOUR_KEY   ← recommended")
        print("  python rag/image_retriever.py --hf                  ← HuggingFace")
        print("  Or copy your own JPEGs into data/interior_images/")


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve_design_references(query: str, top_k: int = 5) -> list[dict]:
    """
    Find interior design images matching a text query using CLIP.

    How it works:
      1. Embed the query text with CLIP text encoder → 512-dim vector
      2. Search FAISS index for the top_k closest image vectors
      3. Return paths + similarity scores

    Args:
        query:  natural language e.g. "scandinavian warm bedroom minimal"
        top_k:  number of results to return

    Returns:
        list of {path, filename, similarity} dicts
        similarity is cosine similarity 0-1, higher = more similar

    Falls back to empty list if index not built yet.
    """
    if not INDEX_PATH.exists():
        print("[rag] CLIP index not built yet.")
        print("Run: python rag/image_retriever.py --download")
        print("Then: python rag/image_retriever.py --build data/interior_images/")
        return []

    import faiss
    import torch
    import numpy as np

    model, _, tokenizer, device = load_clip()
    metadata = json.loads(META_PATH.read_text())
    index    = faiss.read_index(str(INDEX_PATH))

    # embed the text query
    # tokenizer converts text → token IDs, model converts → 512-dim vector
    tokens = tokenizer([query]).to(device)

    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    query_vec = text_emb.cpu().numpy().astype("float32")

    # search — returns (scores array, indices array)
    # scores: cosine similarity values
    # indices: which rows in the FAISS index matched
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:   # FAISS returns -1 for empty slots
            continue
        results.append({
            **metadata[idx],                # path, filename
            "similarity": round(float(score), 4),
        })

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIP image retriever — download, build index, query"
    )
    parser.add_argument("--unsplash", metavar="ACCESS_KEY",
                        help="Download images via Unsplash API (free key at unsplash.com/developers)")
    parser.add_argument("--hf", action="store_true",
                        help="Download images from HuggingFace datasets")
    parser.add_argument("--check", action="store_true",
                        help="Show what images and indexes are already present")
    parser.add_argument("--build", metavar="IMAGE_DIR",
                        help="Build/update CLIP index from images in IMAGE_DIR (incremental by default)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force full rebuild — re-embed all images from scratch")
    parser.add_argument("--query", metavar="TEXT",
                        help="Test retrieval with a text query")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    if args.unsplash:
        download_unsplash(args.unsplash)
    elif args.hf:
        download_huggingface()
    elif args.check:
        check_images()
    elif args.build:
        build_index(args.build, rebuild=args.rebuild)
    elif args.query:
        t0 = time.time()
        results = retrieve_design_references(args.query, top_k=args.topk)
        elapsed = round(time.time() - t0, 3)
        print(f"\nQuery: '{args.query}'")
        print(f"Top {len(results)} results ({elapsed}s):\n")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r['filename']}")
            print(f"     similarity: {r['similarity']}")
            print(f"     path: {r['path']}\n")
    else:
        check_images()
        print()
        parser.print_help()