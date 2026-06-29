"""
rag/inspect_index.py
════════════════════
Inspect the contents of any FAISS index file.
Shows: size, vector dimensions, sample vectors, and runs a test query.

Usage:
  python rag/inspect_index.py --clip          inspect CLIP image index
  python rag/inspect_index.py --ikea          inspect IKEA text index
  python rag/inspect_index.py --both          inspect both
  python rag/inspect_index.py --query "scandinavian bedroom"   live query test
"""

import json
import argparse
import numpy as np
from pathlib import Path

CLIP_INDEX  = Path("data/clip_index.faiss")
CLIP_META   = Path("data/clip_meta.json")
IKEA_INDEX  = Path("data/ikea_index.faiss")
IKEA_META   = Path("data/ikea_meta.json")


def inspect_clip_index():
    print("\n" + "═" * 56)
    print("  CLIP IMAGE INDEX")
    print("═" * 56)

    if not CLIP_INDEX.exists():
        print("  Not built yet.")
        print("  Run: python rag/image_retriever.py --unsplash YOUR_KEY")
        print("       python rag/image_retriever.py --build data/interior_images/")
        return

    import faiss

    index = faiss.read_index(str(CLIP_INDEX))
    meta  = json.loads(CLIP_META.read_text())

    print(f"\n  Index file:     {CLIP_INDEX}")
    print(f"  File size:      {CLIP_INDEX.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Total vectors:  {index.ntotal}  (one per image)")
    print(f"  Vector size:    {index.d} dimensions  (CLIP ViT-B/32 output)")
    print(f"  Index type:     IndexFlatIP  (exact cosine search)")
    print(f"  Metadata rows:  {len(meta)}")

    print(f"\n  What's inside — first 5 entries:")
    print(f"  {'#':<4} {'filename':<35} {'path'}")
    print(f"  {'─'*4} {'─'*35} {'─'*30}")
    for i, m in enumerate(meta[:5]):
        fname = Path(m['path']).name
        print(f"  {i:<4} {fname:<35} {m['path'][:40]}")

    print(f"\n  What a vector looks like (image 0, first 8 of {index.d} numbers):")
    # reconstruct vector for image 0
    vec = index.reconstruct(0)
    print(f"  [{', '.join(f'{v:.4f}' for v in vec[:8])}, ...]")
    print(f"  Vector norm: {np.linalg.norm(vec):.4f}  (should be ~1.0 — normalised)")

    print(f"\n  Style breakdown (from filenames):")
    prefixes = {}
    for m in meta:
        prefix = Path(m['path']).name.split('_')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:8]:
        bar = "█" * (count // max(1, max(prefixes.values()) // 20))
        print(f"  {prefix:<15} {count:>5}  {bar}")

    print()


def inspect_ikea_index():
    print("\n" + "═" * 56)
    print("  IKEA PRODUCT INDEX")
    print("═" * 56)

    if not IKEA_INDEX.exists():
        print("  Not built yet.")
        print("  Run: python rag/ikea_search.py --download")
        print("       python rag/ikea_search.py --build")
        return

    import faiss

    index = faiss.read_index(str(IKEA_INDEX))
    meta  = json.loads(IKEA_META.read_text())

    print(f"\n  Index file:     {IKEA_INDEX}")
    print(f"  File size:      {IKEA_INDEX.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Total vectors:  {index.ntotal}  (one per product)")
    print(f"  Vector size:    {index.d} dimensions  (sentence-transformers output)")
    print(f"  Index type:     IndexFlatIP  (exact cosine search)")
    print(f"  Metadata rows:  {len(meta)}")

    print(f"\n  What's inside — first 5 products:")
    print(f"  {'name':<35} {'category':<15} {'price'}")
    print(f"  {'─'*35} {'─'*15} {'─'*8}")
    for p in meta[:5]:
        name  = p.get("name", "")[:34]
        cat   = p.get("category", "")[:14]
        price = p.get("price_eur", p.get("price", ""))
        print(f"  {name:<35} {cat:<15} {price}")

    print(f"\n  What a vector looks like (product 0, first 8 of {index.d} numbers):")
    vec = index.reconstruct(0)
    print(f"  [{', '.join(f'{v:.4f}' for v in vec[:8])}, ...]")
    print(f"  Vector norm: {np.linalg.norm(vec):.4f}  (should be ~1.0 — normalised)")

    print(f"\n  Category breakdown:")
    cats = {}
    for p in meta:
        c = p.get("top_category", p.get("category", "other"))
        cats[c] = cats.get(c, 0) + 1
    for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * (count // max(1, max(cats.values()) // 20))
        print(f"  {cat:<30} {count:>6}  {bar}")

    print(f"\n  Price range:")
    prices = []
    for p in meta:
        try:
            price_str = p.get("price_eur", p.get("price", ""))
            prices.append(float(price_str.replace("€","").replace("$","").replace(",","")))
        except Exception:
            continue
    if prices:
        print(f"  Min: €{min(prices):.2f}   Max: €{max(prices):.2f}   "
              f"Median: €{sorted(prices)[len(prices)//2]:.2f}")

    print()


def test_query_clip(query: str, top_k: int = 5):
    print(f"\n  CLIP query: '{query}'")
    print(f"  {'─'*50}")

    if not CLIP_INDEX.exists():
        print("  Index not built yet.")
        return

    import faiss
    import torch
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    meta  = json.loads(CLIP_META.read_text())
    index = faiss.read_index(str(CLIP_INDEX))

    tokens = tokenizer([query])
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    scores, indices = index.search(emb.numpy().astype("float32"), top_k)

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        fname = Path(meta[idx]["path"]).name
        print(f"  {rank+1}. {fname:<40} sim={score:.4f}")


def test_query_ikea(query: str, style: str = "scandinavian"):
    print(f"\n  IKEA query: '{style} {query}'")
    print(f"  {'─'*50}")

    if not IKEA_INDEX.exists():
        print("  Index not built yet.")
        return

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rag.ikea_search import search_ikea_products

    results = search_ikea_products(
        style=style,
        furniture_list=[{"name": query}],
        top_k_per_item=3,
    )
    for r in results[:5]:
        print(f"  {r['name'][:40]:<42} {r['price']:<8} sim={r['similarity']:.4f}")
        print(f"     {r['description'][:70]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip",  action="store_true", help="Inspect CLIP index")
    parser.add_argument("--ikea",  action="store_true", help="Inspect IKEA index")
    parser.add_argument("--both",  action="store_true", help="Inspect both indexes")
    parser.add_argument("--query", metavar="TEXT",      help="Run a live test query on both indexes")
    parser.add_argument("--style", default="scandinavian", help="Style for IKEA query")
    args = parser.parse_args()

    if args.both or (not any([args.clip, args.ikea, args.query])):
        inspect_clip_index()
        inspect_ikea_index()
    elif args.clip:
        inspect_clip_index()
    elif args.ikea:
        inspect_ikea_index()

    if args.query:
        test_query_clip(args.query)
        test_query_ikea(args.query, style=args.style)