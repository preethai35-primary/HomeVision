"""
rag/ikea_search.py
══════════════════
Phase 2B — IKEA product semantic search

Dataset: jeffreyszhou/ikea-us-products-2025
  30,500 real IKEA products scraped July 2025. MIT licence.
  Fields: title, description, category_tree, price, image_urls, source_url, materials
  Prices are USD — we convert to EUR (÷1.08) since you're in NL.
  Product URLs are ikea.com/us — we rewrite to ikea.com/nl/en automatically.

Commands:
  Download catalogue:
    python rag/ikea_search.py --download

  Build index (run once after download):
    python rag/ikea_search.py --build

  Test search:
    python rag/ikea_search.py --query "scandinavian oak bed frame"
    python rag/ikea_search.py --query "minimalist pendant lamp"
"""

import json
import csv
import argparse
import time
from pathlib import Path

IKEA_INDEX_PATH = Path("data/ikea_index.faiss")
IKEA_META_PATH  = Path("data/ikea_meta.json")
IKEA_CSV_PATH   = Path("data/ikea_products.csv")

USD_TO_EUR = 0.925   # approximate conversion — update if needed


# ── model loading ─────────────────────────────────────────────────────────────

def load_embedder():
    """Load sentence-transformers model on CPU (avoids MPS memory conflicts after SDXL)."""
    from sentence_transformers import SentenceTransformer
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"sentence-transformers: using {device}")
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


# ── download ──────────────────────────────────────────────────────────────────

def download_ikea_dataset():
    """
    Download jeffreyszhou/ikea-us-products-2025 from HuggingFace.
    30,500 real IKEA products, MIT licence, scraped July 2025.
    Saves to data/ikea_products.csv
    """
    print("Downloading IKEA US products 2025 from HuggingFace...")
    print("Dataset: jeffreyszhou/ikea-us-products-2025 (30.5k products, MIT)")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Run: pip install datasets")
        return

    try:
        ds = load_dataset("jeffreyszhou/ikea-us-products-2025", split="train")
        print(f"Downloaded {len(ds)} products")
        print(f"Columns: {ds.column_names}")

        IKEA_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in ds:
            # extract category from category_tree list
            cat_tree = item.get("category_tree", [])
            category = cat_tree[-1] if cat_tree else "furniture"
            top_category = cat_tree[0] if cat_tree else "furniture"

            # convert USD to EUR, rewrite URL to NL
            price_str = item.get("price", "")
            try:
                price_usd = float(price_str.replace("$", "").replace(",", ""))
                price_eur = f"€{round(price_usd * USD_TO_EUR, 2):.2f}"
            except Exception:
                price_eur = price_str

            url_us = item.get("source_url", "")
            # NL product IDs differ from US — use search URL instead of direct rewrite
            product_name_encoded = item.get("title", "").split(",")[0].strip()
            import urllib.parse
            url_nl = f"https://www.ikea.com/nl/en/search/?q={urllib.parse.quote(product_name_encoded)}"

            # materials as string
            materials = item.get("materials", [])
            materials_str = ", ".join(materials[:5]) if materials else ""

            rows.append({
                "name":        item.get("title", ""),
                "description": item.get("description", ""),
                "category":    category,
                "top_category": top_category,
                "price_eur":   price_eur,
                "price_usd":   price_str,
                "url_nl":      url_nl,
                "url_us":      url_us,
                "materials":   materials_str,
            })

        with open(IKEA_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved {len(rows)} products to {IKEA_CSV_PATH}")
        print(f"Next: python rag/ikea_search.py --build")

    except Exception as e:
        print(f"Download failed: {e}")


# ── index building ─────────────────────────────────────────────────────────────

def build_ikea_index(csv_path: str = str(IKEA_CSV_PATH)):
    """
    Embed IKEA products and build FAISS index.
    Embeds: name + description + category + materials
    """
    import faiss
    import numpy as np

    if not Path(csv_path).exists():
        print(f"CSV not found: {csv_path}")
        print("Run: python rag/ikea_search.py --download")
        return

    embedder = load_embedder()
    device = str(embedder.device)
    products = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            products.append(dict(row))

    print(f"Embedding {len(products)} products on {device}...")

    texts = [
        f"{p.get('name','')} {p.get('description','')} "
        f"{p.get('category','')} {p.get('materials','')}"
        for p in products
    ]

    batch_size = 512 if "cuda" in device else 256
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=batch_size,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    IKEA_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(IKEA_INDEX_PATH))
    IKEA_META_PATH.write_text(json.dumps(products, indent=2, ensure_ascii=False))

    print(f"\nIndex saved: {IKEA_INDEX_PATH}")
    print(f"Products indexed: {len(products)}")
    print(f"Test: python rag/ikea_search.py --query 'scandinavian bed oak'")


# ── search ────────────────────────────────────────────────────────────────────

def search_ikea_products(
    style: str,
    furniture_list: list[dict],
    budget: str = "500-2000",
    top_k_per_item: int = 2,
) -> list[dict]:
    """
    Search IKEA catalogue for products matching room style and furniture needs.

    Builds one query per furniture item: e.g. "scandinavian sofa"
    plus one for lighting. Returns deduplicated results across all queries.

    Falls back to empty list with instructions if index not built.
    """
    if not IKEA_INDEX_PATH.exists():
        print("[rag] IKEA index not built.")
        print("Run: python rag/ikea_search.py --download")
        print("Then: python rag/ikea_search.py --build")
        return []

    import faiss

    embedder = load_embedder()
    meta  = json.loads(IKEA_META_PATH.read_text())
    index = faiss.read_index(str(IKEA_INDEX_PATH))

    # parse budget string "500-2000" → upper limit in EUR
    try:
        budget_max = float(budget.split("-")[-1])
    except Exception:
        budget_max = 99999

    results     = []
    seen_names  = set()   # full name dedup
    seen_series = set()   # IKEA series dedup (e.g. "TVARÖ") — prevents color variants

    def _series(name: str) -> str:
        """Return the IKEA product series name (first all-caps word, e.g. 'TVARÖ')."""
        first = name.split()[0] if name else ""
        return first.upper()

    # Human-readable reason labels per query type — shown in the UI
    QUERY_REASONS = {
        "bed":       "bed / bedroom furniture",
        "sofa":      "seating",
        "chair":     "seating",
        "shelf":     "storage",
        "table":     "tables",
        "wardrobe":  "storage",
        "lamp":      "lighting",
        "pendant":   "lighting",
        "rug":       "floor covering",
        "desk":      "workspace",
    }

    def _reason(query: str) -> str:
        q = query.lower()
        for key, label in QUERY_REASONS.items():
            if key in q:
                return label
        return "room styling"

    # Build queries: one per furniture item, each with a category hint.
    # We separate style from furniture type so sentence-transformers doesn't
    # overweight the style token at the expense of the furniture category.
    # e.g. for bed:     "bed frame oak wood" (not "scandinavian bed")
    #      for lighting: "pendant lamp ceiling warm light"
    #
    # We then use category_filter to only accept results whose top_category
    # matches the intended furniture type — prevents chairs showing up for beds.

    CATEGORY_HINTS = {
        "bed":      ("bed frame mattress", ["Beds", "Bed frames", "Bed"]),
        "sofa":     ("sofa couch seating", ["Sofas", "Sectional sofas", "Sofa"]),
        "chair":    ("chair armchair seat", ["Chairs", "Armchairs", "Chair"]),
        "shelf":    ("shelf bookcase storage", ["Shelving units", "Bookcases", "Storage"]),
        "table":    ("coffee table side table", ["Tables", "Coffee tables", "Side tables"]),
        "wardrobe": ("wardrobe closet storage", ["Wardrobes", "PAX wardrobes"]),
        "lamp":     ("lamp light pendant", ["Lighting", "Lamps", "Pendant lamps"]),
        "rug":      ("rug carpet floor textile", ["Rugs", "Carpets"]),
        "desk":     ("desk workspace writing table", ["Desks", "Work desks"]),
    }

    # build (query_text, allowed_categories) pairs
    query_pairs = []
    for item in furniture_list[:4]:
        fname = item.get("name", "").lower()
        # match to category hint
        hint_text, allowed_cats = None, None
        for key, (hint, cats) in CATEGORY_HINTS.items():
            if key in fname:
                hint_text, allowed_cats = hint, cats
                break
        if hint_text:
            query_pairs.append((hint_text, allowed_cats))
        else:
            query_pairs.append((f"{fname} furniture", None))

    # always add lighting and rug
    query_pairs.append(("pendant lamp ceiling light warm", ["Lighting", "Lamps", "Pendant lamps"]))
    query_pairs.append(("rug carpet floor woven", ["Rugs", "Carpets", "Textiles"]))

    for query, allowed_cats in query_pairs:
        emb = embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # fetch more candidates so category filter doesn't leave us empty
        scores, indices = index.search(emb, top_k_per_item * 8)

        added = 0
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if added >= top_k_per_item:
                break

            product = meta[idx]
            name = product.get("name", "")
            if name in seen_names:
                continue

            # deduplicate by IKEA series name + category (blocks color variants)
            series_key = f"{_series(name)}|{product.get('top_category', '')}"
            if series_key in seen_series:
                continue

            # category filter — only accept matching furniture type
            if allowed_cats:
                top_cat = product.get("top_category", "")
                cat     = product.get("category", "")
                if not any(ac.lower() in top_cat.lower() or ac.lower() in cat.lower()
                           for ac in allowed_cats):
                    continue

            # budget filter
            price_eur_str = product.get("price_eur", "")
            try:
                price_val = float(price_eur_str.replace("€", "").replace(",", ""))
                if price_val > budget_max:
                    continue
            except Exception:
                pass

            # build working NL search URL from product name
            import urllib.parse
            product_base_name = name.split(",")[0].strip()
            url_nl = f"https://www.ikea.com/nl/en/search/?q={urllib.parse.quote(product_base_name)}"

            seen_names.add(name)
            seen_series.add(series_key)
            results.append({
                "name":        name,
                "description": product.get("description", ""),
                "category":    product.get("category", ""),
                "price":       price_eur_str,
                "price_usd":   product.get("price_usd", ""),
                "url":         url_nl,
                "url_us":      product.get("url_us", ""),
                "materials":   product.get("materials", ""),
                "similarity":  round(float(score), 4),
                "query":       query,
                "match_reason": _reason(query),
            })
            added += 1

    return results[:12]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IKEA semantic search — download, build index, query"
    )
    parser.add_argument("--download", action="store_true",
                        help="Download jeffreyszhou/ikea-us-products-2025 from HuggingFace")
    parser.add_argument("--build", action="store_true",
                        help=f"Build FAISS index from {IKEA_CSV_PATH}")
    parser.add_argument("--query", metavar="TEXT",
                        help="Test semantic search e.g. 'scandinavian oak bed'")
    parser.add_argument("--style", default="scandinavian",
                        help="Style prefix for query (default: scandinavian)")
    args = parser.parse_args()

    if args.download:
        download_ikea_dataset()
    elif args.build:
        build_ikea_index()
    elif args.query:
        t0 = time.time()
        results = search_ikea_products(
            style=args.style,
            furniture_list=[{"name": args.query}],
        )
        elapsed = round(time.time() - t0, 3)
        print(f"\nQuery: '{args.style} {args.query}' ({elapsed}s)\n")
        for r in results:
            print(f"  {r['name']}")
            print(f"  {r['description'][:80]}...")
            print(f"  Price: {r['price']}  Category: {r['category']}")
            print(f"  Materials: {r['materials'][:60]}")
            print(f"  Similarity: {r['similarity']}")
            print(f"  URL: {r['url']}\n")
    else:
        # show status
        if IKEA_INDEX_PATH.exists():
            meta = json.loads(IKEA_META_PATH.read_text())
            print(f"IKEA index ready — {len(meta)} products indexed")
            print(f"Test: python rag/ikea_search.py --query 'bed frame oak'")
        else:
            print("IKEA index not built yet.")
            print("  Step 1: python rag/ikea_search.py --download")
            print("  Step 2: python rag/ikea_search.py --build")
        print()
        parser.print_help()