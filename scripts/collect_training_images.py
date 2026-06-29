"""
scripts/collect_training_images.py
════════════════════════════════════
Interactive image collector for LoRA training data.

1. Searches Bing Images for each query phrase
2. Opens a Gradio UI — click images to select/reject (green = keep, red = discard)
3. Saves approved images to the output folder, ready for lora/trainer.py

Usage:
  python scripts/collect_training_images.py \
    --style  "Indian Vintage" \
    --output data/lora/indian_vintage/ \
    --count  30

Then add your own queries in the UI, or use the defaults for the style.

After collecting:
  python lora/trainer.py \
    --images  data/lora/indian_vintage/ \
    --style   "Indian Vintage" \
    --trigger IVI \
    --output  lora/adapters/indian_vintage.safetensors
"""

import argparse
import os
import shutil
import tempfile
import threading
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── style defaults ────────────────────────────────────────────────────────────

STYLE_QUERIES: dict[str, list[str]] = {
    "indian vintage": [
        # South Indian — Chettinad
        "chettinad mansion interior athangudi tiles",
        "athangudi tiles chettinad floor courtyard",
        "chettinad palace interior burma teak pillars",
        "chettinad nalukettu inner courtyard",
        "chettinad heritage home belgian glass windows",
        "chettinad antique furniture carved wood interior",
        # South Indian — Kerala / general
        "kerala nalukettu traditional interior courtyard",
        "kerala traditional wooden palace interior",
        "south indian heritage home teak carved pillars",
        # North Indian — Rajasthani / Mughal
        "rajasthani haveli interior vintage jali screen",
        "indian haveli inner courtyard brass lamp",
        "mughal style room carved rosewood arched niche",
        "rajasthani heritage hotel interior design",
        "block print textile indian vintage living room",
        # Pan-Indian
        "indian heritage home inner courtyard mosaic floor",
        "antique brass urli bowl indian interior decor",
        "india colonial bungalow interior vintage",
    ],
    "indian contemporary": [
        "indian contemporary interior design",
        "modern indian living room teak",
        "india contemporary apartment design",
        "indian craft accent modern home",
    ],
}


def _default_queries(style: str) -> list[str]:
    key = style.lower().strip()
    for k, v in STYLE_QUERIES.items():
        if k in key or key in k:
            return v
    return [f"{style} interior design", f"{style} room decor"]


# ── download ──────────────────────────────────────────────────────────────────

def _download_query(query: str, out_dir: Path, count: int) -> list[Path]:
    """Download images for one search query using DuckDuckGo image search."""
    import requests
    from ddgs import DDGS

    slug = query.replace(" ", "_")[:40]
    query_dir = out_dir / slug
    query_dir.mkdir(parents=True, exist_ok=True)

    existing = list(query_dir.glob("*.jpg")) + list(query_dir.glob("*.png"))
    if len(existing) >= count:
        print(f"  [cache] {query!r} — {len(existing)} already downloaded")
        return existing

    print(f"  Searching: {query!r} ({count} images)...")

    try:
        results = DDGS().images(
            query,
            region="wt-wt",
            safesearch="moderate",
            size="Large",        # filter out thumbnails / icons
            type_image="photo",  # no illustrations, clip art, or line art
            max_results=count * 2,  # fetch extra — some URLs will fail
        )
    except Exception as e:
        print(f"  Search failed: {e}")
        return existing

    saved = 0
    headers = {"User-Agent": "Mozilla/5.0"}
    for i, r in enumerate(results):
        if saved >= count:
            break
        url = r.get("image", "")
        if not url:
            continue
        ext = ".jpg" if "jpg" in url.lower() or "jpeg" in url.lower() else ".png"
        dest = query_dir / f"{i:04d}{ext}"
        if dest.exists():
            saved += 1
            continue
        try:
            resp = requests.get(url, timeout=8, headers=headers, stream=True)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                dest.write_bytes(resp.content)
                saved += 1
        except Exception:
            pass  # skip broken URLs silently

    results = list(query_dir.glob("*.jpg")) + list(query_dir.glob("*.png"))
    print(f"  → {len(results)} downloaded")
    return results


def download_all(queries: list[str], tmp_dir: Path, per_query: int) -> list[Path]:
    all_images: list[Path] = []
    for q in queries:
        imgs = _download_query(q, tmp_dir, per_query)
        all_images.extend(imgs)
    return sorted(set(all_images))


# ── Gradio review UI ──────────────────────────────────────────────────────────

def build_review_ui(
    images: list[Path],
    output_dir: Path,
    style: str,
    tmp_dir: Path,
    per_query: int,
    all_queries: list[str],
):
    import gradio as gr

    selected: set[str] = set()        # paths currently marked "keep"
    all_images: list[Path] = list(images)

    output_dir.mkdir(parents=True, exist_ok=True)

    def _thumb(p: Path) -> str:
        return str(p)

    def _gallery_data(img_list: list[Path]) -> list[tuple[str, str]]:
        return [(_thumb(p), p.name) for p in img_list]

    # ── event handlers ────────────────────────────────────────────────────────

    def toggle_select(evt: gr.SelectData):
        """Toggle an image between selected/deselected."""
        path = str(all_images[evt.index])
        if path in selected:
            selected.discard(path)
        else:
            selected.add(path)

        gallery_items = _gallery_data(all_images)
        count_msg = f"**{len(selected)} selected** out of {len(all_images)}"
        return gallery_items, count_msg

    def save_selected():
        if not selected:
            return "Nothing selected — click images to mark them green first."

        saved = 0
        for src in selected:
            dst = output_dir / Path(src).name
            if not dst.exists():
                shutil.copy2(src, dst)
                saved += 1

        return (
            f"Saved {saved} images to `{output_dir}`.\n\n"
            f"Total in folder: {len(list(output_dir.glob('*.*')))} images.\n\n"
            f"Ready to train:\n"
            f"```\npython lora/trainer.py \\\n"
            f"  --images {output_dir} \\\n"
            f"  --style \"{style}\" \\\n"
            f"  --trigger IVI \\\n"
            f"  --output lora/adapters/{style.lower().replace(' ', '_')}.safetensors\n```"
        )

    def add_query(new_query: str, count: int):
        """Search for additional images and add to the gallery."""
        if not new_query.strip():
            return _gallery_data(all_images), f"{len(all_images)} images total", ""

        new_imgs = _download_query(new_query.strip(), tmp_dir, int(count))
        for p in new_imgs:
            if p not in all_images:
                all_images.append(p)

        return (
            _gallery_data(all_images),
            f"**{len(selected)} selected** out of {len(all_images)}",
            "",
        )

    def select_all():
        for p in all_images:
            selected.add(str(p))
        return _gallery_data(all_images), f"**{len(selected)} selected** out of {len(all_images)}"

    def deselect_all():
        selected.clear()
        return _gallery_data(all_images), f"**0 selected** out of {len(all_images)}"

    # ── layout ────────────────────────────────────────────────────────────────

    with gr.Blocks(title=f"HomeVision — {style} image collector") as app:
        gr.Markdown(f"## {style} — training image collector")
        gr.Markdown(
            "Click images to **select** (green border = keep). "
            "Click again to deselect. Save when done."
        )

        count_label = gr.Markdown(
            f"**0 selected** out of {len(all_images)}"
        )

        gallery = gr.Gallery(
            value=_gallery_data(all_images),
            label="Images — click to select",
            columns=5,
            height=600,
            allow_preview=True,
            show_label=True,
            selected_index=None,
        )

        with gr.Row():
            select_all_btn  = gr.Button("Select all",   variant="secondary", scale=1)
            deselect_all_btn = gr.Button("Deselect all", variant="secondary", scale=1)
            save_btn = gr.Button("Save selected to training folder", variant="primary", scale=2)

        result_box = gr.Markdown("")

        gr.Markdown("---\n### Add more search queries")
        with gr.Row():
            query_input = gr.Textbox(
                placeholder='e.g. "indian haveli courtyard brass lamp"',
                label="Search query",
                scale=3,
            )
            query_count = gr.Slider(5, 50, value=20, step=5, label="Images to fetch", scale=1)
            search_btn  = gr.Button("Search + add", variant="secondary", scale=1)

        gr.Markdown(
            "**Default queries used for this style:**\n" +
            "\n".join(f"- {q}" for q in all_queries)
        )

        # wiring
        gallery.select(toggle_select, outputs=[gallery, count_label])
        save_btn.click(save_selected, outputs=[result_box])
        select_all_btn.click(select_all, outputs=[gallery, count_label])
        deselect_all_btn.click(deselect_all, outputs=[gallery, count_label])
        search_btn.click(
            add_query,
            inputs=[query_input, query_count],
            outputs=[gallery, count_label, query_input],
        )

    return app


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect LoRA training images interactively")
    parser.add_argument("--style",  default="Indian Vintage",
                        help="Style name (used for default queries and UI label)")
    parser.add_argument("--output", default=None,
                        help="Folder to save approved images (default: data/lora/<style>/)")
    parser.add_argument("--count",  type=int, default=20,
                        help="Images to fetch per search query (default 20)")
    parser.add_argument("--port",   type=int, default=7861,
                        help="Gradio port (default 7861 — avoids conflict with main UI)")
    args = parser.parse_args()

    style  = args.style
    output = Path(args.output) if args.output else Path(
        "data/lora/" + style.lower().replace(" ", "_") + "/"
    )

    queries  = _default_queries(style)
    tmp_dir  = Path(tempfile.mkdtemp(prefix="homevision_lora_"))

    print(f"\nHomeVision — {style} image collector")
    print(f"Output folder : {output}")
    print(f"Temp cache    : {tmp_dir}")
    print(f"Queries       : {len(queries)} default\n")

    print("Downloading images (this may take a minute)...")
    images = download_all(queries, tmp_dir, args.count)
    print(f"\n{len(images)} images ready for review\n")

    app = build_review_ui(images, output, style, tmp_dir, args.count, queries)
    app.launch(server_port=args.port, share=False)
