"""
run_phase2.py
═════════════
Run Phase 2 end to end and save all artifacts.

Reads Phase 1 state, runs CLIP image retrieval + IKEA search,
saves results and generates an HTML report showing retrieved images.

Usage:
  python run_phase2.py
  python run_phase2.py --query "scandinavian bedroom minimal"
  python run_phase2.py --topk 8
"""

import argparse
import base64
import json
import time
import shutil
from datetime import datetime
from pathlib import Path

# Phase 1 state file
PHASE1_STATE = Path("data/outputs/phase1/06_state_dict.json")
OUT = Path("data/outputs/phase2")
OUT.mkdir(parents=True, exist_ok=True)


def log(step: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{step}] {msg}")


def save_json(data, filename: str):
    path = OUT / filename
    path.write_text(json.dumps(data, indent=2, default=str))
    log("save", f"{filename}")
    return path


def load_phase1_state() -> dict:
    if not PHASE1_STATE.exists():
        print(f"\nERROR: Phase 1 state not found at {PHASE1_STATE}")
        print("Run Phase 1 first: python run_phase1.py data/my_room.jpg")
        import sys; sys.exit(1)
    state = json.loads(PHASE1_STATE.read_text())
    log("load", f"Phase 1 state loaded — style: {state.get('style_preferences') or state.get('style_preference')}")
    return state


def build_clip_query(state: dict) -> str:
    """
    Build a rich text query for CLIP from the room analysis.
    Combines style preference + room analysis for better retrieval.
    """
    analysis = state.get("room_analysis", {})
    # Phase 1 state stores `style_preferences` (a list); tolerate the old
    # singular key and an empty list so the user's actual style reaches CLIP.
    prefs    = state.get("style_preferences") or state.get("style_preference")
    if isinstance(prefs, list):
        style = prefs[0] if prefs else "scandinavian"
    else:
        style = prefs or "scandinavian"
    mood     = analysis.get("color_mood", "neutral")
    room     = analysis.get("room_type", "bedroom")
    light    = analysis.get("natural_light", "moderate")

    query = f"{style} {room} {mood} tones {light} light interior design"
    log("query", f"CLIP query: '{query}'")
    return query


def step_image_retrieval(state: dict, top_k: int, trace: dict) -> list[dict]:
    log("step1", "Running CLIP image retrieval...")
    t0 = time.time()

    from rag.image_retriever import retrieve_design_references

    query = build_clip_query(state)
    results = retrieve_design_references(query, top_k=top_k)
    elapsed = round(time.time() - t0, 2)

    if results:
        log("step1", f"Retrieved {len(results)} images in {elapsed}s")
        log("step1", f"Top similarity: {results[0]['similarity']}")
    else:
        log("step1", "No results — index not built yet. Using empty list.")

    save_json(results, "01_reference_images.json")

    trace["steps"]["clip_retrieval"] = {
        "status": "success" if results else "no_index",
        "duration_s": elapsed,
        "query": query,
        "results_count": len(results),
        "top_similarity": results[0]["similarity"] if results else None,
    }
    return results


def step_ikea_search(state: dict, trace: dict) -> list[dict]:
    log("step2", "Running IKEA semantic search...")
    t0 = time.time()

    from rag.ikea_search import search_ikea_products

    analysis = state.get("room_analysis", {})
    results = search_ikea_products(
        style=state.get("style_preference", "scandinavian"),
        furniture_list=analysis.get("furniture", []),
        budget=state.get("budget", "500-2000"),
    )
    elapsed = round(time.time() - t0, 2)

    log("step2", f"Found {len(results)} products in {elapsed}s")
    save_json(results, "02_ikea_products.json")

    trace["steps"]["ikea_search"] = {
        "status": "success",
        "duration_s": elapsed,
        "results_count": len(results),
        "using_mock": not Path("data/ikea_index.faiss").exists(),
    }
    return results


def step_update_state(state: dict, ref_images: list, products: list) -> dict:
    """Merge Phase 2 results into state dict."""
    state["reference_images"] = ref_images
    state["ikea_products"] = products
    state["_phase_complete"]["phase_2"] = True
    save_json(state, "03_state_dict.json")
    log("state", "State dict updated with Phase 2 results")
    return state


def step_html_report(state: dict, ref_images: list, products: list, trace: dict):
    """Generate HTML report showing retrieved images and products."""
    log("report", "Generating HTML report...")

    # load original room photo for context
    orig_path = Path("data/outputs/phase1/01_original_room.jpg")
    orig_b64 = ""
    if orig_path.exists():
        orig_b64 = base64.b64encode(orig_path.read_bytes()).decode()

    # build reference image thumbnails
    ref_cards = ""
    for r in ref_images[:6]:
        p = Path(r["path"])
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            ext = p.suffix.lower().replace(".", "")
            ref_cards += f"""
            <div class="img-card">
              <img src="data:image/{ext};base64,{b64}" alt="reference">
              <div class="img-label">sim: {r['similarity']} · {r['filename'][:20]}</div>
            </div>"""
        else:
            ref_cards += f"""
            <div class="img-card no-img">
              <div style="padding:20px;color:#888;font-size:11px">{r['filename']}</div>
              <div class="img-label">sim: {r['similarity']}</div>
            </div>"""

    # build product rows
    product_rows = ""
    for p in products:
        product_rows += f"""
        <tr>
          <td><strong>{p.get('name','')}</strong></td>
          <td>{p.get('category','')}</td>
          <td>{p.get('description','')[:60]}...</td>
          <td><strong>{p.get('price','')}</strong></td>
          <td><a href="{p.get('url','#')}" target="_blank">view ↗</a></td>
          <td style="color:#888">{p.get('similarity','—')}</td>
        </tr>"""

    step_rows = ""
    for name, d in trace["steps"].items():
        status_color = "green" if d["status"] == "success" else "orange"
        step_rows += f"""
        <tr>
          <td>{name}</td>
          <td style="color:{status_color}">{d['status']}</td>
          <td>{d.get('duration_s','—')}s</td>
          <td>{d.get('results_count','—')}</td>
        </tr>"""

    analysis = state.get("room_analysis", {})
    clip_built = Path("data/clip_index.faiss").exists()
    ikea_built = Path("data/ikea_index.faiss").exists()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HomeVision — Phase 2 Report</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#F8F6F3;color:#2C2A27;margin:0;padding:24px}}
  h1{{font-size:22px;font-weight:600;margin-bottom:4px}}
  .sub{{color:#888;font-size:13px;margin-bottom:32px}}
  .section{{background:white;border-radius:12px;padding:20px 24px;
            margin-bottom:20px;border:0.5px solid #E4E0D8}}
  .section h2{{font-size:14px;font-weight:600;color:#534AB7;
               letter-spacing:.05em;text-transform:uppercase;margin:0 0 14px}}
  .img-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
  .img-card{{border-radius:8px;overflow:hidden;border:0.5px solid #E4E0D8}}
  .img-card img{{width:100%;display:block;height:120px;object-fit:cover}}
  .img-card.no-img{{background:#F4F2EF;height:150px;display:flex;
                    flex-direction:column;align-items:center;justify-content:center}}
  .img-label{{font-size:10px;font-weight:500;color:#888;
              padding:5px 8px;background:#F8F6F3;word-break:break-all}}
  .orig-img{{width:200px;border-radius:8px;border:0.5px solid #E4E0D8;float:right;margin-left:16px}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{font-size:11px;font-weight:600;color:#888;text-align:left;
      padding:5px 8px;border-bottom:1px solid #E4E0D8;text-transform:uppercase}}
  td{{padding:7px 8px;border-bottom:0.5px solid #F0EDE8;color:#555}}
  td a{{color:#534AB7;text-decoration:none}}
  .pill{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:99px;
         font-weight:500;margin:2px}}
  .done{{background:#E1F5EE;color:#085041}}
  .todo{{background:#F4F2EF;color:#888}}
  .warn{{background:#FAEEDA;color:#633806}}
  .stat-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}}
  .stat{{background:#F8F6F3;border-radius:8px;padding:10px 16px;border:0.5px solid #E4E0D8}}
  .stat-val{{font-size:20px;font-weight:600;color:#534AB7}}
  .stat-label{{font-size:11px;color:#888}}
  pre{{background:#F4F2EF;border-radius:8px;padding:12px;font-size:11px;
       overflow-x:auto;color:#3C3A36;line-height:1.6;white-space:pre-wrap}}
</style>
</head>
<body>
<h1>HomeVision — Phase 2 Report</h1>
<div class="sub">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

<div class="section">
  <h2>Phase status</h2>
  <span class="pill done">Phase 1 ✓</span>
  <span class="pill done">Phase 2 ✓</span>
  <span class="pill todo">Phase 3 — generation (next)</span>
  <span class="pill todo">Phase 4 — guardrails</span>
  <span class="pill todo">Phase 5 — eval</span>
  <span class="pill todo">Phase 6 — API</span>
  <br><br>
  <span class="pill {'done' if clip_built else 'warn'}">CLIP index {'✓ built' if clip_built else '⚠ not built — mock used'}</span>
  <span class="pill {'done' if ikea_built else 'warn'}">IKEA index {'✓ built' if ikea_built else '⚠ not built — mock used'}</span>
</div>

<div class="section">
  <h2>Context — your room (from Phase 1)</h2>
  {'<img class="orig-img" src="data:image/jpeg;base64,' + orig_b64 + '" alt="your room">' if orig_b64 else ''}
  <div class="stat-row">
    <div class="stat"><div class="stat-val">{analysis.get('style','?')}</div><div class="stat-label">current style</div></div>
    <div class="stat"><div class="stat-val">{state.get('style_preference','?')}</div><div class="stat-label">target style</div></div>
    <div class="stat"><div class="stat-val">{len(ref_images)}</div><div class="stat-label">references found</div></div>
    <div class="stat"><div class="stat-val">{len(products)}</div><div class="stat-label">products found</div></div>
  </div>
  <div style="clear:both"></div>
</div>

<div class="section">
  <h2>Reference images — CLIP retrieval results</h2>
  {'<p style="color:#BA7517;font-size:12px">⚠ CLIP index not built yet — no images retrieved.<br>Run: python rag/image_retriever.py --download<br>Then: python rag/image_retriever.py --build data/interior_images/</p>' if not ref_images else ''}
  <div class="img-grid">{ref_cards}</div>
</div>

<div class="section">
  <h2>IKEA product recommendations</h2>
  {'<p style="color:#BA7517;font-size:12px">⚠ Using mock products — IKEA index not built yet.<br>Run: python rag/ikea_search.py --synthetic<br>Then: python rag/ikea_search.py --build data/ikea_nl.csv</p>' if not Path("data/ikea_index.faiss").exists() else ''}
  <table>
    <thead><tr><th>product</th><th>category</th><th>description</th><th>price</th><th>link</th><th>score</th></tr></thead>
    <tbody>{product_rows}</tbody>
  </table>
</div>

<div class="section">
  <h2>Trace — step timing</h2>
  <table>
    <thead><tr><th>step</th><th>status</th><th>duration</th><th>results</th></tr></thead>
    <tbody>{step_rows}</tbody>
  </table>
</div>

<div class="section">
  <h2>State dict — ready for Phase 3</h2>
  <p style="font-size:12px;color:#888">reference_images and ikea_products are now populated. Phase 3 reads these to drive generation.</p>
  <pre>{json.dumps({"reference_images_count": len(ref_images), "ikea_products_count": len(products), "top_reference": ref_images[0] if ref_images else None, "top_product": products[0] if products else None}, indent=2)}</pre>
</div>

</body></html>"""

    report_path = OUT / "report.html"
    report_path.write_text(html)
    log("report", f"Saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="HomeVision Phase 2 runner")
    parser.add_argument("--topk", type=int, default=6, help="Number of reference images to retrieve")
    parser.add_argument("--style", default=None, help="Override style preference")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  HomeVision — Phase 2 (RAG)")
    print("═" * 60 + "\n")

    trace = {
        "started_at": datetime.now().isoformat(),
        "steps": {},
    }

    state = load_phase1_state()
    if args.style:
        state["style_preference"] = args.style

    ref_images = step_image_retrieval(state, args.topk, trace)
    products   = step_ikea_search(state, trace)
    state      = step_update_state(state, ref_images, products)
    report     = step_html_report(state, ref_images, products, trace)

    print("\n" + "═" * 60)
    print("  PHASE 2 COMPLETE")
    print("═" * 60)
    print(f"  Reference images: {len(ref_images)}")
    print(f"  IKEA products:    {len(products)}")
    print(f"  Report:           open {report}")
    print(f"\n  If indexes not built yet, run:")
    print(f"    python rag/image_retriever.py --download")
    print(f"    python rag/image_retriever.py --build data/interior_images/")
    print(f"    python rag/ikea_search.py --synthetic")
    print(f"    python rag/ikea_search.py --build data/ikea_nl.csv")
    print(f"  Then re-run: python run_phase2.py")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()