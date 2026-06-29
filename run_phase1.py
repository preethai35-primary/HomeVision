"""
run_phase1.py
═════════════
Run Phase 1 end to end and save EVERY intermediate artifact.

After running this you will have:
  data/outputs/phase1/
    ├── 01_original_room.jpg         your room photo (copy)
    ├── 02_depth_map.png             depth map (bright=close, dark=far)
    ├── 03_depth_overlay.png         depth map blended over original
    ├── 04_room_analysis.json        raw GPT-4o output
    ├── 05_room_analysis_pretty.txt  human-readable analysis
    ├── 06_state_dict.json           full LangGraph state at end of phase 1
    ├── 07_langfuse_trace.json       what LangFuse would record (local copy)
    ├── 08_color_palette.png         extracted dominant colours as swatches
    └── report.html                  full visual report — open in browser

Usage:
  python run_phase1.py data/my_room.jpg
  python run_phase1.py data/my_room.jpg --style japandi --budget 500-2000
"""

import argparse
import base64
import json
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path

# ── output folder ─────────────────────────────────────────────────────────────
OUT = Path("data/outputs/phase1")
OUT.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def log(step: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{step}] {msg}")


def save_json(data: dict, filename: str) -> Path:
    path = OUT / filename
    path.write_text(json.dumps(data, indent=2, default=str))
    log("save", f"{filename} ({path.stat().st_size} bytes)")
    return path


def save_text(text: str, filename: str) -> Path:
    path = OUT / filename
    path.write_text(text)
    log("save", f"{filename}")
    return path


# ── step 1: copy original image ───────────────────────────────────────────────

def step_copy_original(image_path: str) -> Path:
    log("step1", "Copying original room photo")
    src = Path(image_path)
    if not src.exists():
        print(f"\nERROR: {image_path} not found.")
        print("Fix: cp ~/your_photo.jpg data/my_room.jpg")
        sys.exit(1)
    dst = OUT / "01_original_room.jpg"
    shutil.copy(src, dst)
    log("step1", f"Saved: {dst}")
    return dst


# ── step 2: room analysis ─────────────────────────────────────────────────────

def step_room_analysis(image_path: str, trace: dict) -> dict:
    log("step2", "Sending room photo to GPT-4o vision...")
    t0 = time.time()

    from vision.room_analyzer import analyze_room
    analysis = analyze_room(image_path)

    elapsed = round(time.time() - t0, 2)
    log("step2", f"Done in {elapsed}s — {analysis['_usage']['input_tokens']} input tokens, "
                 f"{analysis['_usage']['output_tokens']} output tokens, "
                 f"${analysis['_usage']['estimated_cost_usd']}")

    # save raw JSON
    save_json(analysis, "04_room_analysis.json")

    # save human-readable version
    pretty = format_analysis_text(analysis)
    save_text(pretty, "05_room_analysis_pretty.txt")

    # record in trace
    trace["steps"]["analyze_room"] = {
        "status": "success",
        "duration_s": elapsed,
        "input": {"image_path": image_path},
        "output": {k: v for k, v in analysis.items() if k != "_usage"},
        "usage": analysis["_usage"],
    }

    return analysis


def format_analysis_text(a: dict) -> str:
    lines = [
        "═" * 50,
        "ROOM ANALYSIS — GPT-4o output",
        "═" * 50,
        f"Style detected:    {a.get('style', 'unknown')}",
        f"Room type:         {a.get('room_type', 'unknown')}",
        f"Size estimate:     {a.get('size_estimate', 'unknown')}",
        f"Natural light:     {a.get('natural_light', 'unknown')}",
        f"Color mood:        {a.get('color_mood', 'unknown')}",
        "",
        "Furniture:",
    ]
    for f in a.get("furniture", []):
        keep = "keep" if f.get("keep") else "replace"
        lines.append(f"  • {f['name']} ({f.get('condition', '?')}) → {keep}")
    lines += [
        "",
        f"Dominant colors:   {', '.join(a.get('dominant_colors', []))}",
        "",
        "Strengths:",
    ]
    for s in a.get("strengths", []):
        lines.append(f"  ✓ {s}")
    lines += ["", "Opportunities:"]
    for o in a.get("opportunities", []):
        lines.append(f"  → {o}")
    lines += [
        "",
        f"Suggested styles:  {', '.join(a.get('suggested_styles', []))}",
        "",
        "─" * 50,
        f"Tokens used:  {a['_usage']['input_tokens']} in / {a['_usage']['output_tokens']} out",
        f"Cost:         ${a['_usage']['estimated_cost_usd']}",
        f"Model:        {a['_usage']['model']}",
    ]
    return "\n".join(lines)


# ── step 3: depth map ─────────────────────────────────────────────────────────

def step_depth_map(image_path: str, trace: dict) -> dict:
    log("step3", "Running Depth-Anything v2 (first run downloads ~400MB)...")
    t0 = time.time()

    from vision.depth_estimator import estimate_depth
    result = estimate_depth(image_path, output_path=str(OUT / "02_depth_map.png"))

    elapsed = round(time.time() - t0, 2)
    log("step3", f"Done in {elapsed}s — depth map saved")

    # save depth overlay (depth map blended over original)
    _save_depth_overlay(image_path, result)

    trace["steps"]["depth_estimation"] = {
        "status": "success",
        "duration_s": elapsed,
        "model": "depth-anything/Depth-Anything-V2-Small-hf",
        "input": {"image_path": image_path},
        "output": {
            "depth_map_path": str(OUT / "02_depth_map.png"),
            "stats": result["stats"],
            "original_size": result["original_size"],
        },
    }
    return result


def _save_depth_overlay(original_path: str, depth_result: dict):
    """Blend depth map over original image at 50% opacity for visual inspection."""
    from PIL import Image
    import numpy as np

    orig = Image.open(original_path).convert("RGB")
    w, h = orig.size

    # depth_image is already sized to match original
    depth_rgb = depth_result["depth_image"].convert("RGB")

    # blend: 50% original, 50% depth (colourised)
    orig_arr = np.array(orig).astype(float)
    depth_arr = np.array(depth_rgb).astype(float)

    # apply a colourmap to make depth more readable
    # bright areas (close) → warm orange, dark areas (far) → cool blue
    d_norm = np.array(depth_result["depth_image"]).astype(float) / 255.0
    r = (d_norm * 255).clip(0, 255)
    g = ((1 - abs(d_norm - 0.5) * 2) * 200).clip(0, 255)
    b = ((1 - d_norm) * 255).clip(0, 255)
    coloured = np.stack([r, g, b], axis=2)

    overlay = (orig_arr * 0.5 + coloured * 0.5).clip(0, 255).astype("uint8")
    overlay_img = Image.fromarray(overlay)
    overlay_img.save(OUT / "03_depth_overlay.png")
    log("step3", f"Depth overlay saved: 03_depth_overlay.png")




# ── step 4: colour palette ────────────────────────────────────────────────────

def step_color_palette(analysis: dict, trace: dict):
    """Render the detected dominant colours as a swatch PNG."""
    log("step4", "Rendering colour palette")
    from PIL import Image, ImageDraw, ImageFont

    colors = analysis.get("dominant_colors", ["#888888"])
    swatch_w, swatch_h = 120, 80
    padding = 10
    total_w = len(colors) * (swatch_w + padding) + padding
    total_h = swatch_h + 60

    img = Image.new("RGB", (total_w, total_h), (245, 243, 240))
    draw = ImageDraw.Draw(img)

    for i, hex_color in enumerate(colors):
        x = padding + i * (swatch_w + padding)
        try:
            # convert hex to RGB
            h = hex_color.lstrip("#")
            rgb = tuple(int(h[j:j+2], 16) for j in (0, 2, 4))
        except Exception:
            rgb = (180, 180, 180)

        draw.rectangle([x, padding, x + swatch_w, padding + swatch_h], fill=rgb)
        draw.text((x, padding + swatch_h + 8), hex_color,
                  fill=(80, 80, 80))

    img.save(OUT / "08_color_palette.png")
    log("step4", "Colour palette saved: 08_color_palette.png")

    trace["steps"]["color_palette"] = {
        "status": "success",
        "colors": colors,
    }


# ── step 5: build state dict ──────────────────────────────────────────────────

def step_build_state(image_path: str, analysis: dict, depth_result: dict,
                     style: str, budget: str, trace: dict) -> dict:
    """
    Build the LangGraph state dict as it would look after Phase 1 completes.
    This is exactly what gets passed to Phase 2 tools (retrieve_refs, etc.)
    """
    log("step5", "Building LangGraph state dict")

    state = {
        # ── inputs ────────────────────────────────────
        "image_path":      image_path,
        "style_preferences": [style],
        "budget":          budget,
        "focus_areas":     ["furniture", "lighting", "color palette"],
        "user_message":    "",

        # ── vision outputs (populated by Phase 1) ─────
        "room_analysis":   {k: v for k, v in analysis.items() if k != "_usage"},
        "depth_map_path":  str(OUT / "02_depth_map.png"),

        # ── rag outputs (Phase 2 — empty for now) ─────
        "reference_images": None,
        "ikea_products":   None,

        # ── generation (Phase 3 — empty for now) ──────
        "prompts_used":    None,
        "generated_images": None,
        "active_variant":  None,
        "lora_paths":      None,

        # ── evaluation (Phase 3D — empty for now) ─────
        "eval_scores":     None,

        # ── llm outputs (Phase 2+ — empty for now) ────
        "design_explanation":  None,
        "conversation_history": [],

        # ── guardrails (Phase 4 — empty for now) ──────
        "guardrail_flags": None,

        # ── meta ──────────────────────────────────────
        "error":    None,
        "trace_id": trace["trace_id"],

        # ── phase status ──────────────────────────────
        "_phase_complete": {
            "phase_1": True,
            "phase_2": False,
            "phase_3": False,
            "phase_4": False,
        },
    }

    save_json(state, "06_state_dict.json")
    log("step5", "State dict saved: 06_state_dict.json")
    return state


# ── step 6: LangFuse trace ────────────────────────────────────────────────────

def step_save_trace(trace: dict):
    """
    Save a local copy of what LangFuse records.
    If real LangFuse is configured (.env has keys), also flush to cloud.
    """
    log("step6", "Saving LangFuse trace")

    trace["completed_at"] = datetime.now().isoformat()
    trace["total_duration_s"] = round(
        sum(s.get("duration_s", 0) for s in trace["steps"].values()), 2
    )
    trace["total_cost_usd"] = round(
        sum(
            s.get("usage", {}).get("estimated_cost_usd", 0)
            for s in trace["steps"].values()
        ), 4
    )

    save_json(trace, "07_langfuse_trace.json")

    # try to flush to real LangFuse if keys are set
    try:
        from agent.tracing import langfuse
        lf_trace = langfuse.trace(
            name="homevision-phase1",
            input={"image_path": trace["image_path"]},
        )
        for step_name, step_data in trace["steps"].items():
            span = lf_trace.span(
                name=step_name,
                input=step_data.get("input", {}),
            )
            span.end(
                output=step_data.get("output", {}),
                level="DEFAULT" if step_data["status"] == "success" else "ERROR",
            )
        langfuse.flush()
        log("step6", "Flushed to LangFuse cloud (check cloud.langfuse.com)")
    except Exception as e:
        log("step6", f"LangFuse cloud not configured ({e}) — local trace only")


# ── step 7: HTML report ───────────────────────────────────────────────────────

def step_html_report(analysis: dict, state: dict, trace: dict, image_path: str):
    """Generate a self-contained HTML report showing all Phase 1 outputs."""
    log("step7", "Generating HTML report")

    def img_to_b64(path: Path) -> str:
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode()
        return ""

    orig_b64    = img_to_b64(OUT / "01_original_room.jpg")
    depth_b64   = img_to_b64(OUT / "02_depth_map.png")
    overlay_b64 = img_to_b64(OUT / "03_depth_overlay.png")
    palette_b64 = img_to_b64(OUT / "08_color_palette.png")
    furniture_rows = "".join(
        f"<tr><td>{f['name']}</td><td>{f.get('condition','?')}</td>"
        f"<td>{'✓ keep' if f.get('keep') else '→ replace'}</td></tr>"
        for f in analysis.get("furniture", [])
    )

    steps_rows = "".join(
        f"<tr><td>{name}</td>"
        f"<td style='color:{'green' if d['status']=='success' else 'red'}'>{d['status']}</td>"
        f"<td>{d.get('duration_s', '—')}s</td>"
        f"<td>${d.get('usage', {}).get('estimated_cost_usd', '—')}</td></tr>"
        for name, d in trace["steps"].items()
    )

    state_json = json.dumps(
        {k: v for k, v in state.items() if k != "room_analysis"},
        indent=2, default=str
    )
    analysis_json = json.dumps(
        {k: v for k, v in analysis.items() if k != "_usage"},
        indent=2, default=str
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HomeVision — Phase 1 Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #F8F6F3; color: #2C2A27; margin: 0; padding: 24px; }}
  h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; }}
  .sub {{ color: #888; font-size: 13px; margin-bottom: 32px; }}
  .section {{ background: white; border-radius: 12px; padding: 20px 24px;
              margin-bottom: 20px; border: 0.5px solid #E4E0D8; }}
  .section h2 {{ font-size: 14px; font-weight: 600; color: #534AB7;
                 letter-spacing: .05em; text-transform: uppercase; margin: 0 0 14px; }}
  .img-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
  .img-card {{ border-radius: 8px; overflow: hidden; border: 0.5px solid #E4E0D8; }}
  .img-card img {{ width: 100%; display: block; }}
  .img-label {{ font-size: 11px; font-weight: 500; color: #888;
                padding: 6px 10px; background: #F8F6F3; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ font-size: 11px; font-weight: 600; color: #888; text-align: left;
        padding: 5px 8px; border-bottom: 1px solid #E4E0D8;
        text-transform: uppercase; letter-spacing: .04em; }}
  td {{ padding: 7px 8px; border-bottom: 0.5px solid #F0EDE8; color: #555; }}
  .pill {{ display: inline-block; font-size: 11px; padding: 2px 8px;
           border-radius: 99px; background: #EEEDFE; color: #3C3489;
           font-weight: 500; margin: 2px; }}
  .pill-teal {{ background: #E1F5EE; color: #085041; }}
  .pill-amber {{ background: #FAEEDA; color: #633806; }}
  pre {{ background: #F4F2EF; border-radius: 8px; padding: 14px; font-size: 11px;
         overflow-x: auto; color: #3C3A36; line-height: 1.6;
         border: 0.5px solid #E4E0D8; white-space: pre-wrap; }}
  .stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }}
  .stat {{ background: #F8F6F3; border-radius: 8px; padding: 10px 16px;
           border: 0.5px solid #E4E0D8; }}
  .stat-val {{ font-size: 20px; font-weight: 600; color: #534AB7; }}
  .stat-label {{ font-size: 11px; color: #888; margin-top: 2px; }}
  .phase-badge {{ display: inline-block; font-size: 11px; padding: 2px 10px;
                  border-radius: 99px; font-weight: 500; margin-right: 4px; }}
  .done {{ background: #E1F5EE; color: #085041; }}
  .todo {{ background: #F4F2EF; color: #888; }}
</style>
</head>
<body>

<h1>HomeVision — Phase 1 Report</h1>
<div class="sub">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
  Total cost: ${trace.get('total_cost_usd', '?')} ·
  Total time: {trace.get('total_duration_s', '?')}s
</div>

<div class="section">
  <h2>Phase status</h2>
  <span class="phase-badge done">Phase 1 ✓ complete</span>
  <span class="phase-badge todo">Phase 2 — RAG (next)</span>
  <span class="phase-badge todo">Phase 3 — generation</span>
  <span class="phase-badge todo">Phase 4 — guardrails</span>
  <span class="phase-badge todo">Phase 5 — evaluation</span>
  <span class="phase-badge todo">Phase 6 — API</span>
</div>

<div class="section">
  <h2>Images</h2>
  <div class="img-grid">
    <div class="img-card">
      <img src="data:image/jpeg;base64,{orig_b64}" alt="original room">
      <div class="img-label">01 — original room photo</div>
    </div>
    <div class="img-card">
      <img src="data:image/png;base64,{depth_b64}" alt="depth map">
      <div class="img-label">02 — depth map (bright=close, dark=far)</div>
    </div>
    <div class="img-card">
      <img src="data:image/png;base64,{overlay_b64}" alt="depth overlay">
      <div class="img-label">03 — depth overlay on original</div>
    </div>
  </div>
</div>

<div class="section">
  <h2>Room analysis — GPT-4o output</h2>
  <div class="stat-row">
    <div class="stat"><div class="stat-val">{analysis.get('style','?')}</div><div class="stat-label">style detected</div></div>
    <div class="stat"><div class="stat-val">{analysis.get('room_type','?')}</div><div class="stat-label">room type</div></div>
    <div class="stat"><div class="stat-val">{analysis.get('natural_light','?')}</div><div class="stat-label">natural light</div></div>
    <div class="stat"><div class="stat-val">{analysis.get('color_mood','?')}</div><div class="stat-label">colour mood</div></div>
  </div>

  <div style="margin-bottom:12px">
    <div style="font-size:12px;font-weight:500;color:#888;margin-bottom:6px">SUGGESTED STYLES</div>
    {"".join(f'<span class="pill">{s}</span>' for s in analysis.get("suggested_styles", []))}
  </div>

  <div style="margin-bottom:12px">
    <div style="font-size:12px;font-weight:500;color:#888;margin-bottom:6px">STRENGTHS</div>
    {"".join(f'<span class="pill pill-teal">✓ {s}</span>' for s in analysis.get("strengths", []))}
  </div>

  <div style="margin-bottom:12px">
    <div style="font-size:12px;font-weight:500;color:#888;margin-bottom:6px">OPPORTUNITIES</div>
    {"".join(f'<span class="pill pill-amber">→ {s}</span>' for s in analysis.get("opportunities", []))}
  </div>

  <div style="font-size:12px;font-weight:500;color:#888;margin-bottom:6px">FURNITURE</div>
  <table>
    <thead><tr><th>item</th><th>condition</th><th>recommendation</th></tr></thead>
    <tbody>{furniture_rows}</tbody>
  </table>

  <div style="margin-top:14px;font-size:12px;font-weight:500;color:#888;margin-bottom:6px">COLOUR PALETTE</div>
  <img src="data:image/png;base64,{palette_b64}" style="border-radius:8px;border:.5px solid #E4E0D8" alt="colour palette">
</div>

<div class="section">
  <h2>LangGraph state dict — end of Phase 1</h2>
  <div style="font-size:12px;color:#888;margin-bottom:8px">This is the shared state object that Phase 2 (RAG tools) will read from. Keys prefixed with <code>_phase</code> show what's populated. <code>null</code> values will be filled by later phases.</div>
  <pre>{state_json}</pre>
</div>

<div class="section">
  <h2>LangFuse trace — what gets recorded</h2>
  <div style="font-size:12px;color:#888;margin-bottom:10px">Each row = one span in LangFuse. This is exactly what you see in the LangFuse dashboard under this trace.</div>
  <table>
    <thead><tr><th>step</th><th>status</th><th>duration</th><th>cost</th></tr></thead>
    <tbody>{steps_rows}</tbody>
  </table>
  <div style="font-size:12px;color:#888;margin-top:10px">Full trace JSON saved in <code>07_langfuse_trace.json</code></div>
</div>

<div class="section">
  <h2>Raw GPT-4o JSON output</h2>
  <pre>{analysis_json}</pre>
</div>

<div class="section">
  <h2>Files saved</h2>
  <table>
    <thead><tr><th>file</th><th>what it is</th></tr></thead>
    <tbody>
      <tr><td>01_original_room.jpg</td><td>copy of your room photo</td></tr>
      <tr><td>02_depth_map.png</td><td>Depth-Anything v2 output — used by ControlNet in Phase 3</td></tr>
      <tr><td>03_depth_overlay.png</td><td>depth blended over original — visual inspection aid</td></tr>
      <tr><td>04_room_analysis.json</td><td>raw GPT-4o structured output including _usage</td></tr>
      <tr><td>05_room_analysis_pretty.txt</td><td>human readable summary</td></tr>
      <tr><td>06_state_dict.json</td><td>full LangGraph state after Phase 1</td></tr>
      <tr><td>07_langfuse_trace.json</td><td>local copy of LangFuse trace data</td></tr>
      <tr><td>08_color_palette.png</td><td>detected dominant colours as swatches</td></tr>
      <tr><td>report.html</td><td>this report</td></tr>
    </tbody>
  </table>
</div>

</body>
</html>"""

    report_path = OUT / "report.html"
    report_path.write_text(html)
    log("step7", f"HTML report saved: {report_path}")
    log("step7", "Open in browser: open data/outputs/phase1/report.html")
    return report_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HomeVision Phase 1 runner")
    parser.add_argument("image", nargs="?", default="data/my_room.jpg",
                        help="Path to room photo (default: data/my_room.jpg)")
    parser.add_argument("--style", default="scandinavian",
                        help="Style preference (default: scandinavian)")
    parser.add_argument("--budget", default="500-2000",
                        help="Budget range (default: 500-2000)")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  HomeVision — Phase 1")
    print(f"  Image:  {args.image}")
    print(f"  Style:  {args.style}")
    print(f"  Budget: {args.budget}")
    print(f"  Output: {OUT}")
    print("═" * 60 + "\n")

    # initialise trace dict (local equivalent of LangFuse trace)
    trace = {
        "trace_id": f"hv-phase1-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "started_at": datetime.now().isoformat(),
        "image_path": args.image,
        "style_preference": args.style,
        "steps": {},
    }

    # ── run each step ─────────────────────────────────────────────────────────
    step_copy_original(args.image)

    analysis = step_room_analysis(args.image, trace)

    depth_result = step_depth_map(args.image, trace)

    step_color_palette(analysis, trace)

    state = step_build_state(
        args.image, analysis, depth_result,
        args.style, args.budget, trace,
    )

    step_save_trace(trace)

    report_path = step_html_report(analysis, state, trace, args.image)

    # ── final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PHASE 1 COMPLETE")
    print("═" * 60)
    print(f"\n  Style detected:  {analysis.get('style')}")
    print(f"  Room type:       {analysis.get('room_type')}")
    print(f"  Suggested for:   {', '.join(analysis.get('suggested_styles', []))}")
    print(f"  Total cost:      ${trace.get('total_cost_usd', '?')}")
    print(f"  Total time:      {trace.get('total_duration_s', '?')}s")
    print(f"\n  All files in:    {OUT}/")
    print(f"  Open report:     open {report_path}")
    print("\n  Next step: Phase 2 — build RAG index")
    print("  Run:  python rag/image_retriever.py --build data/interior_images/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()