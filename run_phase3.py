"""
run_phase3.py
═════════════
Phase 3 — Image generation + evaluation.

Reads Phase 1 + 2 state, builds SDXL prompts, runs local generation
on MPS/CUDA, evaluates with CLIP score + LLM-as-judge,
saves all artifacts and HTML report.

Usage:
  # Generate with local GPU (MPS or CUDA):
  python run_phase3.py

  # Specific styles (1–4):
  python run_phase3.py --styles japandi "moroccan riad"

  # Specific conditioning variants only (saves time):
  python run_phase3.py --variants depth
  python run_phase3.py --variants text_only depth
  # Available: depth

  # Iterative refinement on an existing generated image:
  python run_phase3.py --refine data/outputs/phase3/japandi_depth.png \
      --instruction "make it warmer, add more plants"

Output:
  data/outputs/phase3/
    ├── {style}_text_only.png        variant A — no spatial conditioning
    ├── {style}_depth.png            variant B — depth map conditioning
    ├── prompts_used.json            exact prompts used
    ├── eval_scores.json             CLIP + LLM judge scores
    ├── state_dict.json              full state after Phase 3
    └── report.html                  visual report — open in browser
"""

import argparse
import base64
import json
import time
from datetime import datetime
from pathlib import Path

PHASE1_STATE = Path("data/outputs/phase1/06_state_dict.json")
PHASE2_STATE = Path("data/outputs/phase2/03_state_dict.json")
OUT          = Path("data/outputs/phase3")
OUT.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def log(step: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{step}] {msg}")


def save_json(data, filename: str):
    p = OUT / filename
    p.write_text(json.dumps(data, indent=2, default=str))
    log("save", filename)
    return p


def load_state() -> dict:
    """Load most recent state — Phase 2 if available, else Phase 1."""
    if PHASE2_STATE.exists():
        state = json.loads(PHASE2_STATE.read_text())
        log("load", f"Phase 2 state loaded")
    elif PHASE1_STATE.exists():
        state = json.loads(PHASE1_STATE.read_text())
        log("load", f"Phase 1 state loaded (Phase 2 not run yet)")
    else:
        print("\nERROR: No phase state found.")
        print("Run Phase 1 first: python run_phase1.py data/my_room.jpg")
        import sys; sys.exit(1)
    return state


# ── step 1: build prompts ─────────────────────────────────────────────────────

def step_build_prompts(state: dict, style_preferences: list[str]) -> dict:
    log("step1", f"Building prompts for styles: {style_preferences or 'auto from analysis'}")

    from generation.prompt_builder import build_all_prompts

    analysis = state.get("room_analysis", {})
    prompts  = build_all_prompts(analysis, style_preferences=style_preferences)

    log("step1", f"Built prompts for {len(prompts)} styles: {list(prompts.keys())}")

    for style, p in prompts.items():
        method = p["lora_validation"]["method"]
        lora   = p["lora_needed"]
        log("step1", f"  {style}: lora_needed={lora} (method={method})")

    save_json(prompts, "prompts_used.json")
    return prompts


# ── step 2: generation ─────────────────────────────────────────────────────────

def step_generate(
    state: dict,
    prompts: dict,
    seed: int,
    variants: list[str] | None = None,
) -> dict:
    from generation.local_pipeline import request_generation, CONDITIONING_VARIANTS

    all_variants  = list(CONDITIONING_VARIANTS.keys())
    use_variants  = variants if variants else all_variants

    invalid = [v for v in use_variants if v not in all_variants]
    if invalid:
        print(f"Unknown variants: {invalid}. Valid: {all_variants}")
        use_variants = [v for v in use_variants if v in all_variants]

    log("step2", f"Generating variants: {use_variants}")

    depth_path = state.get("depth_map_path")
    if not depth_path or not Path(depth_path).exists():
        log("step2", "WARNING: depth map not found — depth conditioning will be skipped")
        use_variants = [v for v in use_variants if "depth" not in v]
        depth_path   = None

    generated = request_generation(
        prompts        = prompts,
        depth_map_path = depth_path or "",
        seed           = seed,
        variants       = use_variants,
    )

    total = sum(
        1 for style_imgs in generated.values()
        for p in style_imgs.values() if p
    )
    log("step2", f"Generated {total} images across {len(generated)} styles")
    save_json(generated, "generated_images.json")
    return generated


# ── step 4: evaluation ────────────────────────────────────────────────────────

def step_evaluate(
    state: dict,
    generated: dict,
    prompts: dict,
    skip_llm: bool = False,
) -> dict:
    log("step4", "Running evaluation (CLIP score + LLM-as-judge)...")

    from evaluation.evaluator import evaluate_variants

    all_scores = {}
    original   = state.get("image_path", "data/my_room.jpg")

    for style, style_images in generated.items():
        log("step4", f"  Evaluating style: {style}")

        variants = [
            {
                "path":         path,
                "variant_name": variant_name,
                "conditioning": variant_name.replace("_", "+"),
            }
            for variant_name, path in style_images.items()
            if path and Path(path).exists()
        ]

        if not variants:
            log("step4", f"  No images found for {style} — skipping")
            continue

        prompt_str = prompts.get(style, {}).get("positive", style)

        try:
            scores = evaluate_variants(
                original_image_path = original,
                variants            = variants,
                style               = style,
                prompts             = {v["variant_name"]: prompt_str for v in variants},
            )
            all_scores[style] = scores
            log("step4", f"  Best style variant:  {scores.get('best_style_variant')}")
            log("step4", f"  Best layout variant: {scores.get('best_layout_variant')}")
        except Exception as e:
            log("step4", f"  Evaluation failed for {style}: {e}")
            all_scores[style] = {"error": str(e)}

    save_json(all_scores, "eval_scores.json")
    return all_scores


# ── step 5: update state ──────────────────────────────────────────────────────

def step_update_state(
    state: dict,
    prompts: dict,
    generated: dict,
    eval_scores: dict,
    style_preferences: list[str],
) -> dict:
    state["style_preferences"]  = style_preferences
    state["prompts_used"]        = prompts
    state["generated_images"]    = generated
    state["eval_scores"]         = eval_scores
    state["_phase_complete"]["phase_3"] = True
    save_json(state, "state_dict.json")
    log("state", "State updated with Phase 3 results")
    return state


# ── step 6: HTML report ───────────────────────────────────────────────────────

def step_html_report(
    state: dict,
    prompts: dict,
    generated: dict,
    eval_scores: dict,
) -> Path:
    log("report", "Generating HTML report...")

    def b64(path: str) -> str:
        p = Path(path)
        if p.exists():
            ext = p.suffix.lower().lstrip(".")
            return f"data:image/{'jpeg' if ext == 'jpg' else ext};base64," + \
                   base64.b64encode(p.read_bytes()).decode()
        return ""

    # original room
    orig_b64 = b64(state.get("image_path", ""))

    # depth map
    depth_b64 = b64(state.get("depth_map_path", ""))

    # build variant grid per style
    variant_labels = {
        "depth":      "depth + text",
        "text_only":  "text only (no layout)",
    }
    variants_html = ""
    for style, style_imgs in generated.items():
        scores = eval_scores.get(style, {}).get("scores", {})
        variants_html += f"<h3 style='margin:20px 0 10px;font-size:14px'>{style}</h3>"
        variants_html += "<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:16px'>"
        for vname, label in variant_labels.items():
            path = style_imgs.get(vname)
            img_html = f"<img src='{b64(path)}' style='width:100%;border-radius:6px'>" if path else \
                       "<div style='background:#f4f2ef;height:140px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:11px;color:#888'>not generated</div>"
            sc = scores.get(vname, {})
            clip  = sc.get("clip_score", "—")
            layout = sc.get("layout", "—")
            style_s = sc.get("style", "—")
            variants_html += f"""
            <div style='border:.5px solid #e4e0d8;border-radius:8px;overflow:hidden'>
              {img_html}
              <div style='padding:6px 8px;font-size:11px;background:#f8f6f3'>
                <div style='font-weight:500'>{label}</div>
                <div style='color:#888;margin-top:2px'>CLIP: {clip} | Layout: {layout} | Style: {style_s}</div>
              </div>
            </div>"""
        variants_html += "</div>"

    # prompts table
    prompts_html = ""
    for style, p in prompts.items():
        lora_badge = "<span style='background:#FAECE7;color:#712B13;padding:1px 6px;border-radius:99px;font-size:10px'>LoRA needed</span>" \
                     if p.get("lora_needed") else ""
        method = p.get("lora_validation", {}).get("method", "unknown")
        prompts_html += f"""
        <tr>
          <td><strong>{style}</strong> {lora_badge}</td>
          <td style='font-size:10px;font-family:monospace'>{p.get('positive','')[:120]}...</td>
          <td style='font-size:11px;color:#888'>{method}</td>
        </tr>"""

    mode_badge = "<span style='background:#E1F5EE;color:#085041;padding:2px 8px;border-radius:99px;font-weight:500'>LOCAL GPU</span>"

    analysis = state.get("room_analysis", {})

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>HomeVision — Phase 3 Report</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#F8F6F3;color:#2C2A27;margin:0;padding:24px}}
  h1{{font-size:22px;font-weight:600;margin-bottom:4px}}
  .sub{{color:#888;font-size:13px;margin-bottom:24px}}
  .section{{background:white;border-radius:12px;padding:20px 24px;
            margin-bottom:20px;border:.5px solid #E4E0D8}}
  .section h2{{font-size:13px;font-weight:600;color:#534AB7;
               letter-spacing:.05em;text-transform:uppercase;margin:0 0 14px}}
  .pill{{display:inline-block;font-size:11px;padding:2px 8px;
         border-radius:99px;font-weight:500;margin:2px}}
  .done{{background:#E1F5EE;color:#085041}}
  .todo{{background:#F4F2EF;color:#888}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{font-size:11px;font-weight:600;color:#888;text-align:left;
      padding:5px 8px;border-bottom:.5px solid #E4E0D8;text-transform:uppercase}}
  td{{padding:6px 8px;border-bottom:.5px solid #F0EDE8;color:#555;vertical-align:top}}
  .ctx-grid{{display:grid;grid-template-columns:160px 1fr;gap:16px;align-items:start}}
  .ctx-img{{border-radius:8px;border:.5px solid #E4E0D8;width:100%}}
  .kv{{display:flex;justify-content:space-between;padding:4px 0;
       border-bottom:.5px solid #f0ede8;font-size:12px}}
  .kv:last-child{{border:none}}
</style>
</head><body>
<h1>HomeVision — Phase 3 Report</h1>
<div class="sub">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · {mode_badge}</div>

<div class="section">
  <h2>Phase status</h2>
  <span class="pill done">Phase 1 ✓</span>
  <span class="pill done">Phase 2 ✓</span>
  <span class="pill done">Phase 3 ✓</span>
  <span class="pill todo">Phase 4 — guardrails</span>
  <span class="pill todo">Phase 5 — eval pipeline</span>
  <span class="pill todo">Phase 6 — API</span>
</div>

<div class="section">
  <h2>Room context</h2>
  <div class="ctx-grid">
    {'<img class="ctx-img" src="' + orig_b64 + '">' if orig_b64 else ''}
    <div>
      <div class="kv"><span>Style detected</span><strong>{analysis.get('style','?')}</strong></div>
      <div class="kv"><span>Room type</span><strong>{analysis.get('room_type','?')}</strong></div>
      <div class="kv"><span>Natural light</span><strong>{analysis.get('natural_light','?')}</strong></div>
      <div class="kv"><span>Styles generated</span><strong>{', '.join(generated.keys())}</strong></div>
      <div class="kv"><span>Conditioning</span><strong>depth map (ControlNet)</strong></div>
    </div>
  </div>
</div>

{'<div class="section"><h2>Depth map used for conditioning</h2><img src="' + depth_b64 + '" style="max-height:200px;border-radius:8px;border:.5px solid #E4E0D8"></div>' if depth_b64 else ''}

<div class="section">
  <h2>Generated variants — {sum(1 for s in generated.values() for p in s.values() if p)} images total</h2>
  {variants_html}
</div>

<div class="section">
  <h2>Prompts used</h2>
  <table>
    <thead><tr><th>style</th><th>positive prompt (truncated)</th><th>lora method</th></tr></thead>
    <tbody>{prompts_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>Next steps</h2>
  <p style="font-size:12px;color:#888;margin:0">
    Generation complete. Review variants above and select best conditioning strategy for Phase 4. Use --refine to iteratively adjust any generated image.
  </p>
</div>

</body></html>"""

    report_path = OUT / "report.html"
    report_path.write_text(html)
    log("report", f"Saved: {report_path}")
    return report_path


# ── main ──────────────────────────────────────────────────────────────────────

def step_refine(
    state: dict,
    image_path: str,
    instruction: str,
    prompts: dict,
    seed: int,
) -> str:
    """Run img2img refinement on an existing generated image."""
    from generation.local_pipeline import request_img2img

    log("refine", f"Refining: {image_path}")
    log("refine", f"Instruction: {instruction}")

    style_guess = Path(image_path).stem.split("_")[0].replace("-", " ")
    prompt_data = prompts.get(style_guess, {})
    positive = prompt_data.get("positive", style_guess)
    negative = prompt_data.get("negative", "")

    refined_path = request_img2img(
        source_image_path=image_path,
        positive_prompt=positive,
        negative_prompt=negative,
        refinement_instruction=instruction,
        strength=0.4,
        seed=seed,
    )
    log("refine", f"Saved: {refined_path}")
    return refined_path


def main():
    parser = argparse.ArgumentParser(description="HomeVision Phase 3 — generation + evaluation")
    parser.add_argument("--styles",   nargs="+", default=None,
                        help="1-4 styles e.g. --styles japandi 'moroccan riad' scandinavian")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--variants", nargs="+",
                        default=None,
                        metavar="VARIANT",
                        help="Which conditioning variants to generate. "
                             "Options: text_only depth. "
                             "Default: both. "
                             "Example: --variants depth")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip LLM-as-judge evaluation (saves ~$0.04 per style)")
    parser.add_argument("--refine", metavar="IMAGE_PATH",
                        help="Path to a generated image to refine with img2img")
    parser.add_argument("--instruction", default="",
                        help="Refinement instruction for --refine e.g. 'make it warmer'")
    args = parser.parse_args()

    # ── img2img refinement mode ──────────────────────────────────────────────
    if args.refine:
        if not Path(args.refine).exists():
            print(f"ERROR: {args.refine} not found")
            import sys; sys.exit(1)
        if not args.instruction:
            print("ERROR: --instruction required with --refine")
            print("Example: --refine path/to/image.png --instruction 'make it warmer'")
            import sys; sys.exit(1)

        print("\n" + "=" * 60)
        print("  HomeVision — Phase 3 (img2img refinement)")
        print("=" * 60 + "\n")

        state   = load_state()
        prompts = step_build_prompts(state, args.styles or [])
        result  = step_refine(state, args.refine, args.instruction, prompts, args.seed)

        print("\n" + "=" * 60)
        print(f"  Refined image: {result}")
        print("=" * 60 + "\n")
        return

    # ── full generation mode ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HomeVision — Phase 3 (local GPU)")
    print(f"  Styles:    {args.styles or 'auto from Phase 1 analysis'}")
    print(f"  Variants:  {args.variants or 'depth'}")
    print(f"  Seed:      {args.seed}")
    print("=" * 60 + "\n")

    state   = load_state()
    prompts = step_build_prompts(state, args.styles or [])
    gen     = step_generate(state, prompts, args.seed, variants=args.variants)
    scores  = step_evaluate(state, gen, prompts, skip_llm=args.skip_eval)
    state   = step_update_state(state, prompts, gen, scores, args.styles or [])
    report  = step_html_report(state, prompts, gen, scores)

    # unload SDXL to free VRAM for evaluation
    from generation.local_pipeline import _pipeline_instance
    if _pipeline_instance:
        _pipeline_instance.unload()

    print("\n" + "=" * 60)
    print("  PHASE 3 COMPLETE")
    print("=" * 60)
    print(f"  Styles:    {list(gen.keys())}")
    print(f"  Images:    {sum(1 for s in gen.values() for p in s.values() if p)}")
    print(f"  Report:    open {report}")
    print(f"\n  Refine any image:")
    print(f"    python run_phase3.py --refine data/outputs/phase3/STYLE_depth.png \\")
    print(f"        --instruction 'make it warmer, add more plants'")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()