"""
evaluation/evaluator.py
═══════════════════════
Phase 3D — Evaluate generated room variants

Two metrics:
  1. CLIP score  — "does this image look like the requested style?"
                   Fast, free, no API cost. Standard in text-to-image papers.

  2. LLM-as-judge — GPT-4o rates layout preservation + style coherence.
                    One API call (~$0.04). Closest to human judgment.

Run after Phase 3C generates images:
  python evaluation/evaluator.py --state data/outputs/phase3/state.json
  python evaluation/evaluator.py --images data/outputs/phase3/ --style japandi
"""

from __future__ import annotations
import json
import base64
import time
from pathlib import Path


# ── CLIP score ────────────────────────────────────────────────────────────────

def clip_score(
    image_path: str,
    prompt: str,
) -> float:
    """
    Compute CLIP score: cosine similarity between image and text embeddings.

    Interpretation:
        > 0.30  excellent — image strongly matches the prompt
        0.25–0.30  good
        0.20–0.25  moderate
        < 0.20  poor — style transfer may not have worked

    These thresholds are empirically calibrated for interior design prompts.
    """
    import torch
    import open_clip
    from PIL import Image

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
        warnings.filterwarnings("ignore", message=".*QuickGELU mismatch.*")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # embed image
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(img)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    # embed text prompt (use just the style + room part, not quality tokens)
    # full prompt has quality anchors that dilute the style signal
    style_prompt = _extract_style_tokens(prompt)
    tokens = tokenizer([style_prompt]).to(device)
    with torch.no_grad():
        txt_emb = model.encode_text(tokens)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    score = (img_emb * txt_emb).sum().item()
    return round(float(score), 4)


def _extract_style_tokens(full_prompt: str) -> str:
    """
    Extract style-relevant tokens from the full SDXL prompt.
    Removes quality anchors (best quality, 8k, etc.) which add noise to CLIP scoring.
    Keeps style name, room type, materials, atmosphere.
    """
    quality_words = {
        "best quality", "masterwork", "ultra-detailed", "8k resolution",
        "professional architectural interior photography",
        "interior design magazine", "shot on canon", "global illumination",
        "physically based rendering", "editorial interior photography",
    }
    parts = [p.strip() for p in full_prompt.split(",")]
    filtered = [
        p for p in parts
        if not any(q in p.lower() for q in quality_words)
    ]
    return ", ".join(filtered[:12])   # cap at 12 tokens for CLIP's 77-token window


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

def llm_judge(
    original_image_path: str,
    generated_images: list[dict],
    style: str,
) -> dict:
    """
    Ask GPT-4o to judge generated room variants.

    generated_images: list of {path, variant_name, conditioning}
    Returns dict: {variant_name: {layout_score, style_score, quality_score, reasoning}}

    One GPT-4o call with all images. ~$0.04 per evaluation run.
    """
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()

    def encode(path: str) -> tuple[str, str]:
        suffix = Path(path).suffix.lower().lstrip(".")
        media  = "jpeg" if suffix in ("jpg", "jpeg") else "png"
        return base64.b64encode(Path(path).read_bytes()).decode(), media

    # build message content — original + all variants
    content = []

    orig_b64, orig_media = encode(original_image_path)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/{orig_media};base64,{orig_b64}", "detail": "low"},
    })
    content.append({"type": "text", "text": "IMAGE 0: Original room (reference for layout)"})

    for i, img in enumerate(generated_images):
        b64, media = encode(img["path"])
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{media};base64,{b64}", "detail": "low"},
        })
        content.append({
            "type": "text",
            "text": f"IMAGE {i+1}: {img['variant_name']} (conditioning: {img['conditioning']})",
        })

    judge_prompt = f"""You are an expert interior design critic and spatial analyst.

The first image is the ORIGINAL room. The following {len(generated_images)} images are \
AI-generated redesigns of the same room in {style} style, each using different \
spatial conditioning methods.

Rate each redesigned image (1–10) on these two criteria:

1. LAYOUT_PRESERVATION: How well does the redesign preserve the original room's \
spatial layout — wall positions, window location, furniture footprint, room proportions?
   10 = identical layout, only style changed
   1  = completely different room layout

2. STYLE_COHERENCE: How well does the redesign match {style} style?
   10 = perfect style execution
   1  = wrong style entirely

Return ONLY this JSON (no prose):
{{
  "IMAGE_1": {{"layout": 8, "style": 7, "reasoning": "one sentence"}},
  "IMAGE_2": {{"layout": 6, "style": 9, "reasoning": "one sentence"}},
  "IMAGE_3": {{"layout": 7, "style": 8, "reasoning": "one sentence"}},
  "IMAGE_4": {{"layout": 5, "style": 6, "reasoning": "one sentence"}}
}}
Only include IMAGE keys for the variants shown."""

    content.append({"type": "text", "text": judge_prompt})

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=600,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    scores = json.loads(raw)

    # attach variant names to scores
    result = {}
    for i, img in enumerate(generated_images):
        key = f"IMAGE_{i+1}"
        if key in scores:
            result[img["variant_name"]] = {
                **scores[key],
                "conditioning": img["conditioning"],
                "path":         img["path"],
            }

    return {
        "scores":       result,
        "usage": {
            "input_tokens":  response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "cost_est_usd":  round(
                response.usage.prompt_tokens  * 0.000005 +
                response.usage.completion_tokens * 0.000015, 4
            ),
        },
    }


# ── Full evaluation run ───────────────────────────────────────────────────────

def evaluate_variants(
    original_image_path: str,
    variants: list[dict],
    style: str,
    prompts: dict,
) -> dict:
    """
    Run full evaluation: CLIP score + LLM-as-judge for all variants.

    variants: list of {path, variant_name, conditioning}
              e.g. [
                {path: "...A.png", variant_name: "text_only",  conditioning: "none"},
                {path: "...B.png", variant_name: "depth",      conditioning: "depth"},
              ]

    prompts: {variant_name: positive_prompt_string}

    Returns combined scores for report.
    """
    print(f"Evaluating {len(variants)} variants for style: {style}")

    results = {}

    # CLIP scores — local, fast, no API cost
    print("  Computing CLIP scores...")
    for v in variants:
        prompt = prompts.get(v["variant_name"], style)
        try:
            score = clip_score(v["path"], prompt)
            results[v["variant_name"]] = {"clip_score": score}
            print(f"    {v['variant_name']}: CLIP = {score}")
        except Exception as e:
            print(f"    {v['variant_name']}: CLIP failed — {e}")
            results[v["variant_name"]] = {"clip_score": None}

    # LLM-as-judge — one API call, ~$0.04
    print("  Running LLM-as-judge...")
    try:
        judge_result = llm_judge(original_image_path, variants, style)
        for name, scores in judge_result["scores"].items():
            if name in results:
                results[name].update({
                    "layout_score": scores["layout"],
                    "style_score":  scores["style"],
                    "reasoning":    scores["reasoning"],
                    "conditioning": scores["conditioning"],
                })
        print(f"  LLM judge cost: ${judge_result['usage']['cost_est_usd']}")
    except Exception as e:
        print(f"  LLM judge failed: {e}")

    # find best variant on each axis
    clip_scores = {k: v["clip_score"] for k, v in results.items() if v.get("clip_score")}
    layout_scores = {k: v.get("layout_score", 0) for k, v in results.items()}

    best_style  = max(clip_scores,   key=clip_scores.get)   if clip_scores   else None
    best_layout = max(layout_scores, key=layout_scores.get) if layout_scores else None

    return {
        "scores":       results,
        "best_style_variant":  best_style,
        "best_layout_variant": best_layout,
        "style":        style,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M"),
    }


# ── Style-specific rubric evaluation for Indian styles ──────────────────────

INDIAN_STYLE_RUBRICS = {
    "indian vintage": {
        "name": "Indian Vintage / Rajasthani",
        "markers": [
            "jali or carved lattice screens",
            "brass or copper accents (lamps, vessels, handles)",
            "block print or hand-woven textiles",
            "jewel tones (ruby, emerald, saffron, sapphire)",
            "carved teak or rosewood furniture",
            "terracotta or stone flooring",
            "arched doorways or niches",
            "traditional motifs (paisley, lotus, mandala)",
        ],
        "anti_markers": [
            "Scandinavian minimal furniture",
            "chrome or steel modern fixtures",
            "stark white bare walls",
        ],
    },
    "indian contemporary": {
        "name": "Indian Contemporary",
        "markers": [
            "clean modern lines with regional craft accents",
            "handloom cotton or khadi textiles",
            "brass drawer pulls or handles",
            "warm white or cream walls",
            "teak or sheesham wood furniture",
            "craft objects as decor accents",
            "natural materials palette",
        ],
        "anti_markers": [
            "heavy ornament overwhelming the space",
            "purely Western aesthetic with no Indian elements",
        ],
    },
}


def style_rubric_eval(
    image_path: str,
    style: str,
    rubric: dict,
) -> dict:
    """
    GPT-4o rates a generated image against style-specific markers.
    Returns marker scores, anti-marker penalties, and overall grade.
    Cost: ~$0.02 per call.
    """
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()

    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    suffix = Path(image_path).suffix.lower().lstrip(".")
    media = "jpeg" if suffix in ("jpg", "jpeg") else "png"

    markers_list = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(rubric["markers"]))
    anti_list = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(rubric["anti_markers"]))

    prompt = f"""You are an expert in {rubric['name']} interior design.

Rate this room image on each style marker (1 = clearly present, 0 = absent):

STYLE MARKERS:
{markers_list}

ANTI-MARKERS (should NOT be present — penalize if found):
{anti_list}

Return ONLY JSON:
{{
  "markers": [1, 0, 1, ...],
  "anti_markers": [0, 0, ...],
  "authenticity_note": "one sentence on how authentic this feels to someone from India"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{media};base64,{b64}", "detail": "low"},
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    result = json.loads(raw)

    markers_found = sum(result.get("markers", []))
    markers_total = len(rubric["markers"])
    anti_found = sum(result.get("anti_markers", []))

    marker_score = markers_found / max(markers_total, 1)
    penalty = anti_found * 0.1
    final_score = max(0, marker_score - penalty)

    grades = {0.8: "A", 0.6: "B", 0.4: "C", 0.2: "D"}
    grade = "F"
    for threshold, g in grades.items():
        if final_score >= threshold:
            grade = g
            break

    return {
        "markers_found": markers_found,
        "markers_total": markers_total,
        "anti_markers_found": anti_found,
        "marker_score": round(marker_score, 3),
        "final_score": round(final_score, 3),
        "grade": grade,
        "authenticity_note": result.get("authenticity_note", ""),
        "cost_usd": round(
            response.usage.prompt_tokens * 0.000005 +
            response.usage.completion_tokens * 0.000015, 4
        ),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generated room variants")
    parser.add_argument("--original", default="data/my_room.jpg")
    parser.add_argument("--style",    default="japandi")
    parser.add_argument("--image",    help="Test CLIP score on a single image")
    parser.add_argument("--prompt",   help="Prompt for single CLIP score test")
    args = parser.parse_args()

    if args.image and args.prompt:
        score = clip_score(args.image, args.prompt)
        print(f"CLIP score: {score}")
        if score > 0.30:
            print("Excellent — image strongly matches the style")
        elif score > 0.25:
            print("Good match")
        elif score > 0.20:
            print("Moderate — some style elements present")
        else:
            print("Poor — style transfer may not have worked well")
    else:
        print("Run with --image and --prompt to test CLIP score")
        print("Or wire into run_phase3.py for full evaluation")