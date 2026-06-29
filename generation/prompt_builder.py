"""
generation/prompt_builder.py
════════════════════════════
Phase 3A — Builds SDXL positive + negative prompts from room_analysis.

Pure Python. No GPU. Fully testable locally.

Takes the room_analysis dict from Phase 1 + a target style string,
returns a dict of prompts ready for the local SDXL pipeline.

Run to test:
  python generation/prompt_builder.py
  python generation/prompt_builder.py --style "moroccan riad" --room bedroom
"""

from __future__ import annotations
import argparse


# ── Style prompt vocabulary ───────────────────────────────────────────────────
# Each style maps to:
#   materials:   physical material tokens (high SDXL weight)
#   atmosphere:  mood / lighting tokens
#   avoid:       things to add to negative prompt for this style
#   lora_needed: True if base SDXL generates this style poorly without LoRA

STYLE_VOCAB: dict[str, dict] = {
    "scandinavian": {
        "materials":   "light pine wood, white walls, linen textiles, birch furniture, "
                       "ceramic accessories, hygge candles",
        "atmosphere":  "warm natural light, airy, clean, cosy Scandinavian interior",
        "avoid":       "dark tones, heavy ornament, clutter",
        "lora_needed": False,
    },
    "japandi": {
        "materials":   "dark oak wood, wabi-sabi ceramic, natural linen, bamboo, "
                       "handcrafted pottery, muted earth tones",
        "atmosphere":  "soft diffused light, serene, austere, mindful calm",
        "avoid":       "bright colours, plastic, chrome, clutter, decoration",
        "lora_needed": False,
    },
    "minimalist": {
        "materials":   "white concrete, glass, chrome, monochrome palette, "
                       "bare surfaces, single accent material",
        "atmosphere":  "stark clean light, silence in space, pure function",
        "avoid":       "patterns, decorative objects, warm tones, plants",
        "lora_needed": False,
    },
    "wabi-sabi": {
        "materials":   "aged plaster, weathered wood, cracked ceramic, raw linen, "
                       "mossy stone, imperfect handmade objects",
        "atmosphere":  "soft grey natural light, transience, gentle decay",
        "avoid":       "perfect surfaces, symmetry, bright colours, new materials",
        "lora_needed": False,
    },
    "zen japanese": {
        "materials":   "tatami mat, shoji paper screen, hinoki cypress, river stones, "
                       "low platform furniture, bamboo",
        "atmosphere":  "filtered morning light, meditative stillness, garden view",
        "avoid":       "western furniture, clutter, bright colours, overhead lighting",
        "lora_needed": False,
    },
    "mediterranean": {
        "materials":   "terracotta floor tiles, white plaster walls, wrought iron, "
                       "hand-painted ceramic, olive wood, mosaic",
        "atmosphere":  "warm golden sunlight, relaxed outdoor-indoor, sea breeze",
        "avoid":       "cold tones, minimalism, industrial materials",
        "lora_needed": False,
    },
    "italian rustic": {
        "materials":   "exposed wooden ceiling beams, stone floor, antique chestnut furniture, "
                       "terracotta urns, linen drapes, rough plaster",
        "atmosphere":  "warm amber afternoon light, farmhouse warmth, aged patina",
        "avoid":       "modern furniture, glass, chrome, white walls",
        "lora_needed": False,
    },
    "spanish colonial": {
        "materials":   "hand-painted azulejo tiles, clay floor tiles, wrought iron grille, "
                       "carved wood, bold terracotta and cobalt",
        "atmosphere":  "bright courtyard light, Andalusian warmth, vivid colour",
        "avoid":       "minimalism, pale tones, Scandinavian furniture",
        "lora_needed": False,
    },
    "portuguese": {
        "materials":   "blue and white azulejo tile panels, limestone floor, cork accents, "
                       "dark wood furniture, fado melancholy palette",
        "atmosphere":  "soft Atlantic light, saudade mood, historic grandeur",
        "avoid":       "bright colours, modern furniture, industrial materials",
        "lora_needed": True,   # azulejo tile patterns need LoRA for quality
    },
    "greek island": {
        "materials":   "whitewash plaster, cobalt blue accents, sea-worn driftwood, "
                       "natural linen, terracotta pots, pebble mosaic",
        "atmosphere":  "intense Mediterranean sunlight, brilliant white and blue",
        "avoid":       "dark tones, heavy furniture, ornate decoration",
        "lora_needed": False,
    },
    "tuscan farmhouse": {
        "materials":   "ochre plastered walls, terracotta roof tile, olive and walnut wood, "
                       "handwoven textiles, rustic ceramics, stone fireplace",
        "atmosphere":  "warm harvest afternoon light, Italian countryside abundance",
        "avoid":       "modern furniture, cool tones, glass, chrome",
        "lora_needed": False,
    },
    "indian vintage": {
        "materials":   "jali carved wood screen, brass accents, block print textiles, "
                       "vivid jewel tone fabrics, teak furniture, terracotta floor",
        "atmosphere":  "warm rich colour, layered texture, artisanal abundance",
        "avoid":       "minimalism, pale tones, plain walls, western furniture",
        "lora_needed": True,   # jali patterns and brass detail need LoRA
    },
    "indian contemporary": {
        "materials":   "warm white walls, natural teak, handloom cotton, brass drawer pulls, "
                       "regional craft objects as accents",
        "atmosphere":  "clean modern space with Indian craft warmth",
        "avoid":       "heavy ornament, overly western look, cold tones",
        "lora_needed": False,
    },
    "moroccan riad": {
        "materials":   "hand-cut zellige mosaic tile, arched stucco doorway, brass lantern, "
                       "jewel-toned silk cushions, carved cedar wood, fountain",
        "atmosphere":  "filtered courtyard light, sensory richness, jewel tones",
        "avoid":       "minimalism, pale tones, Scandinavian furniture, plain walls",
        "lora_needed": True,   # zellige patterns need LoRA
    },
    "persian traditional": {
        "materials":   "hand-knotted geometric carpet, carved stucco muqarnas, "
                       "deep crimson and gold, carved walnut, painted tilework",
        "atmosphere":  "opulent evening light, historical grandeur, intricate pattern",
        "avoid":       "bare walls, minimalism, plain floors, pale tones",
        "lora_needed": True,
    },
    "balinese tropical": {
        "materials":   "open timber pavilion, volcanic andesite stone, rattan, "
                       "tropical plants, batik textiles, teak wood",
        "atmosphere":  "dappled tropical light, open-air flow, lush greenery",
        "avoid":       "closed walls, cold materials, heavy upholstery, grey tones",
        "lora_needed": True,
    },
    "mid-century modern": {
        "materials":   "walnut teak wood, mustard yellow upholstery, geometric pattern, "
                       "splayed tapered legs, Eames-style chair, orange accent",
        "atmosphere":  "warm optimistic 1960s light, open plan, atomic age",
        "avoid":       "heavy ornament, dark walls, rustic materials, chrome excess",
        "lora_needed": False,
    },
    "industrial": {
        "materials":   "exposed red brick wall, raw steel beam, factory window, "
                       "concrete floor, Edison bulb, dark leather",
        "atmosphere":  "cool grey warehouse light, raw urban edge, masculine",
        "avoid":       "warm colours, floral patterns, rustic wood, plush fabrics",
        "lora_needed": False,
    },
    "art deco": {
        "materials":   "geometric gold inlay, velvet upholstery, lacquered black furniture, "
                       "mirrored surfaces, marble, palm leaf motif",
        "atmosphere":  "glamorous evening light, symmetrical grandeur, 1920s opulence",
        "avoid":       "minimalism, rough textures, rustic materials, pale palette",
        "lora_needed": False,
    },
    "contemporary": {
        "materials":   "warm greige walls, mixed wood tones, textured linen, "
                       "statement ceramic lamp, natural stone surface",
        "atmosphere":  "warm neutral light, current trends, comfortable sophistication",
        "avoid":       "dated furniture, heavy ornament, cold greys",
        "lora_needed": False,
    },
    "luxury modern": {
        "materials":   "Calacatta marble surface, brushed brass fixture, "
                       "deep velvet sofa, fluted glass, statement pendant",
        "atmosphere":  "warm gallery lighting, quiet luxury, hotel suite calm",
        "avoid":       "rustic materials, bright colours, cheap finishes, clutter",
        "lora_needed": False,
    },
    "coastal": {
        "materials":   "bleached driftwood, sea-glass blue, white shiplap, "
                       "natural sisal rug, linen curtain, rope accent",
        "atmosphere":  "bright coastal morning light, relaxed nautical ease",
        "avoid":       "dark tones, heavy furniture, ornate decoration",
        "lora_needed": False,
    },
    "boho": {
        "materials":   "layered kilim rug, macramé wall hanging, rattan furniture, "
                       "terracotta pot, trailing plant, mixed warm earthy textiles",
        "atmosphere":  "warm golden hour light, free-spirited, eclectic collector",
        "avoid":       "minimalism, cold tones, matching sets, bare walls",
        "lora_needed": False,
    },
    "cottagecore": {
        "materials":   "vintage floral wallpaper, aged pine dresser, botanical print, "
                       "handmade quilt, soft pastel, dried flower wreath",
        "atmosphere":  "soft English morning light, nostalgic countryside warmth",
        "avoid":       "modern furniture, dark tones, industrial materials",
        "lora_needed": False,
    },
    "maximalist": {
        "materials":   "pattern-on-pattern wallpaper, gallery wall, velvet in multiple "
                       "colours, gilded frame, bold jewel tones, layered rug",
        "atmosphere":  "rich curated abundance, every surface tells a story",
        "avoid":       "bare walls, neutral palette, minimalism, empty surfaces",
        "lora_needed": False,
    },
    "farmhouse": {
        "materials":   "white shiplap wall, galvanised steel, mason jar, "
                       "reclaimed wood beam, cream linen, wicker basket",
        "atmosphere":  "bright airy American farmhouse light, simple honest charm",
        "avoid":       "ornate decoration, dark tones, modern minimalism",
        "lora_needed": False,
    },
    "french country": {
        "materials":   "toile de jouy fabric, distressed painted oak, lavender, "
                       "Provençal ceramic, linen, natural stone floor",
        "atmosphere":  "soft Provençal afternoon light, romantic countryside",
        "avoid":       "industrial materials, modern furniture, cold palette",
        "lora_needed": False,
    },
}

# ── Base prompt templates ─────────────────────────────────────────────────────

QUALITY_TOKENS = "best quality, interior photography"

NEGATIVE_BASE = (
    "low quality, blurry, pixelated, jpeg artifacts, ugly, distorted, "
    "watermark, text, logo, signature, copyright, "
    "extra furniture, clutter, messy, chaotic, "
    "overexposed, underexposed, extreme dark, "
    "cartoon, illustration, 3d render, cgi, unreal engine, video game, "
    "fisheye lens, wide angle distortion, perspective distortion, "
    "people, person, human, face, hands, "
    "bad architecture, impossible geometry, floating objects"
)


# ── LoRA validation ───────────────────────────────────────────────────────────

# CLIP score threshold below which we flag lora_needed empirically.
# Calibrated on interior design prompts: scores below 0.22 indicate
# the base model struggles to represent the style accurately.
LORA_CLIP_THRESHOLD = 0.22


def validate_lora_needed(
    style: str,
    positive_prompt: str,
    test_image_path: str | None = None,
) -> dict:
    """
    Empirically validate whether a style needs LoRA.

    Strategy (in order of preference):
      1. If a test image exists (previously generated without LoRA),
         compute its CLIP score. Low score = confirm LoRA needed.
      2. If GPT-4o flagged lora_needed=True in room_analysis, trust it
         but mark as 'gpt4o_flagged' not 'validated'.
      3. Fall back to the hardcoded heuristic in STYLE_VOCAB.

    Returns:
        {
          "lora_needed":  bool,
          "method":       "clip_score" | "gpt4o_flagged" | "heuristic",
          "clip_score":   float or None,
          "threshold":    LORA_CLIP_THRESHOLD,
        }
    """
    style_lower = style.lower().strip()
    vocab = STYLE_VOCAB.get(style_lower, {})
    heuristic_flag = vocab.get("lora_needed", False)

    # Method 1: empirical CLIP score on an existing test image
    if test_image_path:
        try:
            import torch
            import open_clip
            from PIL import Image as PILImage

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model.eval()

            img = preprocess(
                PILImage.open(test_image_path).convert("RGB")
            ).unsqueeze(0)
            tokens = tokenizer([positive_prompt[:200]])   # truncate for CLIP

            with torch.no_grad():
                img_emb = model.encode_image(img)
                txt_emb = model.encode_text(tokens)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                score   = (img_emb * txt_emb).sum().item()

            lora_needed = score < LORA_CLIP_THRESHOLD
            return {
                "lora_needed": lora_needed,
                "method":      "clip_score",
                "clip_score":  round(float(score), 4),
                "threshold":   LORA_CLIP_THRESHOLD,
            }
        except Exception as e:
            print(f"[lora_validate] CLIP test failed: {e} — falling back to heuristic")

    # Method 2: trust GPT-4o's assessment from room_analysis
    # (caller passes this via room_analysis["lora_needed"][style])
    # Handled in build_all_prompts — not repeated here.

    # Method 3: heuristic fallback
    return {
        "lora_needed": heuristic_flag,
        "method":      "heuristic",
        "clip_score":  None,
        "threshold":   LORA_CLIP_THRESHOLD,
        "note":        "Not empirically validated. Run with a test image to confirm.",
    }


# ── Main builder ──────────────────────────────────────────────────────────────

def build_prompt(
    style: str,
    room_analysis: dict,
    conditioning_strength: float = 0.7,
    test_image_path: str | None = None,
) -> dict:
    """
    Build SDXL positive + negative prompts for a given style and room analysis.

    Args:
        style:                 target style label
        room_analysis:         dict from Phase 1 room_analyzer.py
        conditioning_strength: ControlNet depth strength 0.0–1.0
        test_image_path:       optional — if a previous generation exists,
                               use it to empirically validate lora_needed

    Returns dict with keys:
        positive, negative, style, lora_needed, lora_validation,
        conditioning_strength, tokens_est
    """
    style_lower = style.lower().strip()
    vocab = STYLE_VOCAB.get(style_lower)

    if vocab is None:
        print(f"[prompt_builder] Unknown style '{style}' — using generic prompt")
        vocab = {
            "materials":   f"{style} interior design aesthetic",
            "atmosphere":  "warm natural light, professional interior",
            "avoid":       "low quality",
            "lora_needed": False,
        }

    room_type  = room_analysis.get("room_type",     "room")
    light      = room_analysis.get("natural_light", "moderate")
    color_mood = room_analysis.get("color_mood",    "neutral")
    ceiling    = room_analysis.get("ceiling_height","standard")

    positive_parts = [
        f"{style_lower} style {room_type}",
        f"{vocab['materials']}",
        f"{vocab['atmosphere']}",
        f"{light} natural light" if light != "poor" else "warm artificial lighting",
        f"{color_mood} tones",
        QUALITY_TOKENS,
    ]
    positive = ", ".join(p for p in positive_parts if p)
    # CLIP truncates at 77 tokens — style tokens are first so the
    # quality filler at the end gets cut harmlessly

    style_avoid = vocab.get("avoid", "")
    negative    = f"{NEGATIVE_BASE}, {style_avoid}" if style_avoid else NEGATIVE_BASE

    # validate lora_needed — empirical if test image available, else heuristic
    lora_validation = validate_lora_needed(style_lower, positive, test_image_path)

    # also check what GPT-4o said in room_analysis
    gpt4o_lora_flags = room_analysis.get("lora_needed", {})
    if style_lower in gpt4o_lora_flags and lora_validation["method"] == "heuristic":
        gpt4o_flag = gpt4o_lora_flags[style_lower]
        if gpt4o_flag:
            lora_validation["lora_needed"] = True
            lora_validation["method"]      = "gpt4o_flagged"
            lora_validation["note"]        = "GPT-4o flagged during room analysis"

    return {
        "positive":              positive,
        "negative":              negative,
        "style":                 style_lower,
        "lora_needed":           lora_validation["lora_needed"],
        "lora_validation":       lora_validation,
        "conditioning_strength": conditioning_strength,
        "tokens_est":            len(positive.split(",")),
    }


def build_blend_prompt(
    style1: str,
    style2: str,
    room_analysis: dict,
    conditioning_strength: float = 0.7,
) -> dict:
    """
    Build a blended SDXL prompt from two styles.
    Interleaves the top material tokens from each and merges atmospheres.
    """
    s1, s2 = style1.lower().strip(), style2.lower().strip()
    v1 = STYLE_VOCAB.get(s1, {"materials": s1, "atmosphere": s1, "avoid": "", "lora_needed": False})
    v2 = STYLE_VOCAB.get(s2, {"materials": s2, "atmosphere": s2, "avoid": "", "lora_needed": False})

    mats1 = [m.strip() for m in v1["materials"].split(",")][:2]
    mats2 = [m.strip() for m in v2["materials"].split(",")][:2]
    blended_mats = [item for pair in zip(mats1, mats2) for item in pair]

    room_type  = room_analysis.get("room_type",     "room")
    light      = room_analysis.get("natural_light", "moderate")
    color_mood = room_analysis.get("color_mood",    "neutral")

    positive = (
        f"{s1} and {s2} style {room_type}, "
        f"{', '.join(blended_mats)}, "
        f"{v1['atmosphere']} meets {v2['atmosphere']}, "
        f"{light} natural light, {color_mood} tones, "
        f"{QUALITY_TOKENS}"
    )

    avoid_parts = [v for v in [v1.get("avoid", ""), v2.get("avoid", "")] if v]
    negative = f"{NEGATIVE_BASE}, {', '.join(avoid_parts)}" if avoid_parts else NEGATIVE_BASE

    lora_needed = v1.get("lora_needed", False) or v2.get("lora_needed", False)

    return {
        "positive":              positive,
        "negative":              negative,
        "style":                 f"{s1} × {s2}",
        "lora_needed":           lora_needed,
        "lora_validation":       {"method": "heuristic", "lora_needed": lora_needed, "clip_score": None, "threshold": LORA_CLIP_THRESHOLD},
        "conditioning_strength": conditioning_strength,
        "tokens_est":            len(positive.split(",")),
    }


def build_all_prompts(
    room_analysis: dict,
    style_preferences: list[str] | str | None = None,
) -> dict:
    """
    Build prompts for a list of styles.

    style_preferences can be:
      - A list of 1–4 styles: ["japandi", "moroccan riad"]
      - A single string:      "japandi"
      - None:                 uses room_analysis["suggested_styles"]

    Priority order:
      1. style_preferences (user's explicit choices)
      2. room_analysis["suggested_styles"] (GPT-4o recommendations)
    Combined, deduplicated, capped at 4.

    Returns:
        {
          "japandi":       {positive, negative, lora_needed, lora_validation, ...},
          "moroccan riad": {...},
        }
    """
    # normalise style_preferences to a list
    if style_preferences is None:
        pref_list = []
    elif isinstance(style_preferences, str):
        pref_list = [style_preferences] if style_preferences != "surprise me" else []
    else:
        pref_list = [s for s in style_preferences if s != "surprise me"]

    suggested = room_analysis.get("suggested_styles", [])

    # merge: user choices first, then GPT-4o suggestions, deduplicate
    all_styles = list(dict.fromkeys([*pref_list, *suggested]))[:4]

    if not all_styles:
        all_styles = ["scandinavian"]   # final fallback

    prompts = {}
    for style in all_styles:
        prompts[style] = build_prompt(style, room_analysis)

    return prompts


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description="Build SDXL prompts from room analysis")
    parser.add_argument("--styles", nargs="+", default=["japandi"],
                        help="1–4 target styles e.g. --styles japandi 'moroccan riad' scandinavian")
    parser.add_argument("--room",   default="bedroom",   help="Room type")
    parser.add_argument("--light",  default="good",      help="Natural light level")
    parser.add_argument("--mood",   default="neutral",   help="Colour mood")
    parser.add_argument("--all",    action="store_true", help="List all styles")
    args = parser.parse_args()

    if args.all:
        print(f"Available styles ({len(STYLE_VOCAB)}):\n")
        for style, vocab in STYLE_VOCAB.items():
            lora  = " [LoRA heuristic]" if vocab["lora_needed"] else ""
            print(f"  {style:<25} {vocab['materials'][:55]}...{lora}")
        print("\nNote: lora_needed is a heuristic. Validate empirically with --styles + a test image.")
    else:
        mock_analysis = {
            "room_type":       args.room,
            "natural_light":   args.light,
            "color_mood":      args.mood,
            "ceiling_height":  "standard",
            "spatial_notes":   "Room has good proportions.",
            "suggested_styles": args.styles,
            "lora_needed":     {},
        }

        prompts = build_all_prompts(mock_analysis, style_preferences=args.styles)

        for style, p in prompts.items():
            print(f"\n{'='*60}")
            print(f"STYLE: {style}")
            print(f"LoRA needed:  {p['lora_needed']}")
            print(f"LoRA method:  {p['lora_validation']['method']}")
            if p['lora_validation'].get('note'):
                print(f"Note:         {p['lora_validation']['note']}")
            print(f"Est. tokens:  {p['tokens_est']}")
            print(f"\nPOSITIVE:\n{p['positive'][:200]}...")
            print(f"\nNEGATIVE:\n{p['negative'][:120]}...")