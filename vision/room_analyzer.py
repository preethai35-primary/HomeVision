"""
vision/room_analyzer.py
═══════════════════════
Phase 1 — STEP 1 of 2

What this does:
  Send your room photo to GPT-4o vision.
  Get back structured JSON: style, furniture, colours, strengths, opportunities.

Run directly to test:
  python vision/room_analyzer.py data/my_room.jpg

Cost per call: ~$0.01–0.02 (one GPT-4o call with a high-res image)
"""

import base64
import json
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from .env file
client = OpenAI()


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior interior designer and spatial analyst \
specialising in residential redesign across global aesthetics.

TASK: Analyse the room image and return ONLY valid JSON. No prose, no markdown \
fences, no explanation before or after. Raw JSON object only.

STYLE VOCABULARY — detected_style and suggested_styles MUST use labels from \
this list. Include the 3-4 word descriptor in parentheses when it helps \
disambiguation, but the JSON value must be ONLY the label string itself.

Nordic / East Asian:
  scandinavian        — light wood, white, hygge warmth, functional
  japandi             — dark oak, wabi-sabi, austere, handcrafted
  minimalist          — bare surfaces, monochrome, only essentials
  wabi-sabi           — imperfect textures, aged patina, organic forms
  zen japanese        — tatami, shoji screens, low furniture, stone

Mediterranean / Southern European:
  mediterranean       — terracotta tiles, arched niches, warm plaster
  italian rustic      — exposed beams, stone floors, antique chestnut
  spanish colonial    — wrought iron, clay tiles, azulejo accents
  portuguese          — blue azulejo tiles, limestone, fado melancholy
  greek island        — whitewash, cobalt blue, linen, sea-worn wood
  tuscan farmhouse    — ochre walls, olive wood, rustic ceramics

South / Southeast Asian + Middle Eastern:
  indian vintage      — jali screens, brass accents, block prints, vivid colour
  indian contemporary — neutral base, regional craft accents, clean lines
  moroccan riad       — zellige tiles, arched doorways, lanterns, jewel tones
  persian traditional — geometric carpets, carved stucco, deep reds and golds
  balinese tropical   — open-air, rattan, volcanic stone, lush greenery

Western Modern:
  mid-century modern  — organic curves, walnut, mustard, teak, 1950s optimism
  industrial          — exposed brick, raw steel, factory glass, dark tones
  art deco            — geometric glamour, gold, velvet, lacquer, symmetry
  contemporary        — current trends, warm neutrals, mixed materials
  luxury modern       — marble, brass, statement lighting, rich textures
  coastal             — bleached wood, sea glass, linen, relaxed nautical

Eclectic / Organic:
  boho                — layered rugs, macramé, plants, warm earth tones
  cottagecore         — floral, vintage china, botanical prints, soft pastels
  maximalist          — pattern on pattern, bold colour, curated clutter
  farmhouse           — shiplap, galvanised metal, mason jars, cream tones
  french country      — toile, lavender, distressed oak, Provence palette

STYLE DISAMBIGUATION — when the room is ambiguous between similar styles:
  scandinavian vs japandi: scandinavian is lighter, warmer, more colourful; \
japandi is darker, more austere, celebrates imperfection
  mediterranean vs tuscan: mediterranean is coastal and tiled; tuscan is \
inland, beamed, stone-floored
  indian vintage vs moroccan: indian has jali woodwork and block prints; \
moroccan has zellige tiles and plaster muqarnas
  minimalist vs japandi: minimalist is purely functional with no decoration; \
japandi celebrates natural material quality

Return ONLY the JSON. No commentary."""


# User prompt — inject runtime context so GPT-4o has constraints it cannot
# infer from the image alone (budget, location, focus areas).
# The {placeholders} are filled by analyze_room() before sending.
ANALYSIS_PROMPT = """\
Analyse this {room_type_hint} photo.

Context the owner has provided:
  Preferred style direction: {style_preference}
  Redesign budget:           {budget} EUR
  Focus areas:               {focus_areas}
  Location:                  Netherlands (IKEA NL availability relevant)

Return exactly this JSON (fill real values, no placeholder text):

{{
  "style":           "current detected style — one label from the vocabulary",
  "room_type":       "living room / bedroom / kitchen / bathroom / office / other",
  "size_estimate":   "small / medium / large",
  "natural_light":   "poor / moderate / good / excellent",
  "ceiling_height":  "low / standard / high",
  "spatial_notes":   "1-2 sentences on layout, flow, and spatial relationships",
  "furniture": [
    {{"name": "bed", "condition": "good / fair / poor", "keep": true,
      "placement": "against north wall"}}
  ],
  "dominant_colors": ["#hexcode1", "#hexcode2", "#hexcode3"],
  "color_mood":      "warm / cool / neutral / mixed",
  "strengths":       ["what already works well — be specific"],
  "opportunities":   ["concrete improvements — be specific and actionable"],
  "suggested_styles": ["style-label-1", "style-label-2", "style-label-3"],
  "style_rationale": {{
    "style-label-1": "why this suits the room in 1 sentence",
    "style-label-2": "why this suits the room in 1 sentence"
  }},
  "lora_needed": {{
    "style-label-1": false,
    "style-label-2": false
  }}
}}

For lora_needed: mark true for regional styles the base SDXL model handles \
poorly without fine-tuning (indian vintage, moroccan riad, persian traditional, \
portuguese, balinese tropical). Mark false for all western/nordic styles.\
"""


# ── Image encoding ────────────────────────────────────────────────────────────

def encode_image(image_path: str) -> tuple[str, str]:
    """
    Convert image file to base64 string for the OpenAI API.

    OpenAI's vision API does not accept file paths — it needs the raw image
    data embedded in the request as a base64 string.

    Returns:
        (base64_string, media_type)  e.g. ("iVBOR...", "jpeg")
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            f"Did you copy your room photo to data/my_room.jpg?"
        )

    suffix = path.suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix in ("jpg", "jpeg") else suffix

    with open(path, "rb") as f:           # "rb" = read binary (raw bytes)
        raw_bytes = f.read()
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        # base64 converts binary → ASCII text so it can travel in JSON

    return b64, media_type


# ── Main analysis function ────────────────────────────────────────────────────

def analyze_room(
    image_path: str,
    model: str = "gpt-4o",
    style_preference: str = "surprise me",
    budget: str = "500-2000",
    focus_areas: list | None = None,
    room_type_hint: str = "room",
) -> dict:
    """
    Send room photo to GPT-4o and get back structured room analysis.

    Args:
        image_path:       path to room photo (jpg or png)
        model:            use "gpt-4o" — only this model supports image input
        style_preference: user's preferred style direction (from UI dropdown)
        budget:           redesign budget in EUR e.g. "500-2000"
        focus_areas:      list of focus areas e.g. ["furniture", "lighting"]
        room_type_hint:   optional hint for room type ("bedroom", "living room")

    Returns:
        dict with keys: style, room_type, furniture, dominant_colors,
        suggested_styles, lora_needed, style_rationale, etc.
        Also includes "_usage" key with token counts for cost tracking.
    """
    if focus_areas is None:
        focus_areas = ["furniture", "lighting", "color palette"]

    print(f"Analysing room: {image_path}")
    print(f"Model: {model} | Style preference: {style_preference} | Budget: €{budget}")

    b64_image, media_type = encode_image(image_path)
    print(f"Image encoded ({media_type}), sending to GPT-4o...")

    # fill runtime context into the analysis prompt
    filled_prompt = ANALYSIS_PROMPT.format(
        room_type_hint    = room_type_hint,
        style_preference  = style_preference,
        budget            = budget,
        focus_areas       = ", ".join(focus_areas),
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=1200,   # increased — new schema has more fields
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{media_type};base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": filled_prompt,
                    },
                ],
            },
        ],
    )

    # Step 3: extract the text response
    raw = response.choices[0].message.content.strip()
    print(f"Response received ({response.usage.completion_tokens} output tokens)")

    # Step 4: parse JSON defensively
    # Even with "no markdown fences" in the prompt, models sometimes add ```json
    # Strip them if present so json.loads() doesn't crash.
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])          # remove first line (```json)
        raw = raw.rsplit("```", 1)[0]        # remove trailing ```

    result = json.loads(raw)

    # Step 5: attach token usage for LangFuse / cost tracking
    result["_usage"] = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "model": model,
        "estimated_cost_usd": round(
            (response.usage.prompt_tokens * 0.000005) +
            (response.usage.completion_tokens * 0.000015),
            4
        ),
    }

    return result


# ── Run directly for testing ──────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="data/my_room.jpg")
    parser.add_argument("--style",  default="surprise me")
    parser.add_argument("--budget", default="500-2000")
    args = parser.parse_args()

    try:
        result = analyze_room(
            args.image,
            style_preference=args.style,
            budget=args.budget,
        )
        print("\n" + "─" * 50)
        print("ROOM ANALYSIS RESULT:")
        print("─" * 50)
        pprint.pprint(result, width=60)

        print("\n" + "─" * 50)
        print("SUMMARY:")
        print(f"  Style:          {result.get('style')}")
        print(f"  Room type:      {result.get('room_type')}")
        print(f"  Light:          {result.get('natural_light')}")
        print(f"  Ceiling:        {result.get('ceiling_height')}")
        print(f"  Furniture:      {[f['name'] for f in result.get('furniture', [])]}")
        print(f"  Strengths:      {result.get('strengths', [])}")
        print(f"  Suggested:      {result.get('suggested_styles', [])}")
        print(f"  LoRA needed:    {result.get('lora_needed', {})}")
        print(f"  Cost:           ${result['_usage']['estimated_cost_usd']}")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
    except json.JSONDecodeError as e:
        print(f"\nJSON parse error: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        if "api_key" in str(e).lower():
            print("Check your .env file has OPENAI_API_KEY set correctly.")