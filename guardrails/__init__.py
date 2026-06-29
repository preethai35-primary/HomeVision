"""
guardrails/ — input validation, schema enforcement, cost tracking
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Input validation ─────────────────────────────────────────────────────────

def is_room_image(image_path: str) -> tuple[bool, str, float]:
    """
    GPT-4o vision check: is this actually a room photo?
    Uses detail:"low" to keep cost at ~$0.005.
    Returns (is_valid, reason, cost_usd).
    """
    path = Path(image_path)
    if not path.exists():
        return False, f"File not found: {image_path}", 0.0

    suffix = path.suffix.lower().lstrip(".")
    if suffix not in ("jpg", "jpeg", "png", "webp"):
        return False, f"Unsupported image format: {suffix}", 0.0

    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()

    media_type = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    b64 = base64.b64encode(path.read_bytes()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{media_type};base64,{b64}",
                        "detail": "low",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Is this a photo of an interior room (bedroom, living room, "
                        "kitchen, bathroom, office, etc.)? "
                        "Reply ONLY with JSON: "
                        '{"is_room": true, "reason": "..."}'
                    ),
                },
            ],
        }],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    cost = (
        response.usage.prompt_tokens * 0.000005 +
        response.usage.completion_tokens * 0.000015
    )

    try:
        result = json.loads(raw)
        return result.get("is_room", False), result.get("reason", ""), round(cost, 4)
    except json.JSONDecodeError:
        return True, "Could not parse response — allowing", round(cost, 4)


# ── Schema validation ────────────────────────────────────────────────────────

class FurnitureItem(BaseModel):
    name: str
    condition: str
    keep: bool
    placement: Optional[str] = None


class RoomAnalysisSchema(BaseModel):
    style: str
    room_type: str
    size_estimate: Optional[str] = None
    natural_light: str
    ceiling_height: Optional[str] = None
    furniture: list[FurnitureItem]
    dominant_colors: list[str]
    color_mood: Optional[str] = None
    strengths: list[str]
    opportunities: list[str]
    suggested_styles: list[str]

    @field_validator("dominant_colors", mode="before")
    @classmethod
    def validate_hex_colors(cls, v):
        for c in v:
            if not re.match(r'^#[0-9a-fA-F]{6}$', str(c)):
                raise ValueError(f"Invalid hex color: {c}")
        return v


def validate_analysis(analysis: dict) -> tuple[bool, list[str]]:
    """
    Validate room_analysis JSON against expected schema.
    Returns (is_valid, list_of_error_strings).
    """
    try:
        RoomAnalysisSchema(**analysis)
        return True, []
    except Exception as e:
        errors = []
        if hasattr(e, "errors"):
            for err in e.errors():
                field = ".".join(str(loc) for loc in err["loc"])
                errors.append(f"{field}: {err['msg']}")
        else:
            errors.append(str(e))
        return False, errors


# ── Cost tracking ────────────────────────────────────────────────────────────

COST_LIMIT_USD = 0.50
COST_WARNING_THRESHOLD = 0.80

def check_cost_budget(
    cumulative_cost: float,
    limit: float = COST_LIMIT_USD,
) -> tuple[bool, str]:
    """
    Check if cumulative API cost is within budget.
    Returns (within_budget, message).
    """
    if cumulative_cost > limit:
        return False, f"Cost limit exceeded: ${cumulative_cost:.4f} > ${limit:.2f}"

    if cumulative_cost > limit * COST_WARNING_THRESHOLD:
        return True, f"Warning: approaching cost limit (${cumulative_cost:.4f} / ${limit:.2f})"

    return True, ""
