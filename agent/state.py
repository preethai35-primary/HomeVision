"""
agent/state.py — shared state for the LangGraph agent
"""
from typing import TypedDict, Optional


class HomeVisionState(TypedDict):
    # ── inputs (set by user) ──────────────────────────────
    image_path: str
    style_preferences: list[str]
    budget: str
    focus_areas: list[str]
    user_message: str

    # ── vision outputs ────────────────────────────────────
    room_analysis: Optional[dict]
    depth_map_path: Optional[str]

    # ── rag outputs ───────────────────────────────────────
    reference_images: Optional[dict]
    ikea_products: Optional[list[dict]]

    # ── generation ────────────────────────────────────────
    prompts_used: Optional[dict]
    generated_images: Optional[dict]
    active_variant: Optional[dict]
    lora_paths: Optional[dict]

    # ── evaluation ────────────────────────────────────────
    eval_scores: Optional[dict]
    style_rubric_scores: Optional[dict]

    # ── human-in-the-loop ─────────────────────────────────
    selected_styles: Optional[list[str]]
    user_feedback: Optional[str]
    user_intent: Optional[str]
    refinement_instruction: Optional[str]

    # ── routing / control ─────────────────────────────────
    style_blend: Optional[dict]
    generation_mode: Optional[dict]
    ip_adapter_refs: Optional[dict]
    retry_count: int
    seed: int

    # ── cost tracking ─────────────────────────────────────
    cumulative_cost_usd: float

    # ── llm outputs ───────────────────────────────────────
    design_explanation: Optional[str]
    conversation_history: list[dict]

    # ── guardrails ────────────────────────────────────────
    guardrail_flags: Optional[dict]

    # ── meta ──────────────────────────────────────────────
    error: Optional[str]
    trace_id: Optional[str]