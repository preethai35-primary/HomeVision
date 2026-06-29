"""
agent/graph.py — LangGraph StateGraph for HomeVision

The graph orchestrates room analysis → retrieval → generation → evaluation
with conditional routing, human-in-the-loop interrupts, and guardrails.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import HomeVisionState
from agent.tools import (
    validate_input,
    analyze_room,
    compute_depth_map,
    suggest_styles,
    retrieve_examples,
    select_styles,
    route_generation,
    generate_images,
    evaluate_results,
    search_products,
    present_results,
    classify_intent,
    handle_info,
    refine_image,
)


# ── conditional edge functions ───────────────────────────────────────────────

def check_validation(state: dict) -> str:
    if state.get("error"):
        return "invalid"
    return "valid"


MAX_RETRIES = 2

def check_eval_quality(state: dict) -> str:
    scores = state.get("eval_scores", {})
    retry_count = state.get("retry_count", 0)

    for style, style_scores in scores.items():
        if isinstance(style_scores, dict) and "scores" in style_scores:
            for variant, s in style_scores["scores"].items():
                clip = s.get("clip_score")
                layout = s.get("layout_score", 0)
                style_s = s.get("style_score", 0)

                if clip and clip > 0.20 and layout >= 5 and style_s >= 5:
                    return "good"

    # if eval data is incomplete (nulls), pass through rather than retry
    has_any_score = any(
        isinstance(v, dict) and "scores" in v
        for v in scores.values()
    )
    if not has_any_score:
        return "good"

    if retry_count < MAX_RETRIES:
        return "poor"

    return "good"


def check_user_feedback(state: dict) -> str:
    feedback = state.get("user_feedback", "done")
    if feedback == "refine":
        return "refine"
    return "done"


def check_user_intent(state: dict) -> str:
    return state.get("user_intent", "done")


# ── retry node (bumps seed and retry count) ──────────────────────────────────

def bump_retry(state: dict) -> dict:
    new_count = state.get("retry_count", 0) + 1
    new_seed = state.get("seed", 42) + new_count
    print(f"[retry] Attempt {new_count}/{MAX_RETRIES} — seed={new_seed}")
    return {"retry_count": new_count, "seed": new_seed}


# ── graph builder ────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(HomeVisionState)

    # nodes
    graph.add_node("validate_input", validate_input)
    graph.add_node("analyze_room", analyze_room)
    graph.add_node("compute_depth_map", compute_depth_map)
    graph.add_node("suggest_styles", suggest_styles)
    graph.add_node("retrieve_examples", retrieve_examples)
    graph.add_node("select_styles", select_styles)
    graph.add_node("route_generation", route_generation)
    graph.add_node("generate_images", generate_images)
    graph.add_node("evaluate_results", evaluate_results)
    graph.add_node("bump_retry", bump_retry)
    graph.add_node("search_products", search_products)
    graph.add_node("present_results", present_results)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_info", handle_info)
    graph.add_node("refine_image", refine_image)

    # edges — deterministic pipeline
    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges("validate_input", check_validation, {
        "valid": "analyze_room",
        "invalid": END,
    })
    graph.add_edge("analyze_room", "compute_depth_map")
    graph.add_edge("compute_depth_map", "suggest_styles")
    graph.add_edge("suggest_styles", "retrieve_examples")
    graph.add_edge("retrieve_examples", "select_styles")

    # after HITL style selection
    graph.add_edge("select_styles", "route_generation")
    # search_products runs before SDXL loads — it only needs room_analysis +
    # selected_styles (already available), and running it first avoids the
    # MPS/FAISS segfault that occurs when FAISS threads start after MPS is live.
    graph.add_edge("route_generation", "search_products")
    graph.add_edge("search_products", "generate_images")
    graph.add_edge("generate_images", "evaluate_results")

    # eval routing — good → present, poor → retry
    graph.add_conditional_edges("evaluate_results", check_eval_quality, {
        "good": "present_results",
        "poor": "bump_retry",
    })
    graph.add_edge("bump_retry", "generate_images")

    # present_results captures raw input → classify intent → route
    graph.add_edge("present_results", "classify_intent")
    graph.add_conditional_edges("classify_intent", check_user_intent, {
        "refine": "refine_image",
        "info":   "handle_info",
        "done":   END,
    })
    graph.add_edge("handle_info", "present_results")
    graph.add_edge("refine_image", "evaluate_results")

    return graph


def compile_graph(checkpointer=None):
    """Compile the graph with a checkpointer. Default: MemorySaver."""
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer or MemorySaver())
