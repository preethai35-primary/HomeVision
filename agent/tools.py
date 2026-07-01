"""
agent/tools.py — LangGraph node functions
Each function takes HomeVisionState, returns a partial dict update.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from langgraph.types import interrupt


# ── helpers ──────────────────────────────────────────────────────────────────

OUT = Path("data/outputs")

def _log(node: str, msg: str):
    print(f"[{node}] {msg}")


# ── node 1: validate input ──────────────────────────────────────────────────

def validate_input(state: dict) -> dict:
    _log("validate", f"Checking image: {state['image_path']}")

    image_path = state["image_path"]
    if not Path(image_path).exists():
        return {"error": f"Image not found: {image_path}"}

    from guardrails import is_room_image, check_cost_budget
    is_valid, reason, cost = is_room_image(image_path)

    if not is_valid:
        return {
            "error": f"Not a room image: {reason}",
            "cumulative_cost_usd": state.get("cumulative_cost_usd", 0) + cost,
        }

    within_budget, budget_msg = check_cost_budget(cost)
    _log("validate", f"Image validated (${cost}). {budget_msg}")

    return {
        "guardrail_flags": {"input_validated": True, "reason": reason},
        "cumulative_cost_usd": state.get("cumulative_cost_usd", 0) + cost,
        "error": None,
    }


# ── node 2: analyze room ────────────────────────────────────────────────────

def analyze_room(state: dict) -> dict:
    _log("analyze", "Sending room photo to GPT-4o...")

    from vision.room_analyzer import analyze_room as _analyze
    from guardrails import validate_analysis

    t0 = time.time()
    analysis = _analyze(
        state["image_path"],
        style_preference=state["style_preferences"][0] if state["style_preferences"] else "surprise me",
        budget=state.get("budget", "500-2000"),
        focus_areas=state.get("focus_areas"),
    )
    elapsed = round(time.time() - t0, 1)

    cost = analysis.get("_usage", {}).get("estimated_cost_usd", 0.015)
    _log("analyze", f"Done in {elapsed}s — ${cost}")

    is_valid, errors = validate_analysis(
        {k: v for k, v in analysis.items() if k != "_usage"}
    )
    if not is_valid:
        _log("analyze", f"Schema warnings: {errors}")

    return {
        "room_analysis": {k: v for k, v in analysis.items() if k != "_usage"},
        "cumulative_cost_usd": state.get("cumulative_cost_usd", 0) + cost,
    }


# ── node 3: depth map ───────────────────────────────────────────────────────

def compute_depth_map(state: dict) -> dict:
    _log("depth", "Running Depth-Anything v2...")

    from vision.depth_estimator import estimate_depth

    out_dir = OUT / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_path = str(out_dir / "depth_map.png")

    t0 = time.time()
    estimate_depth(state["image_path"], output_path=depth_path)
    _log("depth", f"Done in {round(time.time() - t0, 1)}s")

    return {"depth_map_path": depth_path}


# ── node 4: suggest styles ──────────────────────────────────────────────────

def suggest_styles(state: dict) -> dict:
    analysis = state.get("room_analysis", {})
    user_prefs = state.get("style_preferences", [])
    suggested = analysis.get("suggested_styles", [])

    # user preferences first, then GPT-4o suggestions, deduplicate, cap at 4
    merged = list(dict.fromkeys(
        [s for s in user_prefs if s != "surprise me"] + suggested
    ))[:4]

    if not merged:
        merged = ["scandinavian"]

    _log("suggest", f"Styles to explore: {merged}")
    return {"style_preferences": merged}


# ── node 5: retrieve examples (with HITL interrupt) ─────────────────────────

def retrieve_examples(state: dict) -> dict:
    """CLIP retrieval only — no interrupt here."""
    styles = state.get("style_preferences", ["scandinavian"])
    _log("retrieve", f"Finding reference images for: {styles}")

    from rag.image_retriever import retrieve_design_references

    ref_images = {}
    for style in styles:
        query = f"{style} interior design room"
        results = retrieve_design_references(query, top_k=5)
        ref_images[style] = results
        count = len(results)
        top_sim = results[0]["similarity"] if results else 0
        _log("retrieve", f"  {style}: {count} images (top sim: {top_sim:.3f})")

    return {"reference_images": ref_images}


def select_styles(state: dict) -> dict:
    """HITL interrupt — show examples, user picks styles."""
    ref_images = state.get("reference_images", {})
    styles = list(ref_images.keys())

    print("\n" + "=" * 60)
    print("  STYLE EXAMPLES RETRIEVED")
    print("=" * 60)
    for style, refs in ref_images.items():
        sim = refs[0]["similarity"] if refs else 0
        quality = "good" if sim > 0.26 else "moderate" if sim > 0.22 else "weak"
        print(f"  {style:<25} {len(refs)} refs  (best match: {sim:.3f} [{quality}])")
    print("=" * 60)

    selected = interrupt({
        "type": "style_selection",
        "message": "Pick styles to generate (comma-separated), or 'all' for all suggested:",
        "options": styles,
    })

    if isinstance(selected, str):
        if selected.strip().lower() == "all":
            selected = styles
        else:
            selected = [s.strip() for s in selected.split(",")]

    _log("select", f"User selected: {selected}")
    return {"selected_styles": selected}


# ── node 6: route generation ────────────────────────────────────────────────

CLIP_REF_THRESHOLD = 0.25

def route_generation(state: dict) -> dict:
    selected = state.get("selected_styles", [])
    ref_images = state.get("reference_images", {})

    mode = {}
    ip_refs = {}

    for style in selected:
        refs = ref_images.get(style, [])
        good_refs = [r for r in refs if r.get("similarity", 0) >= CLIP_REF_THRESHOLD]

        if len(good_refs) >= 2:
            # IP-Adapter ready (deferred — falls through to text_depth for now)
            mode[style] = "text_depth"
            ip_refs[style] = [r["path"] for r in good_refs[:4]]
            _log("route", f"  {style}: {len(good_refs)} good refs (IP-Adapter ready, using text_depth for now)")
        else:
            mode[style] = "text_depth"
            _log("route", f"  {style}: text + depth conditioning")

    return {
        "generation_mode": mode,
        "ip_adapter_refs": ip_refs,
    }


# ── node 7: generate images ─────────────────────────────────────────────────

def generate_images(state: dict) -> dict:
    selected = state.get("selected_styles", [])
    blend = state.get("style_blend")
    _log("generate", f"Generating for styles: {selected}")

    from generation.prompt_builder import build_all_prompts, build_blend_prompt, get_lora
    from generation.local_pipeline import request_generation, _get_pipeline

    analysis = state.get("room_analysis", {})

    if blend and len(blend.get("styles", [])) >= 2:
        s1, s2 = blend["styles"][0], blend["styles"][1]
        blend_name = f"{s1} × {s2}"
        prompts = {blend_name: build_blend_prompt(s1, s2, analysis)}
        selected = [blend_name]
    else:
        prompts = build_all_prompts(analysis, style_preferences=selected)
        prompts = {s: p for s, p in prompts.items() if s in selected}

    # prepend LoRA trigger tokens and load adapters for styles that have one
    lora_loaded = False
    for style, prompt_data in prompts.items():
        lora = get_lora(style)
        if lora:
            trigger, adapter_path = lora
            prompts[style]["positive"] = f"{trigger}, {prompt_data['positive']}"
            pipe = _get_pipeline()
            pipe.load_lora(adapter_path)
            lora_loaded = True
            _log("generate", f"LoRA active for '{style}' — trigger='{trigger}'")

    seed = state.get("seed", 42)
    depth_path = state.get("depth_map_path", "")

    generated = request_generation(
        prompts=prompts,
        depth_map_path=depth_path,
        seed=seed,
        variants=["depth"],
        output_dir="data/outputs/agent",
    )

    # unload LoRA after generation so it doesn't bleed into refinement passes
    if lora_loaded:
        _get_pipeline().unload_lora()

    # keep SDXL loaded for potential img2img refinement in refine_image node

    total = sum(1 for s in generated.values() for p in s.values() if p)
    _log("generate", f"Generated {total} images")

    return {
        "generated_images": generated,
        "prompts_used": prompts,
        "selected_styles": selected,
    }


# ── node 8: evaluate results ────────────────────────────────────────────────

def evaluate_results(state: dict) -> dict:
    generated = state.get("generated_images", {})
    prompts = state.get("prompts_used", {})
    original = state.get("image_path", "")
    _log("evaluate", f"Evaluating {len(generated)} styles...")

    from evaluation.evaluator import evaluate_variants

    all_scores = {}
    total_cost = 0

    for style, style_images in generated.items():
        variants = [
            {
                "path": path,
                "variant_name": vname,
                "conditioning": vname,
            }
            for vname, path in style_images.items()
            if path and Path(path).exists()
        ]
        if not variants:
            continue

        prompt_str = prompts.get(style, {}).get("positive", style)
        try:
            scores = evaluate_variants(
                original_image_path=original,
                variants=variants,
                style=style,
                prompts={v["variant_name"]: prompt_str for v in variants},
            )
            all_scores[style] = scores
            total_cost += 0.04
        except Exception as e:
            _log("evaluate", f"  {style} eval failed: {e}")
            all_scores[style] = {"error": str(e)}

    # run style-specific rubric for Indian styles
    rubric_scores = _run_rubric_eval(state, generated)

    _log("evaluate", f"Evaluation cost: ~${total_cost:.2f}")

    return {
        "eval_scores": all_scores,
        "style_rubric_scores": rubric_scores,
        "cumulative_cost_usd": state.get("cumulative_cost_usd", 0) + total_cost,
    }


def _run_rubric_eval(state: dict, generated: dict) -> dict:
    """Run style-specific rubric evaluation for Indian styles."""
    try:
        from evaluation.evaluator import style_rubric_eval, INDIAN_STYLE_RUBRICS
    except ImportError:
        return {}

    rubric_scores = {}
    for style in generated:
        if style not in INDIAN_STYLE_RUBRICS:
            continue

        # pick the best image (depth variant preferred)
        images = generated[style]
        img_path = images.get("depth") or images.get("text_only")
        if not img_path or not Path(img_path).exists():
            continue

        _log("evaluate", f"  Running Indian style rubric for: {style}")
        try:
            result = style_rubric_eval(img_path, style, INDIAN_STYLE_RUBRICS[style])
            rubric_scores[style] = result
        except Exception as e:
            _log("evaluate", f"  Rubric eval failed for {style}: {e}")

    return rubric_scores


# ── node 9: search products ─────────────────────────────────────────────────

def search_products(state: dict) -> dict:
    _log("products", "Searching IKEA catalogue...")

    from rag.ikea_search import search_ikea_products

    analysis = state.get("room_analysis", {})
    selected = state.get("selected_styles", ["scandinavian"])
    style = selected[0] if selected else "scandinavian"

    products = search_ikea_products(
        style=style,
        furniture_list=analysis.get("furniture", []),
        budget=state.get("budget", "500-2000"),
    )

    _log("products", f"Found {len(products)} products")
    return {"ikea_products": products}


# ── node 10: present results (with HITL interrupt) ──────────────────────────

def present_results(state: dict) -> dict:
    generated = state.get("generated_images", {})
    eval_scores = state.get("eval_scores", {})
    products = state.get("ikea_products", [])
    rubric = state.get("style_rubric_scores", {})

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    for style, images in generated.items():
        print(f"\n  Style: {style}")
        for variant, path in images.items():
            if path:
                scores = eval_scores.get(style, {}).get("scores", {}).get(variant, {})
                clip = scores.get("clip_score", "n/a")
                layout = scores.get("layout_score", "n/a")
                print(f"    {variant:<12} {path}")
                print(f"               CLIP: {clip}  Layout: {layout}")

        if style in rubric:
            r = rubric[style]
            print(f"    Style rubric: {r.get('markers_found', '?')}/{r.get('markers_total', '?')} markers")

    if products:
        print(f"\n  IKEA products: {len(products)} recommendations")
        for p in products[:5]:
            print(f"    {p.get('name', '')[:40]:<42} {p.get('price', ''):<10}  {p.get('url', '')}")

    print(f"\n  Total API cost: ${state.get('cumulative_cost_usd', 0):.4f}")
    print("=" * 60)

    feedback = interrupt({
        "type": "user_feedback",
        "message": "Type 'done' to finish, ask a question, or describe a change:",
    })

    return {
        "user_feedback": "pending",
        "refinement_instruction": str(feedback).strip(),
    }


# ── node 10b: classify intent ───────────────────────────────────────────────

def classify_intent(state: dict) -> dict:
    """Route user input to refine / info / done using GPT-4o-mini."""
    raw = state.get("refinement_instruction", "").strip().lower()

    # fast path for obvious exits
    if raw in ("done", "exit", "quit", "finish", "bye", "q"):
        _log("intent", "done (fast path)")
        return {"user_intent": "done"}

    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "Classify the user's interior design assistant message into exactly one of:\n"
                "- refine: wants a VISUAL change to the generated image "
                "(warmer, brighter, add plants, remove furniture, change colours, different style, etc.)\n"
                "- info: wants INFORMATION without changing the image — this includes product links/prices, "
                "style explanations, material questions, AND any question about where to find something "
                "('where is', 'show me', 'can I see', 'what is the path', 'where can I find', "
                "'is there', 'what file', etc.)\n"
                "- done: finished, wants to exit\n"
                "If the message is a question rather than an instruction, choose info.\n"
                "Reply with ONE word only."
            )},
            {"role": "user", "content": state.get("refinement_instruction", "")},
        ],
    )

    intent = resp.choices[0].message.content.strip().lower()
    if intent not in ("refine", "info", "done"):
        intent = "info"  # safe default — questions are cheaper than accidental img2img

    cost = (
        resp.usage.prompt_tokens * 0.00000015 +
        resp.usage.completion_tokens * 0.0000006
    )
    _log("intent", f"'{state.get('refinement_instruction','')[:50]}' → {intent}")
    return {
        "user_intent": intent,
        "cumulative_cost_usd": state.get("cumulative_cost_usd", 0) + cost,
    }


# ── node 10c: handle info ───────────────────────────────────────────────────

def handle_info(state: dict) -> dict:
    """Answer info queries (links, prices, style details) without regenerating images."""
    query = state.get("refinement_instruction", "")
    products = state.get("ikea_products", []) or []
    analysis = state.get("room_analysis", {}) or {}
    selected = state.get("selected_styles", [])

    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()

    products_ctx = "\n".join(
        f"- {p['name']} | {p.get('price','')} | {p.get('url','')}"
        for p in products
    ) or "No products found."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=400,
        temperature=0.3,
        messages=[
            {"role": "system", "content": (
                "You are a helpful interior design assistant for HomeVision. "
                "Answer the user's question using the context below. "
                "For product links, list them clearly with name, price, and URL. "
                "Be concise and friendly."
            )},
            {"role": "user", "content": (
                f"Style: {', '.join(selected)}\n"
                f"Room: {analysis.get('room_type', 'unknown')}\n\n"
                f"IKEA Products recommended:\n{products_ctx}\n\n"
                f"Question: {query}"
            )},
        ],
    )

    answer = resp.choices[0].message.content.strip()
    cost = (
        resp.usage.prompt_tokens * 0.00000015 +
        resp.usage.completion_tokens * 0.0000006
    )

    print(f"\n  {answer}\n")

    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return {
        "conversation_history": history,
        "cumulative_cost_usd": state.get("cumulative_cost_usd", 0) + cost,
    }


# ── node 11: refine image ───────────────────────────────────────────────────

def refine_image(state: dict) -> dict:
    instruction = state.get("refinement_instruction", "")
    _log("refine", f"Refining: {instruction}")

    from generation.local_pipeline import request_img2img

    generated = state.get("generated_images", {})
    prompts = state.get("prompts_used", {})

    # refine the first selected style's depth variant
    selected = state.get("selected_styles", [])
    if not selected:
        return {"error": "No styles to refine"}

    style = selected[0]
    images = generated.get(style, {})
    source_path = images.get("depth") or images.get("text_only")

    if not source_path or not Path(source_path).exists():
        return {"error": f"No image to refine for {style}"}

    prompt_data = prompts.get(style, {})
    positive = prompt_data.get("positive", style)
    negative = prompt_data.get("negative", "")

    instruction_lower = instruction.lower()

    REMOVAL_TRIGGERS = ("remove ", "delete ", "take out ", "get rid of ", "no ", "without ")
    ADDITION_TRIGGERS = ("add ", "put ", "place ", "include ", "more ", "bring in ")

    is_removal = any(instruction_lower.startswith(t) or f" {t}" in instruction_lower
                     for t in REMOVAL_TRIGGERS)
    is_addition = any(instruction_lower.startswith(t) or f" {t}" in instruction_lower
                      for t in ADDITION_TRIGGERS)

    if is_removal:
        for trigger in REMOVAL_TRIGGERS:
            idx = instruction_lower.find(trigger)
            if idx != -1:
                item = instruction[idx + len(trigger):].strip().rstrip(".,!?")
                negative = f"{negative}, {item}" if negative else item
                break
        strength = 0.65
        _log("refine", f"Removal detected — strength=0.65, added '{item}' to negative")
    elif is_addition:
        strength = 0.55
        _log("refine", f"Addition detected — strength=0.55")
    else:
        strength = 0.4

    refined_path = request_img2img(
        source_image_path=source_path,
        positive_prompt=positive,
        negative_prompt=negative,
        refinement_instruction=instruction,
        strength=strength,
        seed=state.get("seed", 42) + 1,
    )

    # update generated images with refined version
    updated = {**generated}
    updated[style] = {**images, "refined": refined_path}

    _log("refine", f"Saved: {refined_path}")
    return {"generated_images": updated}
