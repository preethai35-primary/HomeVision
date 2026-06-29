"""
ui/gradio_app.py -- Gradio web interface for HomeVision

Run:
  python ui/gradio_app.py
  python ui/gradio_app.py --share   # public URL (runs on your machine)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Must be set before any numpy/faiss import to prevent the macOS OMP segfault.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr


# ── State helpers ──────────────────────────────────────────────────────────────

def _empty_state():
    return dict(
        image_path=None,
        style_preferences=[],
        budget="500-2000",
        focus_areas=["furniture", "lighting", "color palette"],
        user_message="",
        room_analysis=None,
        depth_map_path=None,
        reference_images=None,
        ikea_products=None,
        prompts_used=None,
        generated_images=None,
        active_variant=None,
        lora_paths=None,
        eval_scores=None,
        style_rubric_scores=None,
        selected_styles=None,
        user_feedback=None,
        user_intent=None,
        refinement_instruction=None,
        style_blend=None,
        generation_mode=None,
        ip_adapter_refs=None,
        retry_count=0,
        seed=42,
        cumulative_cost_usd=0.0,
        design_explanation=None,
        conversation_history=[],
        guardrail_flags=None,
        error=None,
        trace_id=f"hv-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )


# ── Display helpers ────────────────────────────────────────────────────────────

def _products_html(products):
    if not products:
        return "<p style='color:#999;padding:8px'>No products found.</p>"
    html = "<div style='display:flex;flex-direction:column;gap:8px'>"
    for p in products:
        name   = p.get("name", "")
        desc   = (p.get("description", "") or "")[:100]
        price  = p.get("price", "")
        url    = p.get("url", "")
        cat    = p.get("category", "")
        reason = p.get("match_reason", "")
        html += (
            f"<div style='border:1px solid #ddd;border-radius:6px;padding:10px;background:#fafafa'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
            f"<b style='font-size:0.95em'>{name}</b>"
            f"{'<span style=\"background:#e8f0fe;color:#1a56db;font-size:0.75em;padding:2px 7px;border-radius:10px;margin-left:8px;white-space:nowrap\">' + reason + '</span>' if reason else ''}"
            f"</div>"
            f"<small style='color:#666;display:block;margin:4px 0'>{desc}{'...' if desc else ''}</small>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:6px'>"
            f"<b style='color:#0058a3;font-size:1.05em'>{price}</b>"
            f"<a href='{url}' target='_blank' "
            f"style='background:#0058a3;color:#fff;padding:4px 12px;border-radius:4px;"
            f"text-decoration:none;font-size:0.82em;font-weight:600'>View on IKEA</a>"
            f"</div>"
            f"<small style='color:#aaa'>{cat}</small>"
            f"</div>"
        )
    html += "</div>"
    return html


def _scores_md(state):
    scores = state.get("eval_scores") or {}
    rubric = state.get("style_rubric_scores") or {}
    styles = state.get("selected_styles") or []
    lines = []
    for style in styles:
        s = scores.get(style, {})
        for variant, sc in s.get("scores", {}).items():
            clip = sc.get("clip_score", "?")
            layout = sc.get("layout_score", "?")
            style_s = sc.get("style_score", "?")
            lines.append(f"**{variant}** -- CLIP {clip} | Layout {layout}/10 | Style {style_s}/10")
        if style in rubric:
            r = rubric[style]
            lines.append(
                f"Rubric {r.get('markers_found')}/{r.get('markers_total')} markers"
                f" (Grade {r.get('grade')})"
            )
    return "  \n".join(lines)


def _analysis_md(analysis, ref_images, styles):
    if not analysis:
        return ""
    lines = [
        f"**Room:** {analysis.get('room_type', '?')} -- **Style:** {analysis.get('style', '?')}",
        f"**Light:** {analysis.get('natural_light', '?')} -- **Mood:** {analysis.get('color_mood', '?')}",
    ]
    opp = analysis.get("opportunities", [])
    if opp:
        lines.append("**Opportunities:** " + " | ".join(opp[:2]))
    if styles and ref_images:
        sim_lines = []
        for st in styles:
            r = (ref_images or {}).get(st, [])
            sim = round(r[0]["similarity"], 3) if r else 0
            quality = "strong" if sim > 0.26 else "good" if sim > 0.22 else "moderate"
            sim_lines.append(f"- **{st}** ({quality} match, sim {sim})")
        lines.append("\n**Suggested styles (select below):**\n" + "\n".join(sim_lines))
    return "\n\n".join(lines)


def _gallery_items(ref_images, styles):
    """Build (path, caption) list for the style reference gallery."""
    items = []
    for style in styles:
        refs = (ref_images or {}).get(style, [])
        for ref in refs[:3]:
            p = ref.get("path", "")
            if p and Path(p).exists():
                sim = ref.get("similarity", 0)
                items.append((p, f"{style}  {sim:.3f}"))
    return items


# ── Step 1: Analyze ────────────────────────────────────────────────────────────

def do_analyze(image, budget, styles_text, state):
    s = _empty_state()

    if image is None:
        return (
            s,
            "Upload a room photo first.",
            None,
            gr.update(choices=[], value=[]),  # style_choices
            gr.update(choices=[]),             # blend_with dropdown
            [],                                # style gallery
            gr.update(visible=False),          # style_section
            gr.update(visible=False),          # results_section
            "Cost: $0.0000",
        )

    s["image_path"] = image
    s["budget"] = (budget or "500-2000").strip()
    s["style_preferences"] = [x.strip() for x in (styles_text or "").split(",") if x.strip()]

    from agent.tools import (
        validate_input, analyze_room, compute_depth_map,
        suggest_styles, retrieve_examples,
    )

    s.update(validate_input(s))
    if s.get("error"):
        return (
            s,
            f"Error: {s['error']}",
            None,
            gr.update(choices=[], value=[]),
            gr.update(choices=[]),
            [],
            gr.update(visible=False),
            gr.update(visible=False),
            f"Cost: ${s.get('cumulative_cost_usd', 0):.4f}",
        )

    s.update(analyze_room(s))
    s.update(compute_depth_map(s))
    s.update(suggest_styles(s))
    s.update(retrieve_examples(s))

    styles = s.get("style_preferences", [])
    ref_images = s.get("reference_images", {})
    gallery = _gallery_items(ref_images, styles)
    analysis_text = _analysis_md(s.get("room_analysis"), ref_images, styles)

    return (
        s,
        analysis_text,
        s.get("depth_map_path"),
        gr.update(choices=styles, value=[]),
        gr.update(choices=styles),
        gallery,
        gr.update(visible=True),
        gr.update(visible=False),
        f"Cost: ${s['cumulative_cost_usd']:.4f}",
    )


# ── Step 2: Generate ──────────────────────────────────────────────────────────

def do_generate(selected_styles, custom_style, blend_with, blend_toggle, state, progress=gr.Progress()):
    s = dict(state)

    custom = (custom_style or "").strip()

    # resolve what to generate
    if custom and blend_with:
        # user typed a style AND picked one to blend with
        s["style_blend"] = {"styles": [custom, blend_with]}
        s["selected_styles"] = [f"{custom} x {blend_with}"]
    elif custom:
        # custom style only, no blend
        s["style_blend"] = None
        s["selected_styles"] = [custom]
    elif blend_toggle and len(selected_styles) >= 2:
        # blend two checked suggestions
        s["style_blend"] = {"styles": list(selected_styles[:2])}
        s["selected_styles"] = [f"{selected_styles[0]} x {selected_styles[1]}"]
    elif selected_styles:
        s["style_blend"] = None
        s["selected_styles"] = list(selected_styles[:1])
    else:
        return (
            s, None, None,
            "Select at least one style or enter a custom style.",
            "<p style='color:#888'>Generate first.</p>",
            gr.update(visible=False),
            f"Cost: ${s.get('cumulative_cost_usd', 0):.4f}",
        )

    from agent.tools import route_generation, search_products, generate_images, evaluate_results

    progress(0.05, desc="Routing...")
    s.update(route_generation(s))

    progress(0.15, desc="Searching IKEA catalogue...")
    s.update(search_products(s))

    progress(0.35, desc="Generating image (60-90s)...")
    s.update(generate_images(s))

    progress(0.85, desc="Evaluating...")
    s.update(evaluate_results(s))

    progress(1.0, desc="Done!")

    generated = s.get("generated_images", {})
    style = (s.get("selected_styles") or [""])[0]
    images = generated.get(style, {})
    after_path = images.get("depth") or images.get("text_only")

    return (
        s,
        s["image_path"],
        after_path,
        _scores_md(s),
        _products_html(s.get("ikea_products", [])),
        gr.update(visible=True),
        f"Cost: ${s['cumulative_cost_usd']:.4f}",
    )


# ── Step 3: Chat (refine or info) ─────────────────────────────────────────────

def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def do_chat(message, chat_history, state):
    s = dict(state)
    message = (message or "").strip()
    if not message:
        return s, chat_history, gr.update(), _scores_md(s), ""

    s["refinement_instruction"] = message

    from agent.tools import classify_intent, handle_info, refine_image, evaluate_results

    s.update(classify_intent(s))
    intent = s.get("user_intent", "info")

    if intent == "done":
        history = chat_history + [_msg("user", message), _msg("assistant", "Done! Your design has been saved.")]
        return s, history, gr.update(), _scores_md(s), ""

    if intent == "info":
        s.update(handle_info(s))
        hist = s.get("conversation_history", [])
        answer = hist[-1]["content"] if hist and hist[-1]["role"] == "assistant" else "Done."
        history = chat_history + [_msg("user", message), _msg("assistant", answer)]
        return s, history, gr.update(), _scores_md(s), ""

    # refine — split multi-sentence instructions; use only the first per pass
    import re
    parts = [p.strip() for p in re.split(r"[.!?]+", message) if p.strip()]
    first = parts[0]
    remainder = ". ".join(parts[1:]) if len(parts) > 1 else ""

    s["refinement_instruction"] = first

    if remainder:
        notice = (
            f"SDXL works best with one change at a time (77-token prompt limit). "
            f"Applying: **\"{first}\"**\n\n"
            f"Send next: *\"{remainder}\"*"
        )
    else:
        notice = f"Applying: **\"{first}\"**"

    history = chat_history + [_msg("user", message), _msg("assistant", notice)]
    s.update(refine_image(s))
    s.update(evaluate_results(s))

    generated = s.get("generated_images", {})
    style = (s.get("selected_styles") or [""])[0]
    images = generated.get(style, {})
    refined_path = images.get("refined") or images.get("depth")

    # resolve to absolute path so Gradio's file server can locate it
    # regardless of which worker thread serves the response
    if refined_path:
        refined_path = str(Path(refined_path).resolve())

    scores = _scores_md(s)
    history[-1] = _msg("assistant", f"{notice}\n\nDone. {scores}")

    return s, history, refined_path, scores, ""


# ── UI layout ──────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="HomeVision") as app:
        gr.Markdown("# HomeVision -- AI Interior Redesign")

        state = gr.State(_empty_state())

        with gr.Row():

            # Left: controls
            with gr.Column(scale=1, min_width=300):

                gr.Markdown("### 1. Analyze Room")
                image_input = gr.Image(type="filepath", label="Room photo")
                budget_input = gr.Textbox(value="500-2000", label="Budget (EUR range)")
                styles_hint = gr.Textbox(
                    label="Style hints (optional)",
                    placeholder="e.g. boho, scandinavian",
                )
                analyze_btn = gr.Button("Analyze Room", variant="primary")

                style_section = gr.Group(visible=False)
                with style_section:
                    gr.Markdown("### 2. Select Style")
                    style_choices = gr.CheckboxGroup(
                        label="Suggested styles (check to select)",
                        choices=[],
                    )
                    blend_toggle = gr.Checkbox(
                        label="Blend two checked styles",
                        value=False,
                    )

                    with gr.Accordion("Or add your own style", open=False):
                        gr.Markdown(
                            "_Type any style not in the list above._  \n"
                            "_Leave 'Blend with' empty to generate it solo._"
                        )
                        custom_style_input = gr.Textbox(
                            label="Custom style",
                            placeholder="e.g. Kerala traditional, wabi-sabi, art nouveau...",
                        )
                        blend_with_input = gr.Dropdown(
                            label="Blend custom style with suggestion",
                            choices=[],
                            value=None,
                        )

                    generate_btn = gr.Button("Generate Design", variant="primary")

                cost_display = gr.Markdown("Cost: $0.0000")

            # Right: outputs
            with gr.Column(scale=2):

                analysis_display = gr.Markdown()
                depth_display = gr.Image(label="Depth map", height=180)

                style_gallery = gr.Gallery(
                    label="Style reference images",
                    columns=3,
                    height=220,
                    visible=False,
                    show_label=True,
                    object_fit="cover",
                )

                results_section = gr.Group(visible=False)
                with results_section:

                    gr.Markdown("### Results")
                    with gr.Row():
                        before_img = gr.Image(label="Before", height=340)
                        after_img  = gr.Image(label="After",  height=340)

                    scores_display = gr.Markdown()

                    with gr.Accordion("IKEA Product Recommendations", open=True):
                        products_display = gr.HTML()

                    gr.Markdown("### Refine or Ask")
                    gr.Markdown(
                        "_Examples: 'make it warmer' &nbsp;|&nbsp;"
                        " 'show me the product links' &nbsp;|&nbsp; 'done'_"
                    )
                    chatbot = gr.Chatbot(height=260, show_label=False)
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Make it warmer | Show links | done",
                            show_label=False,
                            scale=5,
                        )
                        send_btn = gr.Button("Send", scale=1, variant="secondary")

        # Event wiring

        analyze_btn.click(
            do_analyze,
            inputs=[image_input, budget_input, styles_hint, state],
            outputs=[
                state,
                analysis_display,
                depth_display,
                style_choices,
                blend_with_input,
                style_gallery,
                style_section,
                results_section,
                cost_display,
            ],
        ).then(
            # show gallery only if items were returned
            lambda items: gr.update(visible=len(items) > 0),
            inputs=[style_gallery],
            outputs=[style_gallery],
        )

        generate_btn.click(
            do_generate,
            inputs=[style_choices, custom_style_input, blend_with_input, blend_toggle, state],
            outputs=[
                state,
                before_img,
                after_img,
                scores_display,
                products_display,
                results_section,
                cost_display,
            ],
        )

        def _send(msg, history, s):
            s_new, hist, img, scores, _ = do_chat(msg, history, s)
            return s_new, hist, img if img else gr.update(), scores, ""

        send_btn.click(
            _send,
            inputs=[chat_input, chatbot, state],
            outputs=[state, chatbot, after_img, scores_display, chat_input],
        )
        chat_input.submit(
            _send,
            inputs=[chat_input, chatbot, state],
            outputs=[state, chatbot, after_img, scores_display, chat_input],
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link (runs on your machine)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_ui()
    outputs_dir = str(Path("data/outputs").resolve())
    app.launch(
        share=args.share,
        server_port=args.port,
        theme=gr.themes.Soft(),
        allowed_paths=[outputs_dir],
    )
