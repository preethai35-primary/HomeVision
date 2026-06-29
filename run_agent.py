"""
run_agent.py — Interactive CLI for the HomeVision LangGraph agent.

# OMP_NUM_THREADS=1 must be set before any numpy/faiss import to prevent the
# macOS segfault that occurs when sentence-transformers encode() initialises
# OpenMP threads. Set here, before all other imports.

Usage:
  python run_agent.py data/my_room.jpg
  python run_agent.py data/my_room.jpg --styles "indian vintage" japandi
  python run_agent.py data/my_room.jpg --budget 500-2000
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from langgraph.types import Command


def main():
    parser = argparse.ArgumentParser(description="HomeVision — AI interior redesign agent")
    parser.add_argument("image", nargs="?", default="data/my_room.jpg",
                        help="Path to room photo")
    parser.add_argument("--styles", nargs="+", default=None,
                        help="Preferred styles e.g. --styles 'indian vintage' japandi")
    parser.add_argument("--budget", default="500-2000",
                        help="Budget range in EUR (default: 500-2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: {args.image} not found")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  HomeVision Agent")
    print(f"  Image:  {args.image}")
    print(f"  Styles: {args.styles or 'auto (GPT-4o suggests)'}")
    print(f"  Budget: {args.budget}")
    print("=" * 60 + "\n")

    # build initial state
    initial_state = {
        "image_path": args.image,
        "style_preferences": args.styles or [],
        "budget": args.budget,
        "focus_areas": ["furniture", "lighting", "color palette"],
        "user_message": "",
        "room_analysis": None,
        "depth_map_path": None,
        "reference_images": None,
        "ikea_products": None,
        "prompts_used": None,
        "generated_images": None,
        "active_variant": None,
        "lora_paths": None,
        "eval_scores": None,
        "style_rubric_scores": None,
        "selected_styles": None,
        "user_feedback": None,
        "user_intent": None,
        "refinement_instruction": None,
        "style_blend": None,
        "generation_mode": None,
        "ip_adapter_refs": None,
        "retry_count": 0,
        "seed": args.seed,
        "cumulative_cost_usd": 0.0,
        "design_explanation": None,
        "conversation_history": [],
        "guardrail_flags": None,
        "error": None,
        "trace_id": f"hv-agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    }

    # build and run graph with persistent SQLite checkpointer
    from agent.graph import compile_graph
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = "data/agent_state.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = compile_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": initial_state["trace_id"]}}

        # run until first interrupt or completion
        result = graph.invoke(initial_state, config)

        # handle interrupts in a loop
        while True:
            state = graph.get_state(config)

            if not state.next:
                break

            # get the interrupt payload
            tasks = state.tasks
            interrupt_data = None
            for task in tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_data = task.interrupts[0].value
                    break

            if interrupt_data is None:
                break

            interrupt_type = interrupt_data.get("type", "")
            message = interrupt_data.get("message", "Input:")

            print(f"\n  {message}")

            if interrupt_type == "style_selection":
                options = interrupt_data.get("options", [])
                print(f"  Options: {', '.join(options)}")

            user_input = input("\n  > ").strip()

            if not user_input:
                user_input = "all" if interrupt_type == "style_selection" else "done"

            result = graph.invoke(Command(resume=user_input), config)

        final_state = graph.get_state(config).values

    if final_state.get("error"):
        print(f"\nAgent stopped with error: {final_state['error']}")
        sys.exit(1)

    # save final state
    out_dir = Path("data/outputs/agent")
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "final_state.json"

    serializable = {
        k: v for k, v in final_state.items()
        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }
    state_path.write_text(json.dumps(serializable, indent=2, default=str))

    print("\n" + "=" * 60)
    print("  AGENT COMPLETE")
    print("=" * 60)
    print(f"  Cost:    ${final_state.get('cumulative_cost_usd', 0):.4f}")
    print(f"  State:   {state_path}")

    generated = final_state.get("generated_images", {})
    if generated:
        print(f"  Images:")
        for style, images in generated.items():
            for variant, path in images.items():
                if path:
                    print(f"    {style}/{variant}: {path}")

    products = final_state.get("ikea_products", [])
    if products:
        print(f"  Products: {len(products)} IKEA recommendations")

    rubric = final_state.get("style_rubric_scores", {})
    if rubric:
        for style, scores in rubric.items():
            print(f"  Rubric [{style}]: grade={scores.get('grade')} "
                  f"({scores.get('markers_found')}/{scores.get('markers_total')} markers)")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
