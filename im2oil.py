"""Im2Oil: Stroke-based oil painting with configurable palette and brush kit.

Subcommands:
    analyze  - Extract palette and brush widths from an image -> brush kit JSON
    segment  - Semantic layer decomposition (objects, depth, inpainting)
    paint    - Run painting pipeline with a brush kit -> strokes + result
    view     - Generate interactive HTML stroke viewer from output
    run      - Run all steps in sequence
"""

import argparse

from pipeline import cmd_analyze, cmd_paint, cmd_view, cmd_run
from segmentation import cmd_segment


def main():
    parser = argparse.ArgumentParser(
        description="Im2Oil: Stroke-based oil painting pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- analyze ---
    p_analyze = subparsers.add_parser(
        "analyze", help="Analyze image and produce a brush kit JSON"
    )
    p_analyze.add_argument("image", help="Input image path")
    p_analyze.add_argument("--palette-size", type=int, default=12, help="Number of palette colors (default: 12)")
    p_analyze.add_argument("--num-brushes", type=int, default=5, help="Number of discrete brush widths (default: 5)")
    p_analyze.add_argument("--p-max", type=int, default=4, help="Sampling rate reciprocal (default: 4)")
    p_analyze.add_argument("--ssaa", type=int, default=8, help="Super-sampling AA factor (default: 8)")
    p_analyze.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p_analyze.add_argument("--ratio", type=int, default=3, help="max_length/max_width ratio (default: 3)")
    p_analyze.add_argument("--brush", type=str, default="./brush/brush-0.png", help="Brush template path")
    p_analyze.add_argument("-o", "--output", type=str, default=None, help="Output JSON path")

    # --- segment ---
    p_segment = subparsers.add_parser(
        "segment", help="Semantic layer decomposition (objects, depth, inpainting)"
    )
    p_segment.add_argument("image", help="Input image path")
    p_segment.add_argument("kit", help="Path to brush kit JSON")
    p_segment.add_argument("-o", "--output", type=str, default=None, help="Output layers JSON path")

    # --- paint ---
    p_paint = subparsers.add_parser(
        "paint", help="Run painting pipeline with a brush kit"
    )
    p_paint.add_argument("image", help="Input image path")
    p_paint.add_argument("kit", help="Path to brush kit JSON")
    p_paint.add_argument("--layers", type=str, default=None, help="Path to layers JSON (from segment)")
    p_paint.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    p_paint.add_argument("--freq", type=int, default=100, help="Save frame every N strokes (default: 100)")
    p_paint.add_argument("--force", action="store_true", default=True, help="Force recompute anchor map")

    # --- view ---
    p_view = subparsers.add_parser(
        "view", help="Generate HTML stroke viewer from output directory"
    )
    p_view.add_argument("output_dir", help="Path to output directory")

    # --- run ---
    p_run = subparsers.add_parser(
        "run", help="Run full pipeline (analyze + segment + paint + view)"
    )
    p_run.add_argument("image", help="Input image path")
    p_run.add_argument("--palette-size", type=int, default=12, help="Number of palette colors (default: 12)")
    p_run.add_argument("--num-brushes", type=int, default=5, help="Number of discrete brush widths (default: 5)")
    p_run.add_argument("--p-max", type=int, default=4, help="Sampling rate reciprocal (default: 4)")
    p_run.add_argument("--ssaa", type=int, default=8, help="Super-sampling AA factor (default: 8)")
    p_run.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p_run.add_argument("--ratio", type=int, default=3, help="max_length/max_width ratio (default: 3)")
    p_run.add_argument("--brush", type=str, default="./brush/brush-0.png", help="Brush template path")
    p_run.add_argument("--freq", type=int, default=100, help="Save frame every N strokes (default: 100)")
    p_run.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    commands = {
        "analyze": cmd_analyze,
        "segment": cmd_segment,
        "paint": cmd_paint,
        "view": cmd_view,
        "run": cmd_run,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
