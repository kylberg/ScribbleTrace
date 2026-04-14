"""Command-line interface for ScribbleTrace.

This module provides a CLI for converting images to vector drawings.

Usage:
    scribbletrace input.jpg output.svg --algorithm spirals
    scribbletrace input.png output.svg --algorithm hatching --width 60 --levels 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scribbletrace import (
    load_image,
    preprocess,
    compute_gradients,
    SVGWriter,
)
from scribbletrace.algorithms import (
    Spirals,
    SpiralsConfig,
    Circles,
    CirclesConfig,
    Squares,
    SquaresConfig,
    Lines,
    LinesConfig,
    Curves,
    CurvesConfig,
    Hatching,
    HatchingConfig,
    HatchDirection,
)


ALGORITHMS = {
    "spirals": (Spirals, SpiralsConfig),
    "circles": (Circles, CirclesConfig),
    "squares": (Squares, SquaresConfig),
    "lines": (Lines, LinesConfig),
    "curves": (Curves, CurvesConfig),
    "hatching": (Hatching, HatchingConfig),
}


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="scribbletrace",
        description="Convert images to vector drawings for pen plotters",
        epilog="Example: scribbletrace photo.jpg drawing.svg --algorithm spirals",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input image file (JPEG, PNG, etc.)",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output SVG file",
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        choices=list(ALGORITHMS.keys()),
        default="spirals",
        help="Drawing algorithm to use (default: spirals)",
    )

    parser.add_argument(
        "-w",
        "--width",
        type=float,
        default=40.0,
        help="Output width in cells/units (default: 40)",
    )

    parser.add_argument(
        "-l",
        "--levels",
        type=int,
        default=7,
        help="Number of intensity quantization levels (default: 7)",
    )

    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Don't invert image (by default, dark areas produce more marks)",
    )

    parser.add_argument(
        "--stroke-width",
        type=float,
        default=0.5,
        help="SVG stroke width in mm (default: 0.5)",
    )

    parser.add_argument(
        "--randomness",
        type=float,
        default=0.1,
        help="Randomness for vertex positions (default: 0.1)",
    )

    # Hatching-specific options
    parser.add_argument(
        "--hatch-directions",
        nargs="+",
        choices=["horizontal", "vertical", "diagonal_right", "diagonal_left"],
        default=["diagonal_right"],
        help="Hatching directions (for hatching algorithm)",
    )

    parser.add_argument(
        "--cross-hatch",
        action="store_true",
        help="Use cross-hatching (diagonal_right + diagonal_left)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 2.0.0",
    )

    return parser


def get_hatch_directions(args) -> list[HatchDirection]:
    """Get hatching directions from arguments.

    Args:
        args: Parsed arguments.

    Returns:
        List of HatchDirection enums.
    """
    if args.cross_hatch:
        return [HatchDirection.DIAGONAL_RIGHT, HatchDirection.DIAGONAL_LEFT]

    direction_map = {
        "horizontal": HatchDirection.HORIZONTAL,
        "vertical": HatchDirection.VERTICAL,
        "diagonal_right": HatchDirection.DIAGONAL_RIGHT,
        "diagonal_left": HatchDirection.DIAGONAL_LEFT,
    }

    return [direction_map[d] for d in args.hatch_directions]


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        if args.verbose:
            print(f"Loading image: {args.input}")

        # Load and preprocess image
        image = load_image(args.input)

        if args.verbose:
            print(f"  Original size: {image.shape[1]}x{image.shape[0]}")

        processed = preprocess(
            image,
            output_width=args.width,
            levels=args.levels,
            invert=not args.no_invert,
        )

        if args.verbose:
            print(f"  Processed size: {processed.width}x{processed.height}")
            print(f"  Quantization levels: {processed.levels}")

        # Get algorithm class and config
        algo_class, config_class = ALGORITHMS[args.algorithm]

        # Check if algorithm needs gradients
        needs_gradients = args.algorithm in ["lines", "curves"]

        gradients = None
        if needs_gradients:
            if args.verbose:
                print("Computing image gradients...")
            gradients = compute_gradients(processed.original)

        # Create config
        config_kwargs = {
            "stroke_width": args.stroke_width,
            "randomness_vertex": args.randomness,
        }

        # Add hatching-specific config
        if args.algorithm == "hatching":
            config_kwargs["directions"] = get_hatch_directions(args)

        config = config_class(**config_kwargs)

        if args.verbose:
            print(f"Using algorithm: {args.algorithm}")

        # Create and run algorithm
        algorithm = algo_class(processed, config=config, gradients=gradients)
        paths = algorithm.process()

        if args.verbose:
            print(f"Generated {len(paths)} paths")

        # Create SVG writer
        svg_width, svg_height = algorithm.get_svg_dimensions()
        writer = SVGWriter(paths, width=svg_width, height=svg_height)

        # Save output
        writer.save(args.output)

        if args.verbose:
            print(f"Saved to: {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
