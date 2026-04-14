"""ScribbleTrace - Convert images to vector drawings for pen plotters.

This package provides algorithms to transform raster images into vector
graphics suitable for pen plotters like the Axidraw. Various artistic
styles are supported including spirals, circles, squares, lines, curves,
and hatching patterns.

Example usage:
    from scribbletrace import load_image, preprocess
    from scribbletrace.algorithms import Spirals
    from scribbletrace.svg_output import SVGWriter

    # Load and preprocess image
    img = load_image("photo.jpg")
    processed = preprocess(img, output_width=40, levels=7)

    # Apply algorithm
    algorithm = Spirals(processed)
    paths = algorithm.process()

    # Export to SVG
    writer = SVGWriter(paths, width=200, height=150)
    writer.save("output.svg")
"""

from scribbletrace.image_processing import load_image, preprocess, compute_gradients
from scribbletrace.svg_output import SVGWriter
from scribbletrace.algorithms import (
    Spirals,
    Circles,
    Squares,
    Lines,
    Curves,
    Hatching,
)

__version__ = "2.0.0"
__author__ = "Gustaf Kylberg"
__all__ = [
    "load_image",
    "preprocess",
    "compute_gradients",
    "SVGWriter",
    "Spirals",
    "Circles",
    "Squares",
    "Lines",
    "Curves",
    "Hatching",
]
