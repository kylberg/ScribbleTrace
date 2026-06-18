"""Vectorization algorithms for ScribbleTrace.

This package contains various algorithms for converting raster images
to vector paths suitable for pen plotters.

Available algorithms:
    - Spirals: Archimedean spirals based on intensity
    - Circles: Concentric circles based on intensity
    - Squares: Nested squares with optional randomness
    - Lines: Gradient-oriented lines
    - Curves: Bézier curves following gradients
    - Hatching: Cross-hatching patterns based on intensity
"""

from scribbletrace.algorithms.base import Algorithm, AlgorithmConfig
from scribbletrace.algorithms.circles import Circles, CirclesConfig
from scribbletrace.algorithms.curves import Curves, CurvesConfig
from scribbletrace.algorithms.hatching import HatchDirection, Hatching, HatchingConfig
from scribbletrace.algorithms.lines import Lines, LinesConfig
from scribbletrace.algorithms.spirals import Spirals, SpiralsConfig
from scribbletrace.algorithms.squares import Squares, SquaresConfig

__all__ = [
    "Algorithm",
    "AlgorithmConfig",
    "Spirals",
    "SpiralsConfig",
    "Circles",
    "CirclesConfig",
    "Squares",
    "SquaresConfig",
    "Lines",
    "LinesConfig",
    "Curves",
    "CurvesConfig",
    "Hatching",
    "HatchingConfig",
    "HatchDirection",
]
