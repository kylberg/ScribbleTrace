"""Nested squares algorithm for ScribbleTrace.

This algorithm draws nested squares in each cell, where the
number of squares is based on the image intensity.
Inspired by the original scribbleSquare.py implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scribbletrace.algorithms.base import Algorithm, AlgorithmConfig
from scribbletrace.svg_output import PathSegment


@dataclass
class SquaresConfig(AlgorithmConfig):
    """Configuration for the Squares algorithm.

    Attributes:
        small_first: If True, draw smaller squares first (inside out).
    """

    small_first: bool = True
    randomness_vertex: float = 0.1  # Slight vertex randomness for organic look


class Squares(Algorithm):
    """Nested squares algorithm.

    Each image cell is rendered as a set of nested squares (or slightly
    irregular quadrilaterals), where the number of squares is proportional
    to the intensity. The vertex randomness gives an organic, hand-drawn feel.
    """

    def __init__(self, image, config: SquaresConfig | None = None, gradients=None):
        super().__init__(image, config, gradients)
        if config is None:
            self.config = SquaresConfig()

    @classmethod
    def _default_config(cls) -> SquaresConfig:
        return SquaresConfig()

    def _generate_square(
        self, cx: float, cy: float, size: float
    ) -> list[tuple[float, float]]:
        """Generate square vertices with optional randomness.

        Args:
            cx: Center x coordinate.
            cy: Center y coordinate.
            size: Square size (side length).

        Returns:
            List of (x, y) vertices forming a closed square.
        """
        cfg = self.config

        # Base square vertices (counterclockwise from bottom-left)
        base_x = np.array([0.0, 0.0, 1.0, 1.0])
        base_y = np.array([0.0, 1.0, 1.0, 0.0])

        # Add random vertex displacement
        if cfg.randomness_vertex > 0:
            base_x = base_x + np.random.uniform(
                -cfg.randomness_vertex, cfg.randomness_vertex, 4
            )
            base_y = base_y + np.random.uniform(
                -cfg.randomness_vertex, cfg.randomness_vertex, 4
            )

        # Scale and center
        vert_x = (base_x - 0.5) * size + cx
        vert_y = (base_y - 0.5) * size + cy

        # Create closed path
        vertices = [(vert_x[i], vert_y[i]) for i in range(4)]
        vertices.append(vertices[0])  # Close the square

        return vertices

    def _process_cell(self, row: int, col: int, value: int) -> list[PathSegment]:
        """Process a single cell and return its square paths.

        Args:
            row: Row index.
            col: Column index.
            value: Intensity value for this cell.

        Returns:
            List of PathSegment objects for this cell's squares.
        """
        cfg = self.config
        paths = []

        if value <= 0:
            return paths

        # Generate pixel sizes for each level
        pixel_sizes = np.linspace(cfg.min_element_size, cfg.max_element_size, self.levels)

        # Apply random variation
        if cfg.randomness_position > 0:
            pixel_sizes = pixel_sizes * np.random.uniform(
                1 - cfg.randomness_position,
                1 + cfg.randomness_position,
                self.levels,
            )

        # Order squares
        if not cfg.small_first:
            pixel_sizes = pixel_sizes[::-1]

        # Draw squares up to intensity value
        pixel_sizes = pixel_sizes[:value]

        for size in pixel_sizes:
            vertices = self._generate_square(col, row, size)
            paths.append(
                PathSegment(
                    points=vertices,
                    closed=True,
                    stroke_width=cfg.stroke_width,
                )
            )

        return paths

    def process(self) -> list[PathSegment]:
        """Process the image and generate square paths.

        Returns:
            List of PathSegment objects.
        """
        paths = []

        for c in range(self.width):
            for r in range(self.height):
                value = self.get_value(r, c)
                cell_paths = self._process_cell(r, c, value)
                paths.extend(cell_paths)

        return paths
