"""Concentric circles algorithm for ScribbleTrace.

This algorithm draws concentric circles in each cell, where the
number of circles is based on the image intensity.
Inspired by the original scribbleCircle.py implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scribbletrace.algorithms.base import Algorithm, AlgorithmConfig
from scribbletrace.svg_output import PathSegment


@dataclass
class CirclesConfig(AlgorithmConfig):
    """Configuration for the Circles algorithm.

    Attributes:
        small_first: If True, draw smaller circles first (inside out).
        circle_points: Number of points used to approximate each circle.
    """

    small_first: bool = True
    circle_points: int = 36


class Circles(Algorithm):
    """Concentric circles algorithm.

    Each image cell is rendered as a set of concentric circles, where
    the number of circles is proportional to the intensity. This creates
    a halftone-like effect with circular patterns.
    """

    def __init__(self, image, config: CirclesConfig | None = None, gradients=None):
        super().__init__(image, config, gradients)
        if config is None:
            self.config = CirclesConfig()

    @classmethod
    def _default_config(cls) -> CirclesConfig:
        return CirclesConfig()

    def _generate_circle(
        self, cx: float, cy: float, radius: float
    ) -> list[tuple[float, float]]:
        """Generate circle vertices.

        Args:
            cx: Center x coordinate.
            cy: Center y coordinate.
            radius: Circle radius.

        Returns:
            List of (x, y) vertices forming a closed circle.
        """
        n_points = self.config.circle_points
        angles = np.linspace(0, 2 * np.pi, n_points + 1)
        return [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]

    def _process_cell(self, row: int, col: int, value: int) -> list[PathSegment]:
        """Process a single cell and return its circle paths.

        Args:
            row: Row index.
            col: Column index.
            value: Intensity value for this cell.

        Returns:
            List of PathSegment objects for this cell's circles.
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

        # Order circles
        if not cfg.small_first:
            pixel_sizes = pixel_sizes[::-1]

        # Draw circles up to intensity value
        pixel_sizes = pixel_sizes[:value]

        for size in pixel_sizes:
            radius = size / 2
            vertices = self._generate_circle(col, row, radius)
            paths.append(
                PathSegment(
                    points=vertices,
                    closed=True,
                    stroke_width=cfg.stroke_width,
                )
            )

        return paths

    def process(self) -> list[PathSegment]:
        """Process the image and generate circle paths.

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
