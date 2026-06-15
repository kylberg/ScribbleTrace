"""Gradient-oriented lines algorithm for ScribbleTrace.

This algorithm draws short lines in each cell, oriented perpendicular
to the local gradient direction. Line length is based on gradient
magnitude, and the number of lines is based on intensity.
Inspired by the original scribbleLines.py implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scribbletrace.algorithms.base import Algorithm, AlgorithmConfig, rotate_point
from scribbletrace.svg_output import PathSegment


@dataclass
class LinesConfig(AlgorithmConfig):
    """Configuration for the Lines algorithm.

    Attributes:
        segment_length: Base multiplier for line segment length.
        randomness_length: Amount of random variation in line length.
        min_gradient_scale: Minimum line scale based on gradient.
        max_gradient_scale: Maximum line scale based on gradient.
    """

    randomness_vertex: float = 0.1
    randomness_position: float = 0.5
    segment_length: float = 1.0
    randomness_length: float = 0.0
    min_gradient_scale: float = 0.1
    max_gradient_scale: float = 10.0


class Lines(Algorithm):
    """Gradient-oriented lines algorithm.

    Each image cell is rendered with short lines that are oriented
    perpendicular to the local gradient direction. The line length
    is scaled by the gradient magnitude (strong edges = longer lines),
    and the number of lines per cell is proportional to intensity.

    Requires gradient data to be provided.
    """

    def __init__(self, image, config: LinesConfig | None = None, gradients=None):
        super().__init__(image, config, gradients)
        if config is None:
            self.config = LinesConfig()

    @classmethod
    def _default_config(cls) -> LinesConfig:
        return LinesConfig()

    def _generate_line(
        self,
        cx: float,
        cy: float,
        angle: float,
        length: float,
    ) -> list[tuple[float, float]]:
        """Generate a line centered at a point with given angle and length.

        Args:
            cx: Center x coordinate.
            cy: Center y coordinate.
            angle: Rotation angle in radians.
            length: Line length.

        Returns:
            List of two (x, y) vertices forming the line.
        """
        cfg = self.config
        half_len = length / 2

        # Base line endpoints (horizontal)
        x1, y1 = -half_len, 0
        x2, y2 = half_len, 0

        # Add vertex randomness
        if cfg.randomness_vertex > 0:
            x1 += self.random_offset(cfg.randomness_vertex)
            y1 += self.random_offset(cfg.randomness_vertex)
            x2 += self.random_offset(cfg.randomness_vertex)
            y2 += self.random_offset(cfg.randomness_vertex)

        # Rotate to gradient direction
        x1, y1 = rotate_point(x1, y1, angle)
        x2, y2 = rotate_point(x2, y2, angle)

        # Apply position randomness and translate to cell center
        pos_offset_x = self.random_offset(cfg.randomness_position)
        pos_offset_y = self.random_offset(cfg.randomness_position)

        return [
            (cx + x1 + pos_offset_x, cy + y1 + pos_offset_y),
            (cx + x2 + pos_offset_x, cy + y2 + pos_offset_y),
        ]

    def _process_cell(
        self,
        row: int,
        col: int,
        value: int,
        angle: float,
        grad_mag: float,
    ) -> list[PathSegment]:
        """Process a single cell and return its line paths.

        Args:
            row: Row index.
            col: Column index.
            value: Intensity value for this cell.
            angle: Gradient angle at this cell.
            grad_mag: Gradient magnitude at this cell.

        Returns:
            List of PathSegment objects for this cell's lines.
        """
        cfg = self.config
        paths = []

        if value <= 0:
            return paths

        # Scale line length by gradient magnitude
        length = cfg.segment_length * max(cfg.min_gradient_scale, grad_mag * cfg.max_gradient_scale)

        # Apply length randomness
        if cfg.randomness_length > 0:
            length *= self.random_scale(1.0, cfg.randomness_length)

        # Draw multiple lines based on intensity
        for _ in range(value):
            vertices = self._generate_line(col, row, angle, length)
            paths.append(
                PathSegment(
                    points=vertices,
                    closed=False,
                    stroke_width=cfg.stroke_width,
                )
            )

        return paths

    def process(self) -> list[PathSegment]:
        """Process the image and generate line paths.

        Returns:
            List of PathSegment objects.

        Raises:
            ValueError: If gradient data is not provided.
        """
        if self.gradients is None:
            raise ValueError("Lines algorithm requires gradient data")

        paths = []

        for c in range(self.width):
            for r in range(self.height):
                value = self.get_value(r, c)

                # Get gradient info at this position
                dx = self.gradients.dx[r, c]
                dy = self.gradients.dy[r, c]
                grad_angle = np.arctan2(dy, dx)
                # Draw along image contours (isophotes): perpendicular to gradient.
                angle = grad_angle - np.pi / 2
                grad_mag = dx * dx + dy * dy

                cell_paths = self._process_cell(r, c, value, angle, grad_mag)
                paths.extend(cell_paths)

        return paths
