"""Archimedean spiral algorithm for ScribbleTrace.

This algorithm draws connected Archimedean spirals in each cell,
where the spiral size/complexity is based on the image intensity.
Inspired by the original ArchimedeanSpiral.py implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scribbletrace.algorithms.base import Algorithm, AlgorithmConfig
from scribbletrace.svg_output import PathSegment


@dataclass
class SpiralsConfig(AlgorithmConfig):
    """Configuration for the Spirals algorithm.

    Attributes:
        theta_resolution: Number of points per 2π rotation.
        spiral_a: Inner radius parameter (typically 0).
        spiral_b: Spiral growth rate parameter.
        normalize_spirals: If True, normalize spiral size to fit cell.
        connect_cells: If True, draw connecting lines between cells.
    """

    theta_resolution: int = 50
    spiral_a: float = 0.0
    spiral_b: float = 1.0
    normalize_spirals: bool = True
    connect_cells: bool = True


class Spirals(Algorithm):
    """Archimedean spiral algorithm.

    Each image cell is rendered as an Archimedean spiral, where the
    number of turns is proportional to the intensity. Cells can be
    connected horizontally for efficient pen plotter drawing.

    The spiral is defined by:
        r = a + b * θ

    Where a is the inner radius (typically 0) and b controls the
    growth rate.
    """

    def __init__(self, image, config: SpiralsConfig | None = None, gradients=None):
        super().__init__(image, config, gradients)
        if config is None:
            self.config = SpiralsConfig()

    @classmethod
    def _default_config(cls) -> SpiralsConfig:
        return SpiralsConfig()

    def _vertices_on_spiral(self, theta_range: float) -> list[tuple[float, float]]:
        """Generate vertices along an Archimedean spiral.

        Args:
            theta_range: Total angle range in radians.

        Returns:
            List of (x, y) vertex coordinates.
        """
        cfg = self.config
        n_vertices = max(
            3, round(cfg.theta_resolution * theta_range / (2 * np.pi))
        )

        vertices = []
        thetas = np.linspace(-theta_range, theta_range, n_vertices)

        for theta in thetas:
            if theta > 0:
                radius = cfg.spiral_a + cfg.spiral_b * theta
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
            else:
                radius = cfg.spiral_a + cfg.spiral_b * (-theta)
                x = -radius * np.cos(-theta)
                y = -radius * np.sin(-theta)
            vertices.append((x, y))

        return vertices

    def _vertices_on_sine(self) -> list[tuple[float, float]]:
        """Generate a sine wave for zero-intensity cells.

        Returns:
            List of (x, y) vertex coordinates.
        """
        n_vertices = self.config.theta_resolution
        thetas = np.linspace(0, 2 * np.pi, n_vertices)
        y = np.sin(thetas) * 0.1

        vertices = []
        x_vals = np.linspace(-0.5, 0.5, n_vertices)
        for i in range(n_vertices):
            vertices.append((x_vals[i], y[i]))

        return vertices

    def _process_cell(self, row: int, col: int, value: int) -> list[tuple[float, float]]:
        """Process a single cell and return its path vertices.

        Args:
            row: Row index.
            col: Column index.
            value: Intensity value for this cell.

        Returns:
            List of (x, y) vertices for this cell's path.
        """
        cfg = self.config

        if value > 0:
            theta_range = value * np.pi
            vertices = self._vertices_on_spiral(theta_range)

            # Normalize spiral to fit in cell
            if vertices and cfg.normalize_spirals:
                max_coord = max(max(abs(x), abs(y)) for x, y in vertices)
                if max_coord > 0:
                    vertices = [(x / max_coord * 0.5, y / max_coord * 0.5) for x, y in vertices]
            elif vertices:
                # Fixed maximum size
                max_size = 22
                vertices = [(x / max_size * 0.5, y / max_size * 0.5) for x, y in vertices]
        else:
            vertices = self._vertices_on_sine()

        # Translate to cell position
        vertices = [(x + col, y + row) for x, y in vertices]

        return vertices

    def process(self) -> list[PathSegment]:
        """Process the image and generate spiral paths.

        Returns:
            List of PathSegment objects.
        """
        paths = []
        cfg = self.config

        # Process row by row
        for r in range(self.height):
            row_path_points = []

            for c in range(self.width):
                value = self.get_value(r, c)
                cell_vertices = self._process_cell(r, c, value)

                if cfg.connect_cells and cell_vertices:
                    # Add connecting line from previous cell
                    if row_path_points:
                        # Connect from end of previous to start of current
                        row_path_points.append(cell_vertices[0])

                    # Add cell vertices
                    row_path_points.extend(cell_vertices)
                else:
                    # Individual cell path
                    if cell_vertices:
                        paths.append(
                            PathSegment(
                                points=cell_vertices,
                                closed=False,
                                stroke_width=cfg.stroke_width,
                            )
                        )

            # Add row path if connecting cells
            if cfg.connect_cells and row_path_points:
                paths.append(
                    PathSegment(
                        points=row_path_points,
                        closed=False,
                        stroke_width=cfg.stroke_width,
                    )
                )

        return paths
