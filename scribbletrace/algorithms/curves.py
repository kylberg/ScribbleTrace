"""Bézier curves algorithm for ScribbleTrace.

This algorithm draws Bézier curves that follow the local gradient
direction. Curves are traced by stepping along the gradient field.
Inspired by the original scribbleCurves.py implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scribbletrace.algorithms.base import Algorithm, AlgorithmConfig
from scribbletrace.svg_output import PathSegment


@dataclass
class CurvesConfig(AlgorithmConfig):
    """Configuration for the Curves algorithm.

    Attributes:
        max_steps: Maximum number of steps in each direction.
        step_size: Distance to travel per step.
        segment_length: Base multiplier for traced segment length.
        randomness_length: Random variation in traced segment length.
        randomness_angle: Random variation in step direction.
        bezier_samples: Number of points to sample on the Bézier curve.
        gradient_mag_levels: Quantization levels for gradient magnitude.
    """

    max_steps: int = 4
    step_size: float = 2.0
    segment_length: float = 1.0
    randomness_length: float = 0.0
    randomness_angle: float = 0.01
    randomness_position: float = 0.5
    bezier_samples: int = 15
    gradient_mag_levels: int = 12


class Curves(Algorithm):
    """Bézier curves following gradients algorithm.

    Each image cell generates one or more Bézier curves that follow
    the local gradient direction. Starting from the cell center, points
    are traced in both directions along the gradient, then smoothed
    into a Bézier curve.

    Requires gradient data to be provided.
    """

    def __init__(self, image, config: CurvesConfig | None = None, gradients=None):
        super().__init__(image, config, gradients)
        if config is None:
            self.config = CurvesConfig()

    @classmethod
    def _default_config(cls) -> CurvesConfig:
        return CurvesConfig()

    def _trace_gradient_path(
        self, start_col: float, start_row: float, grad_mag: float
    ) -> list[tuple[float, float]]:
        """Trace a path following the gradient from a starting point.

        Args:
            start_col: Starting column position.
            start_row: Starting row position.
            grad_mag: Local gradient magnitude.

        Returns:
            List of (x, y) points along the traced path.
        """
        cfg = self.config

        # Initialize path with starting point (with random offset)
        start_x = start_col + self.random_offset(cfg.randomness_position)
        start_y = start_row + self.random_offset(cfg.randomness_position)

        path_forward = [(start_x, start_y)]
        path_backward = []

        # Calculate number of steps based on gradient magnitude
        step_count = int(min(cfg.max_steps, grad_mag))
        step_length = cfg.segment_length * min(cfg.step_size, grad_mag / 10)
        if cfg.randomness_length > 0:
            step_length *= self.random_scale(1.0, cfg.randomness_length)

        if step_count <= 0:
            return path_forward

        # Trace forward direction
        loc_c, loc_r = int(round(start_x)), int(round(start_y))

        for _ in range(step_count):
            # Get gradient direction (perpendicular to gradient for flow lines)
            angle = self._get_safe_angle(loc_r, loc_c) - np.pi / 2

            # Apply random angle variation
            if cfg.randomness_angle > 0:
                angle *= self.random_scale(1.0, cfg.randomness_angle)

            # Step in gradient direction
            dx = step_length * np.cos(angle)
            dy = step_length * np.sin(angle)

            new_x = path_forward[-1][0] + dx
            new_y = path_forward[-1][1] + dy

            path_forward.append((new_x, new_y))

            # Update location for next iteration
            loc_c = int(np.clip(round(new_x), 0, self.width - 1))
            loc_r = int(np.clip(round(new_y), 0, self.height - 1))

        # Trace backward direction
        loc_c, loc_r = int(round(start_x)), int(round(start_y))

        for _ in range(step_count):
            # Get gradient direction (opposite direction)
            angle = self._get_safe_angle(loc_r, loc_c) - 3 * np.pi / 2

            # Apply random angle variation
            if cfg.randomness_angle > 0:
                angle *= self.random_scale(1.0, cfg.randomness_angle)

            # Step in gradient direction
            dx = step_length * np.cos(angle)
            dy = step_length * np.sin(angle)

            if path_backward:
                new_x = path_backward[-1][0] + dx
                new_y = path_backward[-1][1] + dy
            else:
                new_x = start_x + dx
                new_y = start_y + dy

            path_backward.append((new_x, new_y))

            # Update location for next iteration
            loc_c = int(np.clip(round(new_x), 0, self.width - 1))
            loc_r = int(np.clip(round(new_y), 0, self.height - 1))

        # Combine paths (backward reversed + forward)
        path_backward.reverse()
        return path_backward + path_forward

    def _get_safe_angle(self, row: int, col: int) -> float:
        """Get gradient angle at a position, handling boundaries.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Gradient angle in radians.
        """
        row = int(np.clip(row, 0, self.height - 1))
        col = int(np.clip(col, 0, self.width - 1))

        if self.gradients is None:
            return 0.0

        return np.arctan2(
            self.gradients.dy[row, col],
            self.gradients.dx[row, col],
        )

    def _bezier_smooth(
        self, points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Smooth a path using quadratic Bézier interpolation.

        This is a simplified Bézier smoothing that creates a smooth
        curve through the control points.

        Args:
            points: List of control points.

        Returns:
            Smoothed list of points.
        """
        if len(points) < 3:
            return points

        cfg = self.config
        n_samples = cfg.bezier_samples

        # Simple B-spline-like smoothing
        smoothed = []
        t_values = np.linspace(0, 1, n_samples)

        for i in range(len(points) - 2):
            p0 = np.array(points[i])
            p1 = np.array(points[i + 1])
            p2 = np.array(points[i + 2])

            # Quadratic Bézier for this segment
            for t in t_values[:-1]:  # Skip last to avoid duplicates
                point = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
                smoothed.append((float(point[0]), float(point[1])))

        # Add final point
        smoothed.append(points[-1])

        return smoothed

    def _process_cell(
        self,
        row: int,
        col: int,
        value: int,
        grad_mag: float,
    ) -> list[PathSegment]:
        """Process a single cell and return its curve paths.

        Args:
            row: Row index.
            col: Column index.
            value: Intensity value for this cell.
            grad_mag: Gradient magnitude at this cell.

        Returns:
            List of PathSegment objects for this cell's curves.
        """
        cfg = self.config
        paths = []

        if value <= 0:
            return paths

        # Draw multiple curves based on intensity
        for _ in range(value):
            # Trace path following gradient
            raw_path = self._trace_gradient_path(col, row, grad_mag)

            if len(raw_path) >= 2:
                # Smooth with Bézier
                smoothed = self._bezier_smooth(raw_path)

                paths.append(
                    PathSegment(
                        points=smoothed,
                        closed=False,
                        stroke_width=cfg.stroke_width,
                    )
                )

        return paths

    def process(self) -> list[PathSegment]:
        """Process the image and generate curve paths.

        Returns:
            List of PathSegment objects.

        Raises:
            ValueError: If gradient data is not provided.
        """
        if self.gradients is None:
            raise ValueError("Curves algorithm requires gradient data")

        paths = []

        # Quantize gradient magnitude if needed
        grad_mag = self.gradients.magnitude.copy()
        if self.config.gradient_mag_levels > 0:
            grad_mag = np.round(grad_mag * (self.config.gradient_mag_levels - 1)) + 1

        for c in range(self.width):
            for r in range(self.height):
                value = self.get_value(r, c)
                local_grad_mag = grad_mag[r, c]

                cell_paths = self._process_cell(r, c, value, local_grad_mag)
                paths.extend(cell_paths)

        return paths
