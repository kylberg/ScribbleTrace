"""Hatching algorithm for ScribbleTrace.

This algorithm creates cross-hatching patterns based on image intensity.
Multiple hatching directions can be combined to achieve different shading
densities. This is a new algorithm designed specifically for pen plotters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from scribbletrace.algorithms.base import Algorithm, AlgorithmConfig
from scribbletrace.svg_output import PathSegment


class HatchDirection(Enum):
    """Hatching direction options."""

    HORIZONTAL = 0
    VERTICAL = 90
    DIAGONAL_RIGHT = 45
    DIAGONAL_LEFT = -45


@dataclass
class HatchingConfig(AlgorithmConfig):
    """Configuration for the Hatching algorithm.

    Attributes:
        directions: List of hatching directions to use.
        base_spacing: Base line spacing in output units.
        min_spacing: Minimum spacing (for darkest areas).
        max_spacing: Maximum spacing (for lightest areas).
        line_extension: How much lines extend beyond cell boundaries.
        use_gradient_angle: If True, modulate angle based on gradient.
        gradient_angle_weight: How much gradient affects angle (0-1).
        optimize_path: If True, optimize path order for pen plotter.
        margin: Margin around image edges.
    """

    directions: list[HatchDirection] = field(
        default_factory=lambda: [HatchDirection.DIAGONAL_RIGHT]
    )
    base_spacing: float = 1.0
    min_spacing: float = 0.3
    max_spacing: float = 2.0
    line_extension: float = 0.5
    use_gradient_angle: bool = False
    gradient_angle_weight: float = 0.3
    optimize_path: bool = True
    margin: float = 0.0


class Hatching(Algorithm):
    """Cross-hatching algorithm.

    Creates hatching patterns where line density varies with image
    intensity. Multiple hatching directions can be layered to create
    cross-hatching effects. Dark areas have denser (closer) lines,
    while light areas have sparser lines or no lines at all.

    This algorithm is optimized for pen plotters by:
    - Drawing continuous lines across the image
    - Minimizing pen lifts through path ordering
    - Using configurable line spacing based on intensity
    """

    def __init__(self, image, config: HatchingConfig | None = None, gradients=None):
        super().__init__(image, config, gradients)
        if config is None:
            self.config = HatchingConfig()

    @classmethod
    def _default_config(cls) -> HatchingConfig:
        return HatchingConfig()

    def _intensity_to_spacing(self, intensity: float) -> float:
        """Convert image intensity to line spacing.

        Lower intensity (darker) = smaller spacing (denser lines).
        Higher intensity (lighter) = larger spacing (sparser lines).

        Args:
            intensity: Normalized intensity value (0=dark, 1=light).

        Returns:
            Line spacing for this intensity.
        """
        cfg = self.config
        # Linear interpolation between min and max spacing
        return cfg.min_spacing + intensity * (cfg.max_spacing - cfg.min_spacing)

    def _generate_hatch_lines(
        self,
        direction: HatchDirection,
    ) -> list[PathSegment]:
        """Generate hatching lines for a single direction.

        Args:
            direction: Hatching direction enum.

        Returns:
            List of PathSegment objects for this hatching direction.
        """
        cfg = self.config
        paths = []

        angle_rad = np.radians(direction.value)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Calculate image diagonal for line coverage
        diagonal = np.sqrt(self.width**2 + self.height**2)
        margin = cfg.margin

        # Determine line positions based on angle
        # We need to cover the entire image when lines are rotated

        if abs(direction.value) in [0, 180]:
            # Horizontal lines
            lines = self._generate_horizontal_lines()
        elif abs(direction.value) in [90, 270]:
            # Vertical lines
            lines = self._generate_vertical_lines()
        else:
            # Diagonal lines
            lines = self._generate_diagonal_lines(direction)

        return lines

    def _generate_horizontal_lines(self) -> list[PathSegment]:
        """Generate horizontal hatching lines.

        Returns:
            List of PathSegment objects.
        """
        cfg = self.config
        paths = []

        y = cfg.margin
        while y < self.height - cfg.margin:
            # Sample intensity along this row
            row = int(np.clip(y, 0, self.height - 1))

            # Get average intensity for this row
            row_intensity = np.mean(self.original[row, :])

            # Determine spacing for next line
            spacing = self._intensity_to_spacing(row_intensity)

            # Generate line if area is dark enough
            if row_intensity < 0.95:  # Skip very light areas
                x_start = -cfg.line_extension
                x_end = self.width + cfg.line_extension

                # Create line points with variable intensity
                points = self._create_intensity_modulated_line(
                    x_start, y, x_end, y, horizontal=True
                )

                if points:
                    paths.append(
                        PathSegment(
                            points=points,
                            closed=False,
                            stroke_width=cfg.stroke_width,
                            group="hatch_horizontal",
                        )
                    )

            y += spacing

        return paths

    def _generate_vertical_lines(self) -> list[PathSegment]:
        """Generate vertical hatching lines.

        Returns:
            List of PathSegment objects.
        """
        cfg = self.config
        paths = []

        x = cfg.margin
        while x < self.width - cfg.margin:
            # Sample intensity along this column
            col = int(np.clip(x, 0, self.width - 1))

            # Get average intensity for this column
            col_intensity = np.mean(self.original[:, col])

            # Determine spacing for next line
            spacing = self._intensity_to_spacing(col_intensity)

            # Generate line if area is dark enough
            if col_intensity < 0.95:
                y_start = -cfg.line_extension
                y_end = self.height + cfg.line_extension

                points = self._create_intensity_modulated_line(
                    x, y_start, x, y_end, horizontal=False
                )

                if points:
                    paths.append(
                        PathSegment(
                            points=points,
                            closed=False,
                            stroke_width=cfg.stroke_width,
                            group="hatch_vertical",
                        )
                    )

            x += spacing

        return paths

    def _generate_diagonal_lines(
        self, direction: HatchDirection
    ) -> list[PathSegment]:
        """Generate diagonal hatching lines.

        Args:
            direction: Diagonal direction (45 or -45 degrees).

        Returns:
            List of PathSegment objects.
        """
        cfg = self.config
        paths = []

        angle_rad = np.radians(direction.value)
        diagonal = np.sqrt(self.width**2 + self.height**2)

        # For 45 degrees: lines go from bottom-left to top-right
        # For -45 degrees: lines go from top-left to bottom-right

        # Calculate perpendicular direction for spacing
        perp_angle = angle_rad + np.pi / 2
        cos_perp = np.cos(perp_angle)
        sin_perp = np.sin(perp_angle)

        # Start from corner and move perpendicular to line direction
        if direction == HatchDirection.DIAGONAL_RIGHT:
            # Start from bottom-left corner area
            start_offset = -diagonal / 2
        else:
            # Start from top-left corner area
            start_offset = -diagonal / 2

        offset = start_offset
        while offset < diagonal:
            # Calculate line start and end points
            cx = self.width / 2 + offset * cos_perp
            cy = self.height / 2 + offset * sin_perp

            # Line direction
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)

            # Line endpoints (extend beyond image)
            x1 = cx - diagonal * dx
            y1 = cy - diagonal * dy
            x2 = cx + diagonal * dx
            y2 = cy + diagonal * dy

            # Clip line to image bounds and check intensity
            clipped_points = self._clip_and_sample_line(x1, y1, x2, y2)

            if clipped_points:
                # Calculate average intensity along line
                avg_intensity = self._sample_line_intensity(clipped_points)

                # Generate line if dark enough
                if avg_intensity < 0.95:
                    paths.append(
                        PathSegment(
                            points=clipped_points,
                            closed=False,
                            stroke_width=cfg.stroke_width,
                            group=f"hatch_{direction.name.lower()}",
                        )
                    )

            # Determine spacing based on local intensity
            sample_x = int(np.clip(cx, 0, self.width - 1))
            sample_y = int(np.clip(cy, 0, self.height - 1))
            local_intensity = self._get_intensity_at(sample_y, sample_x)
            spacing = self._intensity_to_spacing(local_intensity)

            offset += spacing

        return paths

    def _create_intensity_modulated_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        horizontal: bool,
    ) -> list[tuple[float, float]]:
        """Create a line that skips over light areas.

        Args:
            x1, y1: Start point.
            x2, y2: End point.
            horizontal: True for horizontal lines.

        Returns:
            List of points, potentially with gaps.
        """
        points = []
        num_samples = int(max(abs(x2 - x1), abs(y2 - y1)) * 2) + 2

        in_stroke = False

        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            # Check if point is within image bounds
            if 0 <= x < self.width and 0 <= y < self.height:
                intensity = self._get_intensity_at(int(y), int(x))

                if intensity < 0.9:  # Dark enough to draw
                    if not in_stroke:
                        in_stroke = True
                    points.append((x, y))
                else:
                    in_stroke = False
            elif in_stroke:
                # Extend slightly beyond bounds
                points.append((x, y))

        return points

    def _clip_and_sample_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> list[tuple[float, float]]:
        """Clip a line to image bounds and return sample points.

        Args:
            x1, y1: Start point.
            x2, y2: End point.

        Returns:
            List of clipped and sampled points.
        """
        # Simple clipping - find intersections with image bounds
        points = []
        cfg = self.config
        ext = cfg.line_extension

        # Sample along line
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length == 0:
            return points

        num_samples = int(length * 2) + 2
        in_bounds = False

        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            # Check if within extended bounds
            if -ext <= x <= self.width + ext and -ext <= y <= self.height + ext:
                if not in_bounds and (-ext <= x <= self.width + ext):
                    in_bounds = True
                if in_bounds:
                    points.append((x, y))
            elif in_bounds:
                break  # Exited bounds

        return points

    def _sample_line_intensity(
        self, points: list[tuple[float, float]]
    ) -> float:
        """Get average intensity along a line.

        Args:
            points: List of points on the line.

        Returns:
            Average intensity (0-1).
        """
        if not points:
            return 1.0

        intensities = []
        for x, y in points:
            if 0 <= x < self.width and 0 <= y < self.height:
                intensities.append(self._get_intensity_at(int(y), int(x)))

        return np.mean(intensities) if intensities else 1.0

    def _get_intensity_at(self, row: int, col: int) -> float:
        """Get original image intensity at a position.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Intensity value (0=dark, 1=light).
        """
        row = int(np.clip(row, 0, self.height - 1))
        col = int(np.clip(col, 0, self.width - 1))
        return float(self.original[row, col])

    def process(self) -> list[PathSegment]:
        """Process the image and generate hatching paths.

        Returns:
            List of PathSegment objects.
        """
        cfg = self.config
        all_paths = []

        # Generate lines for each direction
        for direction in cfg.directions:
            paths = self._generate_hatch_lines(direction)
            all_paths.extend(paths)

        # Optimize path order if requested
        if cfg.optimize_path:
            all_paths = self._optimize_paths(all_paths)

        return all_paths

    def _optimize_paths(self, paths: list[PathSegment]) -> list[PathSegment]:
        """Optimize path order to minimize pen travel.

        Simple greedy nearest-neighbor optimization.

        Args:
            paths: List of paths to optimize.

        Returns:
            Reordered list of paths.
        """
        if len(paths) <= 1:
            return paths

        remaining = list(paths)
        optimized = [remaining.pop(0)]

        while remaining:
            # Get end point of last path
            last_end = optimized[-1].points[-1] if optimized[-1].points else (0, 0)

            # Find nearest path start
            min_dist = float("inf")
            min_idx = 0

            for i, path in enumerate(remaining):
                if path.points:
                    # Check distance to start
                    start = path.points[0]
                    dist_to_start = (start[0] - last_end[0]) ** 2 + (
                        start[1] - last_end[1]
                    ) ** 2

                    # Also check distance to end (path can be reversed)
                    end = path.points[-1]
                    dist_to_end = (end[0] - last_end[0]) ** 2 + (
                        end[1] - last_end[1]
                    ) ** 2

                    min_local = min(dist_to_start, dist_to_end)
                    if min_local < min_dist:
                        min_dist = min_local
                        min_idx = i
                        # Mark if we should reverse
                        should_reverse = dist_to_end < dist_to_start

            # Get the nearest path
            nearest = remaining.pop(min_idx)

            # Reverse if needed
            if should_reverse and nearest.points:
                nearest = PathSegment(
                    points=list(reversed(nearest.points)),
                    closed=nearest.closed,
                    stroke_width=nearest.stroke_width,
                    group=nearest.group,
                )

            optimized.append(nearest)

        return optimized
