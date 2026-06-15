"""SVG output generation for ScribbleTrace.

This module provides SVG generation using the svgwrite library,
optimized for pen plotters like the Axidraw.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import svgwrite
from svgwrite import Drawing


@dataclass
class PathSegment:
    """A single path segment for SVG output.

    Attributes:
        points: List of (x, y) tuples defining the path.
        closed: If True, the path is closed (returns to start).
        stroke_width: Line width for this path.
        group: Optional group name for organizing paths.
    """

    points: list[tuple[float, float]]
    closed: bool = False
    stroke_width: float = 0.5
    group: str = ""


@dataclass
class SVGConfig:
    """Configuration for SVG output.

    Attributes:
        width: SVG document width in mm.
        height: SVG document height in mm.
        stroke_color: Default stroke color.
        stroke_width: Default stroke width in mm.
        background: Background color (None for transparent).
        margin: Margin around the drawing in mm.
        units: SVG units (mm, px, etc.).
    """

    width: float = 200.0
    height: float = 150.0
    stroke_color: str = "black"
    stroke_width: float = 0.5
    background: str | None = None
    margin: float = 10.0
    units: str = "mm"


class SVGWriter:
    """SVG file writer optimized for pen plotters.

    This class takes path data and generates clean SVG output suitable
    for pen plotters. It supports grouping paths, optimizing drawing
    order, and configuring stroke properties.

    Example:
        paths = [PathSegment([(0, 0), (10, 10), (20, 0)])]
        writer = SVGWriter(paths, width=100, height=100)
        writer.save("output.svg")
    """

    def __init__(
        self,
        paths: Sequence[PathSegment] | None = None,
        width: float = 200.0,
        height: float = 150.0,
        config: SVGConfig | None = None,
    ):
        """Initialize the SVG writer.

        Args:
            paths: Sequence of PathSegment objects to include.
            width: Document width (overrides config if provided).
            height: Document height (overrides config if provided).
            config: SVG configuration object.
        """
        self.config = config or SVGConfig(width=width, height=height)
        if width != 200.0:
            self.config.width = width
        if height != 150.0:
            self.config.height = height

        self.paths: list[PathSegment] = list(paths) if paths else []
        self._groups: dict[str, list[PathSegment]] = {}

    def add_path(self, path: PathSegment) -> None:
        """Add a single path to the drawing.

        Args:
            path: PathSegment to add.
        """
        self.paths.append(path)

    def add_paths(self, paths: Sequence[PathSegment]) -> None:
        """Add multiple paths to the drawing.

        Args:
            paths: Sequence of PathSegment objects.
        """
        self.paths.extend(paths)

    def add_polyline(
        self,
        points: list[tuple[float, float]],
        closed: bool = False,
        stroke_width: float | None = None,
        group: str = "",
    ) -> None:
        """Convenience method to add a polyline path.

        Args:
            points: List of (x, y) coordinate tuples.
            closed: If True, close the path.
            stroke_width: Line width (uses default if None).
            group: Optional group name.
        """
        path = PathSegment(
            points=points,
            closed=closed,
            stroke_width=stroke_width or self.config.stroke_width,
            group=group,
        )
        self.add_path(path)

    def add_circle(
        self,
        center: tuple[float, float],
        radius: float,
        num_points: int = 36,
        stroke_width: float | None = None,
        group: str = "",
    ) -> None:
        """Add a circle as a polyline approximation.

        Args:
            center: (x, y) center coordinates.
            radius: Circle radius.
            num_points: Number of points to approximate the circle.
            stroke_width: Line width.
            group: Optional group name.
        """
        import numpy as np

        cx, cy = center
        angles = np.linspace(0, 2 * np.pi, num_points + 1)
        points = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]

        self.add_polyline(
            points=points,
            closed=True,
            stroke_width=stroke_width,
            group=group,
        )

    def _create_drawing(self) -> Drawing:
        """Create the svgwrite Drawing object.

        Returns:
            Configured Drawing object.
        """
        cfg = self.config
        size = (f"{cfg.width}{cfg.units}", f"{cfg.height}{cfg.units}")
        view_min_x = 0.0
        view_min_y = 0.0
        view_w = cfg.width
        view_h = cfg.height

        # Build a stroke-aware viewBox so paths at x=0/y=0 are not clipped.
        if self.paths:
            min_x, min_y, max_x, max_y = self.get_bounds()
            max_stroke = max((p.stroke_width for p in self.paths), default=cfg.stroke_width)
            pad = max(0.25, max_stroke / 2.0)

            view_min_x = min(min_x - pad, 0.0)
            view_min_y = min(min_y - pad, 0.0)
            view_max_x = max(max_x + pad, cfg.width)
            view_max_y = max(max_y + pad, cfg.height)

            view_w = max(1e-6, view_max_x - view_min_x)
            view_h = max(1e-6, view_max_y - view_min_y)

        viewbox = f"{view_min_x} {view_min_y} {view_w} {view_h}"

        dwg = svgwrite.Drawing(size=size, viewBox=viewbox)

        # Add background if specified
        if cfg.background:
            dwg.add(
                dwg.rect(
                    insert=(view_min_x, view_min_y),
                    size=(view_w, view_h),
                    fill=cfg.background,
                )
            )

        return dwg

    def _path_to_svg_d(self, segment: PathSegment) -> str:
        """Convert a PathSegment to SVG path data string.

        Args:
            segment: PathSegment to convert.

        Returns:
            SVG path data string (d attribute).
        """
        if not segment.points:
            return ""

        parts = []
        # Move to first point
        x, y = segment.points[0]
        parts.append(f"M {x:.4f} {y:.4f}")

        # Line to subsequent points
        for x, y in segment.points[1:]:
            parts.append(f"L {x:.4f} {y:.4f}")

        # Close path if needed
        if segment.closed:
            parts.append("Z")

        return " ".join(parts)

    def _add_paths_to_drawing(self, dwg: Drawing) -> None:
        """Add all paths to the drawing.

        Args:
            dwg: Drawing to add paths to.
        """
        # Group paths by their group attribute
        groups: dict[str, list[PathSegment]] = {}
        ungrouped: list[PathSegment] = []

        for path in self.paths:
            if path.group:
                if path.group not in groups:
                    groups[path.group] = []
                groups[path.group].append(path)
            else:
                ungrouped.append(path)

        # Add grouped paths
        for group_name, group_paths in groups.items():
            g = dwg.g(id=group_name)
            for segment in group_paths:
                d = self._path_to_svg_d(segment)
                if d:
                    path_elem = dwg.path(
                        d=d,
                        stroke=self.config.stroke_color,
                        stroke_width=segment.stroke_width,
                        fill="none",
                    )
                    g.add(path_elem)
            dwg.add(g)

        # Add ungrouped paths
        for segment in ungrouped:
            d = self._path_to_svg_d(segment)
            if d:
                path_elem = dwg.path(
                    d=d,
                    stroke=self.config.stroke_color,
                    stroke_width=segment.stroke_width,
                    fill="none",
                )
                dwg.add(path_elem)

    def to_string(self) -> str:
        """Generate SVG content as a string.

        Returns:
            SVG document as string.
        """
        dwg = self._create_drawing()
        self._add_paths_to_drawing(dwg)
        return dwg.tostring()

    def save(self, path: str | Path) -> None:
        """Save the SVG to a file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        dwg = self._create_drawing()
        self._add_paths_to_drawing(dwg)
        dwg.saveas(str(path), pretty=True)

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Get the bounding box of all paths.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y).
        """
        if not self.paths:
            return (0, 0, self.config.width, self.config.height)

        all_x = []
        all_y = []
        for path in self.paths:
            for x, y in path.points:
                all_x.append(x)
                all_y.append(y)

        return (min(all_x), min(all_y), max(all_x), max(all_y))


def optimize_path_order(paths: list[PathSegment]) -> list[PathSegment]:
    """Optimize path order to minimize pen travel (greedy nearest neighbor).

    This simple optimization helps reduce plotting time by minimizing
    the distance the pen travels between paths.

    Args:
        paths: List of paths to optimize.

    Returns:
        Reordered list of paths.
    """
    if len(paths) <= 1:
        return paths

    import numpy as np

    remaining = list(paths)
    optimized = [remaining.pop(0)]

    while remaining:
        # Get end point of last path
        last_point = optimized[-1].points[-1] if optimized[-1].points else (0, 0)

        # Find nearest path start
        min_dist = float("inf")
        min_idx = 0

        for i, path in enumerate(remaining):
            if path.points:
                start = path.points[0]
                dist = np.sqrt((start[0] - last_point[0]) ** 2 + (start[1] - last_point[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i

        optimized.append(remaining.pop(min_idx))

    return optimized
