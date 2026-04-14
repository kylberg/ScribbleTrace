"""Tests for SVG output module."""

import pytest

from scribbletrace.svg_output import (
    SVGWriter,
    SVGConfig,
    PathSegment,
    optimize_path_order,
)


class TestPathSegment:
    """Tests for PathSegment dataclass."""

    def test_path_segment_creation(self):
        """Test creating a path segment."""
        points = [(0, 0), (10, 10), (20, 0)]
        segment = PathSegment(points=points)

        assert segment.points == points
        assert segment.closed is False
        assert segment.stroke_width == 0.5

    def test_path_segment_closed(self):
        """Test creating a closed path segment."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        segment = PathSegment(points=points, closed=True)

        assert segment.closed is True


class TestSVGWriter:
    """Tests for SVGWriter class."""

    def test_svg_writer_creation(self):
        """Test creating an SVG writer."""
        writer = SVGWriter(width=100, height=100)

        assert writer.config.width == 100
        assert writer.config.height == 100

    def test_svg_writer_with_paths(self):
        """Test SVG writer with paths."""
        paths = [
            PathSegment(points=[(0, 0), (10, 10)]),
            PathSegment(points=[(20, 20), (30, 30)]),
        ]
        writer = SVGWriter(paths, width=50, height=50)

        assert len(writer.paths) == 2

    def test_svg_writer_add_path(self):
        """Test adding paths to writer."""
        writer = SVGWriter(width=50, height=50)
        writer.add_path(PathSegment(points=[(0, 0), (10, 10)]))

        assert len(writer.paths) == 1

    def test_svg_writer_to_string(self):
        """Test generating SVG string."""
        paths = [PathSegment(points=[(0, 0), (10, 10)])]
        writer = SVGWriter(paths, width=50, height=50)

        svg_string = writer.to_string()

        assert "<svg" in svg_string
        assert "<path" in svg_string
        assert "M 0.0000 0.0000" in svg_string or "M 0" in svg_string

    def test_svg_writer_add_circle(self):
        """Test adding circle to writer."""
        writer = SVGWriter(width=50, height=50)
        writer.add_circle(center=(25, 25), radius=10)

        assert len(writer.paths) == 1
        assert writer.paths[0].closed is True

    def test_svg_writer_get_bounds(self):
        """Test getting bounds of paths."""
        paths = [
            PathSegment(points=[(5, 10), (15, 20)]),
            PathSegment(points=[(0, 0), (25, 30)]),
        ]
        writer = SVGWriter(paths, width=50, height=50)

        bounds = writer.get_bounds()

        assert bounds == (0, 0, 25, 30)


class TestOptimizePathOrder:
    """Tests for path optimization."""

    def test_optimize_empty(self):
        """Test optimizing empty list."""
        result = optimize_path_order([])
        assert result == []

    def test_optimize_single(self):
        """Test optimizing single path."""
        paths = [PathSegment(points=[(0, 0), (10, 10)])]
        result = optimize_path_order(paths)
        assert len(result) == 1

    def test_optimize_multiple(self):
        """Test optimizing multiple paths."""
        # Create paths that benefit from reordering
        paths = [
            PathSegment(points=[(0, 0), (10, 10)]),
            PathSegment(points=[(100, 100), (110, 110)]),
            PathSegment(points=[(10, 10), (20, 20)]),
        ]
        result = optimize_path_order(paths)

        assert len(result) == 3
        # First path should stay first
        assert result[0].points[0] == (0, 0)
        # Third path should be second (nearest to end of first)
        assert result[1].points[0] == (10, 10)
