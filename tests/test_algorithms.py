"""Tests for algorithm modules."""

import numpy as np
import pytest

from scribbletrace.image_processing import preprocess, compute_gradients
from scribbletrace.svg_output import PathSegment
from scribbletrace.algorithms import (
    Spirals,
    SpiralsConfig,
    Circles,
    CirclesConfig,
    Squares,
    SquaresConfig,
    Lines,
    LinesConfig,
    Curves,
    CurvesConfig,
    Hatching,
    HatchingConfig,
    HatchDirection,
)


@pytest.fixture
def test_image():
    """Create a test image for algorithm tests."""
    # Create a gradient image
    image = np.zeros((100, 100))
    for i in range(100):
        image[i, :] = i / 100.0
    return image


@pytest.fixture
def processed_image(test_image):
    """Create a processed image for tests."""
    return preprocess(test_image, output_width=20, levels=5)


@pytest.fixture
def gradients(processed_image):
    """Create gradient data for tests."""
    return compute_gradients(processed_image.original)


class TestSpirals:
    """Tests for Spirals algorithm."""

    def test_spirals_produces_paths(self, processed_image):
        """Test that spirals produces valid paths."""
        algo = Spirals(processed_image)
        paths = algo.process()

        assert isinstance(paths, list)
        assert len(paths) > 0
        assert all(isinstance(p, PathSegment) for p in paths)

    def test_spirals_with_config(self, processed_image):
        """Test spirals with custom config."""
        config = SpiralsConfig(
            theta_resolution=30,
            normalize_spirals=True,
            connect_cells=False,
        )
        algo = Spirals(processed_image, config=config)
        paths = algo.process()

        # Without connect_cells, should have more individual paths
        assert len(paths) > 0


class TestCircles:
    """Tests for Circles algorithm."""

    def test_circles_produces_paths(self, processed_image):
        """Test that circles produces valid paths."""
        algo = Circles(processed_image)
        paths = algo.process()

        assert isinstance(paths, list)
        assert len(paths) > 0
        assert all(isinstance(p, PathSegment) for p in paths)

    def test_circles_closed(self, processed_image):
        """Test that circles are closed paths."""
        algo = Circles(processed_image)
        paths = algo.process()

        # All circles should be closed
        assert all(p.closed for p in paths)


class TestSquares:
    """Tests for Squares algorithm."""

    def test_squares_produces_paths(self, processed_image):
        """Test that squares produces valid paths."""
        algo = Squares(processed_image)
        paths = algo.process()

        assert isinstance(paths, list)
        assert len(paths) > 0
        assert all(isinstance(p, PathSegment) for p in paths)


class TestLines:
    """Tests for Lines algorithm."""

    def test_lines_requires_gradients(self, processed_image):
        """Test that lines raises error without gradients."""
        algo = Lines(processed_image)

        with pytest.raises(ValueError, match="gradient"):
            algo.process()

    def test_lines_produces_paths(self, processed_image, gradients):
        """Test that lines produces valid paths with gradients."""
        algo = Lines(processed_image, gradients=gradients)
        paths = algo.process()

        assert isinstance(paths, list)
        assert len(paths) > 0


class TestCurves:
    """Tests for Curves algorithm."""

    def test_curves_requires_gradients(self, processed_image):
        """Test that curves raises error without gradients."""
        algo = Curves(processed_image)

        with pytest.raises(ValueError, match="gradient"):
            algo.process()

    def test_curves_produces_paths(self, processed_image, gradients):
        """Test that curves produces valid paths with gradients."""
        algo = Curves(processed_image, gradients=gradients)
        paths = algo.process()

        assert isinstance(paths, list)
        assert len(paths) > 0


class TestHatching:
    """Tests for Hatching algorithm."""

    def test_hatching_produces_paths(self, processed_image):
        """Test that hatching produces valid paths."""
        algo = Hatching(processed_image)
        paths = algo.process()

        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_hatching_directions(self, processed_image):
        """Test hatching with multiple directions."""
        config = HatchingConfig(
            directions=[
                HatchDirection.DIAGONAL_RIGHT,
                HatchDirection.DIAGONAL_LEFT,
            ]
        )
        algo = Hatching(processed_image, config=config)
        paths = algo.process()

        assert len(paths) > 0

    def test_hatching_path_optimization(self, processed_image):
        """Test that path optimization works."""
        config = HatchingConfig(optimize_path=True)
        algo = Hatching(processed_image, config=config)
        paths = algo.process()

        assert len(paths) > 0
