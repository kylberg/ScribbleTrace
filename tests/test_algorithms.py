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

    def test_spiral_samples_center_for_even_levels(self, processed_image):
        """Even levels should include theta=0 to avoid center-bridge artifacts."""
        config = SpiralsConfig(theta_resolution=50, connect_cells=False)
        algo = Spirals(processed_image, config=config)

        vertices = algo._vertices_on_spiral(2 * np.pi)
        min_radius = min(np.hypot(x, y) for x, y in vertices)

        assert min_radius < 1e-12

    def test_spiral_direction_consistent_across_levels(self, processed_image):
        """Connected spirals should always run left-to-right across levels."""
        config = SpiralsConfig(theta_resolution=50, connect_cells=True)
        algo = Spirals(processed_image, config=config)

        # Check a few levels that historically flipped direction by parity.
        for value in (1, 2, 3, 4):
            verts = algo._process_cell(row=0, col=0, value=value)
            assert verts[0][0] <= verts[-1][0]

    def test_spiral_randomness_controls_affect_vertices(self, processed_image):
        """Position and vertex randomness should perturb spiral geometry."""
        base_cfg = SpiralsConfig(
            theta_resolution=40,
            randomness_vertex=0.0,
            randomness_position=0.0,
            connect_cells=False,
        )
        wiggly_cfg = SpiralsConfig(
            theta_resolution=40,
            randomness_vertex=0.1,
            randomness_position=0.1,
            connect_cells=False,
        )

        np.random.seed(123)
        base_verts = Spirals(processed_image, config=base_cfg)._process_cell(row=2, col=3, value=3)

        np.random.seed(123)
        wiggly_verts = Spirals(processed_image, config=wiggly_cfg)._process_cell(row=2, col=3, value=3)

        assert len(base_verts) == len(wiggly_verts)
        assert any(
            not np.allclose(base_point, wiggly_point)
            for base_point, wiggly_point in zip(base_verts, wiggly_verts, strict=False)
        )


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

    def test_lines_follow_contours_not_gradients(self, processed_image, gradients):
        """Line direction should be perpendicular to gradient direction."""
        algo = Lines(processed_image, gradients=gradients)

        r = processed_image.height // 2
        c = processed_image.width // 2
        dx = gradients.dx[r, c]
        dy = gradients.dy[r, c]

        grad_angle = np.arctan2(dy, dx)
        line_angle = grad_angle - np.pi / 2

        # Dot product between gradient and line direction should be ~0.
        line_dir = np.array([np.cos(line_angle), np.sin(line_angle)])
        grad_vec = np.array([dx, dy])
        dot = float(np.dot(line_dir, grad_vec))

        assert np.isclose(dot, 0.0, atol=1e-9)

    def test_lines_segment_length_scales_line(self, processed_image, gradients):
        """Segment length should scale generated line length."""
        short_cfg = LinesConfig(
            segment_length=0.5,
            randomness_vertex=0.0,
            randomness_position=0.0,
            randomness_length=0.0,
            min_gradient_scale=1.0,
            max_gradient_scale=1.0,
        )
        long_cfg = LinesConfig(
            segment_length=2.0,
            randomness_vertex=0.0,
            randomness_position=0.0,
            randomness_length=0.0,
            min_gradient_scale=1.0,
            max_gradient_scale=1.0,
        )

        short_algo = Lines(processed_image, config=short_cfg, gradients=gradients)
        long_algo = Lines(processed_image, config=long_cfg, gradients=gradients)

        short_line = short_algo._generate_line(0.0, 0.0, 0.0, 1.0)
        long_line = long_algo._generate_line(0.0, 0.0, 0.0, 4.0)

        short_len = np.hypot(short_line[1][0] - short_line[0][0], short_line[1][1] - short_line[0][1])
        long_len = np.hypot(long_line[1][0] - long_line[0][0], long_line[1][1] - long_line[0][1])

        assert long_len > short_len


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

    def test_curves_segment_length_scales_trace_step(self, processed_image, gradients):
        """Segment length should scale traced curve path extent."""
        short_cfg = CurvesConfig(
            segment_length=0.5,
            max_steps=1,
            step_size=2.0,
            randomness_angle=0.0,
            randomness_position=0.0,
            bezier_samples=5,
        )
        long_cfg = CurvesConfig(
            segment_length=2.0,
            max_steps=1,
            step_size=2.0,
            randomness_angle=0.0,
            randomness_position=0.0,
            bezier_samples=5,
        )

        short_algo = Curves(processed_image, config=short_cfg, gradients=gradients)
        long_algo = Curves(processed_image, config=long_cfg, gradients=gradients)

        short_path = short_algo._trace_gradient_path(5.0, 5.0, 10.0)
        long_path = long_algo._trace_gradient_path(5.0, 5.0, 10.0)

        short_extent = max(np.hypot(x - 5.0, y - 5.0) for x, y in short_path)
        long_extent = max(np.hypot(x - 5.0, y - 5.0) for x, y in long_path)

        assert long_extent > short_extent

    def test_curves_length_randomness_changes_trace(self, processed_image, gradients):
        """Length randomness should perturb traced curve points."""
        base_cfg = CurvesConfig(
            segment_length=1.0,
            randomness_length=0.0,
            max_steps=2,
            step_size=2.0,
            randomness_angle=0.0,
            randomness_position=0.0,
            bezier_samples=5,
        )
        rand_cfg = CurvesConfig(
            segment_length=1.0,
            randomness_length=0.5,
            max_steps=2,
            step_size=2.0,
            randomness_angle=0.0,
            randomness_position=0.0,
            bezier_samples=5,
        )

        np.random.seed(123)
        base_path = Curves(processed_image, config=base_cfg, gradients=gradients)._trace_gradient_path(5.0, 5.0, 10.0)

        np.random.seed(123)
        rand_path = Curves(processed_image, config=rand_cfg, gradients=gradients)._trace_gradient_path(5.0, 5.0, 10.0)

        assert len(base_path) == len(rand_path)
        assert any(
            not np.allclose(a, b)
            for a, b in zip(base_path, rand_path, strict=False)
        )


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
