"""Tests for image processing module."""

import numpy as np
import pytest

from scribbletrace.image_processing import (
    preprocess,
    compute_gradients,
    ProcessedImage,
    GradientData,
)


class TestPreprocess:
    """Tests for the preprocess function."""

    def test_preprocess_basic(self):
        """Test basic preprocessing."""
        # Create a simple test image
        image = np.random.rand(100, 80)

        result = preprocess(image, output_width=20, levels=5)

        assert isinstance(result, ProcessedImage)
        assert result.levels == 5
        assert result.data.max() <= 4  # 0 to levels-1
        assert result.data.min() >= 0

    def test_preprocess_invert(self):
        """Test that inversion works correctly."""
        # Create an image with known values
        image = np.zeros((40, 40))
        image[:20, :] = 1.0  # Top half is white

        result_inverted = preprocess(image, output_width=10, levels=5, invert=True)
        result_normal = preprocess(image, output_width=10, levels=5, invert=False)

        # Inverted: white areas should have low values
        # Normal: white areas should have high values
        assert result_inverted.data[0, 0] < result_inverted.data[-1, 0]
        assert result_normal.data[0, 0] > result_normal.data[-1, 0]

    def test_preprocess_dimensions(self):
        """Test that dimensions are calculated correctly."""
        image = np.random.rand(200, 150)

        result = preprocess(image, output_width=40, levels=7)

        assert result.height <= 50  # Should be scaled down
        assert result.width <= 50


class TestComputeGradients:
    """Tests for gradient computation."""

    def test_gradient_computation(self):
        """Test basic gradient computation."""
        # Create image with known gradients
        image = np.zeros((50, 50))
        # Horizontal gradient
        for i in range(50):
            image[:, i] = i / 50.0

        gradients = compute_gradients(image)

        assert isinstance(gradients, GradientData)
        assert gradients.dx.shape == image.shape
        assert gradients.dy.shape == image.shape
        assert gradients.magnitude.shape == image.shape
        assert gradients.angle.shape == image.shape

    def test_gradient_magnitude_range(self):
        """Test that magnitude is normalized."""
        image = np.random.rand(50, 50)

        gradients = compute_gradients(image)

        assert gradients.magnitude.max() <= 1.0
        assert gradients.magnitude.min() >= 0.0

    def test_gradient_quantization(self):
        """Test gradient magnitude quantization."""
        image = np.random.rand(50, 50)

        gradients = compute_gradients(image, quantize_magnitude=True, magnitude_levels=10)

        # Should have discrete values
        unique_values = np.unique(gradients.magnitude)
        assert len(unique_values) <= 10

    def test_gradient_sigma_smoothing(self):
        """Test that higher Gaussian sigma smooths gradient response."""
        image = np.zeros((51, 51))
        image[25, 25] = 1.0

        sharp = compute_gradients(image, sigma=0.0)
        smooth = compute_gradients(image, sigma=2.0)

        # Smoothing should spread edge energy over a wider area.
        sharp_active = np.count_nonzero(sharp.magnitude > 0.01)
        smooth_active = np.count_nonzero(smooth.magnitude > 0.01)

        assert smooth_active > sharp_active
        assert not np.allclose(sharp.magnitude, smooth.magnitude)
