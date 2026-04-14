"""Base classes for ScribbleTrace algorithms.

This module provides the abstract base class and common configuration
for all vectorization algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scribbletrace.image_processing import ProcessedImage, GradientData
    from scribbletrace.svg_output import PathSegment


@dataclass
class AlgorithmConfig:
    """Base configuration for algorithms.

    Attributes:
        randomness_vertex: Amount of random displacement for vertices (0-1).
        randomness_position: Amount of random displacement for positions (0-1).
        min_element_size: Minimum size of drawing elements.
        max_element_size: Maximum size of drawing elements.
        stroke_width: Line width for output paths.
    """

    randomness_vertex: float = 0.1
    randomness_position: float = 0.0
    min_element_size: float = 0.1
    max_element_size: float = 1.0
    stroke_width: float = 0.5


class Algorithm(ABC):
    """Abstract base class for vectorization algorithms.

    All algorithms inherit from this class and implement the process()
    method to generate vector paths from a processed image.

    Example:
        class MyAlgorithm(Algorithm):
            def process(self) -> list[PathSegment]:
                paths = []
                for c in range(self.width):
                    for r in range(self.height):
                        value = self.get_value(r, c)
                        # Generate paths based on value
                        ...
                return paths
    """

    def __init__(
        self,
        image: ProcessedImage,
        config: AlgorithmConfig | None = None,
        gradients: GradientData | None = None,
    ):
        """Initialize the algorithm.

        Args:
            image: Processed image data.
            config: Algorithm configuration.
            gradients: Optional gradient data for gradient-based algorithms.
        """
        self.image = image
        self.config = config or self._default_config()
        self.gradients = gradients

        # Convenience attributes
        self.data = image.data
        self.original = image.original
        self.width = image.width
        self.height = image.height
        self.levels = image.levels

    @classmethod
    def _default_config(cls) -> AlgorithmConfig:
        """Return the default configuration for this algorithm.

        Override in subclasses to provide algorithm-specific defaults.
        """
        return AlgorithmConfig()

    def get_value(self, row: int, col: int) -> int:
        """Get the quantized intensity value at a position.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Integer intensity value (0 to levels-1).
        """
        return int(self.data[row, col])

    def get_original(self, row: int, col: int) -> float:
        """Get the original (non-quantized) intensity at a position.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Float intensity value (0.0 to 1.0).
        """
        return float(self.original[row, col])

    def get_gradient_magnitude(self, row: int, col: int) -> float:
        """Get gradient magnitude at a position.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Gradient magnitude, or 0 if gradients not available.
        """
        if self.gradients is None:
            return 0.0
        return float(self.gradients.magnitude[row, col])

    def get_gradient_angle(self, row: int, col: int) -> float:
        """Get gradient angle at a position.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Gradient angle in radians, or 0 if gradients not available.
        """
        if self.gradients is None:
            return 0.0
        return float(self.gradients.angle[row, col])

    def random_offset(self, amount: float = 0.1) -> float:
        """Generate a random offset value.

        Args:
            amount: Maximum offset magnitude.

        Returns:
            Random value in [-amount, amount].
        """
        return np.random.uniform(-amount, amount)

    def random_scale(self, base: float = 1.0, variation: float = 0.1) -> float:
        """Generate a random scale factor.

        Args:
            base: Base scale value.
            variation: Relative variation (0.1 = ±10%).

        Returns:
            Random scale factor.
        """
        return base * np.random.uniform(1 - variation, 1 + variation)

    @abstractmethod
    def process(self) -> list[PathSegment]:
        """Process the image and generate vector paths.

        This is the main method that each algorithm must implement.

        Returns:
            List of PathSegment objects representing the vector drawing.
        """
        pass

    def get_svg_dimensions(self) -> tuple[float, float]:
        """Get recommended SVG dimensions for this image.

        Returns:
            Tuple of (width, height) in output units.
        """
        return (float(self.width), float(self.height))


def rotate_point(
    x: float, y: float, angle: float, cx: float = 0, cy: float = 0
) -> tuple[float, float]:
    """Rotate a point around a center.

    Args:
        x: X coordinate of point.
        y: Y coordinate of point.
        angle: Rotation angle in radians.
        cx: X coordinate of rotation center.
        cy: Y coordinate of rotation center.

    Returns:
        Rotated (x, y) coordinates.
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = x - cx
    dy = y - cy
    return (
        cx + dx * cos_a - dy * sin_a,
        cy + dx * sin_a + dy * cos_a,
    )


def normalize_vector(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a vector to unit length.

    Args:
        v: Input vector.

    Returns:
        Normalized vector, or original if zero-length.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
