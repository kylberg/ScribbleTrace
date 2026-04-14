"""Image processing utilities for ScribbleTrace.

This module provides functions for loading, preprocessing, and analyzing
images to prepare them for vectorization algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skimage import filters, io
from skimage.color import rgb2gray
from skimage.transform import resize


@dataclass
class ProcessedImage:
    """Container for a processed image with its metadata.

    Attributes:
        data: The quantized grayscale image data (0 to levels-1).
        original: The original grayscale image (0.0 to 1.0).
        width: Width of the processed image in pixels.
        height: Height of the processed image in pixels.
        levels: Number of quantization levels.
        output_width: Target output width used for scaling.
    """

    data: NDArray[np.float64]
    original: NDArray[np.float64]
    width: int
    height: int
    levels: int
    output_width: float


@dataclass
class GradientData:
    """Container for gradient information.

    Attributes:
        dx: Horizontal gradient component (Sobel).
        dy: Vertical gradient component (Sobel).
        magnitude: Gradient magnitude.
        angle: Gradient angle in radians.
    """

    dx: NDArray[np.float64]
    dy: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    angle: NDArray[np.float64]


def load_image(path: str | Path) -> NDArray[np.float64]:
    """Load an image from file and convert to grayscale.

    Args:
        path: Path to the image file.

    Returns:
        Grayscale image as a 2D numpy array with values in [0, 1].

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file cannot be read as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        img = io.imread(str(path))
    except Exception as e:
        raise ValueError(f"Could not read image file: {path}") from e

    # Convert to grayscale if needed
    if img.ndim == 3:
        if img.shape[2] == 4:
            # RGBA - ignore alpha channel
            img = rgb2gray(img[:, :, :3])
        else:
            img = rgb2gray(img)

    # Normalize to [0, 1]
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    elif img.max() > 1.0:
        img = img.astype(np.float64) / img.max()
    else:
        img = img.astype(np.float64)

    return img


def preprocess(
    image: NDArray[np.float64],
    output_width: float = 40.0,
    levels: int = 7,
    invert: bool = True,
) -> ProcessedImage:
    """Preprocess an image for vectorization.

    This function scales the image to the target output width and quantizes
    the intensity values to the specified number of levels.

    Args:
        image: Grayscale image array with values in [0, 1].
        output_width: Target width in output units (pixels in downsampled image).
        levels: Number of quantization levels for intensity.
        invert: If True, invert the image (dark areas become dense patterns).

    Returns:
        ProcessedImage containing the processed data and metadata.
    """
    # Calculate scale factor to achieve target output width
    scale_factor = round(image.shape[0] / output_width)
    if scale_factor < 1:
        scale_factor = 1

    # Resize image
    new_height = round(image.shape[0] / scale_factor)
    new_width = round(image.shape[1] / scale_factor)
    resized = resize(image, (new_height, new_width), anti_aliasing=True)

    # Store original (resized but not quantized)
    original = resized.copy()

    # Invert if requested (so dark areas produce more marks)
    if invert:
        resized = 1 - resized

    # Quantize to levels
    quantized = np.round(resized * (levels - 1))

    return ProcessedImage(
        data=quantized,
        original=original,
        width=new_width,
        height=new_height,
        levels=levels,
        output_width=output_width,
    )


def compute_gradients(
    image: NDArray[np.float64],
    quantize_magnitude: bool = False,
    magnitude_levels: int = 12,
) -> GradientData:
    """Compute image gradients using Sobel filters.

    Args:
        image: Grayscale image array.
        quantize_magnitude: If True, quantize the gradient magnitude.
        magnitude_levels: Number of levels for magnitude quantization.

    Returns:
        GradientData containing dx, dy, magnitude, and angle.
    """
    # Compute gradients using Sobel filters
    dy = filters.sobel_h(image)
    dx = filters.sobel_v(image)

    # Compute magnitude
    magnitude = np.sqrt(dx**2 + dy**2)

    # Normalize magnitude
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    # Optionally quantize magnitude
    if quantize_magnitude:
        magnitude = np.round(magnitude * (magnitude_levels - 1)) + 1

    # Compute angle
    angle = np.arctan2(dy, dx)

    return GradientData(dx=dx, dy=dy, magnitude=magnitude, angle=angle)
