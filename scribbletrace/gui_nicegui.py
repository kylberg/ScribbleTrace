"""NiceGUI-based web interface for ScribbleTrace.

This module provides a browser-based UI for the ScribbleTrace pen plotter
tool, using the NiceGUI framework.

Usage:
    scribbletrace-gui
    # or
    python -m scribbletrace.gui_nicegui
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import re
import tempfile
from datetime import datetime
from xml.dom import minidom
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scribbletrace import SVGWriter, compute_gradients, preprocess
from scribbletrace.image_processing import ProcessedImage
from scribbletrace.svg_output import SVGConfig
from scribbletrace.algorithms import (
    Circles,
    CirclesConfig,
    Curves,
    CurvesConfig,
    HatchDirection,
    Hatching,
    HatchingConfig,
    Lines,
    LinesConfig,
    Spirals,
    SpiralsConfig,
    Squares,
    SquaresConfig,
)

ui = None

# ---------------------------------------------------------------------------
# Pipeline functions (moved from gui.py to eliminate Gradio dependency)
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent.parent / "examples" / "MagrittePipe.jpg"

PIPELINE_STEP_LABELS = [
    "1. Preprocess",
    "2. Gradient Magnitude",
    "3. Final SVG",
]


def create_sample_image() -> np.ndarray:
    """Create a simple gradient sample image for testing."""
    size = 200
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    img = 1 - np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 1.5
    img = np.clip(img, 0, 1)
    return img


def get_default_gui_image() -> np.ndarray | None:
    """Load the default GUI image if available, otherwise return None."""
    if DEFAULT_IMAGE_PATH.exists():
        try:
            return np.array(Image.open(DEFAULT_IMAGE_PATH).convert("RGB"))
        except Exception:
            return None
    return None


def normalize_input_image(input_image: np.ndarray | None) -> np.ndarray:
    """Normalize an input image to a 2D grayscale float array in [0, 1]."""
    if input_image is None:
        input_image = create_sample_image()

    img = np.asarray(input_image, dtype=np.float64)

    if img.max() > 1.0:
        img = img / 255.0

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        elif img.shape[2] == 3:
            img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        else:
            img = img[:, :, 0]

    img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Could not convert image to 2D grayscale. Shape: {img.shape}")

    return np.clip(img, 0.0, 1.0)


def to_display_uint8(image: np.ndarray, levels: int | None = None) -> np.ndarray:
    """Convert an image array to uint8 for display."""
    arr = np.asarray(image, dtype=np.float64)
    if levels is not None and levels > 1:
        arr = np.clip(arr / float(levels - 1), 0.0, 1.0)
    else:
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def resize_preview_nearest(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a preview image with nearest-neighbor interpolation."""
    target_h, target_w = target_shape
    arr = np.asarray(image, dtype=np.uint8)

    if arr.ndim != 2:
        return arr

    if arr.shape == (target_h, target_w):
        return arr

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.NEAREST
    else:
        resample = Image.NEAREST

    return np.asarray(
        Image.fromarray(arr).resize((target_w, target_h), resample=resample),
        dtype=np.uint8,
    )


def image_signature(image: np.ndarray) -> str:
    """Create a stable signature for a normalized image."""
    digest = hashlib.sha1(image.tobytes()).hexdigest()  # noqa: S324
    return f"{image.shape}:{digest}"


def _normalize_histogram_knots(h_min: float, h_mid: float, h_max: float) -> tuple[float, float, float]:
    """Return safe, ordered histogram transform knot positions in [0, 1]."""
    min_v = float(np.clip(h_min, 0.0, 1.0))
    mid_v = float(np.clip(h_mid, 0.0, 1.0))
    max_v = float(np.clip(h_max, 0.0, 1.0))

    eps = 1e-4
    if min_v > max_v:
        min_v, max_v = max_v, min_v
    mid_v = min(max(mid_v, min_v + eps), max_v - eps)
    if max_v - min_v < 2 * eps:
        min_v = max(0.0, min_v - eps)
        max_v = min(1.0, max_v + eps)
        mid_v = (min_v + max_v) / 2

    return min_v, mid_v, max_v


def apply_histogram_transform(
    image: np.ndarray,
    hist_min: float,
    hist_mid: float,
    hist_max: float,
) -> np.ndarray:
    """Apply a three-knot tonal transform (min->0, mid->0.5, max->1)."""
    min_v, mid_v, max_v = _normalize_histogram_knots(hist_min, hist_mid, hist_max)
    in_levels = np.array([min_v, mid_v, max_v], dtype=np.float64)
    out_levels = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    transformed = np.interp(np.asarray(image, dtype=np.float64), in_levels, out_levels)
    return np.clip(transformed, 0.0, 1.0)


def render_histogram_preview(
    image: np.ndarray,
    hist_min: float,
    hist_mid: float,
    hist_max: float,
    bins: int = 128,
    width: int = 720,
    height: int = 220,
) -> np.ndarray:
    """Render a simple histogram preview with vertical marker lines."""
    values = np.clip(np.asarray(image, dtype=np.float64).ravel(), 0.0, 1.0)
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    max_count = max(1, int(counts.max()))

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for idx, count in enumerate(counts):
        x0 = int(idx * width / bins)
        x1 = int((idx + 1) * width / bins)
        bar_h = int((count / max_count) * (height - 24))
        y0 = height - 1
        y1 = max(0, y0 - bar_h)
        canvas[y1:y0, x0:max(x0 + 1, x1), :] = 40

    min_v, mid_v, max_v = _normalize_histogram_knots(hist_min, hist_mid, hist_max)
    markers = [
        (min_v, np.array([230, 57, 70], dtype=np.uint8)),
        (mid_v, np.array([241, 160, 17], dtype=np.uint8)),
        (max_v, np.array([69, 123, 157], dtype=np.uint8)),
    ]
    for pos, color in markers:
        x = int(np.clip(round(pos * (width - 1)), 0, width - 1))
        canvas[:, max(0, x - 1) : min(width, x + 2), :] = color

    return canvas


def resolve_svg_theme(color_theme: str) -> tuple[str, str]:
    """Return stroke/background colors for a named SVG theme."""
    if color_theme == "White Lines on Black":
        return ("white", "black")
    return ("black", "white")


def estimate_vertex_count(
    processed: ProcessedImage,
    algorithm: str,
    theta_resolution: int,
    circle_points: int,
    max_steps: int,
    bezier_samples: int,
    hatch_horizontal: bool,
    hatch_vertical: bool,
    hatch_diag_right: bool,
    hatch_diag_left: bool,
    min_spacing: float,
) -> int:
    """Estimate vertex count for safety/preview purposes."""
    data = processed.data.astype(int)
    intensity_sum = int(np.sum(data))
    h, w = data.shape
    cells = int(h * w)

    if algorithm == "spirals":
        values = data.ravel()
        zeros = int(np.sum(values == 0))
        nonzero = values[values > 0]
        nz_vertices = np.maximum(3, np.round(theta_resolution * nonzero / 2.0).astype(int))
        nz_vertices = nz_vertices + (nz_vertices % 2 == 0)
        return int(zeros * theta_resolution + int(np.sum(nz_vertices)))

    if algorithm == "circles":
        return int(intensity_sum * (circle_points + 1))

    if algorithm == "squares":
        return int(intensity_sum * 5)

    if algorithm == "lines":
        return int(intensity_sum * 2)

    if algorithm == "curves":
        return int(max(1, intensity_sum) * max(3, max_steps * max(2, bezier_samples // 2)))

    if algorithm == "hatching":
        directions = sum([hatch_horizontal, hatch_vertical, hatch_diag_right, hatch_diag_left])
        directions = max(1, directions)
        spacing = max(0.1, float(min_spacing))
        lines_est = ((w + h) / spacing) * directions
        verts_per_line = max(w, h)
        return int(lines_est * verts_per_line)

    return int(cells)


def generate_svg_content(
    processed: ProcessedImage,
    gradients,
    algorithm: str,
    color_theme: str,
    stroke_width: float,
    randomness_vertex: float,
    randomness_position: float,
    theta_resolution: int,
    spiral_b: float,
    connect_cells: bool,
    circle_points: int,
    small_first: bool,
    hatch_horizontal: bool,
    hatch_vertical: bool,
    hatch_diag_right: bool,
    hatch_diag_left: bool,
    min_spacing: float,
    max_spacing: float,
    lines_segment_length: float,
    randomness_length: float,
    min_gradient_scale: float,
    max_gradient_scale: float,
    curves_segment_length: float,
    curves_randomness_length: float,
    max_steps: int,
    step_size: float,
    bezier_samples: int,
) -> tuple[str, np.ndarray]:
    """Generate SVG content and gradient magnitude from already computed intermediates."""
    if algorithm == "spirals":
        config = SpiralsConfig(
            randomness_vertex=randomness_vertex,
            randomness_position=randomness_position,
            stroke_width=stroke_width,
            theta_resolution=theta_resolution,
            spiral_b=spiral_b,
            connect_cells=connect_cells,
        )
        algo = Spirals(processed, config)

    elif algorithm == "circles":
        config = CirclesConfig(
            randomness_vertex=randomness_vertex,
            randomness_position=randomness_position,
            stroke_width=stroke_width,
            circle_points=circle_points,
            small_first=small_first,
        )
        algo = Circles(processed, config)

    elif algorithm == "squares":
        config = SquaresConfig(
            randomness_vertex=randomness_vertex,
            randomness_position=randomness_position,
            stroke_width=stroke_width,
            small_first=small_first,
        )
        algo = Squares(processed, config)

    elif algorithm == "lines":
        config = LinesConfig(
            randomness_vertex=randomness_vertex,
            randomness_position=randomness_position,
            stroke_width=stroke_width,
            segment_length=lines_segment_length,
            randomness_length=randomness_length,
            min_gradient_scale=min_gradient_scale,
            max_gradient_scale=max_gradient_scale,
        )
        algo = Lines(processed, config, gradients)

    elif algorithm == "curves":
        config = CurvesConfig(
            randomness_vertex=randomness_vertex,
            randomness_position=randomness_position,
            stroke_width=stroke_width,
            segment_length=curves_segment_length,
            randomness_length=curves_randomness_length,
            max_steps=max_steps,
            step_size=step_size,
            bezier_samples=bezier_samples,
        )
        algo = Curves(processed, config, gradients)

    elif algorithm == "hatching":
        directions = []
        if hatch_horizontal:
            directions.append(HatchDirection.HORIZONTAL)
        if hatch_vertical:
            directions.append(HatchDirection.VERTICAL)
        if hatch_diag_right:
            directions.append(HatchDirection.DIAGONAL_RIGHT)
        if hatch_diag_left:
            directions.append(HatchDirection.DIAGONAL_LEFT)
        if not directions:
            directions = [HatchDirection.DIAGONAL_RIGHT]

        config = HatchingConfig(
            randomness_vertex=randomness_vertex,
            randomness_position=randomness_position,
            stroke_width=stroke_width,
            directions=directions,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
        )
        algo = Hatching(processed, config, gradients)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    paths = algo.process()
    stroke_color, background_color = resolve_svg_theme(color_theme)
    svg_config = SVGConfig(
        width=processed.width + 2,
        height=processed.height + 2,
        stroke_color=stroke_color,
        background=background_color,
    )
    writer = SVGWriter(paths, config=svg_config)
    return writer.to_string(), gradients.magnitude


def run_pipeline_to_stage(
    pipeline_cache: dict[str, Any] | None,
    input_image: np.ndarray | None,
    target_stage_index: int,
    algorithm: str,
    color_theme: str,
    hist_min: float,
    hist_mid: float,
    hist_max: float,
    output_width: float,
    levels: int,
    invert: bool,
    gradient_sigma: float,
    stroke_width: float,
    randomness_vertex: float,
    randomness_position: float,
    theta_resolution: int,
    spiral_b: float,
    connect_cells: bool,
    circle_points: int,
    small_first: bool,
    hatch_horizontal: bool,
    hatch_vertical: bool,
    hatch_diag_right: bool,
    hatch_diag_left: bool,
    min_spacing: float,
    max_spacing: float,
    lines_segment_length: float,
    randomness_length: float,
    min_gradient_scale: float,
    max_gradient_scale: float,
    curves_segment_length: float,
    curves_randomness_length: float,
    max_steps: int,
    step_size: float,
    bezier_samples: int,
    enable_vertex_guard: bool,
    max_estimated_vertices: float,
) -> tuple[
    None,  # svg_preview_image (not used in NiceGUI)
    str,  # svg_code
    None,  # download_update (not used in NiceGUI)
    np.ndarray | None,  # gradient_display
    np.ndarray | None,  # grayscale_display
    np.ndarray | None,  # downscaled_display
    np.ndarray | None,  # quantized_display
    np.ndarray | None,  # hist_preview
    str,  # complexity_text
    str,  # status
    dict[str, Any],  # cache
]:
    """Run the image pipeline up to a target stage and return GUI-ready outputs."""
    cache = pipeline_cache if isinstance(pipeline_cache, dict) else {}
    target_stage_index = int(np.clip(target_stage_index, 0, len(PIPELINE_STEP_LABELS) - 1))

    hist_min, hist_mid, hist_max = _normalize_histogram_knots(hist_min, hist_mid, hist_max)

    grayscale_original = normalize_input_image(input_image)
    grayscale = apply_histogram_transform(grayscale_original, hist_min, hist_mid, hist_max)
    image_key = image_signature(grayscale_original)
    if cache.get("image_key") != image_key:
        cache = {"image_key": image_key, "grayscale": grayscale_original}
    else:
        cache["grayscale"] = grayscale_original

    hist_preview = render_histogram_preview(grayscale_original, hist_min, hist_mid, hist_max)

    grayscale_display = to_display_uint8(grayscale)
    preview_shape = grayscale_display.shape

    downscaled_display = None
    quantized_display = None
    gradient_display = None
    complexity_text = ""  # No longer displayed

    processed: ProcessedImage | None = None
    gradients = None

    if target_stage_index >= 0:
        processed_key = (
            float(output_width),
            int(levels),
            bool(invert),
            round(float(hist_min), 6),
            round(float(hist_mid), 6),
            round(float(hist_max), 6),
        )
        if cache.get("processed_key") != processed_key:
            cache["processed"] = preprocess(grayscale, output_width=output_width, levels=levels, invert=invert)
            cache["processed_key"] = processed_key
            cache["gradients_by_sigma"] = {}
            cache["svg_by_key"] = {}
        processed = cache["processed"]
        resized_inverted = 1.0 - processed.original if invert else processed.original
        step1_display = to_display_uint8(resized_inverted)
        step1_display = resize_preview_nearest(step1_display, preview_shape)
        grayscale_display = step1_display
        downscaled_display = step1_display

    if target_stage_index >= 0 and processed is not None:
        quantized_display = to_display_uint8(processed.data, levels=processed.levels)
        quantized_display = resize_preview_nearest(quantized_display, preview_shape)

    if target_stage_index >= 1 and processed is not None:
        sigma_key = round(float(gradient_sigma), 4)
        gradients_by_sigma = cache.setdefault("gradients_by_sigma", {})
        gradients = gradients_by_sigma.get(sigma_key)
        if gradients is None:
            gradients = compute_gradients(processed.original, sigma=gradient_sigma)
            gradients_by_sigma[sigma_key] = gradients
        gradient_display = to_display_uint8(gradients.magnitude)
        gradient_display = resize_preview_nearest(gradient_display, preview_shape)

    svg_code = ""

    if target_stage_index >= 2:
        if processed is None:
            processed = preprocess(grayscale, output_width=output_width, levels=levels, invert=invert)

        estimated_vertices = estimate_vertex_count(
            processed,
            algorithm,
            theta_resolution,
            circle_points,
            max_steps,
            bezier_samples,
            hatch_horizontal,
            hatch_vertical,
            hatch_diag_right,
            hatch_diag_left,
            min_spacing,
        )

        if enable_vertex_guard and estimated_vertices > int(max_estimated_vertices):
            status = f"Skipped: {estimated_vertices:,} vertices > {int(max_estimated_vertices):,} limit"
            return (
                None,
                "",
                None,
                gradient_display,
                grayscale_display,
                downscaled_display,
                quantized_display,
                hist_preview,
                complexity_text,
                status,
                cache,
            )

        sigma_key = round(float(gradient_sigma), 4)
        gradients_by_sigma = cache.setdefault("gradients_by_sigma", {})
        gradients = gradients_by_sigma.get(sigma_key)
        if gradients is None:
            gradients = compute_gradients(processed.original, sigma=gradient_sigma)
            gradients_by_sigma[sigma_key] = gradients

        svg_key = (
            algorithm,
            color_theme,
            float(stroke_width),
            float(randomness_vertex),
            float(randomness_position),
            int(theta_resolution),
            float(spiral_b),
            bool(connect_cells),
            int(circle_points),
            bool(small_first),
            bool(hatch_horizontal),
            bool(hatch_vertical),
            bool(hatch_diag_right),
            bool(hatch_diag_left),
            float(min_spacing),
            float(max_spacing),
            float(lines_segment_length),
            float(randomness_length),
            float(min_gradient_scale),
            float(max_gradient_scale),
            float(curves_segment_length),
            float(curves_randomness_length),
            int(max_steps),
            float(step_size),
            int(bezier_samples),
            sigma_key,
        )
        svg_by_key = cache.setdefault("svg_by_key", {})
        cached_svg = svg_by_key.get(svg_key)
        if cached_svg is None:
            cached_svg = generate_svg_content(
                processed,
                gradients,
                algorithm,
                color_theme,
                stroke_width,
                randomness_vertex,
                randomness_position,
                theta_resolution,
                spiral_b,
                connect_cells,
                circle_points,
                small_first,
                hatch_horizontal,
                hatch_vertical,
                hatch_diag_right,
                hatch_diag_left,
                min_spacing,
                max_spacing,
                lines_segment_length,
                randomness_length,
                min_gradient_scale,
                max_gradient_scale,
                curves_segment_length,
                curves_randomness_length,
                max_steps,
                step_size,
                bezier_samples,
            )
            svg_by_key[svg_key] = cached_svg

        svg_content, gradient_magnitude = cached_svg
        svg_code = svg_content

        # Keep gradient tab meaningful for all algorithms once gradients are computed
        if gradient_display is None:
            gradient_display = to_display_uint8(gradient_magnitude)
            gradient_display = resize_preview_nearest(gradient_display, preview_shape)

    status = ""
    return (
        None,
        svg_code,
        None,
        gradient_display,
        grayscale_display,
        downscaled_display,
        quantized_display,
        hist_preview,
        complexity_text,
        status,
        cache,
    )


# ---------------------------------------------------------------------------
# NiceGUI-specific code
# ---------------------------------------------------------------------------


@dataclass
class NiceGuiState:
    """Mutable UI state for the NiceGUI app."""

    input_image: np.ndarray | None = None
    input_image_name: str = ""
    pipeline_cache: dict[str, Any] = field(default_factory=dict)
    current_stage: int = 0
    latest_svg_content: str = ""
    suppress_hist_callbacks: bool = False
    hist_drag_target: str | None = None
    hist_source_width: int = 720
    using_default_input: bool = True
    is_processing: bool = False
    # Uploaded images library: {name: np.ndarray}
    uploaded_images: dict[str, np.ndarray] = field(default_factory=dict)
    # Backdrop images for preview overlay (all at processed resolution)
    backdrop_gradient: np.ndarray | None = None
    backdrop_quantized: np.ndarray | None = None
    backdrop_grayscale: np.ndarray | None = None
    backdrop_original: np.ndarray | None = None  # Original RGB at processed resolution
    # Processed image dimensions for SVG alignment
    processed_width: int = 0
    processed_height: int = 0


HIST_WIDTH = 480
HIST_HEIGHT = 150
PREVIEW_CONTAINER_SIZES = {
    "Small": 420,
    "Medium": 560,
    "Large": 720,
    "X-Large": 900,
}
UI_STEP_LABELS = ["1. Image Processing", "2. Vector Generation"]
DEFAULT_PREVIEW_WIDTH = "w-[420px]"
PREVIEW_WIDTH_CLASSES = {
    "Small": "w-[320px]",
    "Medium": "w-[420px]",
    "Large": "w-[560px]",
}
ALGORITHM_THUMB_DIR = Path(__file__).parent / "assets" / "algorithm_thumbs"
ALGORITHM_ORDER = ["spirals", "circles", "squares", "lines", "curves", "hatching"]
PAPER_SIZES_MM = {
    "A3": (297, 420),
    "A4": (210, 297),
    "A5": (148, 210),
    "A6": (105, 148),
}
# Rosé Pine color palette
ROSEPINE_COLORS = {
    "Rose": "#ebbcba",
    "Pine": "#31748f",
    "Foam": "#9ccfd8",
    "Iris": "#c4a7e7",
    "Gold": "#f6c177",
    "Love": "#eb6f92",
    "Text": "#e0def4",
    "Subtle": "#908caa",
    "Muted": "#6e6a86",
    "Base": "#191724",
    "Surface": "#1f1d2e",
    "Overlay": "#26233a",
}

# Stroke colors for preview (includes Black/White + Rosé Pine)
STROKE_COLORS = {
    "Black": "#000000",
    "White": "#ffffff",
    **ROSEPINE_COLORS,
}


class _ValueProxy:
    """Small value holder to keep non-visual control values in the same API shape."""

    def __init__(self, value: Any, on_change=None) -> None:
        self.value = value
        self._on_change = on_change

    def set_value(self, value: Any) -> None:
        self.value = value
        if self._on_change is not None:
            self._on_change(value)


def _require_nicegui():
    global ui
    if ui is not None:
        return ui
    try:
        from nicegui import ui as _ui
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("NiceGUI is not installed. Install with: pip install nicegui") from exc
    ui = _ui
    return ui


def _pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _np_to_data_url(image: np.ndarray | None) -> str:
    if image is None:
        return ""

    arr = np.asarray(image)
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.zeros((32, 32, 3), dtype=np.uint8)

    return _pil_to_data_url(Image.fromarray(arr))


def _downscale_for_ui(image: np.ndarray | None, max_dim: int = 720) -> np.ndarray | None:
    if image is None:
        return None

    arr = np.asarray(image)
    if arr.ndim < 2:
        return arr

    h, w = arr.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return arr

    scale = float(max_dim) / float(largest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resized = Image.fromarray(arr).resize((new_w, new_h), resample=resampling)
    return np.asarray(resized)


def _safe_value(ctrl):
    return ctrl.value


def _svg_file_to_data_url(file_path: Path) -> str:
    raw = file_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _svg_text_to_data_url(svg_text: str) -> str:
    encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _format_svg_code(svg_text: str) -> str:
    if not svg_text.strip():
        return ""
    try:
        parsed = minidom.parseString(svg_text.encode("utf-8"))
        pretty = parsed.toprettyxml(indent="  ")
        # Remove empty lines introduced by minidom pretty printer.
        return "\n".join(line for line in pretty.splitlines() if line.strip())
    except Exception:
        return svg_text


def _slider_precision(step: float) -> int:
    step_str = f"{float(step):.10f}".rstrip("0").rstrip(".")
    if "." not in step_str:
        return 0
    return len(step_str.split(".", maxsplit=1)[1])


def _format_slider_value(value: float | int, precision: int) -> str:
    if precision <= 0:
        return str(int(round(float(value))))
    return f"{float(value):.{precision}f}"


def _style_histogram_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    arr = np.asarray(image).copy()
    if arr.ndim != 3 or arr.shape[2] != 3:
        return arr

    # Dark UI-style background with muted accents for bars and marker lines.
    bg = np.array([20, 26, 36], dtype=np.uint8)
    dist = np.array([110, 132, 148], dtype=np.uint8)

    marker_map = {
        (230, 57, 70): np.array([156, 94, 101], dtype=np.uint8),
        (241, 160, 17): np.array([173, 144, 93], dtype=np.uint8),
        (69, 123, 157): np.array([98, 129, 147], dtype=np.uint8),
    }

    white_bg = np.all(arr >= 245, axis=2)
    dark_bars = np.all(arr <= 80, axis=2)

    arr[white_bg] = bg
    arr[dark_bars] = dist

    for rgb, mapped in marker_map.items():
        color_mask = np.all(arr == np.array(rgb, dtype=np.uint8), axis=2)
        arr[color_mask] = mapped

    return arr


def _build_settings_snapshot(controls: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "algorithm",
        "color_theme",
        "paper_size",
        "paper_orientation",
        "hist_min",
        "hist_mid",
        "hist_max",
        "output_width",
        "levels",
        "gradient_sigma",
        "invert",
        "stroke_width",
        "randomness_vertex",
        "randomness_position",
        "theta_resolution",
        "spiral_b",
        "connect_cells",
        "circle_points",
        "small_first",
        "hatch_horizontal",
        "hatch_vertical",
        "hatch_diag_right",
        "hatch_diag_left",
        "min_spacing",
        "max_spacing",
        "lines_segment_length",
        "randomness_length",
        "min_gradient_scale",
        "max_gradient_scale",
        "curves_segment_length",
        "curves_randomness_length",
        "max_steps",
        "step_size",
        "bezier_samples",
        "enable_vertex_guard",
        "max_estimated_vertices",
    ]
    return {k: _safe_value(controls[k]) for k in keys}


def _fit_svg_dimensions(svg_text: str, paper_size: str, orientation: str = "Portrait") -> tuple[str, dict]:
    """Scale SVG to fit paper size while preserving stroke widths.
    
    Directly scales path coordinates to fit the paper, keeping stroke-width
    unchanged so the physical line thickness remains as specified.
    Follows Inkscape convention: viewBox matches document mm dimensions.
    
    Returns:
        Tuple of (transformed_svg, placement_info) where placement_info contains:
        - offset_x, offset_y: centering offset in mm
        - scaled_width, scaled_height: content size in mm
        - paper_width, paper_height: paper dimensions in mm
    """
    placement_info = {}
    dims = PAPER_SIZES_MM.get(paper_size)
    if dims is None:
        return svg_text, placement_info

    svg_match = re.search(r"<svg\b[^>]*>", svg_text)
    if not svg_match:
        return svg_text, placement_info

    svg_tag = svg_match.group(0)
    # dims is (width, height) in portrait; swap for landscape
    paper_width_mm, paper_height_mm = dims
    if orientation == "Landscape":
        paper_width_mm, paper_height_mm = paper_height_mm, paper_width_mm

    placement_info["paper_width"] = paper_width_mm
    placement_info["paper_height"] = paper_height_mm

    # Extract current viewBox to get content bounds
    viewbox_match = re.search(r'viewBox\s*=\s*["\']([^"\']+)["\']', svg_tag)
    if not viewbox_match:
        return svg_text, placement_info
    
    vb_parts = viewbox_match.group(1).split()
    if len(vb_parts) != 4:
        return svg_text, placement_info
    
    vb_min_x, vb_min_y, vb_width, vb_height = map(float, vb_parts)
    if vb_width <= 0 or vb_height <= 0:
        return svg_text, placement_info
    
    # Calculate scale to fit content in paper while maintaining aspect ratio
    content_aspect = vb_width / vb_height
    paper_aspect = paper_width_mm / paper_height_mm
    
    if content_aspect > paper_aspect:
        # Content is wider - fit to paper width
        scale = paper_width_mm / vb_width
    else:
        # Content is taller - fit to paper height  
        scale = paper_height_mm / vb_height
    
    # Calculate offset to center content
    scaled_width = vb_width * scale
    scaled_height = vb_height * scale
    offset_x = (paper_width_mm - scaled_width) / 2
    offset_y = (paper_height_mm - scaled_height) / 2

    placement_info["offset_x"] = offset_x
    placement_info["offset_y"] = offset_y
    placement_info["scaled_width"] = scaled_width
    placement_info["scaled_height"] = scaled_height
    placement_info["scale"] = scale

    def _set_attr(tag: str, name: str, value: str) -> str:
        pattern = rf"\b{name}\s*=\s*(\"[^\"]*\"|'[^']*')"
        if re.search(pattern, tag):
            return re.sub(pattern, f'{name}="{value}"', tag, count=1)
        return tag[:-1] + f' {name}="{value}">'

    # Update document dimensions and viewBox to paper size
    new_viewbox = f"0 0 {paper_width_mm} {paper_height_mm}"
    new_svg_tag = _set_attr(svg_tag, "width", f"{paper_width_mm}mm")
    new_svg_tag = _set_attr(new_svg_tag, "height", f"{paper_height_mm}mm")
    new_svg_tag = _set_attr(new_svg_tag, "viewBox", new_viewbox)
    
    result = svg_text[: svg_match.start()] + new_svg_tag + svg_text[svg_match.end() :]
    
    # Scale path coordinates: transform each number in d="..." attributes
    def scale_path_d(m):
        d = m.group(1)
        result_parts = []
        i = 0
        while i < len(d):
            # Check for command letter
            if d[i].isalpha():
                result_parts.append(d[i])
                i += 1
            # Check for number (including negative and decimal)
            elif d[i].isdigit() or d[i] == '-' or d[i] == '.':
                # Extract the full number
                j = i
                if d[i] == '-':
                    j += 1
                while j < len(d) and (d[j].isdigit() or d[j] == '.'):
                    j += 1
                num_str = d[i:j]
                try:
                    val = float(num_str)
                    scaled_val = val * scale
                    result_parts.append(f"{scaled_val:.4f}")
                except ValueError:
                    result_parts.append(num_str)
                i = j
            else:
                # Whitespace or comma - preserve as-is
                result_parts.append(d[i])
                i += 1
        return f'd="{"".join(result_parts)}"'
    
    result = re.sub(r'd="([^"]+)"', scale_path_d, result)
    
    # Handle background rect separately - it should fill the entire paper
    # Extract and remove original rect, replace with full-paper rect
    bg_rect_match = re.search(r'<rect\b[^>]*fill="([^"]+)"[^>]*/?>(?:</rect>)?', result)
    bg_fill = None
    if bg_rect_match:
        bg_fill = bg_rect_match.group(1)
        result = result[:bg_rect_match.start()] + result[bg_rect_match.end():]
    
    # Add offset to center content using a wrapper group
    # Find the closing </svg> and wrap content
    svg_end_match = re.search(r"</svg\s*>", result)
    if svg_end_match:
        # Find where content starts (after opening svg tag)
        new_svg_match = re.search(r"<svg\b[^>]*>", result)
        if new_svg_match:
            before = result[:new_svg_match.end()]
            content = result[new_svg_match.end():svg_end_match.start()]
            after = result[svg_end_match.start():]
            
            # Add full-paper background rect if there was one
            bg_rect = ""
            if bg_fill:
                bg_rect = f'<rect x="0" y="0" width="{paper_width_mm}" height="{paper_height_mm}" fill="{bg_fill}"/>'
            
            # Wrap paths in translate group if needed
            if abs(offset_x) > 0.01 or abs(offset_y) > 0.01:
                result = before + bg_rect + f'<g transform="translate({offset_x:.4f},{offset_y:.4f})">' + content + '</g>' + after
            else:
                result = before + bg_rect + content + after
    
    return result, placement_info


def _paper_size_icon_svg(size: str) -> str:
    """Generate an SVG icon representing a paper size with relative dimensions."""
    # Base sizes in px (A3 is largest) - portrait orientation
    size_map = {
        "A3": (60, 85),
        "A4": (42, 60),
        "A5": (30, 42),
        "A6": (21, 30),
    }
    w, h = size_map.get(size, (42, 60))
    
    # Always use unselected styling; selection is handled via CSS classes
    border_color = "#39414f"
    bg_color = "#141a24"
    border_width = "1"
    
    return (
        f'<svg viewBox="0 0 {w + 4} {h + 4}" width="{w + 4}" height="{h + 4}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="2" y="2" width="{w}" height="{h}" fill="{bg_color}" '
        f'stroke="{border_color}" stroke-width="{border_width}" rx="2"/>'
        f'<text x="{(w + 4) / 2}" y="{(h + 4) / 2}" font-size="10" font-weight="bold" '
        f'fill="#908caa" text-anchor="middle" dominant-baseline="middle">{size}</text>'
        f'</svg>'
    )


def _orientation_icon_svg(orientation: str) -> str:
    """Generate an SVG icon representing paper orientation."""
    border_color = "#39414f"
    bg_color = "#141a24"
    
    if orientation == "Portrait":
        w, h = 24, 32
    else:  # Landscape
        w, h = 32, 24
    
    return (
        f'<svg viewBox="0 0 {w + 8} {h + 8}" width="{w + 8}" height="{h + 8}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="4" y="4" width="{w}" height="{h}" fill="{bg_color}" '
        f'stroke="{border_color}" stroke-width="1" rx="2"/>'
        # Add line icon to indicate text direction
        f'<line x1="8" y1="{4 + h//3}" x2="{w}" y2="{4 + h//3}" stroke="#6e6a86" stroke-width="1"/>'
        f'<line x1="8" y1="{4 + h//2}" x2="{w - 4}" y2="{4 + h//2}" stroke="#6e6a86" stroke-width="1"/>'
        f'<line x1="8" y1="{4 + 2*h//3}" x2="{w - 8}" y2="{4 + 2*h//3}" stroke="#6e6a86" stroke-width="1"/>'
        f'</svg>'
    )


def _inject_svg_settings_metadata(svg_text: str, settings: dict[str, Any]) -> str:
    if not svg_text.strip():
        return svg_text

    metadata_json = json.dumps({"version": 1, "settings": settings}, separators=(",", ":"))
    metadata_block = (
        "<metadata id=\"scribbletrace-settings\">"
        f"<![CDATA[{metadata_json}]]>"
        "</metadata>"
    )

    # Replace existing ScribbleTrace metadata if present, otherwise insert after opening <svg>.
    existing_pattern = re.compile(
        r"<metadata\b[^>]*id\s*=\s*([\"'])scribbletrace-settings\1[^>]*>.*?</metadata>",
        re.DOTALL,
    )
    if existing_pattern.search(svg_text):
        return existing_pattern.sub(metadata_block, svg_text, count=1)

    match = re.search(r"<svg\b[^>]*>", svg_text)
    if not match:
        return svg_text
    insert_at = match.end()
    return svg_text[:insert_at] + metadata_block + svg_text[insert_at:]


def _slugify_filename_part(value: str, fallback: str = "output") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _build_download_base_name(input_name: str, algorithm: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    input_base = _slugify_filename_part(Path(input_name or "input").stem, fallback="input")
    algo_base = _slugify_filename_part(algorithm or "algorithm", fallback="algorithm")
    return f"{input_base}_{algo_base}_{stamp}"


def create_gui() -> None:
    _require_nicegui()

    # Initialize with default image already in the gallery
    default_img = get_default_gui_image()
    state = NiceGuiState(
        input_image=default_img,
        input_image_name="default",
        using_default_input=True,
        uploaded_images={"default": default_img} if default_img is not None else {},
    )

    controls: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    algorithm_tiles: dict[str, Any] = {}
    theme_tiles: dict[str, Any] = {}
    vector_groups: dict[str, Any] = {}
    paper_size_tiles: dict[str, Any] = {}
    orientation_tiles: dict[str, Any] = {}
    source_tiles: dict[str, Any] = {}
    backdrop_tiles: dict[str, Any] = {}

    def _update_backdrop_tile_styles() -> None:
        selected = str(_safe_value(controls.get("preview_backdrop", _ValueProxy("Transparent"))))
        for name, tile in backdrop_tiles.items():
            if name == selected:
                tile.classes(remove="st-source-tile", add="st-source-tile-active")
            else:
                tile.classes(remove="st-source-tile-active", add="st-source-tile")

    def _set_backdrop(name: str) -> None:
        controls["preview_backdrop"].set_value(name)
        _update_svg_preview_html()

    def _set_stroke_color(color: str) -> None:
        controls["preview_stroke_color"].set_value(color)
        _update_svg_preview_html()

    def _update_source_tile_styles() -> None:
        selected = str(_safe_value(controls.get("vector_source", _ValueProxy("levels"))))
        for name, tile in source_tiles.items():
            if name == selected:
                tile.classes(remove="st-source-tile", add="st-source-tile-active")
            else:
                tile.classes(remove="st-source-tile-active", add="st-source-tile")

    def _set_vector_source(name: str) -> None:
        controls["vector_source"].set_value(name)

    def _update_algorithm_tile_styles() -> None:
        selected = str(_safe_value(controls.get("algorithm", _ValueProxy("spirals"))))
        for name, tile in algorithm_tiles.items():
            if name == selected:
                tile.classes(remove="st-alg-tile", add="st-alg-tile-active")
            else:
                tile.classes(remove="st-alg-tile-active", add="st-alg-tile")

    def _set_algorithm(name: str) -> None:
        controls["algorithm"].set_value(name)

    def _update_vector_setting_visibility() -> None:
        selected = str(_safe_value(controls.get("algorithm", _ValueProxy("spirals"))))
        selected_group = {
            "spirals": "spirals",
            "circles": "circles_squares",
            "squares": "circles_squares",
            "lines": "lines",
            "curves": "curves",
            "hatching": "hatching",
        }.get(selected, "spirals")
        for name, group in vector_groups.items():
            group.set_visibility(name == selected_group)

    def _update_theme_tile_styles() -> None:
        selected = str(_safe_value(controls.get("color_theme", _ValueProxy("Black Lines on White"))))
        for name, tile in theme_tiles.items():
            if name == selected:
                tile.classes(remove="st-theme-tile", add="st-theme-tile-active")
            else:
                tile.classes(remove="st-theme-tile-active", add="st-theme-tile")

    def _set_color_theme(name: str) -> None:
        controls["color_theme"].set_value(name)

    def _update_paper_size_tile_styles() -> None:
        selected = str(_safe_value(controls.get("paper_size", _ValueProxy("A4"))))
        for name, tile in paper_size_tiles.items():
            if name == selected:
                tile.classes(remove="cursor-pointer", add="ring-2 ring-offset-1 ring-offset-[#141a24] ring-[#9ccfd8]")
            else:
                tile.classes(remove="ring-2 ring-offset-1 ring-offset-[#141a24] ring-[#9ccfd8]", add="cursor-pointer")

    def _set_paper_size(name: str) -> None:
        controls["paper_size"].set_value(name)
        _update_paper_size_tile_styles()

    def _update_orientation_tile_styles() -> None:
        selected = str(_safe_value(controls.get("paper_orientation", _ValueProxy("Portrait"))))
        for name, tile in orientation_tiles.items():
            if name == selected:
                tile.classes(remove="cursor-pointer", add="ring-2 ring-offset-1 ring-offset-[#141a24] ring-[#9ccfd8]")
            else:
                tile.classes(remove="ring-2 ring-offset-1 ring-offset-[#141a24] ring-[#9ccfd8]", add="cursor-pointer")

    def _set_orientation(name: str) -> None:
        controls["paper_orientation"].set_value(name)
        _update_orientation_tile_styles()

    def _apply_preview_size() -> None:
        size = str(_safe_value(controls.get("preview_size", _ValueProxy("Medium"))))
        width_class = PREVIEW_WIDTH_CLASSES.get(size, DEFAULT_PREVIEW_WIDTH)
        preview_keys = [
            "preview_grayscale",
            "preview_hist_downsampled",
            "preview_invert_levels",
            "preview_gradmag",
        ]
        for key in preview_keys:
            if key in outputs:
                outputs[key].classes(remove="w-[320px] w-[420px] w-[560px]", add=width_class)

    def _update_svg_preview_html() -> None:
        # Get preview container size
        preview_size = str(_safe_value(controls.get("preview_size", _ValueProxy("Medium"))))
        container_size = PREVIEW_CONTAINER_SIZES.get(preview_size, 560)
        
        # Show processing message while generating vectors
        if state.is_processing:
            outputs["svg_preview_html"].set_content(
                f'<div class="st-card" style="width:{container_size}px; height:{container_size}px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:16px;">'
                '<div style="font-size:48px;">⏳</div>'
                '<div style="font-size:20px; font-weight:600; color:#9ccfd8;">Generating Vectors...</div>'
                '<div style="font-size:14px; color:#908caa;">Processing your image (1-15 seconds)</div>'
                '</div>'
            )
            return

        svg_code = state.latest_svg_content or ""
        if not svg_code.strip():
            outputs["svg_preview_html"].set_content(
                f'<div class="st-card" style="width:{container_size}px; height:{container_size}px; padding:12px; color:#908caa; display:flex; align-items:center; justify-content:center;">No SVG generated yet.</div>'
            )
            return

        # Get backdrop preference
        preview_backdrop = str(_safe_value(controls.get("preview_backdrop", _ValueProxy("Transparent"))))
        
        # Get paper size and orientation for proper scaling
        paper_size = str(_safe_value(controls.get("paper_size", _ValueProxy("A4"))))
        orientation = str(_safe_value(controls.get("paper_orientation", _ValueProxy("Portrait"))))
        
        # Zoom level from preset toggle (default to "Fit" if not set)
        zoom_val = str(_safe_value(controls.get("svg_zoom", _ValueProxy("Fit"))))
        zoom_fit = zoom_val == "Fit"
        zoom = int(zoom_val.rstrip("%")) if not zoom_fit else 100
        
        # Get backdrop image data URL if needed
        backdrop_url = ""
        if preview_backdrop in ("Original", "Grayscale", "Levels", "Gradient Magnitude"):
            backdrop_img = None
            if preview_backdrop == "Gradient Magnitude" and state.backdrop_gradient is not None:
                backdrop_img = state.backdrop_gradient
            elif preview_backdrop == "Levels" and state.backdrop_quantized is not None:
                backdrop_img = state.backdrop_quantized
            elif preview_backdrop == "Grayscale" and state.backdrop_grayscale is not None:
                backdrop_img = state.backdrop_grayscale
            elif preview_backdrop == "Original" and state.backdrop_original is not None:
                backdrop_img = state.backdrop_original
            
            if backdrop_img is not None:
                backdrop_url = _np_to_data_url(backdrop_img)
        
        # Scale SVG to paper dimensions - always use the same coordinate system
        svg_for_preview, placement = _fit_svg_dimensions(svg_code, paper_size, orientation)
        
        # Interpolation: pixelated (nearest neighbor) vs smooth (auto/linear)
        pixelated = bool(_safe_value(controls.get("pixelated", _ValueProxy(True))))
        img_rendering = "pixelated" if pixelated else "auto"
        
        # Get positioning info
        offset_x = placement.get("offset_x", 0) if placement else 0
        offset_y = placement.get("offset_y", 0) if placement else 0
        scale = placement.get("scale", 1) if placement else 1
        paper_w = placement.get("paper_width", 210) if placement else 210
        paper_h = placement.get("paper_height", 297) if placement else 297
        
        # Always remove original background rect (we'll add our own)
        svg_for_preview = re.sub(r'<rect[^>]*fill="[^"]+"[^>]*/?>(?:</rect>)?', '', svg_for_preview)
        
        # Build background/backdrop element based on selection
        backdrop_elem = ""
        margin_adjust = 1 * scale if scale > 0 else 0
        
        # Checkerboard pattern definition (used as base for all non-transparent backdrops too)
        checker_size = 4  # mm per checker square
        checker_defs = (
            f'<defs>'
            f'<pattern id="checker" width="{checker_size * 2}" height="{checker_size * 2}" patternUnits="userSpaceOnUse">'
            f'<rect width="{checker_size * 2}" height="{checker_size * 2}" fill="#2a2e38"/>'
            f'<rect width="{checker_size}" height="{checker_size}" fill="#1e2228"/>'
            f'<rect x="{checker_size}" y="{checker_size}" width="{checker_size}" height="{checker_size}" fill="#1e2228"/>'
            f'</pattern>'
            f'</defs>'
        )
        checker_rect = f'<rect x="0" y="0" width="{paper_w}" height="{paper_h}" fill="url(#checker)"/>'
        
        if preview_backdrop == "Transparent":
            # Just checkerboard for entire paper
            backdrop_elem = checker_defs + checker_rect
        elif preview_backdrop in ("White", "Black") and state.processed_width > 0 and state.processed_height > 0 and scale > 0:
            # Checkerboard base, then solid color only in image area
            img_w_mm = state.processed_width * scale
            img_h_mm = state.processed_height * scale
            img_x = offset_x + margin_adjust
            img_y = offset_y + margin_adjust
            fill_color = "#ffffff" if preview_backdrop == "White" else "#000000"
            backdrop_elem = (
                checker_defs + checker_rect +
                f'<rect x="{img_x:.4f}" y="{img_y:.4f}" width="{img_w_mm:.4f}" height="{img_h_mm:.4f}" fill="{fill_color}"/>'
            )
        elif preview_backdrop in ("White", "Black"):
            # Fallback if no image dimensions - full paper solid color
            fill_color = "#ffffff" if preview_backdrop == "White" else "#000000"
            backdrop_elem = f'<rect x="0" y="0" width="{paper_w}" height="{paper_h}" fill="{fill_color}"/>'
        elif backdrop_url and state.processed_width > 0 and state.processed_height > 0 and scale > 0:
            # Image backdrop - checkerboard base + image
            # All backdrops use same bounding box (processed dimensions)
            # Original just has more pixels (full res) in that same area
            img_w_mm = state.processed_width * scale
            img_h_mm = state.processed_height * scale
            img_x = offset_x + margin_adjust
            img_y = offset_y + margin_adjust
            
            backdrop_elem = (
                checker_defs + checker_rect +
                f'<image href="{backdrop_url}" '
                f'x="{img_x:.4f}" y="{img_y:.4f}" '
                f'width="{img_w_mm:.4f}" height="{img_h_mm:.4f}" '
                f'preserveAspectRatio="none" '
                f'style="image-rendering:{img_rendering}"/>'
            )
        
        # Insert backdrop after opening <svg> tag
        if backdrop_elem:
            svg_for_preview = re.sub(
                r'(<svg\b[^>]*>)',
                r'\1' + backdrop_elem,
                svg_for_preview
            )
        
        # Apply vector positioning shift for all backdrop types
        # This ensures consistent positioning regardless of backdrop selection
        # Apply same shift to all backdrops so vectors don't move when switching
        if state.processed_width > 0 and state.processed_height > 0 and scale > 0:
            pixel_shift = 0.5 * scale  # Always apply for consistency
            total_shift = margin_adjust + pixel_shift
            if total_shift > 0:
                svg_for_preview = re.sub(
                    r'transform="translate\(([0-9.-]+),([0-9.-]+)\)"',
                    lambda m: f'transform="translate({float(m.group(1)) + total_shift:.4f},{float(m.group(2)) + total_shift:.4f})"',
                    svg_for_preview
                )
        
        # Apply custom stroke color if set
        preview_color = str(_safe_value(controls.get("preview_stroke_color", _ValueProxy(""))))
        if preview_color and preview_color in STROKE_COLORS:
            hex_color = STROKE_COLORS[preview_color]
            svg_for_preview = re.sub(r'stroke="[^"]+"', f'stroke="{hex_color}"', svg_for_preview)
        
        # Add page outline rectangle to show paper boundaries (insert at end so it's on top)
        if placement:
            paper_w = placement.get("paper_width", 0)
            paper_h = placement.get("paper_height", 0)
            if paper_w > 0 and paper_h > 0:
                # Dashed outline to show page boundaries
                page_outline = (
                    f'<rect x="0" y="0" width="{paper_w}" height="{paper_h}" '
                    f'fill="none" stroke="#908caa" stroke-width="0.5" '
                    f'stroke-dasharray="2,2"/>'
                )
                # Insert before closing </svg> tag so it renders on top
                svg_for_preview = re.sub(
                    r'(</svg\s*>)',
                    page_outline + r'\1',
                    svg_for_preview
                )
        
        svg_url = _svg_text_to_data_url(svg_for_preview)
        
        # Single consistent preview - just the SVG (with or without embedded backdrop)
        # Square container for consistent viewing, content centered
        if zoom_fit:
            # Fit mode: scale to fit container without scrolling
            img_style = f"display:block; max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; image-rendering:{img_rendering};"
        else:
            # Percentage zoom: fixed width, allow scrolling
            img_style = f"display:block; width:{zoom}%; max-width:none; height:auto; image-rendering:{img_rendering}; flex-shrink:0;"
        
        outputs["svg_preview_html"].set_content(
            "".join(
                [
                    f'<div class="st-card" style="width:{container_size}px; height:{container_size}px; overflow:auto; display:flex; align-items:center; justify-content:center;">',
                    f'<img src="{svg_url}" style="{img_style}"/>',
                    "</div>",
                ]
            )
        )

    def _slider_with_readout(
        *,
        min_v: float,
        max_v: float,
        value: float,
        step: float,
        label: str = "",
        classes: str = "w-full",
    ):
        precision = _slider_precision(step)
        with ui.column().classes(classes):
            if label:
                value_display = ui.label(f"{label}: {_format_slider_value(value, precision)}").classes(
                    "text-sm font-medium"
                )
            slider = ui.slider(min=min_v, max=max_v, value=value, step=step)
            if label:
                def update_value_display():
                    v = _safe_value(slider)
                    value_display.set_text(f"{label}: {_format_slider_value(v, precision)}")
                slider.on_value_change(update_value_display)
            with ui.row().classes("w-full justify-between items-center text-xs st-muted"):
                ui.label(_format_slider_value(min_v, precision))
                ui.label(_format_slider_value(max_v, precision))
        return slider

    def _update_hist_overlay() -> None:
        active = str(_safe_value(controls["hist_target"]))
        outputs["hist_marker_info"].set_text(f"Active marker: {active.upper()}")

    def _sync_hist_controls(min_v: float, mid_v: float, max_v: float) -> None:
        min_v, mid_v, max_v = _normalize_histogram_knots(min_v, mid_v, max_v)
        state.suppress_hist_callbacks = True
        controls["hist_min"].set_value(min_v)
        controls["hist_mid"].set_value(mid_v)
        controls["hist_max"].set_value(max_v)
        state.suppress_hist_callbacks = False
        _update_hist_overlay()

    def _set_hist_from_percentiles(p_min: float, p_max: float) -> None:
        gray = normalize_input_image(state.input_image).ravel()
        min_v = float(np.percentile(gray, p_min))
        max_v = float(np.percentile(gray, p_max))
        mid_v = 0.5 * (min_v + max_v)
        _sync_hist_controls(min_v, mid_v, max_v)
        asyncio.create_task(run_to_stage(1))

    def _tone_from_hist_event(e) -> float:
        x = float(getattr(e, "image_x", 0.0))
        hist_width = max(2, int(state.hist_source_width))
        return float(np.clip(x / float(hist_width - 1), 0.0, 1.0))

    def _target_from_hist_event(e) -> str:
        return str(_safe_value(controls["hist_target"]))

    def _set_hist_target_value(target: str, tone: float) -> None:
        min_v = float(_safe_value(controls["hist_min"]))
        mid_v = float(_safe_value(controls["hist_mid"]))
        max_v = float(_safe_value(controls["hist_max"]))

        if target == "min":
            min_v = tone
        elif target == "max":
            max_v = tone
        else:
            mid_v = tone

        _sync_hist_controls(min_v, mid_v, max_v)

    def _on_hist_mouse(e) -> None:
        event_type = str(getattr(e, "type", ""))

        if event_type in {"mousedown", "touchstart"}:
            state.hist_drag_target = _target_from_hist_event(e)
            _set_hist_target_value(state.hist_drag_target, _tone_from_hist_event(e))
            return

        if event_type in {"mousemove", "touchmove"}:
            if state.hist_drag_target is None:
                return
            _set_hist_target_value(state.hist_drag_target, _tone_from_hist_event(e))
            return

        if event_type in {"mouseup", "mouseleave", "touchend", "touchcancel"}:
            if state.hist_drag_target is not None:
                asyncio.create_task(run_to_stage(1))
            state.hist_drag_target = None
            return

        if event_type == "click":
            target = _target_from_hist_event(e)
            _set_hist_target_value(target, _tone_from_hist_event(e))
            asyncio.create_task(run_to_stage(1))

    def update_input_preview() -> None:
        # Gallery handles display - nothing else to update
        pass

    def _update_image_gallery() -> None:
        """Rebuild the image gallery with clickable thumbnails."""
        if "image_gallery" not in outputs:
            return
        gallery = outputs["image_gallery"]
        gallery.clear()
        
        def _select_image(name: str):
            """Select an image as current input."""
            if name in state.uploaded_images:
                state.input_image = state.uploaded_images[name]
                state.input_image_name = name
                state.using_default_input = (name == "default")
                outputs["input_name"].set_text(f"Input: {name}")
                state.pipeline_cache = {}
                update_input_preview()
                _update_image_gallery()  # Update selection highlight
                asyncio.create_task(run_to_stage(1))
        
        def _remove_image(name: str):
            """Remove an image from the library (cannot remove default)."""
            if name == "default":
                return  # Don't allow removing default
            if name in state.uploaded_images:
                del state.uploaded_images[name]
                # If this was the current image, switch to default
                if state.input_image_name == name:
                    _select_image("default")
                else:
                    _update_image_gallery()
        
        with gallery:
            for name, img in state.uploaded_images.items():
                is_current = name == state.input_image_name
                border_class = "border-2 border-[#9ccfd8]" if is_current else "border border-[#39414f]"
                with ui.column().classes(f"gap-1 p-2 rounded-lg bg-[#141a24] {border_class} cursor-pointer"):
                    thumb_url = _np_to_data_url(_downscale_for_ui(img, max_dim=100))
                    ui.image(thumb_url).classes("w-[80px] h-[60px] object-cover rounded").on(
                        "click", lambda _, n=name: _select_image(n)
                    )
                    with ui.row().classes("items-center gap-1"):
                        short_name = name[:12] + "..." if len(name) > 15 else name
                        ui.label(short_name).classes("text-xs st-muted").on(
                            "click", lambda _, n=name: _select_image(n)
                        )
                        # Only show delete button for non-default images
                        if name != "default":
                            ui.button(icon="close", on_click=lambda _, n=name: _remove_image(n)).props(
                                "flat dense round size=xs"
                            ).classes("text-xs")

    async def run_to_stage(stage_index: int) -> None:
        state.is_processing = True
        outputs["gen_vectors_spinner"].set_visibility(True)
        outputs["gen_vectors_status"].set_visibility(True)
        outputs["gen_vectors_btn"].enabled = False
        _update_svg_preview_html()  # Show processing message immediately
        await asyncio.sleep(0)  # Yield to allow UI to update
        (
            svg_preview,
            svg_code,
            _download,
            gradient,
            _grayscale,
            _downscaled,
            quantized,
            histogram,
            complexity_text,
            status_text,
            cache,
        ) = run_pipeline_to_stage(
            state.pipeline_cache,
            state.input_image,
            stage_index,
            str(_safe_value(controls["algorithm"])),
            str(_safe_value(controls["color_theme"])),
            float(_safe_value(controls["hist_min"])),
            float(_safe_value(controls["hist_mid"])),
            float(_safe_value(controls["hist_max"])),
            float(_safe_value(controls["output_width"])),
            int(_safe_value(controls["levels"])),
            bool(_safe_value(controls["invert"])),
            float(_safe_value(controls["gradient_sigma"])),
            float(_safe_value(controls["stroke_width"])),
            float(_safe_value(controls["randomness_vertex"])),
            float(_safe_value(controls["randomness_position"])),
            int(_safe_value(controls["theta_resolution"])),
            float(_safe_value(controls["spiral_b"])),
            bool(_safe_value(controls["connect_cells"])),
            int(_safe_value(controls["circle_points"])),
            bool(_safe_value(controls["small_first"])),
            bool(_safe_value(controls["hatch_horizontal"])),
            bool(_safe_value(controls["hatch_vertical"])),
            bool(_safe_value(controls["hatch_diag_right"])),
            bool(_safe_value(controls["hatch_diag_left"])),
            float(_safe_value(controls["min_spacing"])),
            float(_safe_value(controls["max_spacing"])),
            float(_safe_value(controls["lines_segment_length"])),
            float(_safe_value(controls["randomness_length"])),
            float(_safe_value(controls["min_gradient_scale"])),
            float(_safe_value(controls["max_gradient_scale"])),
            float(_safe_value(controls["curves_segment_length"])),
            float(_safe_value(controls["curves_randomness_length"])),
            int(_safe_value(controls["max_steps"])),
            float(_safe_value(controls["step_size"])),
            int(_safe_value(controls["bezier_samples"])),
            bool(_safe_value(controls["enable_vertex_guard"])),
            float(_safe_value(controls["max_estimated_vertices"])),
        )

        state.pipeline_cache = cache
        state.latest_svg_content = svg_code
        if isinstance(histogram, np.ndarray) and histogram.ndim >= 2:
            state.hist_source_width = int(histogram.shape[1])

        gray_input = normalize_input_image(state.input_image)
        gray_display = to_display_uint8(gray_input)
        preview_shape = gray_display.shape

        hist_transformed = apply_histogram_transform(
            gray_input,
            float(_safe_value(controls["hist_min"])),
            float(_safe_value(controls["hist_mid"])),
            float(_safe_value(controls["hist_max"])),
        )
        hist_result_display = to_display_uint8(hist_transformed)

        processed = cache.get("processed")
        invert_display = None
        width_display = None
        if processed is not None and getattr(processed, "original", None) is not None:
            base = np.asarray(processed.original)
            invert_arr = 1.0 - base if bool(_safe_value(controls["invert"])) else base
            invert_display = to_display_uint8(invert_arr)
            width_display = to_display_uint8(base)
            invert_display = resize_preview_nearest(invert_display, preview_shape)
            width_display = resize_preview_nearest(width_display, preview_shape)

        hist_result_display = resize_preview_nearest(hist_result_display, preview_shape)

        outputs["status"].set_text(status_text)
        outputs["histogram"].set_source(_np_to_data_url(_style_histogram_image(_downscale_for_ui(histogram))))
        _update_hist_overlay()
        outputs["preview_grayscale"].set_source(_np_to_data_url(_downscale_for_ui(gray_display)))
        outputs["preview_hist_downsampled"].set_source(_np_to_data_url(_downscale_for_ui(width_display)))
        outputs["preview_invert_levels"].set_source(_np_to_data_url(_downscale_for_ui(quantized)))
        outputs["preview_gradmag"].set_source(_np_to_data_url(_downscale_for_ui(gradient)))
        
        # Update source selector thumbnails in Vector tab
        # "Levels" = quantized levels (dark = more marks if invert is on)
        # "Inverted Levels" = inverted quantized (bright = more marks)
        # "Gradient Magnitude" = edge detection result
        if "source_thumb_levels" in outputs:
            outputs["source_thumb_levels"].set_source(_np_to_data_url(_downscale_for_ui(quantized)))
        if "source_thumb_inverted" in outputs and quantized is not None:
            # Invert the quantized levels for the thumbnail
            inverted_levels = 255 - np.asarray(quantized).astype(np.uint8)
            outputs["source_thumb_inverted"].set_source(_np_to_data_url(_downscale_for_ui(inverted_levels)))
        if "source_thumb_gradmag" in outputs:
            outputs["source_thumb_gradmag"].set_source(_np_to_data_url(_downscale_for_ui(gradient)))
        
        # Update backdrop selector thumbnails
        if "backdrop_thumb_grayscale" in outputs:
            outputs["backdrop_thumb_grayscale"].set_source(_np_to_data_url(_downscale_for_ui(gray_display)))
        if "backdrop_thumb_levels" in outputs:
            outputs["backdrop_thumb_levels"].set_source(_np_to_data_url(_downscale_for_ui(quantized)))
        if "backdrop_thumb_gradmag" in outputs:
            outputs["backdrop_thumb_gradmag"].set_source(_np_to_data_url(_downscale_for_ui(gradient)))
        if "backdrop_thumb_original" in outputs and state.input_image is not None:
            outputs["backdrop_thumb_original"].set_source(_np_to_data_url(_downscale_for_ui(state.input_image)))
        
        # Store backdrop images for preview overlay
        # All backdrops must be at the PROCESSED resolution (same as SVG coordinates)
        # The gradient/quantized from the pipeline are resized to preview, so we need
        # to get the actual processed-resolution data from the cache
        processed = cache.get("processed")
        if processed is not None:
            state.processed_width = processed.width
            state.processed_height = processed.height
            
            # Grayscale: resize original to processed dimensions
            gray_resized = np.array(
                Image.fromarray(gray_display).resize(
                    (processed.width, processed.height),
                    resample=getattr(getattr(Image, "Resampling", Image), "BILINEAR")
                )
            )
            state.backdrop_grayscale = gray_resized
            
            # Original RGB: keep at native resolution (not resized)
            # This allows seeing full-res original aligned with vectors
            state.backdrop_original = state.input_image.copy() if state.input_image is not None else None
            
            # Quantized (levels): use processed.data at native resolution
            quantized_native = to_display_uint8(processed.data, levels=processed.levels)
            state.backdrop_quantized = quantized_native
            
            # Gradient magnitude: get from cache at native resolution
            sigma_key = round(float(_safe_value(controls["gradient_sigma"])), 4)
            gradients_by_sigma = cache.get("gradients_by_sigma", {})
            gradients_data = gradients_by_sigma.get(sigma_key)
            if gradients_data is not None:
                gradient_native = to_display_uint8(gradients_data.magnitude)
                state.backdrop_gradient = gradient_native
            else:
                state.backdrop_gradient = None
        else:
            state.backdrop_grayscale = gray_display
            state.backdrop_gradient = gradient
            state.backdrop_quantized = quantized
            state.backdrop_original = state.input_image  # Use original input as fallback
        
        outputs["svg_source"].set_content(_format_svg_code(svg_code))
        settings_snapshot = _build_settings_snapshot(controls)
        outputs["settings_source"].set_content(json.dumps(settings_snapshot, indent=2))

        # Mark processing complete BEFORE updating preview so it shows the SVG, not "Processing..."
        state.current_stage = max(0, min(stage_index, len(PIPELINE_STEP_LABELS) - 1))
        state.is_processing = False
        outputs["gen_vectors_spinner"].set_visibility(False)
        outputs["gen_vectors_status"].set_visibility(False)
        outputs["gen_vectors_btn"].enabled = True
        _update_svg_preview_html()

    async def on_open_image(e) -> None:
        """Handle image upload - adds to library and selects it."""
        try:
            data = await e.file.read()
            img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            name = e.file.name
            # Add to images library
            state.uploaded_images[name] = img
            # Select this image as current
            state.input_image = img
            state.input_image_name = name
            state.using_default_input = False
            outputs["input_name"].set_text(f"Input: {name}")
            state.pipeline_cache = {}
            update_input_preview()
            _update_image_gallery()
            await run_to_stage(1)
        except Exception as err:  # noqa: BLE001
            ui.notify(f"Failed to load image: {err}", color="negative")

    def on_run_one() -> None:
        asyncio.create_task(run_to_stage(min(state.current_stage + 1, len(PIPELINE_STEP_LABELS) - 1)))

    def on_run_selected() -> None:
        step_label = str(_safe_value(controls.get("target_step", _ValueProxy(UI_STEP_LABELS[0]))))
        target_stage = 1 if step_label == UI_STEP_LABELS[0] else len(PIPELINE_STEP_LABELS) - 1
        asyncio.create_task(run_to_stage(target_stage))

    def on_run_all() -> None:
        asyncio.create_task(run_to_stage(len(PIPELINE_STEP_LABELS) - 1))

    def download_svg() -> None:
        if not state.latest_svg_content:
            ui.notify("No SVG generated yet. Run vector generation first.", color="warning")
            return
        try:
            paper_size = str(_safe_value(controls.get("paper_size", _ValueProxy("A4"))))
            orientation = str(_safe_value(controls.get("paper_orientation", _ValueProxy("Portrait"))))
            svg_to_save, _ = _fit_svg_dimensions(state.latest_svg_content, paper_size, orientation)
            base = _build_download_base_name(
                state.input_image_name,
                str(_safe_value(controls.get("algorithm", _ValueProxy("spirals")))),
            )
            ui.download(svg_to_save.encode("utf-8"), filename=f"{base}.svg", media_type="image/svg+xml")
        except Exception as err:  # noqa: BLE001
            ui.notify(f"Failed to download SVG: {err}", color="negative")

    def download_settings() -> None:
        try:
            payload = {"version": 1, "settings": _build_settings_snapshot(controls)}
            base = _build_download_base_name(
                state.input_image_name,
                str(_safe_value(controls.get("algorithm", _ValueProxy("spirals")))),
            )
            ui.download(
                json.dumps(payload, indent=2).encode("utf-8"),
                filename=f"{base}_settings.json",
                media_type="application/json",
            )
        except Exception as err:  # noqa: BLE001
            ui.notify(f"Failed to download settings: {err}", color="negative")

    async def on_load_preset(e) -> None:
        try:
            content = await e.file.read()
            payload = json.loads(content.decode("utf-8"))
            settings = payload.get("settings", payload)
            if not isinstance(settings, dict):
                raise ValueError("Preset must contain a JSON object")

            for key, value in settings.items():
                if key in controls:
                    controls[key].set_value(value)

            state.pipeline_cache = {}
            await run_to_stage(state.current_stage)
            ui.notify(f"Loaded preset: {e.file.name}", color="positive")
        except Exception as err:  # noqa: BLE001
            ui.notify(f"Failed to load preset: {err}", color="negative")

    ui.add_head_html(
        """
    <style>
            body { background: radial-gradient(circle at 20% 0%, #1f2430 0%, #11151c 60%, #0b0e13 100%); }
            .st-card { border: 1px solid #39414f; border-radius: 10px; background: #141a24; }
      .st-title { font-weight: 700; font-size: 1.2rem; }
            .st-muted { color: #908caa; }
                        .st-code textarea {
                            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace !important;
                            font-size: 12px;
                            line-height: 1.4;
                            white-space: pre;
                        }
                        /* Tab styling - better active tab highlighting */
                        .q-tab--active {
                            background: linear-gradient(135deg, #1c2a34 0%, #192530 100%) !important;
                            border-left: 3px solid #9ccfd8 !important;
                            color: #e0def4 !important;
                            font-weight: 600 !important;
                        }
                        .q-tab {
                            border-left: 3px solid transparent;
                            transition: all 0.2s ease;
                        }
                        .q-tab:hover:not(.q-tab--active) {
                            background: #1a2028 !important;
                            border-left: 3px solid #39414f;
                        }
                        .st-alg-tile {
                            border: 1px solid #39414f;
                            border-radius: 10px;
                            background: #141a24;
                            padding: 8px;
                            cursor: pointer;
                        }
                        .st-alg-tile-active {
                            border: 1px solid #9ccfd8;
                            border-radius: 10px;
                            background: #1c2a34;
                            padding: 8px;
                            cursor: pointer;
                            box-shadow: 0 0 0 1px rgba(156, 207, 216, 0.2) inset;
                        }
                        .st-theme-tile {
                            border: 1px solid #39414f;
                            border-radius: 10px;
                            background: #141a24;
                            padding: 10px;
                            cursor: pointer;
                        }
                        .st-theme-tile-active {
                            border: 1px solid #c4a7e7;
                            border-radius: 10px;
                            background: #2a2540;
                            padding: 10px;
                            cursor: pointer;
                            box-shadow: 0 0 0 1px rgba(196, 167, 231, 0.25) inset;
                        }
                        .st-source-tile {
                            border: 2px solid #39414f;
                            border-radius: 8px;
                            background: #141a24;
                            padding: 4px;
                            cursor: pointer;
                            transition: all 0.15s ease;
                        }
                        .st-source-tile:hover {
                            border-color: #56526e;
                        }
                        .st-source-tile-active {
                            border: 2px solid #f6c177;
                            border-radius: 8px;
                            background: #2a2418;
                            padding: 4px;
                            cursor: pointer;
                            box-shadow: 0 0 8px rgba(246, 193, 119, 0.3);
                        }
                        .st-tabs-sticky {
                            position: sticky;
                            top: 16px;
                            align-self: flex-start;
                            z-index: 100;
                        }
    </style>
    """
    )

    # Set Rosé Pine colors for UI elements
    ui.colors(
        primary="#31748f",    # Pine - primary actions (buttons, sliders)
        secondary="#c4a7e7",  # Subtle - secondary elements
        positive="#9ccfd8",   # Foam - success/download actions
    )

    dark_mode = ui.dark_mode()
    dark_mode.enable()

    with ui.row().classes("w-full justify-between items-center mb-4"):
        ui.label("ScribbleTrace NiceGUI").classes("st-title")
        with ui.row().classes("items-center gap-2"):
            # GitHub repo link
            with ui.link(target="https://github.com/kylberg/ScribbleTrace", new_tab=True):
                ui.html(
                    '<svg viewBox="0 0 24 24" width="24" height="24" fill="#908caa" style="cursor:pointer;">'
                    '<path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>'
                    '</svg>'
                ).tooltip("View on GitHub")
            outputs["dark_mode_toggle"] = ui.button(icon="dark_mode").props("flat round")
            def toggle_dark_mode():
                dark_mode.toggle()
                if dark_mode.value:
                    outputs["dark_mode_toggle"].props(remove="icon=light_mode", add="icon=dark_mode")
                else:
                    outputs["dark_mode_toggle"].props(remove="icon=dark_mode", add="icon=light_mode")
            outputs["dark_mode_toggle"].on_click(toggle_dark_mode)

    with ui.row().classes("w-full items-start no-wrap gap-4"):
        with ui.tabs().props("vertical").classes("w-[180px] st-card p-2 st-tabs-sticky") as tabs:
            tab_input = ui.tab("Input").props("icon=upload")
            tab_processing = ui.tab("Proc").props("icon=tune")
            tab_vector = ui.tab("Vector").props("icon=gesture")

        with ui.tab_panels(tabs, value=tab_processing).props("vertical animated transition-prev=fade transition-next=fade").classes("flex-1 min-w-0"):
            with ui.tab_panel(tab_input):
                with ui.column().classes("w-full max-w-[720px] gap-4"):
                    outputs["input_name"] = ui.label("Input: default").classes("st-title")
                    outputs["status"] = ui.label("").classes("st-muted")
                    
                    # Image gallery (default + uploaded)
                    outputs["image_gallery"] = ui.row().classes("w-full gap-2 flex-wrap")
                    
                    # Upload button
                    ui.upload(
                        on_upload=on_open_image, 
                        auto_upload=True, 
                        multiple=True,
                        label="Add images"
                    ).classes("max-w-[200px]").props('accept="image/*" flat')

                    # Presets
                    with ui.row().classes("w-full gap-2 items-center"):
                        ui.button("Save Preset", icon="save", on_click=download_settings).props("flat")
                        ui.upload(on_upload=on_load_preset, auto_upload=True, label="Load Preset").props('accept=".json" flat').classes("max-w-[200px]")

            with ui.tab_panel(tab_processing):
                with ui.column().classes("w-full gap-3"):
                    ui.label("Step 1 - Image Processing").classes("st-title")
                    controls["hist_min"] = _ValueProxy(0.0)
                    controls["hist_mid"] = _ValueProxy(0.5)
                    controls["hist_max"] = _ValueProxy(1.0)

                    with ui.row().classes("w-full items-center gap-4"):
                        ui.label("Preview Size").classes("st-muted")
                        controls["preview_size"] = ui.toggle(["Small", "Medium", "Large"], value="Medium").props(
                            "dense"
                        )
                        controls["preview_size"].on_value_change(lambda _: _apply_preview_size())

                    with ui.row().classes("w-full gap-6 items-start flex-wrap"):
                        with ui.column().classes("min-w-[320px] max-w-[640px] gap-2"):
                            ui.label("Histogram Transform").classes("st-title")
                            controls["hist_target"] = ui.toggle(
                                ["min", "mid", "max"],
                                value="mid",
                            ).props("dense")
                            controls["hist_target"].on_value_change(lambda _: asyncio.create_task(run_to_stage(1)))
                            ui.label("Click histogram to place selected marker").classes("st-muted")
                            outputs["hist_marker_info"] = ui.label("Active marker: MID").classes("st-muted")
                            outputs["histogram"] = ui.interactive_image(
                                source="",
                                on_mouse=_on_hist_mouse,
                                events=["click", "mousedown", "mousemove", "mouseup", "mouseleave"],
                                size=(HIST_WIDTH, HIST_HEIGHT),
                                cross=False,
                            ).classes("w-[520px] st-card")
                            with ui.row().classes("w-full gap-2"):
                                ui.button(
                                    "Auto 1/99 (center mid)", on_click=lambda: _set_hist_from_percentiles(1, 99)
                                )
                                ui.button(
                                    "Auto 2/98 (center mid)", on_click=lambda: _set_hist_from_percentiles(2, 98)
                                )
                                ui.button(
                                    "Auto 10/90 (center mid)", on_click=lambda: _set_hist_from_percentiles(10, 90)
                                )
                                ui.button(
                                    "Full 0/100 (center mid)", on_click=lambda: _set_hist_from_percentiles(0, 100)
                                )

                        with ui.column().classes("min-w-[280px] max-w-[420px] gap-2"):
                            ui.label("Image Processing Settings").classes("st-title")
                            controls["output_width"] = _slider_with_readout(
                                label="Output Width", min_v=10, max_v=200, value=40, step=1
                            )
                            controls["output_width"].on_value_change(lambda _: asyncio.create_task(run_to_stage(1)))
                            controls["levels"] = _slider_with_readout(
                                label="Levels", min_v=2, max_v=16, value=7, step=1
                            )
                            controls["levels"].on_value_change(lambda _: asyncio.create_task(run_to_stage(1)))
                            controls["invert"] = ui.switch("Invert", value=True)
                            controls["invert"].on_value_change(lambda _: asyncio.create_task(run_to_stage(1)))
                            controls["gradient_sigma"] = _slider_with_readout(
                                label="Gaussian Gradient Magnitude Sigma", min_v=0.0, max_v=6.0, value=1.0, step=0.1
                            )
                            controls["gradient_sigma"].on_value_change(lambda _: asyncio.create_task(run_to_stage(1)))

                    ui.separator()
                    ui.label("Image Processing Previews").classes("st-title")
                    with ui.row().classes("w-full gap-4 flex-wrap"):
                        with ui.column().classes("gap-1"):
                            ui.label("1) Grayscale").classes("st-muted")
                            outputs["preview_grayscale"] = ui.image().classes(f"{DEFAULT_PREVIEW_WIDTH} max-w-full st-card")
                        with ui.column().classes("gap-1"):
                            ui.label("2) Histogram Transformed + Downsampled").classes("st-muted")
                            outputs["preview_hist_downsampled"] = ui.image().classes(
                                f"{DEFAULT_PREVIEW_WIDTH} max-w-full st-card"
                            )
                        with ui.column().classes("gap-1"):
                            ui.label("3) Inverted + Levels").classes("st-muted")
                            outputs["preview_invert_levels"] = ui.image().classes(
                                f"{DEFAULT_PREVIEW_WIDTH} max-w-full st-card"
                            )
                        with ui.column().classes("gap-1"):
                            ui.label("4) Gradient Magnitude").classes("st-muted")
                            outputs["preview_gradmag"] = ui.image().classes(f"{DEFAULT_PREVIEW_WIDTH} max-w-full st-card")

                    _apply_preview_size()

            with ui.tab_panel(tab_vector):
                with ui.column().classes("w-full gap-3"):
                    ui.label("Step 2 - Vector Generation").classes("st-title")

                    controls["algorithm"] = _ValueProxy(
                        "spirals", on_change=lambda _: (_update_algorithm_tile_styles(), _update_vector_setting_visibility())
                    )

                    ui.label("Algorithm").classes("st-muted")
                    with ui.row().classes("w-full gap-3 flex-wrap"):
                        for name in ALGORITHM_ORDER:
                            thumb = ALGORITHM_THUMB_DIR / f"{name}.svg"
                            with ui.column().classes("items-center gap-1 w-[120px] st-alg-tile") as tile:
                                if thumb.exists():
                                    ui.image(_svg_file_to_data_url(thumb)).classes("w-[92px] h-[92px] st-card")
                                else:
                                    ui.label("No thumb").classes("st-muted")
                                ui.label(name.capitalize()).classes("st-muted")
                            tile.on("click", lambda _e, n=name: _set_algorithm(n))
                            algorithm_tiles[name] = tile
                    _update_algorithm_tile_styles()

                    # Source data selector - clickable thumbnails from processing step
                    controls["vector_source"] = _ValueProxy(
                        "levels", on_change=lambda _: _update_source_tile_styles()
                    )
                    ui.label("Source Data (for vector generation)").classes("st-muted")
                    with ui.row().classes("w-full gap-4 items-end"):
                        with ui.column().classes("items-center gap-1 st-source-tile") as levels_tile:
                            outputs["source_thumb_levels"] = ui.image().classes("w-[120px] h-[90px] object-cover")
                            ui.label("Levels").classes("st-muted text-sm")
                        levels_tile.on("click", lambda _e: _set_vector_source("levels"))
                        source_tiles["levels"] = levels_tile
                        
                        with ui.column().classes("items-center gap-1 st-source-tile") as inverted_tile:
                            outputs["source_thumb_inverted"] = ui.image().classes("w-[120px] h-[90px] object-cover")
                            ui.label("Inverted Levels").classes("st-muted text-sm")
                        inverted_tile.on("click", lambda _e: _set_vector_source("inverted"))
                        source_tiles["inverted"] = inverted_tile
                        
                        with ui.column().classes("items-center gap-1 st-source-tile") as gradmag_tile:
                            outputs["source_thumb_gradmag"] = ui.image().classes("w-[120px] h-[90px] object-cover")
                            ui.label("Gradient Magnitude").classes("st-muted text-sm")
                        gradmag_tile.on("click", lambda _e: _set_vector_source("gradmag"))
                        source_tiles["gradmag"] = gradmag_tile
                    _update_source_tile_styles()

                    with ui.row().classes("w-full gap-4"):
                        controls["color_theme"] = _ValueProxy(
                            "White Lines on Black", on_change=lambda _: _update_theme_tile_styles()
                        )

                    ui.label("SVG Theme").classes("st-muted")
                    with ui.row().classes("w-full gap-3 flex-wrap"):
                        with ui.column().classes("items-start gap-2 w-[200px] st-theme-tile") as light_tile:
                            ui.label("Bright").classes("st-title")
                            ui.label("Black lines on white").classes("st-muted")
                            ui.html(
                                '<svg viewBox="0 0 120 64" width="120" height="64">'
                                '<rect width="120" height="64" fill="#f6f2ff" />'
                                '<path d="M10 54 C30 12, 52 14, 82 48" fill="none" stroke="#222" stroke-width="2" />'
                                '<path d="M26 56 C48 20, 78 18, 108 54" fill="none" stroke="#222" stroke-width="2" />'
                                '</svg>'
                            )
                        light_tile.on("click", lambda _e: _set_color_theme("Black Lines on White"))
                        theme_tiles["Black Lines on White"] = light_tile

                        with ui.column().classes("items-start gap-2 w-[200px] st-theme-tile") as dark_tile:
                            ui.label("Dark").classes("st-title")
                            ui.label("White lines on black").classes("st-muted")
                            ui.html(
                                '<svg viewBox="0 0 120 64" width="120" height="64">'
                                '<rect width="120" height="64" fill="#121018" />'
                                '<path d="M10 54 C30 12, 52 14, 82 48" fill="none" stroke="#ece8ff" stroke-width="2" />'
                                '<path d="M26 56 C48 20, 78 18, 108 54" fill="none" stroke="#ece8ff" stroke-width="2" />'
                                '</svg>'
                            )
                        dark_tile.on("click", lambda _e: _set_color_theme("White Lines on Black"))
                        theme_tiles["White Lines on Black"] = dark_tile
                    _update_theme_tile_styles()

                    with ui.column().classes("st-card p-3 gap-2"):
                        ui.label("General SVG Settings").classes("st-title")
                        controls["stroke_width"] = _slider_with_readout(
                            label="Line Width (mm)", min_v=0.01, max_v=2.0, value=0.2, step=0.001
                        )
                        ui.label("Fit SVG drawing in").classes("st-muted text-sm")
                        with ui.row().classes("gap-2 items-center justify-start"):
                            controls["paper_size"] = _ValueProxy("A4")
                            for size in ["A3", "A4", "A5", "A6"]:
                                tile = ui.html(_paper_size_icon_svg(size)).classes(
                                    "cursor-pointer"
                                )
                                tile.on("click", lambda s=size: _set_paper_size(s))
                                paper_size_tiles[size] = tile
                            _update_paper_size_tile_styles()
                        
                        ui.label("Orientation").classes("st-muted text-sm")
                        with ui.row().classes("gap-3 items-center justify-start"):
                            controls["paper_orientation"] = _ValueProxy("Portrait")
                            for orient in ["Portrait", "Landscape"]:
                                tile = ui.html(_orientation_icon_svg(orient)).classes(
                                    "cursor-pointer"
                                )
                                tile.on("click", lambda o=orient: _set_orientation(o))
                                orientation_tiles[orient] = tile
                            _update_orientation_tile_styles()

                    ui.label("Vector Settings Matrix").classes("st-muted")
                    with ui.row().classes("w-full gap-3 items-start flex-wrap"):
                        with ui.column().classes("st-card p-3 gap-2 min-w-[320px] max-w-[420px]"):
                            ui.label("Safety Limits").classes("st-title")
                            controls["enable_vertex_guard"] = ui.switch("Enable Vertex Guard", value=True)
                            controls["max_estimated_vertices"] = ui.number(
                                label="Max Estimated Vertices",
                                value=300000,
                                min=1000,
                                step=1000,
                                format="%.0f",
                            )

                        with ui.column().classes("st-card p-3 gap-2 min-w-[320px] max-w-[420px]"):
                            ui.label("Shared Randomness").classes("st-title")
                            controls["randomness_vertex"] = _slider_with_readout(
                                label="Vertex Randomness", min_v=0.0, max_v=1.0, value=0.0, step=0.01
                            )
                            controls["randomness_position"] = _slider_with_readout(
                                label="Position Randomness", min_v=0.0, max_v=1.0, value=0.0, step=0.01
                            )

                        vector_groups["spirals"] = ui.column().classes("st-card p-3 gap-2 min-w-[320px] max-w-[420px]")
                        with vector_groups["spirals"]:
                            ui.label("Spirals Settings").classes("st-title")
                            controls["theta_resolution"] = _slider_with_readout(
                                label="Theta Resolution", min_v=16, max_v=360, value=60, step=1
                            )
                            controls["spiral_b"] = _slider_with_readout(
                                label="Spiral B", min_v=0.1, max_v=3.0, value=1.0, step=0.1
                            )
                            controls["connect_cells"] = ui.switch("Connect Cells", value=True)

                        vector_groups["circles_squares"] = ui.column().classes(
                            "st-card p-3 gap-2 min-w-[320px] max-w-[420px]"
                        )
                        with vector_groups["circles_squares"]:
                            ui.label("Circles/Squares Settings").classes("st-title")
                            controls["circle_points"] = _slider_with_readout(
                                label="Circle Points", min_v=3, max_v=72, value=36, step=1
                            )
                            controls["small_first"] = ui.switch("Small First", value=True)

                        vector_groups["lines"] = ui.column().classes("st-card p-3 gap-2 min-w-[320px] max-w-[420px]")
                        with vector_groups["lines"]:
                            ui.label("Lines Settings").classes("st-title")
                            controls["lines_segment_length"] = _slider_with_readout(
                                label="Segment Length", min_v=0.1, max_v=5.0, value=1.0, step=0.1
                            )
                            controls["randomness_length"] = _slider_with_readout(
                                label="Randomness", min_v=0.0, max_v=1.0, value=0.0, step=0.01
                            )
                            controls["min_gradient_scale"] = _slider_with_readout(
                                label="Min Gradient Scale", min_v=0.01, max_v=1.0, value=0.1, step=0.01
                            )
                            controls["max_gradient_scale"] = _slider_with_readout(
                                label="Max Gradient Scale", min_v=1.0, max_v=20.0, value=10.0, step=0.5
                            )

                        vector_groups["curves"] = ui.column().classes("st-card p-3 gap-2 min-w-[320px] max-w-[420px]")
                        with vector_groups["curves"]:
                            ui.label("Curves Settings").classes("st-title")
                            controls["curves_segment_length"] = _slider_with_readout(
                                label="Segment Length", min_v=0.1, max_v=5.0, value=1.0, step=0.1
                            )
                            controls["curves_randomness_length"] = _slider_with_readout(
                                label="Randomness", min_v=0.0, max_v=1.0, value=0.0, step=0.01
                            )
                            controls["max_steps"] = _slider_with_readout(
                                label="Max Steps", min_v=1, max_v=20, value=4, step=1
                            )
                            controls["step_size"] = _slider_with_readout(
                                label="Step Size", min_v=0.5, max_v=5.0, value=2.0, step=0.1
                            )
                            controls["bezier_samples"] = _slider_with_readout(
                                label="Bezier Samples", min_v=5, max_v=50, value=15, step=1
                            )

                        vector_groups["hatching"] = ui.column().classes("st-card p-3 gap-2 min-w-[320px] max-w-[420px]")
                        with vector_groups["hatching"]:
                            ui.label("Hatching Settings").classes("st-title")
                            controls["hatch_horizontal"] = ui.switch("Horizontal", value=False)
                            controls["hatch_vertical"] = ui.switch("Vertical", value=False)
                            controls["hatch_diag_right"] = ui.switch("Diagonal Right", value=True)
                            controls["hatch_diag_left"] = ui.switch("Diagonal Left", value=False)
                            controls["min_spacing"] = _slider_with_readout(
                                label="Min Spacing", min_v=0.1, max_v=2.0, value=0.3, step=0.1
                            )
                            controls["max_spacing"] = _slider_with_readout(
                                label="Max Spacing", min_v=0.5, max_v=5.0, value=2.0, step=0.1
                            )

                    _update_vector_setting_visibility()

                    with ui.row().classes("w-full items-center gap-4"):
                        outputs["gen_vectors_btn"] = ui.button(
                            "Generate Vectors", on_click=lambda: asyncio.create_task(run_to_stage(2))
                        ).classes("grow")
                        with ui.row().classes("items-center gap-2"):
                            outputs["gen_vectors_spinner"] = ui.spinner(size="xl").set_visibility(False)
                            outputs["gen_vectors_status"] = ui.label("Processing...").classes("st-muted").set_visibility(False)
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label("Zoom").classes("st-muted")
                        controls["svg_zoom"] = ui.toggle(
                            ["Fit", "25%", "50%", "75%", "100%", "200%", "400%"],
                            value="Fit"
                        ).props("dense")
                        controls["svg_zoom"].on_value_change(lambda _: _update_svg_preview_html())
                    
                    # Preview Background selector with thumbnails
                    controls["preview_backdrop"] = _ValueProxy(
                        "Transparent", on_change=lambda _: _update_backdrop_tile_styles()
                    )
                    ui.label("Preview Background").classes("st-muted")
                    with ui.row().classes("w-full gap-3 items-end"):
                        # Transparent checkerboard
                        with ui.column().classes("items-center gap-1 st-source-tile") as trans_tile:
                            ui.html(
                                '<svg viewBox="0 0 80 60" width="80" height="60">'
                                '<defs><pattern id="chk" width="10" height="10" patternUnits="userSpaceOnUse">'
                                '<rect width="10" height="10" fill="#2a2e38"/>'
                                '<rect width="5" height="5" fill="#1e2228"/>'
                                '<rect x="5" y="5" width="5" height="5" fill="#1e2228"/>'
                                '</pattern></defs>'
                                '<rect width="80" height="60" fill="url(#chk)"/>'
                                '</svg>'
                            )
                            ui.label("Transparent").classes("st-muted text-xs")
                        trans_tile.on("click", lambda _e: _set_backdrop("Transparent"))
                        backdrop_tiles["Transparent"] = trans_tile
                        
                        # White
                        with ui.column().classes("items-center gap-1 st-source-tile") as white_tile:
                            ui.html(
                                '<svg viewBox="0 0 80 60" width="80" height="60">'
                                '<rect width="80" height="60" fill="#ffffff"/>'
                                '</svg>'
                            )
                            ui.label("White").classes("st-muted text-xs")
                        white_tile.on("click", lambda _e: _set_backdrop("White"))
                        backdrop_tiles["White"] = white_tile
                        
                        # Black
                        with ui.column().classes("items-center gap-1 st-source-tile") as black_tile:
                            ui.html(
                                '<svg viewBox="0 0 80 60" width="80" height="60">'
                                '<rect width="80" height="60" fill="#000000" stroke="#39414f" stroke-width="1"/>'
                                '</svg>'
                            )
                            ui.label("Black").classes("st-muted text-xs")
                        black_tile.on("click", lambda _e: _set_backdrop("Black"))
                        backdrop_tiles["Black"] = black_tile
                        
                        # Grayscale
                        with ui.column().classes("items-center gap-1 st-source-tile") as gray_tile:
                            outputs["backdrop_thumb_grayscale"] = ui.image().classes("w-[80px] h-[60px] object-cover")
                            ui.label("Grayscale").classes("st-muted text-xs")
                        gray_tile.on("click", lambda _e: _set_backdrop("Grayscale"))
                        backdrop_tiles["Grayscale"] = gray_tile
                        
                        # Levels
                        with ui.column().classes("items-center gap-1 st-source-tile") as levels_tile:
                            outputs["backdrop_thumb_levels"] = ui.image().classes("w-[80px] h-[60px] object-cover")
                            ui.label("Levels").classes("st-muted text-xs")
                        levels_tile.on("click", lambda _e: _set_backdrop("Levels"))
                        backdrop_tiles["Levels"] = levels_tile
                        
                        # Gradient Magnitude
                        with ui.column().classes("items-center gap-1 st-source-tile") as gradmag_tile:
                            outputs["backdrop_thumb_gradmag"] = ui.image().classes("w-[80px] h-[60px] object-cover")
                            ui.label("Grad Mag").classes("st-muted text-xs")
                        gradmag_tile.on("click", lambda _e: _set_backdrop("Gradient Magnitude"))
                        backdrop_tiles["Gradient Magnitude"] = gradmag_tile
                        
                        # Original (RGB)
                        with ui.column().classes("items-center gap-1 st-source-tile") as orig_tile:
                            outputs["backdrop_thumb_original"] = ui.image().classes("w-[80px] h-[60px] object-cover")
                            ui.label("Original").classes("st-muted text-xs")
                        orig_tile.on("click", lambda _e: _set_backdrop("Original"))
                        backdrop_tiles["Original"] = orig_tile
                    _update_backdrop_tile_styles()
                    
                    with ui.row().classes("w-full items-center gap-4"):
                        controls["pixelated"] = ui.switch("Pixelated", value=True)
                        controls["pixelated"].on_value_change(lambda _: _update_svg_preview_html())
                        
                        ui.label("Preview Size").classes("st-muted")
                        controls["preview_size"] = ui.toggle(
                            list(PREVIEW_CONTAINER_SIZES.keys()), value="Medium"
                        ).props("dense")
                        controls["preview_size"].on_value_change(lambda _: _update_svg_preview_html())
                    
                    # Preview Stroke Color - clickable swatches
                    controls["preview_stroke_color"] = _ValueProxy("")
                    ui.label("Preview Stroke Color").classes("st-muted")
                    with ui.row().classes("gap-1 flex-wrap"):
                        for name, hex_color in STROKE_COLORS.items():
                            # Add border for visibility on dark colors
                            border = "border:1px solid #39414f;" if hex_color in ("#000000", "#191724", "#1f1d2e", "#26233a") else ""
                            ui.html(
                                f'<div style="width:24px; height:24px; background:{hex_color}; '
                                f'border-radius:4px; cursor:pointer; {border}" title="{name}"></div>'
                            ).on("click", lambda _, n=name: _set_stroke_color(n))
                    
                    outputs["svg_preview_html"] = ui.html().classes("w-full")

                    with ui.row().classes("w-full gap-2"):
                        ui.button("Download SVG", icon="download", on_click=download_svg).props(
                            "color=positive unelevated"
                        ).classes("grow")
                        ui.button("Download Settings", icon="tune", on_click=download_settings).props(
                            "color=secondary unelevated"
                        ).classes("grow")
                    with ui.expansion("SVG Source", icon="code").classes("w-full"):
                        outputs["svg_source"] = ui.code("", language="xml").classes("w-full")
                    with ui.expansion("Settings Source", icon="tune").classes("w-full"):
                        outputs["settings_source"] = ui.code("", language="json").classes("w-full")

    # Initial defaults and first run.
    _update_image_gallery()
    ui.timer(0.1, lambda: asyncio.create_task(run_to_stage(1)), once=True)


def main() -> None:
    """Launch the NiceGUI app."""
    _require_nicegui()
    create_gui()
    # Support PORT env var for cloud hosting (render.com, etc.)
    import os
    port = int(os.environ.get("PORT", 8080))
    ui.run(title="ScribbleTrace NiceGUI", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
