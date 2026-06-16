"""Interactive GUI for ScribbleTrace parameter exploration.

This module provides a Gradio-based web interface for exploring
vectorization parameters visually and interactively.

Usage:
    scribbletrace-gui
    # or
    python -m scribbletrace.gui
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import xml.dom.minidom
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from scribbletrace import SVGWriter, compute_gradients, load_image, preprocess
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

# Default sample image (a simple gradient for testing)
DEFAULT_IMAGE = None
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent.parent / "examples" / "MagrittePipe.jpg"

PIPELINE_STEP_LABELS = [
    "1. Preprocess",
    "2. Gradient Magnitude",
    "3. Final SVG",
]

PIPELINE_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(PIPELINE_STEP_LABELS)}

HUB_THEME_ID = "Nymbo/Nymbo_Theme"

GUI_EXTRA_CSS = """
#svg-source-code .cm-scroller {
    max-height: 360px !important;
    min-height: 360px !important;
    overflow-y: auto !important;
}
"""

# Legacy custom CSS kept for reference, but not applied when using hub themes.
VSCODE_DARK_CSS = """
:root {
    --vsc-bg: #1e1e1e;
    --vsc-panel: #252526;
    --vsc-panel-elev: #2d2d30;
    --vsc-border: #3c3c3c;
    --vsc-text: #d4d4d4;
    --vsc-muted: #9da0a6;
    --vsc-accent: #007acc;
    --vsc-accent-strong: #1592ff;
    --vsc-success: #4ec9b0;

    /* Override Gradio theme tokens that default to indigo/purple. */
    --color-accent: #007acc;
    --color-accent-soft: #17354d;
    --background-fill-secondary: #252526;
    --block-background-fill: #252526;
    --block-border-color: #3c3c3c;
    --block-label-background-fill: #252526;
    --block-label-border-color: #3c3c3c;
    --block-title-text-color: #d4d4d4;
}

body,
.gradio-container {
    background:
        radial-gradient(1100px 520px at 8% -12%, rgba(0, 122, 204, 0.16) 0%, transparent 58%),
        radial-gradient(900px 420px at 92% -15%, rgba(21, 146, 255, 0.1) 0%, transparent 52%),
        var(--vsc-bg);
    color: var(--vsc-text);
}

.gradio-container {
    max-width: 1400px !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container label,
.gradio-container p,
.gradio-container span,
.gradio-container .prose {
    color: var(--vsc-text) !important;
}

.gradio-container .gr-block,
.gradio-container .block,
.gradio-container .panel,
.gradio-container .tabs,
.gradio-container .tabitem {
    background: color-mix(in srgb, var(--vsc-panel) 92%, black 8%) !important;
    border: 1px solid color-mix(in srgb, var(--vsc-border) 88%, var(--vsc-accent) 12%) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28);
}

/* Subsection containers (accordions) */
.gradio-container .gr-accordion,
.gradio-container .accordion,
.gradio-container .accordion-header,
.gradio-container .accordion-content {
    background: color-mix(in srgb, var(--vsc-panel) 90%, black 10%) !important;
    border-color: color-mix(in srgb, var(--vsc-border) 82%, var(--vsc-accent) 18%) !important;
    color: var(--vsc-text) !important;
}

.gradio-container .accordion-header:hover,
.gradio-container .accordion-header:focus {
    background: color-mix(in srgb, var(--vsc-panel-elev) 88%, var(--vsc-accent) 12%) !important;
}

.gradio-container .gr-button-primary,
.gradio-container button.primary {
    background: linear-gradient(135deg, var(--vsc-accent), var(--vsc-accent-strong)) !important;
    color: #ffffff !important;
    border: 0 !important;
    font-weight: 700;
}

.gradio-container .gr-button-primary:hover,
.gradio-container button.primary:hover {
    filter: brightness(1.08);
}

.gradio-container .gr-button-secondary,
.gradio-container button.secondary {
    background: color-mix(in srgb, var(--vsc-panel-elev) 85%, black 15%) !important;
    color: var(--vsc-text) !important;
    border: 1px solid color-mix(in srgb, var(--vsc-border) 70%, var(--vsc-accent) 30%) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap {
    background: color-mix(in srgb, var(--vsc-panel-elev) 92%, black 8%) !important;
    color: var(--vsc-text) !important;
    border-color: color-mix(in srgb, var(--vsc-border) 85%, var(--vsc-accent) 15%) !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
    outline: 1px solid var(--vsc-accent) !important;
    box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.25) !important;
}

.gradio-container .tab-nav button.selected,
.gradio-container [role="tab"][aria-selected="true"] {
    color: var(--vsc-accent-strong) !important;
    border-color: var(--vsc-accent) !important;
}

.gradio-container .gr-markdown a {
    color: var(--vsc-accent-strong) !important;
}

.gradio-container .gr-markdown strong {
    color: var(--vsc-success) !important;
}

/* Slider styling: remove default purple and align with blue accents */
.gradio-container input[type="range"] {
    accent-color: var(--vsc-accent) !important;
}

.gradio-container input[type="range"]::-webkit-slider-runnable-track {
    background: linear-gradient(90deg, color-mix(in srgb, var(--vsc-accent) 55%, #0b2538 45%), #1f2a36) !important;
    height: 6px;
    border-radius: 999px;
}

.gradio-container input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    margin-top: -4px;
    border-radius: 50%;
    background: var(--vsc-accent-strong) !important;
    border: 2px solid #0f111a;
    box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.25);
}

.gradio-container input[type="range"]::-moz-range-track {
    background: linear-gradient(90deg, color-mix(in srgb, var(--vsc-accent) 55%, #0b2538 45%), #1f2a36) !important;
    height: 6px;
    border-radius: 999px;
}

.gradio-container input[type="range"]::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--vsc-accent-strong) !important;
    border: 2px solid #0f111a;
    box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.25);
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 8px !important;
    }
}
"""


def create_sample_image() -> np.ndarray:
    """Create a simple gradient sample image for testing."""
    size = 200
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    # Create a radial gradient
    img = 1 - np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 1.5
    img = np.clip(img, 0, 1)
    return img


def get_default_gui_image() -> np.ndarray | None:
    """Load the default GUI image if available, otherwise return None."""
    if DEFAULT_IMAGE_PATH.exists():
        try:
            # Keep the input preview as the original image (including color).
            return np.array(Image.open(DEFAULT_IMAGE_PATH).convert("RGB"))
        except Exception:  # noqa: BLE001
            return None
    return None


def svg_to_png(svg_content: str, width: int = 800, height: int = 600) -> Image.Image:
    """Convert SVG content to PNG image for display.

    Uses cairosvg if available, otherwise returns a placeholder.
    """
    try:
        import cairosvg

        png_data = cairosvg.svg2png(
            bytestring=svg_content.encode("utf-8"),
            output_width=width,
            output_height=height,
        )
        return Image.open(io.BytesIO(png_data))
    except ImportError:
        # cairosvg not available, try svglib + reportlab
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM

            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
                f.write(svg_content.encode("utf-8"))
                f.flush()
                drawing = svg2rlg(f.name)
                if drawing:
                    png_data = renderPM.drawToString(drawing, fmt="PNG")
                    return Image.open(io.BytesIO(png_data))
        except ImportError:
            pass

        # Fallback: return a simple placeholder
        img = Image.new("RGB", (width, height), color="white")
        return img


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
    """Convert an image array to uint8 for Gradio display."""
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


def pretty_format_svg(svg_content: str) -> str:
    """Return an indented SVG string suitable for source preview."""
    try:
        dom = xml.dom.minidom.parseString(svg_content.encode("utf-8"))
        pretty = dom.toprettyxml(indent="  ")
        # Remove empty lines introduced by minidom formatting.
        return "\n".join(line for line in pretty.splitlines() if line.strip())
    except Exception:
        return svg_content


def resolve_svg_theme(color_theme: str) -> tuple[str, str]:
    """Return stroke/background colors for a named SVG theme."""
    if color_theme == "White Lines on Black":
        return ("white", "black")
    return ("black", "white")


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
        canvas[:, max(0, x - 1):min(width, x + 2), :] = color

    return canvas


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

        # n ~= theta_resolution * value / 2 (odd-forced sample count).
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


def process_image(
    input_image: np.ndarray | None,
    algorithm: str,
    color_theme: str,
    output_width: float,
    levels: int,
    invert: bool,
    gradient_sigma: float,
    stroke_width: float,
    randomness_vertex: float,
    randomness_position: float,
    # Spirals parameters
    theta_resolution: int,
    spiral_b: float,
    connect_cells: bool,
    # Circles parameters
    circle_points: int,
    small_first: bool,
    # Hatching parameters
    hatch_horizontal: bool,
    hatch_vertical: bool,
    hatch_diag_right: bool,
    hatch_diag_left: bool,
    min_spacing: float,
    max_spacing: float,
    # Lines parameters
    lines_segment_length: float,
    randomness_length: float,
    min_gradient_scale: float,
    max_gradient_scale: float,
    # Curves parameters
    curves_segment_length: float,
    curves_randomness_length: float,
    max_steps: int,
    step_size: float,
    bezier_samples: int,
) -> tuple[str, str | None, np.ndarray]:
    """Process image with selected algorithm and parameters.

    Returns:
        Tuple of (SVG content string, path to temp SVG file for download, gradient magnitude image).
    """
    input_image = normalize_input_image(input_image)

    # Preprocess image
    processed = preprocess(
        input_image,
        output_width=output_width,
        levels=levels,
        invert=invert,
    )

    # Compute gradients for algorithms that need them
    gradients = compute_gradients(processed.original, sigma=gradient_sigma)

    svg_content, gradient_magnitude = generate_svg_content(
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

    # Save to temp file for download
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w") as f:
        f.write(svg_content)
        temp_path = f.name

    return svg_content, temp_path, gradient_magnitude


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
    Image.Image | None,
    str,
    Any,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    str,
    str,
    dict[str, Any],
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
    complexity_text = "Estimated vertices: -"

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

    svg_preview_image = None
    svg_code = ""
    download_update = gr.update(value=None, interactive=False)

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
        complexity_text = f"Estimated vertices: {estimated_vertices:,}"

        if enable_vertex_guard and estimated_vertices > int(max_estimated_vertices):
            status = (
                f"Skipped final SVG at {PIPELINE_STEP_LABELS[target_stage_index]}: "
                f"estimate {estimated_vertices:,} > limit {int(max_estimated_vertices):,}"
            )
            return (
                None,
                "",
                gr.update(value=None, interactive=False),
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

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w") as f:
            f.write(svg_content)
            temp_path = f.name

        # Render a high-resolution raster preview so Gradio's built-in
        # image expansion/zoom can be used reliably.
        width_match = re.search(r'width="([\d.]+)mm"', svg_content)
        height_match = re.search(r'height="([\d.]+)mm"', svg_content)

        preview_width_px = 1600
        preview_height_px = 1200
        if width_match and height_match:
            w = float(width_match.group(1))
            h = float(height_match.group(1))

            longest = max(w, h, 1.0)
            scale = 2200.0 / longest
            preview_width_px = int(np.clip(round(w * scale), 800, 2600))
            preview_height_px = int(np.clip(round(h * scale), 600, 2600))

        svg_preview_image = svg_to_png(
            svg_content,
            width=preview_width_px,
            height=preview_height_px,
        )
        svg_code = pretty_format_svg(svg_content)
        download_update = gr.update(value=temp_path, interactive=True)

        # Keep gradient tab meaningful for all algorithms once gradients are computed.
        if gradient_display is None:
            gradient_display = to_display_uint8(gradient_magnitude)
            gradient_display = resize_preview_nearest(gradient_display, preview_shape)

    status = f"Pipeline at {PIPELINE_STEP_LABELS[target_stage_index]}"
    return (
        svg_preview_image,
        svg_code,
        download_update,
        gradient_display,
        grayscale_display,
        downscaled_display,
        quantized_display,
        hist_preview,
        complexity_text,
        status,
        cache,
    )


def create_gui() -> gr.Blocks:
    """Create the Gradio interface."""
    with gr.Blocks(title="ScribbleTrace - Interactive Parameter Explorer", css=GUI_EXTRA_CSS) as app:
        gr.Markdown(
            """
            # ScribbleTrace Parameter Explorer

            Notebook-style pipeline where each step has local settings, output, and rerun controls.
            """
        )

        with gr.Accordion("Input", open=True, elem_id="step-input"):
            input_image = gr.Image(
                label="Input Image",
                type="numpy",
                sources=["upload", "clipboard"],
                value=get_default_gui_image(),
            )

        with gr.Accordion("Step Navigation", open=False):
            gr.Markdown(
                """
                Jump like a document: [Step 1](#step-1) | [Step 2](#step-2) | [Step 3](#step-3)
                """
            )
            with gr.Row():
                run_step_btn = gr.Button("Run 1 Step", variant="secondary")
                run_to_btn = gr.Button("Run To Selected", variant="secondary")
                run_all_btn = gr.Button("Run All", variant="primary")

            target_step = gr.Dropdown(
                choices=PIPELINE_STEP_LABELS,
                value=PIPELINE_STEP_LABELS[-1],
                label="Selected Pipeline Step",
            )
            pipeline_status = gr.Markdown("Pipeline at 1. Preprocess")

        with gr.Accordion("Step 1 - Preprocess (Resize, Invert, Quantize)", open=False, elem_id="step-1"):
            output_width = gr.Slider(
                minimum=10,
                maximum=200,
                value=40,
                step=1,
                label="Output Width (cells)",
                info="Number of cells across the image",
            )
            invert = gr.Checkbox(
                value=True,
                label="Invert",
                info="Dark areas produce more marks",
            )
            levels = gr.Slider(
                minimum=2,
                maximum=16,
                value=7,
                step=1,
                label="Intensity Levels",
                info="Number of quantization levels",
            )
            histogram_output = gr.Image(
                label="Histogram + Transform Markers",
                type="numpy",
                interactive=False,
            )
            with gr.Row():
                hist_min = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    label="Histogram Min",
                )
                hist_max = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.01,
                    label="Histogram Max",
                )
            hist_mid = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.01,
                label="Histogram Midpoint",
                info="Controls the midtone mapping between min and max",
            )
            run_gray_btn = gr.Button("Play Until Here", variant="secondary")
            grayscale_output = gr.Image(
                label="Resized + Invert Output",
                type="numpy",
                interactive=False,
            )
            quantized_output = gr.Image(
                label="Quantized Output",
                type="numpy",
                interactive=False,
            )
            # Keep this hidden component for stable callback outputs.
            downscaled_output = gr.Image(
                label="",
                type="numpy",
                interactive=False,
                visible=False,
            )

        with gr.Accordion("Step 2 - Gradient Magnitude", open=False, elem_id="step-2"):
            gradient_sigma = gr.Slider(
                minimum=0.0,
                maximum=6.0,
                value=1.0,
                step=0.1,
                label="Gradient Sigma",
                info="Gaussian smoothing before gradient magnitude",
            )
            run_gradient_btn = gr.Button("Play Until Here", variant="secondary")
            gradient_output = gr.Image(
                label="Step 2 Output",
                type="numpy",
                interactive=False,
            )

        with gr.Accordion("Step 3 - Final SVG", open=True, elem_id="step-3"):
            algorithm = gr.Dropdown(
                choices=["spirals", "circles", "squares", "lines", "curves", "hatching"],
                value="spirals",
                label="Algorithm",
            )
            color_theme = gr.Dropdown(
                choices=["Black Lines on White", "White Lines on Black"],
                value="Black Lines on White",
                label="SVG Color Theme",
            )
            stroke_width = gr.Slider(
                minimum=0.01,
                maximum=2.0,
                value=0.5,
                step=0.01,
                label="Stroke Width (mm)",
            )

            with gr.Accordion("Safety Limits", open=False):
                enable_vertex_guard = gr.Checkbox(
                    value=True,
                    label="Prevent Overly Large SVG Jobs",
                    info="Skips rendering when estimated vertices exceed the limit",
                )
                max_estimated_vertices = gr.Slider(
                    minimum=50000,
                    maximum=5000000,
                    value=900000,
                    step=50000,
                    label="Max Estimated Vertices",
                )
                complexity_status = gr.Markdown("Estimated vertices: -")

            with gr.Accordion("Shared Randomness", open=False):
                randomness_vertex = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.1,
                    step=0.01,
                    label="Vertex Randomness",
                    info="Random displacement of vertices",
                )
                randomness_position = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="Position Randomness",
                    info="Random offset of element positions",
                )

            with gr.Accordion("Spirals Settings", open=False, visible=True) as spirals_accordion:
                theta_resolution = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Theta Resolution",
                    info="Points per rotation",
                )
                spiral_b = gr.Slider(
                    minimum=0.1,
                    maximum=3.0,
                    value=1.0,
                    step=0.1,
                    label="Spiral Growth Rate",
                )
                connect_cells = gr.Checkbox(
                    value=True,
                    label="Connect Cells",
                    info="Draw lines between cells",
                )

            with gr.Accordion("Circles/Squares Settings", open=False, visible=False) as circles_accordion:
                circle_points = gr.Slider(
                    minimum=12,
                    maximum=72,
                    value=36,
                    step=6,
                    label="Circle Points",
                    info="Points per circle (circles only)",
                )
                small_first = gr.Checkbox(
                    value=True,
                    label="Small First",
                    info="Draw smaller shapes first",
                )

            with gr.Accordion("Lines Settings", open=False, visible=False) as lines_accordion:
                lines_segment_length = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=1.0,
                    step=0.1,
                    label="Segment Length",
                )
                randomness_length = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="Length Randomness",
                )
                min_gradient_scale = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.1,
                    step=0.01,
                    label="Min Gradient Scale",
                )
                max_gradient_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=10.0,
                    step=0.5,
                    label="Max Gradient Scale",
                )

            with gr.Accordion("Curves Settings", open=False, visible=False) as curves_accordion:
                curves_segment_length = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=1.0,
                    step=0.1,
                    label="Segment Length",
                )
                curves_randomness_length = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="Length Randomness",
                )
                max_steps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=4,
                    step=1,
                    label="Max Steps",
                )
                step_size = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=2.0,
                    step=0.25,
                    label="Step Size",
                )
                bezier_samples = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=15,
                    step=5,
                    label="Bezier Samples",
                )

            with gr.Accordion("Hatching Settings", open=False, visible=False) as hatching_accordion:
                gr.Markdown("**Hatching Directions:**")
                hatch_horizontal = gr.Checkbox(value=False, label="Horizontal")
                hatch_vertical = gr.Checkbox(value=False, label="Vertical")
                hatch_diag_right = gr.Checkbox(value=True, label="Diagonal Right (\\)")
                hatch_diag_left = gr.Checkbox(value=False, label="Diagonal Left (/)")
                min_spacing = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.3,
                    step=0.1,
                    label="Min Spacing (dark areas)",
                )
                max_spacing = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=2.0,
                    step=0.25,
                    label="Max Spacing (light areas)",
                )

            run_final_btn = gr.Button("Play Until Here", variant="primary")
            svg_output = gr.Image(
                label="SVG Preview",
                type="pil",
                interactive=False,
            )
            svg_code = gr.Code(
                label="SVG Source",
                language="html",
                lines=20,
                elem_id="svg-source-code",
            )
            download_btn = gr.DownloadButton(
                "Download SVG",
                value=None,
                variant="primary",
                size="lg",
                interactive=False,
            )

        with gr.Accordion("Presets", open=False):
            save_preset_btn = gr.DownloadButton(
                "Download Settings JSON",
                value=None,
                variant="secondary",
                size="lg",
            )
            load_preset_file = gr.File(
                label="Load Settings JSON",
                file_types=[".json"],
                type="filepath",
            )
            apply_preset_btn = gr.Button("Apply Loaded Preset", variant="secondary")
            preset_status = gr.Markdown("")

        current_stage_state = gr.State(value=0)
        pipeline_cache_state = gr.State(value={})

        # Visibility toggling based on algorithm selection
        def update_visibility(algo: str):
            return {
                spirals_accordion: gr.update(visible=algo == "spirals"),
                circles_accordion: gr.update(visible=algo in ("circles", "squares")),
                lines_accordion: gr.update(visible=algo == "lines"),
                curves_accordion: gr.update(visible=algo == "curves"),
                hatching_accordion: gr.update(visible=algo == "hatching"),
            }

        algorithm.change(
            fn=update_visibility,
            inputs=[algorithm],
            outputs=[spirals_accordion, circles_accordion, lines_accordion, curves_accordion, hatching_accordion],
        )

        # Process functions
        def on_run_to_stage(
            target_stage_idx,
            cache,
            img,
            algo,
            theme,
            h_min,
            h_mid,
            h_max,
            width,
            lvls,
            inv,
            grad_sigma,
            stroke,
            rand_v,
            rand_p,
            theta_res,
            spiral_growth,
            connect,
            circ_pts,
            sm_first,
            hatch_h,
            hatch_v,
            hatch_dr,
            hatch_dl,
            min_sp,
            max_sp,
            line_seg_len,
            rand_len,
            min_grad,
            max_grad,
            curve_seg_len,
            curve_rand_len,
            m_steps,
            s_size,
            bez_samp,
            guard_enabled,
            max_vertices,
        ):
            return run_pipeline_to_stage(
                cache,
                img,
                target_stage_idx,
                algo,
                theme,
                h_min,
                h_mid,
                h_max,
                width,
                lvls,
                inv,
                grad_sigma,
                stroke,
                rand_v,
                rand_p,
                theta_res,
                spiral_growth,
                connect,
                circ_pts,
                sm_first,
                hatch_h,
                hatch_v,
                hatch_dr,
                hatch_dl,
                min_sp,
                max_sp,
                line_seg_len,
                rand_len,
                min_grad,
                max_grad,
                curve_seg_len,
                curve_rand_len,
                m_steps,
                s_size,
                bez_samp,
                guard_enabled,
                max_vertices,
            )

        def run_step(current_stage, *params):
            next_stage = min(int(current_stage) + 1, len(PIPELINE_STEP_LABELS) - 1)
            outputs = on_run_to_stage(next_stage, *params)
            ui_outputs = outputs[:-1]
            cache = outputs[-1]
            return (*ui_outputs, next_stage, cache)

        def run_to_selected(step_label, *params):
            target_stage = PIPELINE_LABEL_TO_INDEX.get(step_label, len(PIPELINE_STEP_LABELS) - 1)
            outputs = on_run_to_stage(target_stage, *params)
            ui_outputs = outputs[:-1]
            cache = outputs[-1]
            return (*ui_outputs, target_stage, cache)

        def run_all(*params):
            target_stage = len(PIPELINE_STEP_LABELS) - 1
            outputs = on_run_to_stage(target_stage, *params)
            ui_outputs = outputs[:-1]
            cache = outputs[-1]
            return (*ui_outputs, target_stage, cache)

        def run_to_fixed_stage(stage_index: int, *params):
            outputs = on_run_to_stage(stage_index, *params)
            ui_outputs = outputs[:-1]
            cache = outputs[-1]
            return (*ui_outputs, stage_index, cache)

        # Collect all processing inputs
        pipeline_inputs = [
            input_image,
            algorithm,
            color_theme,
            hist_min,
            hist_mid,
            hist_max,
            output_width,
            levels,
            invert,
            gradient_sigma,
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
            enable_vertex_guard,
            max_estimated_vertices,
        ]

        pipeline_outputs = [
            svg_output,
            svg_code,
            download_btn,
            gradient_output,
            grayscale_output,
            downscaled_output,
            quantized_output,
            histogram_output,
            complexity_status,
            pipeline_status,
        ]

        setting_keys = [
            "algorithm",
            "color_theme",
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

        setting_components = [
            algorithm,
            color_theme,
            hist_min,
            hist_mid,
            hist_max,
            output_width,
            levels,
            gradient_sigma,
            invert,
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
            enable_vertex_guard,
            max_estimated_vertices,
        ]

        def create_preset_file(*settings):
            preset = {
                "version": 1,
                "settings": dict(zip(setting_keys, settings, strict=False)),
            }
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as f:
                json.dump(preset, f, indent=2)
                temp_path = f.name
            return gr.update(value=temp_path)

        def apply_preset_file(preset_file_path: str | None, *current_settings):
            current_values = list(current_settings)
            current_algo = current_values[0] if current_values else "spirals"

            if not preset_file_path:
                vis = update_visibility(current_algo)
                return (
                    *current_values,
                    vis[spirals_accordion],
                    vis[circles_accordion],
                    vis[lines_accordion],
                    vis[curves_accordion],
                    vis[hatching_accordion],
                    "Upload a preset JSON file first.",
                )

            try:
                with open(preset_file_path, encoding="utf-8") as f:
                    payload = json.load(f)

                settings = payload.get("settings", payload)
                if not isinstance(settings, dict):
                    raise ValueError("Preset content must be a JSON object.")

                for idx, key in enumerate(setting_keys):
                    if key in settings:
                        current_values[idx] = settings[key]

                selected_algo = current_values[0]
                vis = update_visibility(selected_algo)
                status = f"Loaded preset: {Path(preset_file_path).name}"

                return (
                    *current_values,
                    vis[spirals_accordion],
                    vis[circles_accordion],
                    vis[lines_accordion],
                    vis[curves_accordion],
                    vis[hatching_accordion],
                    status,
                )
            except Exception as err:  # noqa: BLE001
                vis = update_visibility(current_algo)
                return (
                    *current_values,
                    vis[spirals_accordion],
                    vis[circles_accordion],
                    vis[lines_accordion],
                    vis[curves_accordion],
                    vis[hatching_accordion],
                    f"Failed to load preset: {err}",
                )

        run_step_btn.click(
            fn=run_step,
            inputs=[current_stage_state, pipeline_cache_state, *pipeline_inputs],
            outputs=[*pipeline_outputs, current_stage_state, pipeline_cache_state],
        )

        run_to_btn.click(
            fn=run_to_selected,
            inputs=[target_step, pipeline_cache_state, *pipeline_inputs],
            outputs=[*pipeline_outputs, current_stage_state, pipeline_cache_state],
        )

        run_all_btn.click(
            fn=run_all,
            inputs=[pipeline_cache_state, *pipeline_inputs],
            outputs=[*pipeline_outputs, current_stage_state, pipeline_cache_state],
        )

        run_gray_btn.click(
            fn=lambda *params: run_to_fixed_stage(0, *params),
            inputs=[pipeline_cache_state, *pipeline_inputs],
            outputs=[*pipeline_outputs, current_stage_state, pipeline_cache_state],
        )

        run_gradient_btn.click(
            fn=lambda *params: run_to_fixed_stage(1, *params),
            inputs=[pipeline_cache_state, *pipeline_inputs],
            outputs=[*pipeline_outputs, current_stage_state, pipeline_cache_state],
        )

        run_final_btn.click(
            fn=lambda *params: run_to_fixed_stage(2, *params),
            inputs=[pipeline_cache_state, *pipeline_inputs],
            outputs=[*pipeline_outputs, current_stage_state, pipeline_cache_state],
        )

        # New images start at step 1 preview to support step-by-step workflows.
        input_image.change(
            fn=lambda *params: run_to_selected(PIPELINE_STEP_LABELS[0], *params),
            inputs=[pipeline_cache_state, *pipeline_inputs],
            outputs=[*pipeline_outputs, current_stage_state, pipeline_cache_state],
        )

        save_preset_btn.click(
            fn=create_preset_file,
            inputs=setting_components,
            outputs=[save_preset_btn],
        )

        apply_preset_btn.click(
            fn=apply_preset_file,
            inputs=[load_preset_file, *setting_components],
            outputs=[
                *setting_components,
                spirals_accordion,
                circles_accordion,
                lines_accordion,
                curves_accordion,
                hatching_accordion,
                preset_status,
            ],
        )

    return app


def main():
    """Launch the GUI."""
    app = create_gui()
    app.queue()  # Enable queuing for better handling of rapid changes
    theme = gr.themes.ThemeClass.from_hub(HUB_THEME_ID)
    app.launch(
        theme=theme,
        share=False,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
