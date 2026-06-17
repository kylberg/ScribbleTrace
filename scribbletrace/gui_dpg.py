"""Dear PyGUI desktop interface for ScribbleTrace.

This module provides a parallel GUI implementation to the Gradio app,
including staged pipeline execution and interactive histogram controls.

Usage:
    scribbletrace-dpg
    # or
    python -m scribbletrace.gui_dpg
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import io
import json
import os
import re
import tempfile
import xml.dom.minidom
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scribbletrace import SVGWriter, compute_gradients, load_image, preprocess
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
from scribbletrace.image_processing import ProcessedImage
from scribbletrace.svg_output import SVGConfig

dpg = None


def _validate_graphics_environment() -> None:
    """Validate that a GUI/OpenGL environment is available before GLFW init.

    Dear PyGUI uses GLFW/OpenGL. On headless systems or environments missing
    GLX libraries, GLFW can abort the process. We preflight here to return a
    clear actionable error instead.
    """
    if os.name != "posix":
        return

    display = os.environ.get("DISPLAY")
    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    if not display and not wayland_display:
        raise RuntimeError(
            "No GUI display detected. Set DISPLAY/WAYLAND_DISPLAY or run from a desktop session.\n"
            "If you are on a server, use the web GUI instead: scribbletrace-gui"
        )

    # On Linux/X11, GLFW needs OpenGL/GLX runtime libraries.
    missing_libs = []
    for lib_name in ("GL", "GLX"):
        resolved = ctypes.util.find_library(lib_name)
        if resolved is None:
            missing_libs.append(lib_name)

    if missing_libs:
        raise RuntimeError(
            "Missing OpenGL runtime libraries for Dear PyGUI: "
            f"{', '.join(missing_libs)}.\n"
            "Install system packages (Ubuntu/Debian):\n"
            "  sudo apt install libgl1 libglx0 libgl1-mesa-dri\n"
            "Then retry: python -m scribbletrace.gui_dpg"
        )

    # Force-load libs early so failures are reported as Python exceptions,
    # not later as a GLFW abort.
    for soname in ("libGL.so.1", "libGLX.so.0"):
        try:
            ctypes.CDLL(soname)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load {soname}: {exc}\n"
                "Install/repair OpenGL runtime packages, then retry."
            ) from exc


def _require_dpg():
    global dpg
    if dpg is not None:
        return dpg
    try:
        import dearpygui.dearpygui as _dpg
    except Exception as exc:  # pragma: no cover - runtime dependency
        err = str(exc)

        # Some conda environments ship an older libstdc++ that cannot load
        # Dear PyGUI wheels built with newer C++ runtimes.
        if "GLIBCXX_" in err and "libstdc++.so.6" in err:
            system_libstdcpp = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
            if system_libstdcpp.exists():
                try:
                    ctypes.CDLL(str(system_libstdcpp), mode=ctypes.RTLD_GLOBAL)
                    import dearpygui.dearpygui as _dpg
                    dpg = _dpg
                    return dpg
                except Exception as retry_exc:  # noqa: BLE001
                    err = f"{err}\nRetry with system libstdc++ failed: {retry_exc}"

            raise ImportError(
                "Dear PyGUI failed to load due to a C++ runtime mismatch (GLIBCXX).\n"
                "Suggested fixes:\n"
                "1) In conda env: conda install -n leafbright -c conda-forge 'libstdcxx-ng>=12' 'libgcc-ng>=12'\n"
                "2) Immediate workaround: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python -m scribbletrace.gui_dpg\n"
                "3) Reopen shell and retry: python -m scribbletrace.gui_dpg\n"
                f"Original error: {err}"
            ) from exc

        raise ImportError(
            "Dear PyGUI is not installed or failed to load. Install with: pip install dearpygui\n"
            f"Original error: {err}"
        ) from exc
    dpg = _dpg
    return dpg

DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent.parent / "examples" / "MagrittePipe.jpg"

PIPELINE_STEP_LABELS = [
    "1. Preprocess",
    "2. Gradient Magnitude",
    "3. Final SVG",
]

PIPELINE_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(PIPELINE_STEP_LABELS)}

PREVIEW_GRAY_SIZE = (512, 512)
PREVIEW_HIST_SIZE = (720, 220)
PREVIEW_SVG_SIZE = (1000, 750)


@dataclass
class AppState:
    """Runtime state for the Dear PyGUI app."""

    input_image: np.ndarray | None = None
    image_path: str = ""
    cache: dict[str, Any] = field(default_factory=dict)
    current_stage: int = 0
    svg_content: str = ""
    suppress_hist_callbacks: bool = False
    suppress_step_callbacks: bool = False


def create_sample_image() -> np.ndarray:
    """Create a simple gradient sample image for testing."""
    size = 200
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    img = 1 - np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 1.5
    return np.clip(img, 0, 1)


def get_default_gui_image() -> np.ndarray | None:
    """Load the default GUI image if available, otherwise return None."""
    if DEFAULT_IMAGE_PATH.exists():
        try:
            return np.array(Image.open(DEFAULT_IMAGE_PATH).convert("RGB"))
        except Exception:  # noqa: BLE001
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
        if img.shape[2] >= 3:
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


def pretty_format_svg(svg_content: str) -> str:
    """Return an indented SVG string suitable for source preview."""
    try:
        dom = xml.dom.minidom.parseString(svg_content.encode("utf-8"))
        pretty = dom.toprettyxml(indent="  ")
        return "\n".join(line for line in pretty.splitlines() if line.strip())
    except Exception:  # noqa: BLE001
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
    """Generate SVG content and gradient magnitude from computed intermediates."""
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


def svg_to_png(svg_content: str, width: int = 1000, height: int = 750) -> Image.Image:
    """Convert SVG content to PNG image for display."""
    try:
        import cairosvg

        png_data = cairosvg.svg2png(
            bytestring=svg_content.encode("utf-8"),
            output_width=width,
            output_height=height,
        )
        return Image.open(io.BytesIO(png_data)).convert("RGB")
    except Exception:  # noqa: BLE001
        return Image.new("RGB", (width, height), color="white")


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.asarray(np.clip(arr, 0.0, 255.0), dtype=np.uint8)
    return arr


def _resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    pil = Image.fromarray(_to_rgb_uint8(image), mode="RGB")
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.BILINEAR
    else:
        resample = Image.BILINEAR
    return np.asarray(pil.resize((w, h), resample=resample), dtype=np.uint8)


def _to_texture_data(image: np.ndarray, size: tuple[int, int]) -> list[float]:
    rgb = _resize_rgb(image, size)
    alpha = np.full((size[0], size[1], 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=2).astype(np.float32) / 255.0
    return rgba.ravel().tolist()


class DearPyGuiApp:
    """Desktop GUI built with Dear PyGUI."""

    def __init__(self) -> None:
        self.state = AppState(input_image=get_default_gui_image())

    def _slider(self, tag: str) -> Any:
        return dpg.get_value(tag)

    def _run_pipeline_to_stage(self, target_stage_index: int) -> dict[str, Any]:
        cache = self.state.cache if isinstance(self.state.cache, dict) else {}
        target_stage_index = int(np.clip(target_stage_index, 0, len(PIPELINE_STEP_LABELS) - 1))

        hist_min = float(self._slider("hist_min"))
        hist_mid = float(self._slider("hist_mid"))
        hist_max = float(self._slider("hist_max"))
        hist_min, hist_mid, hist_max = _normalize_histogram_knots(hist_min, hist_mid, hist_max)

        if not self.state.suppress_hist_callbacks:
            self.state.suppress_hist_callbacks = True
            dpg.set_value("hist_min", hist_min)
            dpg.set_value("hist_mid", hist_mid)
            dpg.set_value("hist_max", hist_max)
            dpg.set_value("hist_line_min", hist_min)
            dpg.set_value("hist_line_mid", hist_mid)
            dpg.set_value("hist_line_max", hist_max)
            self.state.suppress_hist_callbacks = False

        grayscale_original = normalize_input_image(self.state.input_image)
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
        svg_preview_image = np.full((PREVIEW_SVG_SIZE[0], PREVIEW_SVG_SIZE[1], 3), 255, dtype=np.uint8)
        svg_code = ""
        status = f"Pipeline at {PIPELINE_STEP_LABELS[target_stage_index]}"

        processed: ProcessedImage | None = None
        gradients = None

        output_width = float(self._slider("output_width"))
        levels = int(self._slider("levels"))
        invert = bool(self._slider("invert"))
        gradient_sigma = float(self._slider("gradient_sigma"))

        if target_stage_index >= 0:
            processed_key = (
                output_width,
                levels,
                invert,
                round(hist_min, 6),
                round(hist_mid, 6),
                round(hist_max, 6),
            )
            if cache.get("processed_key") != processed_key:
                cache["processed"] = preprocess(
                    grayscale,
                    output_width=output_width,
                    levels=levels,
                    invert=invert,
                )
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
            sigma_key = round(gradient_sigma, 4)
            gradients_by_sigma = cache.setdefault("gradients_by_sigma", {})
            gradients = gradients_by_sigma.get(sigma_key)
            if gradients is None:
                gradients = compute_gradients(processed.original, sigma=gradient_sigma)
                gradients_by_sigma[sigma_key] = gradients
            gradient_display = to_display_uint8(gradients.magnitude)
            gradient_display = resize_preview_nearest(gradient_display, preview_shape)

        if target_stage_index >= 2:
            if processed is None:
                processed = preprocess(
                    grayscale,
                    output_width=output_width,
                    levels=levels,
                    invert=invert,
                )

            algorithm = self._slider("algorithm")
            estimated_vertices = estimate_vertex_count(
                processed,
                algorithm,
                int(self._slider("theta_resolution")),
                int(self._slider("circle_points")),
                int(self._slider("max_steps")),
                int(self._slider("bezier_samples")),
                bool(self._slider("hatch_horizontal")),
                bool(self._slider("hatch_vertical")),
                bool(self._slider("hatch_diag_right")),
                bool(self._slider("hatch_diag_left")),
                float(self._slider("min_spacing")),
            )
            complexity_text = f"Estimated vertices: {estimated_vertices:,}"

            if bool(self._slider("enable_vertex_guard")) and estimated_vertices > int(
                self._slider("max_estimated_vertices")
            ):
                status = (
                    f"Skipped final SVG at {PIPELINE_STEP_LABELS[target_stage_index]}: "
                    f"estimate {estimated_vertices:,} > limit {int(self._slider('max_estimated_vertices')):,}"
                )
                self.state.cache = cache
                return {
                    "svg_preview": svg_preview_image,
                    "svg_code": svg_code,
                    "gradient": gradient_display,
                    "grayscale": grayscale_display,
                    "quantized": quantized_display,
                    "hist": hist_preview,
                    "complexity": complexity_text,
                    "status": status,
                }

            sigma_key = round(gradient_sigma, 4)
            gradients_by_sigma = cache.setdefault("gradients_by_sigma", {})
            gradients = gradients_by_sigma.get(sigma_key)
            if gradients is None:
                gradients = compute_gradients(processed.original, sigma=gradient_sigma)
                gradients_by_sigma[sigma_key] = gradients

            svg_key = (
                self._slider("algorithm"),
                self._slider("color_theme"),
                float(self._slider("stroke_width")),
                float(self._slider("randomness_vertex")),
                float(self._slider("randomness_position")),
                int(self._slider("theta_resolution")),
                float(self._slider("spiral_b")),
                bool(self._slider("connect_cells")),
                int(self._slider("circle_points")),
                bool(self._slider("small_first")),
                bool(self._slider("hatch_horizontal")),
                bool(self._slider("hatch_vertical")),
                bool(self._slider("hatch_diag_right")),
                bool(self._slider("hatch_diag_left")),
                float(self._slider("min_spacing")),
                float(self._slider("max_spacing")),
                float(self._slider("lines_segment_length")),
                float(self._slider("randomness_length")),
                float(self._slider("min_gradient_scale")),
                float(self._slider("max_gradient_scale")),
                float(self._slider("curves_segment_length")),
                float(self._slider("curves_randomness_length")),
                int(self._slider("max_steps")),
                float(self._slider("step_size")),
                int(self._slider("bezier_samples")),
                sigma_key,
            )

            svg_by_key = cache.setdefault("svg_by_key", {})
            cached_svg = svg_by_key.get(svg_key)
            if cached_svg is None:
                cached_svg = generate_svg_content(
                    processed,
                    gradients,
                    self._slider("algorithm"),
                    self._slider("color_theme"),
                    float(self._slider("stroke_width")),
                    float(self._slider("randomness_vertex")),
                    float(self._slider("randomness_position")),
                    int(self._slider("theta_resolution")),
                    float(self._slider("spiral_b")),
                    bool(self._slider("connect_cells")),
                    int(self._slider("circle_points")),
                    bool(self._slider("small_first")),
                    bool(self._slider("hatch_horizontal")),
                    bool(self._slider("hatch_vertical")),
                    bool(self._slider("hatch_diag_right")),
                    bool(self._slider("hatch_diag_left")),
                    float(self._slider("min_spacing")),
                    float(self._slider("max_spacing")),
                    float(self._slider("lines_segment_length")),
                    float(self._slider("randomness_length")),
                    float(self._slider("min_gradient_scale")),
                    float(self._slider("max_gradient_scale")),
                    float(self._slider("curves_segment_length")),
                    float(self._slider("curves_randomness_length")),
                    int(self._slider("max_steps")),
                    float(self._slider("step_size")),
                    int(self._slider("bezier_samples")),
                )
                svg_by_key[svg_key] = cached_svg

            svg_content, gradient_magnitude = cached_svg
            self.state.svg_content = svg_content
            svg_code = pretty_format_svg(svg_content)

            width_match = re.search(r'width="([\d.]+)mm"', svg_content)
            height_match = re.search(r'height="([\d.]+)mm"', svg_content)

            preview_width_px = PREVIEW_SVG_SIZE[1]
            preview_height_px = PREVIEW_SVG_SIZE[0]
            if width_match and height_match:
                w = float(width_match.group(1))
                h = float(height_match.group(1))
                longest = max(w, h, 1.0)
                scale = 2000.0 / longest
                preview_width_px = int(np.clip(round(w * scale), 900, 2200))
                preview_height_px = int(np.clip(round(h * scale), 700, 2200))

            svg_preview_image = np.asarray(
                svg_to_png(svg_content, width=preview_width_px, height=preview_height_px)
            )

            if gradient_display is None:
                gradient_display = to_display_uint8(gradient_magnitude)
                gradient_display = resize_preview_nearest(gradient_display, preview_shape)

        self.state.cache = cache
        return {
            "svg_preview": svg_preview_image,
            "svg_code": svg_code,
            "gradient": gradient_display,
            "grayscale": grayscale_display,
            "quantized": quantized_display,
            "hist": hist_preview,
            "complexity": complexity_text,
            "status": status,
        }

    def _set_texture(self, texture_tag: str, image: np.ndarray, size: tuple[int, int]) -> None:
        dpg.set_value(texture_tag, _to_texture_data(image, size))

    def _update_hist_plot(self, image: np.ndarray, h_min: float, h_mid: float, h_max: float) -> None:
        values = np.clip(np.asarray(image, dtype=np.float64).ravel(), 0.0, 1.0)
        bins = 128
        counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
        x = ((edges[:-1] + edges[1:]) * 0.5).tolist()
        y = counts.astype(float).tolist()
        dpg.set_value("hist_bars", [x, y])
        dpg.fit_axis_data("hist_x")
        dpg.fit_axis_data("hist_y")

        self.state.suppress_hist_callbacks = True
        dpg.set_value("hist_line_min", h_min)
        dpg.set_value("hist_line_mid", h_mid)
        dpg.set_value("hist_line_max", h_max)
        self.state.suppress_hist_callbacks = False

    def _update_outputs(self, outputs: dict[str, Any]) -> None:
        if outputs.get("grayscale") is not None:
            self._set_texture("tex_step1", outputs["grayscale"], PREVIEW_GRAY_SIZE)
        if outputs.get("quantized") is not None:
            self._set_texture("tex_quant", outputs["quantized"], PREVIEW_GRAY_SIZE)
        if outputs.get("gradient") is not None:
            self._set_texture("tex_grad", outputs["gradient"], PREVIEW_GRAY_SIZE)
        if outputs.get("hist") is not None:
            self._set_texture("tex_hist", outputs["hist"], PREVIEW_HIST_SIZE)

        svg_preview = outputs.get("svg_preview")
        if svg_preview is not None:
            self._set_texture("tex_svg", svg_preview, PREVIEW_SVG_SIZE)

        dpg.set_value("svg_code", outputs.get("svg_code", ""))
        dpg.set_value("complexity_text", outputs.get("complexity", "Estimated vertices: -"))
        dpg.set_value("status_text", outputs.get("status", ""))

    def _run_to_stage(self, stage_idx: int) -> None:
        outputs = self._run_pipeline_to_stage(stage_idx)
        self.state.current_stage = stage_idx
        self._set_step_selection(PIPELINE_STEP_LABELS[stage_idx])
        self._update_outputs(outputs)

        base_image = normalize_input_image(self.state.input_image)
        h_min = float(dpg.get_value("hist_min"))
        h_mid = float(dpg.get_value("hist_mid"))
        h_max = float(dpg.get_value("hist_max"))
        self._update_hist_plot(base_image, h_min, h_mid, h_max)

    def _on_run_one(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        next_stage = min(self.state.current_stage + 1, len(PIPELINE_STEP_LABELS) - 1)
        self._run_to_stage(next_stage)

    def _on_run_to_selected(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        stage = PIPELINE_LABEL_TO_INDEX.get(dpg.get_value("target_step"), len(PIPELINE_STEP_LABELS) - 1)
        self._run_to_stage(stage)

    def _on_run_all(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        self._run_to_stage(len(PIPELINE_STEP_LABELS) - 1)

    def _on_run_step_fixed(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        self._run_to_stage(int(user_data))

    def _set_step_selection(self, step_label: str) -> None:
        """Synchronize left step list and right step tab without callback loops."""
        if step_label not in PIPELINE_LABEL_TO_INDEX:
            return

        idx = PIPELINE_LABEL_TO_INDEX[step_label]
        self.state.suppress_step_callbacks = True
        dpg.set_value("target_step", step_label)
        if dpg.does_item_exist("step_tabs"):
            dpg.set_value("step_tabs", f"tab_step_{idx}")
        self.state.suppress_step_callbacks = False

    def _on_step_selected(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        if self.state.suppress_step_callbacks:
            return
        step_label = str(dpg.get_value("target_step"))
        self._set_step_selection(step_label)

    def _on_step_tab_changed(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        if self.state.suppress_step_callbacks:
            return

        selected_tab = str(app_data)
        for idx, label in enumerate(PIPELINE_STEP_LABELS):
            if selected_tab == f"tab_step_{idx}":
                self._set_step_selection(label)
                break

    def _on_hist_slider_changed(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        if self.state.suppress_hist_callbacks:
            return

        h_min, h_mid, h_max = _normalize_histogram_knots(
            float(dpg.get_value("hist_min")),
            float(dpg.get_value("hist_mid")),
            float(dpg.get_value("hist_max")),
        )

        self.state.suppress_hist_callbacks = True
        dpg.set_value("hist_min", h_min)
        dpg.set_value("hist_mid", h_mid)
        dpg.set_value("hist_max", h_max)
        dpg.set_value("hist_line_min", h_min)
        dpg.set_value("hist_line_mid", h_mid)
        dpg.set_value("hist_line_max", h_max)
        self.state.suppress_hist_callbacks = False

        self._run_to_stage(min(self.state.current_stage, 1))

    def _on_hist_drag_line_changed(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        if self.state.suppress_hist_callbacks:
            return

        h_min, h_mid, h_max = _normalize_histogram_knots(
            float(dpg.get_value("hist_line_min")),
            float(dpg.get_value("hist_line_mid")),
            float(dpg.get_value("hist_line_max")),
        )

        self.state.suppress_hist_callbacks = True
        dpg.set_value("hist_line_min", h_min)
        dpg.set_value("hist_line_mid", h_mid)
        dpg.set_value("hist_line_max", h_max)
        dpg.set_value("hist_min", h_min)
        dpg.set_value("hist_mid", h_mid)
        dpg.set_value("hist_max", h_max)
        self.state.suppress_hist_callbacks = False

        self._run_to_stage(min(self.state.current_stage, 1))

    def _on_algorithm_or_input_changed(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        self._run_to_stage(min(self.state.current_stage, 0))

    def _on_open_image_dialog(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        selected = app_data.get("file_path_name", "") if isinstance(app_data, dict) else ""
        if not selected:
            return
        try:
            self.state.input_image = load_image(selected)
            self.state.image_path = selected
            dpg.set_value("input_path", selected)
            self.state.cache = {}
            self._set_texture("tex_input", self.state.input_image, PREVIEW_GRAY_SIZE)
            self._run_to_stage(0)
        except Exception as err:  # noqa: BLE001
            dpg.set_value("status_text", f"Failed to load image: {err}")

    def _on_save_svg(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        path = str(dpg.get_value("save_svg_path") or "").strip()
        if not path:
            dpg.set_value("status_text", "Set a save path first.")
            return
        if not self.state.svg_content:
            dpg.set_value("status_text", "No SVG generated yet. Run Step 3 first.")
            return
        try:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(self.state.svg_content, encoding="utf-8")
            dpg.set_value("status_text", f"Saved SVG to: {out}")
        except Exception as err:  # noqa: BLE001
            dpg.set_value("status_text", f"Failed to save SVG: {err}")

    def _on_save_preset(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        path = str(dpg.get_value("preset_path") or "").strip()
        if not path:
            dpg.set_value("status_text", "Set a preset file path first.")
            return

        setting_tags = [
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

        payload = {"version": 1, "settings": {tag: dpg.get_value(tag) for tag in setting_tags}}

        try:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            dpg.set_value("status_text", f"Preset saved: {out}")
        except Exception as err:  # noqa: BLE001
            dpg.set_value("status_text", f"Failed to save preset: {err}")

    def _on_load_preset(self, sender, app_data, user_data) -> None:  # noqa: ANN001
        path = str(dpg.get_value("preset_path") or "").strip()
        if not path:
            dpg.set_value("status_text", "Set a preset file path first.")
            return

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            settings = payload.get("settings", payload)
            if not isinstance(settings, dict):
                raise ValueError("Preset must contain a JSON object")

            for key, value in settings.items():
                if dpg.does_item_exist(key):
                    dpg.set_value(key, value)

            self.state.cache = {}
            self._run_to_stage(self.state.current_stage)
            dpg.set_value("status_text", f"Preset loaded: {path}")
        except Exception as err:  # noqa: BLE001
            dpg.set_value("status_text", f"Failed to load preset: {err}")

    def _build_textures(self) -> None:
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=PREVIEW_GRAY_SIZE[1],
                height=PREVIEW_GRAY_SIZE[0],
                default_value=_to_texture_data(np.zeros((*PREVIEW_GRAY_SIZE, 3), np.uint8), PREVIEW_GRAY_SIZE),
                tag="tex_input",
            )
            dpg.add_dynamic_texture(
                width=PREVIEW_GRAY_SIZE[1],
                height=PREVIEW_GRAY_SIZE[0],
                default_value=_to_texture_data(np.zeros((*PREVIEW_GRAY_SIZE, 3), np.uint8), PREVIEW_GRAY_SIZE),
                tag="tex_step1",
            )
            dpg.add_dynamic_texture(
                width=PREVIEW_GRAY_SIZE[1],
                height=PREVIEW_GRAY_SIZE[0],
                default_value=_to_texture_data(np.zeros((*PREVIEW_GRAY_SIZE, 3), np.uint8), PREVIEW_GRAY_SIZE),
                tag="tex_quant",
            )
            dpg.add_dynamic_texture(
                width=PREVIEW_GRAY_SIZE[1],
                height=PREVIEW_GRAY_SIZE[0],
                default_value=_to_texture_data(np.zeros((*PREVIEW_GRAY_SIZE, 3), np.uint8), PREVIEW_GRAY_SIZE),
                tag="tex_grad",
            )
            dpg.add_dynamic_texture(
                width=PREVIEW_HIST_SIZE[1],
                height=PREVIEW_HIST_SIZE[0],
                default_value=_to_texture_data(np.zeros((*PREVIEW_HIST_SIZE, 3), np.uint8), PREVIEW_HIST_SIZE),
                tag="tex_hist",
            )
            dpg.add_dynamic_texture(
                width=PREVIEW_SVG_SIZE[1],
                height=PREVIEW_SVG_SIZE[0],
                default_value=_to_texture_data(np.full((*PREVIEW_SVG_SIZE, 3), 255, np.uint8), PREVIEW_SVG_SIZE),
                tag="tex_svg",
            )

    def _build_ui(self) -> None:
        _require_dpg()
        dpg.create_context()
        self._build_textures()

        with dpg.window(label="ScribbleTrace (Dear PyGUI)", tag="main_window"):
            dpg.add_text("Notebook-style pipeline with Dear PyGUI")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                with dpg.child_window(width=320, autosize_y=True, border=True):
                    dpg.add_text("Pipeline Steps")
                    dpg.add_listbox(
                        PIPELINE_STEP_LABELS,
                        default_value=PIPELINE_STEP_LABELS[0],
                        tag="target_step",
                        callback=self._on_step_selected,
                        num_items=3,
                    )
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Run 1 Step", callback=self._on_run_one)
                        dpg.add_button(label="Run To Selected", callback=self._on_run_to_selected)
                    dpg.add_button(label="Run All", callback=self._on_run_all)

                    dpg.add_separator()
                    dpg.add_text("Input")
                    dpg.add_input_text(tag="input_path", label="Image Path", width=280, readonly=True)
                    dpg.add_button(label="Open Image...", callback=lambda: dpg.show_item("open_image_dialog"))
                    dpg.add_image("tex_input")

                    dpg.add_separator()
                    dpg.add_text("Pipeline at 1. Preprocess", tag="status_text")
                    dpg.add_text("Estimated vertices: -", tag="complexity_text")

                    dpg.add_separator()
                    dpg.add_text("Presets")
                    dpg.add_input_text(tag="preset_path", default_value="scribbletrace_preset.json", width=280)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Save", callback=self._on_save_preset)
                        dpg.add_button(label="Load", callback=self._on_load_preset)

                with dpg.child_window(width=-1, autosize_y=True, border=False):
                    with dpg.tab_bar(tag="step_tabs", callback=self._on_step_tab_changed):
                        with dpg.tab(label="Step 1 - Preprocess", tag="tab_step_0"):
                            dpg.add_slider_int(
                                tag="output_width",
                                label="Output Width (cells)",
                                min_value=10,
                                max_value=200,
                                default_value=40,
                            )
                            dpg.add_checkbox(tag="invert", label="Invert", default_value=True)
                            dpg.add_slider_int(
                                tag="levels",
                                label="Intensity Levels",
                                min_value=2,
                                max_value=16,
                                default_value=7,
                            )

                            dpg.add_image("tex_hist")
                            with dpg.plot(label="Histogram (drag lines to adjust)", height=260, width=760):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="Tone", tag="hist_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="Count", tag="hist_y"):
                                    dpg.add_bar_series(
                                        [0.0],
                                        [0.0],
                                        label="Histogram",
                                        tag="hist_bars",
                                        weight=0.006,
                                    )
                                dpg.add_drag_line(
                                    label="Min",
                                    tag="hist_line_min",
                                    default_value=0.0,
                                    color=(230, 57, 70, 255),
                                    callback=self._on_hist_drag_line_changed,
                                )
                                dpg.add_drag_line(
                                    label="Mid",
                                    tag="hist_line_mid",
                                    default_value=0.5,
                                    color=(241, 160, 17, 255),
                                    callback=self._on_hist_drag_line_changed,
                                )
                                dpg.add_drag_line(
                                    label="Max",
                                    tag="hist_line_max",
                                    default_value=1.0,
                                    color=(69, 123, 157, 255),
                                    callback=self._on_hist_drag_line_changed,
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_slider_float(
                                    tag="hist_min",
                                    label="Histogram Min",
                                    min_value=0.0,
                                    max_value=1.0,
                                    default_value=0.0,
                                    callback=self._on_hist_slider_changed,
                                )
                                dpg.add_slider_float(
                                    tag="hist_max",
                                    label="Histogram Max",
                                    min_value=0.0,
                                    max_value=1.0,
                                    default_value=1.0,
                                    callback=self._on_hist_slider_changed,
                                )
                            dpg.add_slider_float(
                                tag="hist_mid",
                                label="Histogram Midpoint",
                                min_value=0.0,
                                max_value=1.0,
                                default_value=0.5,
                                callback=self._on_hist_slider_changed,
                            )

                            dpg.add_button(label="Play Until Here", callback=self._on_run_step_fixed, user_data=0)
                            dpg.add_text("Resized + Invert Output")
                            dpg.add_image("tex_step1")
                            dpg.add_text("Quantized Output")
                            dpg.add_image("tex_quant")

                        with dpg.tab(label="Step 2 - Gradient Magnitude", tag="tab_step_1"):
                            dpg.add_slider_float(
                                tag="gradient_sigma",
                                label="Gradient Sigma",
                                min_value=0.0,
                                max_value=6.0,
                                default_value=1.0,
                            )
                            dpg.add_button(label="Play Until Here", callback=self._on_run_step_fixed, user_data=1)
                            dpg.add_text("Gradient Magnitude")
                            dpg.add_image("tex_grad")

                        with dpg.tab(label="Step 3 - Final SVG", tag="tab_step_2"):
                            dpg.add_combo(
                                ["spirals", "circles", "squares", "lines", "curves", "hatching"],
                                default_value="spirals",
                                label="Algorithm",
                                tag="algorithm",
                            )
                            dpg.add_combo(
                                ["Black Lines on White", "White Lines on Black"],
                                default_value="Black Lines on White",
                                label="SVG Color Theme",
                                tag="color_theme",
                            )
                            dpg.add_slider_float(
                                tag="stroke_width",
                                label="Stroke Width (mm)",
                                min_value=0.01,
                                max_value=2.0,
                                default_value=0.5,
                            )

                            with dpg.tree_node(label="Safety Limits", default_open=False):
                                dpg.add_checkbox(
                                    tag="enable_vertex_guard",
                                    label="Enable Vertex Guard",
                                    default_value=True,
                                )
                                dpg.add_input_int(
                                    tag="max_estimated_vertices",
                                    label="Max Estimated Vertices",
                                    default_value=300000,
                                    min_value=1000,
                                )

                            with dpg.tree_node(label="Shared Randomness", default_open=False):
                                dpg.add_slider_float(
                                    tag="randomness_vertex",
                                    label="Vertex Randomness",
                                    min_value=0.0,
                                    max_value=1.0,
                                    default_value=0.0,
                                )
                                dpg.add_slider_float(
                                    tag="randomness_position",
                                    label="Position Randomness",
                                    min_value=0.0,
                                    max_value=1.0,
                                    default_value=0.0,
                                )

                            with dpg.tree_node(label="Spirals Settings", default_open=False):
                                dpg.add_slider_int(
                                    tag="theta_resolution",
                                    label="Theta Resolution",
                                    min_value=16,
                                    max_value=360,
                                    default_value=60,
                                )
                                dpg.add_slider_float(
                                    tag="spiral_b",
                                    label="Spiral Growth Rate",
                                    min_value=0.1,
                                    max_value=3.0,
                                    default_value=1.0,
                                )
                                dpg.add_checkbox(tag="connect_cells", label="Connect Cells", default_value=True)

                            with dpg.tree_node(label="Circles/Squares Settings", default_open=False):
                                dpg.add_slider_int(
                                    tag="circle_points",
                                    label="Circle Points",
                                    min_value=12,
                                    max_value=72,
                                    default_value=36,
                                )
                                dpg.add_checkbox(tag="small_first", label="Small First", default_value=True)

                            with dpg.tree_node(label="Lines Settings", default_open=False):
                                dpg.add_slider_float(
                                    tag="lines_segment_length",
                                    label="Segment Length",
                                    min_value=0.1,
                                    max_value=5.0,
                                    default_value=1.0,
                                )
                                dpg.add_slider_float(
                                    tag="randomness_length",
                                    label="Length Randomness",
                                    min_value=0.0,
                                    max_value=1.0,
                                    default_value=0.0,
                                )
                                dpg.add_slider_float(
                                    tag="min_gradient_scale",
                                    label="Min Gradient Scale",
                                    min_value=0.01,
                                    max_value=1.0,
                                    default_value=0.1,
                                )
                                dpg.add_slider_float(
                                    tag="max_gradient_scale",
                                    label="Max Gradient Scale",
                                    min_value=1.0,
                                    max_value=20.0,
                                    default_value=10.0,
                                )

                            with dpg.tree_node(label="Curves Settings", default_open=False):
                                dpg.add_slider_float(
                                    tag="curves_segment_length",
                                    label="Segment Length",
                                    min_value=0.1,
                                    max_value=5.0,
                                    default_value=1.0,
                                )
                                dpg.add_slider_float(
                                    tag="curves_randomness_length",
                                    label="Length Randomness",
                                    min_value=0.0,
                                    max_value=1.0,
                                    default_value=0.0,
                                )
                                dpg.add_slider_int(
                                    tag="max_steps",
                                    label="Max Steps",
                                    min_value=1,
                                    max_value=20,
                                    default_value=4,
                                )
                                dpg.add_slider_float(
                                    tag="step_size",
                                    label="Step Size",
                                    min_value=0.5,
                                    max_value=5.0,
                                    default_value=2.0,
                                )
                                dpg.add_slider_int(
                                    tag="bezier_samples",
                                    label="Bezier Samples",
                                    min_value=5,
                                    max_value=50,
                                    default_value=15,
                                )

                            with dpg.tree_node(label="Hatching Settings", default_open=False):
                                dpg.add_checkbox(tag="hatch_horizontal", label="Horizontal", default_value=False)
                                dpg.add_checkbox(tag="hatch_vertical", label="Vertical", default_value=False)
                                dpg.add_checkbox(
                                    tag="hatch_diag_right",
                                    label="Diagonal Right (\\)",
                                    default_value=True,
                                )
                                dpg.add_checkbox(
                                    tag="hatch_diag_left",
                                    label="Diagonal Left (/)",
                                    default_value=False,
                                )
                                dpg.add_slider_float(
                                    tag="min_spacing",
                                    label="Min Spacing",
                                    min_value=0.1,
                                    max_value=2.0,
                                    default_value=0.3,
                                )
                                dpg.add_slider_float(
                                    tag="max_spacing",
                                    label="Max Spacing",
                                    min_value=0.5,
                                    max_value=5.0,
                                    default_value=2.0,
                                )

                            dpg.add_button(label="Play Until Here", callback=self._on_run_step_fixed, user_data=2)
                            dpg.add_text("SVG Preview")
                            dpg.add_image("tex_svg")
                            dpg.add_input_text(tag="svg_code", multiline=True, readonly=True, width=960, height=280)

                            with dpg.group(horizontal=True):
                                dpg.add_input_text(
                                    tag="save_svg_path",
                                    default_value="scribbletrace_output.svg",
                                    width=560,
                                )
                                dpg.add_button(label="Save SVG", callback=self._on_save_svg)

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_open_image_dialog,
            tag="open_image_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".jpeg")
            dpg.add_file_extension(".bmp")
            dpg.add_file_extension(".tif")
            dpg.add_file_extension(".tiff")
            dpg.add_file_extension(".webp")

        dpg.create_viewport(title="ScribbleTrace - Dear PyGUI", width=1280, height=920)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self) -> None:
        """Launch the Dear PyGUI app."""
        self._build_ui()

        # Apply startup image and run stage 1 preview.
        if self.state.input_image is None:
            self.state.input_image = create_sample_image()
        self._set_texture("tex_input", self.state.input_image, PREVIEW_GRAY_SIZE)
        if self.state.image_path:
            dpg.set_value("input_path", self.state.image_path)

        self._run_to_stage(0)

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


def main() -> None:
    """Entrypoint for the Dear PyGUI desktop app."""
    _validate_graphics_environment()
    _require_dpg()
    app = DearPyGuiApp()
    app.run()


if __name__ == "__main__":
    main()
