"""NiceGUI-based web interface for ScribbleTrace.

This module provides a browser UI alternative to the Gradio app while
reusing the same processing pipeline and parameter model.

Usage:
    scribbletrace-nicegui
    # or
    python -m scribbletrace.gui_nicegui
"""

from __future__ import annotations

import base64
import io
import json
import re
from datetime import datetime
from xml.dom import minidom
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scribbletrace.gui import (
    PIPELINE_STEP_LABELS,
    apply_histogram_transform,
    get_default_gui_image,
    normalize_input_image,
    resize_preview_nearest,
    run_pipeline_to_stage,
    to_display_uint8,
)

ui = None


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


HIST_WIDTH = 480
HIST_HEIGHT = 150
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


def _normalize_hist_knots(min_v: float, mid_v: float, max_v: float) -> tuple[float, float, float]:
    min_v = float(np.clip(min_v, 0.0, 1.0))
    mid_v = float(np.clip(mid_v, 0.0, 1.0))
    max_v = float(np.clip(max_v, 0.0, 1.0))

    eps = 1e-4
    if min_v > max_v:
        min_v, max_v = max_v, min_v
    if max_v - min_v < 2 * eps:
        min_v = max(0.0, min_v - eps)
        max_v = min(1.0, max_v + eps)
    mid_v = min(max(mid_v, min_v + eps), max_v - eps)
    return min_v, mid_v, max_v


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


def _fit_svg_dimensions(svg_text: str, paper_size: str) -> str:
    dims = PAPER_SIZES_MM.get(paper_size)
    if dims is None:
        return svg_text

    match = re.search(r"<svg\b[^>]*>", svg_text)
    if not match:
        return svg_text

    tag = match.group(0)
    width_mm, height_mm = dims

    def _set_attr(svg_tag: str, name: str, value: str) -> str:
        pattern = rf"\b{name}\s*=\s*(\"[^\"]*\"|'[^']*')"
        if re.search(pattern, svg_tag):
            return re.sub(pattern, f'{name}="{value}"', svg_tag, count=1)
        return svg_tag[:-1] + f' {name}="{value}">'

    tag = _set_attr(tag, "width", f"{width_mm}mm")
    tag = _set_attr(tag, "height", f"{height_mm}mm")
    return svg_text[: match.start()] + tag + svg_text[match.end() :]


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

    state = NiceGuiState(input_image=get_default_gui_image(), input_image_name="default", using_default_input=True)

    controls: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    algorithm_tiles: dict[str, Any] = {}
    theme_tiles: dict[str, Any] = {}
    vector_groups: dict[str, Any] = {}

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
        zoom = int(_safe_value(controls.get("svg_zoom", _ValueProxy(100))))
        svg_code = state.latest_svg_content or ""
        if not svg_code.strip():
            outputs["svg_preview_html"].set_content(
                '<div class="st-card" style="padding:12px; color:#aab5c8;">No SVG generated yet.</div>'
            )
            return

        svg_url = _svg_text_to_data_url(svg_code)
        outputs["svg_preview_html"].set_content(
            "".join(
                [
                    '<div class="st-card" style="width:100%; height:520px; overflow:auto;">',
                    f'<img src="{svg_url}" style="display:block; width:{zoom}%; max-width:none; height:auto;"/>',
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
        classes: str = "w-full",
    ):
        precision = _slider_precision(step)
        with ui.column().classes(classes):
            slider = ui.slider(min=min_v, max=max_v, value=value, step=step)
            slider.props("label label-always")
            with ui.row().classes("w-full justify-between items-center text-xs st-muted"):
                ui.label(_format_slider_value(min_v, precision))
                ui.label(_format_slider_value(max_v, precision))
        return slider

    def _update_hist_overlay() -> None:
        active = str(_safe_value(controls["hist_target"]))
        outputs["hist_marker_info"].set_text(f"Active marker: {active.upper()}")

    def _sync_hist_controls(min_v: float, mid_v: float, max_v: float) -> None:
        min_v, mid_v, max_v = _normalize_hist_knots(min_v, mid_v, max_v)
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
        run_to_stage(1)

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
                run_to_stage(1)
            state.hist_drag_target = None
            return

        if event_type == "click":
            target = _target_from_hist_event(e)
            _set_hist_target_value(target, _tone_from_hist_event(e))
            run_to_stage(1)

    def _set_default_visibility() -> None:
        if "default_group" in outputs:
            outputs["default_group"].set_visibility(state.using_default_input)

    def update_input_preview() -> None:
        if state.input_image is None:
            return
        src = np.asarray(state.input_image)
        gray = normalize_input_image(state.input_image)
        outputs["current_original_preview"].set_source(_np_to_data_url(_downscale_for_ui(src)))
        outputs["current_grayscale_preview"].set_source(
            _np_to_data_url(_downscale_for_ui((gray * 255.0).astype(np.uint8)))
        )
        _set_default_visibility()

    def on_load_default() -> None:
        state.input_image = get_default_gui_image()
        state.input_image_name = "default"
        state.using_default_input = True
        outputs["input_name"].set_text("Input: default")
        state.pipeline_cache = {}
        update_input_preview()
        run_to_stage(1)

    def run_to_stage(stage_index: int) -> None:
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
        outputs["complexity"].set_text(complexity_text)
        outputs["histogram"].set_source(_np_to_data_url(_style_histogram_image(_downscale_for_ui(histogram))))
        _update_hist_overlay()
        outputs["preview_grayscale"].set_source(_np_to_data_url(_downscale_for_ui(gray_display)))
        outputs["preview_hist_downsampled"].set_source(_np_to_data_url(_downscale_for_ui(width_display)))
        outputs["preview_invert_levels"].set_source(_np_to_data_url(_downscale_for_ui(quantized)))
        outputs["preview_gradmag"].set_source(_np_to_data_url(_downscale_for_ui(gradient)))
        outputs["svg_source"].value = _format_svg_code(svg_code)
        _update_svg_preview_html()

        if svg_preview is None:
            state.latest_svg_content = ""
            _update_svg_preview_html()

        state.current_stage = max(0, min(stage_index, len(PIPELINE_STEP_LABELS) - 1))

    async def on_open_image(e) -> None:
        try:
            data = await e.file.read()
            state.input_image = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            state.input_image_name = e.file.name
            state.using_default_input = False
            outputs["input_name"].set_text(f"Input: {state.input_image_name}")
            state.pipeline_cache = {}
            update_input_preview()
            run_to_stage(1)
        except Exception as err:  # noqa: BLE001
            ui.notify(f"Failed to load image: {err}", color="negative")

    def on_run_one() -> None:
        run_to_stage(min(state.current_stage + 1, len(PIPELINE_STEP_LABELS) - 1))

    def on_run_selected() -> None:
        step_label = str(_safe_value(controls.get("target_step", _ValueProxy(UI_STEP_LABELS[0]))))
        target_stage = 1 if step_label == UI_STEP_LABELS[0] else len(PIPELINE_STEP_LABELS) - 1
        run_to_stage(target_stage)

    def on_run_all() -> None:
        run_to_stage(len(PIPELINE_STEP_LABELS) - 1)

    def download_svg() -> None:
        if not state.latest_svg_content:
            ui.notify("No SVG generated yet. Run vector generation first.", color="warning")
            return
        try:
            paper_size = str(_safe_value(controls.get("paper_size", _ValueProxy("A4"))))
            svg_to_save = _fit_svg_dimensions(state.latest_svg_content, paper_size)
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

    def save_preset() -> None:
        path = str(_safe_value(controls["preset_path"]) or "").strip()
        if not path:
            ui.notify("Set a preset path first.", color="warning")
            return
        payload = {"version": 1, "settings": _build_settings_snapshot(controls)}
        try:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            ui.notify(f"Saved preset: {out}", color="positive")
        except Exception as err:  # noqa: BLE001
            ui.notify(f"Failed to save preset: {err}", color="negative")

    def load_preset() -> None:
        path = str(_safe_value(controls["preset_path"]) or "").strip()
        if not path:
            ui.notify("Set a preset path first.", color="warning")
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            settings = payload.get("settings", payload)
            if not isinstance(settings, dict):
                raise ValueError("Preset must contain a JSON object")

            for key, value in settings.items():
                if key in controls:
                    controls[key].set_value(value)

            state.pipeline_cache = {}
            run_to_stage(state.current_stage)
            ui.notify(f"Loaded preset: {path}", color="positive")
        except Exception as err:  # noqa: BLE001
            ui.notify(f"Failed to load preset: {err}", color="negative")

    ui.add_head_html(
        """
    <style>
            body { background: radial-gradient(circle at 20% 0%, #1f2430 0%, #11151c 60%, #0b0e13 100%); }
            .st-card { border: 1px solid #39414f; border-radius: 10px; background: #141a24; }
      .st-title { font-weight: 700; font-size: 1.2rem; }
            .st-muted { color: #aab5c8; }
                        .st-code textarea {
                            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace !important;
                            font-size: 12px;
                            line-height: 1.4;
                            white-space: pre;
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
    </style>
    """
    )

    ui.dark_mode().enable()

    ui.label("ScribbleTrace NiceGUI").classes("st-title")

    with ui.row().classes("w-full items-start no-wrap gap-4"):
        with ui.tabs().props("vertical").classes("w-[180px] st-card p-2") as tabs:
            tab_input = ui.tab("Input").props("icon=upload")
            tab_processing = ui.tab("Proc").props("icon=tune")
            tab_vector = ui.tab("Vector").props("icon=gesture")

        with ui.tab_panels(tabs, value=tab_processing).props("vertical animated transition-prev=fade transition-next=fade").classes("flex-1 min-w-0"):
            with ui.tab_panel(tab_input):
                with ui.column().classes("w-full max-w-[720px] gap-3"):
                    outputs["input_name"] = ui.label("Input: default")
                    outputs["status"] = ui.label("Pipeline at 1. Image Processing").classes("st-muted")
                    outputs["complexity"] = ui.label("Estimated vertices: -").classes("st-muted")
                    ui.upload(on_upload=on_open_image, auto_upload=True, label="Upload Image").classes("w-full")
                    ui.button("Load Default", on_click=on_load_default)

                    with ui.column().classes("w-full gap-2"):
                        ui.label("Current Input").classes("st-title")
                        with ui.row().classes("w-full gap-4 flex-wrap"):
                            with ui.column().classes("gap-1"):
                                ui.label("Original").classes("st-muted")
                                outputs["current_original_preview"] = ui.image().classes("w-[320px] max-w-full st-card")
                            with ui.column().classes("gap-1"):
                                ui.label("Grayscale").classes("st-muted")
                                outputs["current_grayscale_preview"] = ui.image().classes("w-[320px] max-w-full st-card")

                    outputs["default_group"] = ui.column().classes("w-full gap-2")
                    with outputs["default_group"]:
                        ui.label("Default Reference").classes("st-title")
                        default_img = get_default_gui_image()
                        default_gray = normalize_input_image(default_img)
                        with ui.row().classes("w-full gap-4 flex-wrap"):
                            with ui.column().classes("gap-1"):
                                ui.label("Default Original").classes("st-muted")
                                ui.image(_np_to_data_url(_downscale_for_ui(default_img))).classes("w-[320px] max-w-full st-card")
                            with ui.column().classes("gap-1"):
                                ui.label("Default Grayscale").classes("st-muted")
                                ui.image(_np_to_data_url(_downscale_for_ui((default_gray * 255.0).astype(np.uint8)))).classes(
                                    "w-[320px] max-w-full st-card"
                                )

                    controls["preset_path"] = ui.input("Preset Path", value="scribbletrace_preset.json").classes("w-full")
                    with ui.row().classes("w-full gap-2"):
                        ui.button("Save Preset", on_click=save_preset)
                        ui.button("Load Preset", on_click=load_preset)

                    with ui.row().classes("w-full gap-2"):
                        ui.button("Run Image Processing", on_click=lambda: run_to_stage(1))
                        ui.button("Run Full Pipeline", on_click=lambda: run_to_stage(2))

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
                            controls["hist_target"].on_value_change(lambda _: run_to_stage(1))
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
                            ui.label("Output Width").classes("st-muted")
                            controls["output_width"] = _slider_with_readout(min_v=10, max_v=200, value=40, step=1)
                            controls["output_width"].on_value_change(lambda _: run_to_stage(1))
                            ui.label("Levels").classes("st-muted")
                            controls["levels"] = _slider_with_readout(min_v=2, max_v=16, value=7, step=1)
                            controls["levels"].on_value_change(lambda _: run_to_stage(1))
                            controls["invert"] = ui.switch("Invert", value=True)
                            controls["invert"].on_value_change(lambda _: run_to_stage(1))
                            ui.label("Gaussian Gradient Magnitude Sigma").classes("st-muted")
                            controls["gradient_sigma"] = _slider_with_readout(
                                min_v=0.0, max_v=6.0, value=1.0, step=0.1
                            )
                            controls["gradient_sigma"].on_value_change(lambda _: run_to_stage(1))

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

                    with ui.row().classes("w-full gap-4"):
                        controls["color_theme"] = _ValueProxy(
                            "Black Lines on White", on_change=lambda _: _update_theme_tile_styles()
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

                    ui.label("Line Width (mm)").classes("st-muted")
                    controls["stroke_width"] = _slider_with_readout(min_v=0.01, max_v=2.0, value=0.5, step=0.01)

                    controls["paper_size"] = ui.select(
                        ["A3", "A4", "A5", "A6"],
                        value="A4",
                        label="Fit SVG drawing in",
                    ).classes("w-[240px]")

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
                                min_v=0.0, max_v=1.0, value=0.0, step=0.01
                            )
                            controls["randomness_position"] = _slider_with_readout(
                                min_v=0.0, max_v=1.0, value=0.0, step=0.01
                            )

                        vector_groups["spirals"] = ui.column().classes("st-card p-3 gap-2 min-w-[320px] max-w-[420px]")
                        with vector_groups["spirals"]:
                            ui.label("Spirals Settings").classes("st-title")
                            controls["theta_resolution"] = _slider_with_readout(min_v=16, max_v=360, value=60, step=1)
                            controls["spiral_b"] = _slider_with_readout(min_v=0.1, max_v=3.0, value=1.0, step=0.1)
                            controls["connect_cells"] = ui.switch("Connect Cells", value=True)

                        vector_groups["circles_squares"] = ui.column().classes(
                            "st-card p-3 gap-2 min-w-[320px] max-w-[420px]"
                        )
                        with vector_groups["circles_squares"]:
                            ui.label("Circles/Squares Settings").classes("st-title")
                            controls["circle_points"] = _slider_with_readout(min_v=12, max_v=72, value=36, step=1)
                            controls["small_first"] = ui.switch("Small First", value=True)

                        vector_groups["lines"] = ui.column().classes("st-card p-3 gap-2 min-w-[320px] max-w-[420px]")
                        with vector_groups["lines"]:
                            ui.label("Lines Settings").classes("st-title")
                            controls["lines_segment_length"] = _slider_with_readout(
                                min_v=0.1, max_v=5.0, value=1.0, step=0.1
                            )
                            controls["randomness_length"] = _slider_with_readout(
                                min_v=0.0, max_v=1.0, value=0.0, step=0.01
                            )
                            controls["min_gradient_scale"] = _slider_with_readout(
                                min_v=0.01, max_v=1.0, value=0.1, step=0.01
                            )
                            controls["max_gradient_scale"] = _slider_with_readout(
                                min_v=1.0, max_v=20.0, value=10.0, step=0.5
                            )

                        vector_groups["curves"] = ui.column().classes("st-card p-3 gap-2 min-w-[320px] max-w-[420px]")
                        with vector_groups["curves"]:
                            ui.label("Curves Settings").classes("st-title")
                            controls["curves_segment_length"] = _slider_with_readout(
                                min_v=0.1, max_v=5.0, value=1.0, step=0.1
                            )
                            controls["curves_randomness_length"] = _slider_with_readout(
                                min_v=0.0, max_v=1.0, value=0.0, step=0.01
                            )
                            ui.label("Max Steps").classes("st-muted")
                            controls["max_steps"] = _slider_with_readout(min_v=1, max_v=20, value=4, step=1)
                            controls["step_size"] = _slider_with_readout(min_v=0.5, max_v=5.0, value=2.0, step=0.1)
                            controls["bezier_samples"] = _slider_with_readout(min_v=5, max_v=50, value=15, step=1)

                        vector_groups["hatching"] = ui.column().classes("st-card p-3 gap-2 min-w-[320px] max-w-[420px]")
                        with vector_groups["hatching"]:
                            ui.label("Hatching Settings").classes("st-title")
                            controls["hatch_horizontal"] = ui.switch("Horizontal", value=False)
                            controls["hatch_vertical"] = ui.switch("Vertical", value=False)
                            controls["hatch_diag_right"] = ui.switch("Diagonal Right", value=True)
                            controls["hatch_diag_left"] = ui.switch("Diagonal Left", value=False)
                            controls["min_spacing"] = _slider_with_readout(min_v=0.1, max_v=2.0, value=0.3, step=0.1)
                            controls["max_spacing"] = _slider_with_readout(min_v=0.5, max_v=5.0, value=2.0, step=0.1)

                    _update_vector_setting_visibility()

                    ui.button("Generate Vectors", on_click=lambda: run_to_stage(2))
                    ui.label("SVG Preview").classes("st-muted")
                    controls["svg_zoom"] = _slider_with_readout(min_v=50, max_v=300, value=100, step=5)
                    controls["svg_zoom"].on_value_change(lambda _: _update_svg_preview_html())
                    outputs["svg_preview_html"] = ui.html().classes("w-full")

                    with ui.row().classes("w-full gap-2"):
                        ui.button("Download SVG", icon="download", on_click=download_svg).props(
                            "color=positive unelevated"
                        ).classes("grow text-white")
                        ui.button("Download Settings", icon="tune", on_click=download_settings).props(
                            "color=secondary unelevated"
                        ).classes("grow")
                    outputs["svg_source"] = ui.textarea(label="SVG Source").props("rows=18 readonly").classes(
                        "w-full st-code"
                    )

    # Initial defaults and first run.
    update_input_preview()
    run_to_stage(1)


def main() -> None:
    """Launch the NiceGUI app."""
    _require_nicegui()
    create_gui()
    ui.run(title="ScribbleTrace NiceGUI", reload=False)


if __name__ == "__main__":
    main()
