"""Interactive GUI for ScribbleTrace parameter exploration.

This module provides a Gradio-based web interface for exploring
vectorization parameters visually and interactively.

Usage:
    scribbletrace-gui
    # or
    python -m scribbletrace.gui
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
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

# Default sample image (a simple gradient for testing)
DEFAULT_IMAGE = None


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


def process_image(
    input_image: np.ndarray | None,
    algorithm: str,
    output_width: float,
    levels: int,
    invert: bool,
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
    randomness_length: float,
    min_gradient_scale: float,
    max_gradient_scale: float,
    # Curves parameters
    max_steps: int,
    step_size: float,
    bezier_samples: int,
) -> tuple[str, str | None]:
    """Process image with selected algorithm and parameters.

    Returns:
        Tuple of (SVG content string, path to temp SVG file for download).
    """
    if input_image is None:
        input_image = create_sample_image()

    # Handle different input formats from Gradio
    input_image = np.asarray(input_image, dtype=np.float64)
    
    # Normalize to [0, 1] first
    if input_image.max() > 1.0:
        input_image = input_image / 255.0

    # Ensure grayscale (2D array)
    if input_image.ndim == 3:
        if input_image.shape[2] == 4:
            # RGBA - use luminosity method, ignore alpha
            input_image = (
                0.2989 * input_image[:, :, 0] +
                0.5870 * input_image[:, :, 1] +
                0.1140 * input_image[:, :, 2]
            )
        elif input_image.shape[2] == 3:
            # RGB - use luminosity method
            input_image = (
                0.2989 * input_image[:, :, 0] +
                0.5870 * input_image[:, :, 1] +
                0.1140 * input_image[:, :, 2]
            )
        else:
            # Unknown format, just take first channel
            input_image = input_image[:, :, 0]
    
    # Ensure it's 2D
    input_image = np.squeeze(input_image)
    if input_image.ndim != 2:
        raise ValueError(f"Could not convert image to 2D grayscale. Shape: {input_image.shape}")

    # Preprocess image
    processed = preprocess(
        input_image,
        output_width=output_width,
        levels=levels,
        invert=invert,
    )

    # Compute gradients for algorithms that need them
    gradients = compute_gradients(processed.original)

    # Create algorithm config based on selection
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

    # Generate paths
    paths = algo.process()

    # Create SVG
    svg_width = processed.width + 2
    svg_height = processed.height + 2
    writer = SVGWriter(paths, width=svg_width, height=svg_height)

    # Save to temp file for download
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w") as f:
        svg_content = writer.to_string()
        f.write(svg_content)
        temp_path = f.name

    return svg_content, temp_path


def create_gui() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(
        title="ScribbleTrace - Interactive Parameter Explorer",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # ScribbleTrace Parameter Explorer

            Upload an image and explore different vectorization algorithms and their parameters.
            The preview updates automatically as you adjust settings.

            **Tip:** Start with a simple image and low output width for faster preview.
            """
        )

        with gr.Row():
            # Left column: Input and parameters
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )

                algorithm = gr.Dropdown(
                    choices=["spirals", "circles", "squares", "lines", "curves", "hatching"],
                    value="spirals",
                    label="Algorithm",
                )

                with gr.Accordion("General Settings", open=True):
                    output_width = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=40,
                        step=1,
                        label="Output Width (cells)",
                        info="Number of cells across the image",
                    )
                    levels = gr.Slider(
                        minimum=2,
                        maximum=16,
                        value=7,
                        step=1,
                        label="Intensity Levels",
                        info="Number of quantization levels",
                    )
                    invert = gr.Checkbox(
                        value=True,
                        label="Invert",
                        info="Dark areas produce more marks",
                    )
                    stroke_width = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        label="Stroke Width (mm)",
                    )

                with gr.Accordion("Randomness", open=False):
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

                # Algorithm-specific parameters
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

                generate_btn = gr.Button("Generate Preview", variant="primary", size="lg")

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

            # Right column: Output preview
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("SVG Preview"):
                        svg_output = gr.HTML(
                            label="SVG Preview",
                            elem_id="svg-preview",
                        )
                    with gr.TabItem("SVG Code"):
                        svg_code = gr.Code(
                            label="SVG Source",
                            language="html",
                            lines=20,
                        )

                download_btn = gr.DownloadButton(
                    "Download SVG",
                    value=None,
                    variant="primary",
                    size="lg",
                    interactive=False,
                )

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

        # Process function
        def on_generate(
            img, algo, width, lvls, inv, stroke, rand_v, rand_p,
            theta_res, spiral_growth, connect,
            circ_pts, sm_first,
            hatch_h, hatch_v, hatch_dr, hatch_dl, min_sp, max_sp,
            rand_len, min_grad, max_grad,
            m_steps, s_size, bez_samp,
        ):
            svg_content, temp_path = process_image(
                img, algo, width, lvls, inv, stroke, rand_v, rand_p,
                theta_res, spiral_growth, connect,
                circ_pts, sm_first,
                hatch_h, hatch_v, hatch_dr, hatch_dl, min_sp, max_sp,
                rand_len, min_grad, max_grad,
                m_steps, s_size, bez_samp,
            )

            # Scale SVG to fill panel width while preserving aspect ratio
            import re
            svg_display = svg_content
            
            # Extract width/height values (they are in mm)
            width_match = re.search(r'width="([\d.]+)mm"', svg_content)
            height_match = re.search(r'height="([\d.]+)mm"', svg_content)
            
            if width_match and height_match:
                w = float(width_match.group(1))
                h = float(height_match.group(1))
                
                # Add viewBox attribute if not present, and set width to 100%
                if 'viewBox' not in svg_content:
                    svg_display = re.sub(
                        r'<svg([^>]*)width="[\d.]+mm"([^>]*)height="[\d.]+mm"',
                        f'<svg\\1width="100%"\\2viewBox="0 0 {w} {h}"',
                        svg_display
                    )
                else:
                    svg_display = re.sub(r'width="[\d.]+mm"', 'width="100%"', svg_display)
                
                # Remove height to let aspect ratio control it
                svg_display = re.sub(r'\s*height="[\d.]+mm"', '', svg_display)

            # Create HTML to display SVG inline with white background
            html_preview = f"""
            <div style="background: white; padding: 10px; border: 1px solid #ddd; border-radius: 4px;">
                {svg_display}
            </div>
            """

            return html_preview, svg_content, gr.update(value=temp_path, interactive=True)

        # Collect all inputs
        all_inputs = [
            input_image, algorithm, output_width, levels, invert, stroke_width,
            randomness_vertex, randomness_position,
            theta_resolution, spiral_b, connect_cells,
            circle_points, small_first,
            hatch_horizontal, hatch_vertical, hatch_diag_right, hatch_diag_left,
            min_spacing, max_spacing,
            randomness_length, min_gradient_scale, max_gradient_scale,
            max_steps, step_size, bezier_samples,
        ]

        setting_keys = [
            "algorithm",
            "output_width",
            "levels",
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
            "randomness_length",
            "min_gradient_scale",
            "max_gradient_scale",
            "max_steps",
            "step_size",
            "bezier_samples",
        ]

        setting_components = [
            algorithm,
            output_width,
            levels,
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
            randomness_length,
            min_gradient_scale,
            max_gradient_scale,
            max_steps,
            step_size,
            bezier_samples,
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

        generate_btn.click(
            fn=on_generate,
            inputs=all_inputs,
            outputs=[svg_output, svg_code, download_btn],
        )

        # Auto-generate on parameter changes (with debounce via queue)
        for component in all_inputs:
            if component != input_image:  # Don't auto-trigger on image upload
                component.change(
                    fn=on_generate,
                    inputs=all_inputs,
                    outputs=[svg_output, svg_code, download_btn],
                )

        # Initial generation when image is uploaded
        input_image.change(
            fn=on_generate,
            inputs=all_inputs,
            outputs=[svg_output, svg_code, download_btn],
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
    app.launch(
        share=False,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
