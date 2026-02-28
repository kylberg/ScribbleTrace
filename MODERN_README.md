# ScribbleTrace - Modern Python Version

A modern Python tool for converting images to vector drawings optimized for pen plotters like the AxiDraw.

## Features

- ✨ **Modern Python 3.12+** with type hints and clean architecture
- 🎨 **Multiple Rendering Modes**:
  - **Hatch**: Parallel and cross-hatch shading based on image intensity
  - **Lines**: Short lines following image gradients
  - **Curves**: Smooth flowing curves following gradients
- 📐 **Clean SVG Output** optimized for pen plotters
- 🤖 **AxiDraw Compatible** with proper units and formatting
- 🎯 **Single-Line Vector Paths** for efficient plotting

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Basic usage:

```bash
# Hatch rendering (default)
python scribbletrace.py input.jpg output.svg

# Specify rendering mode
python scribbletrace.py input.jpg output.svg --mode hatch
python scribbletrace.py input.jpg output.svg --mode lines
python scribbletrace.py input.jpg output.svg --mode curves
```

## Usage Examples

### Hatch Mode (Recommended for Photos)

Generate smooth hatch shading with cross-hatching for darker areas:

```bash
python scribbletrace.py photo.jpg output.svg \
  --mode hatch \
  --width 100 \
  --levels 10 \
  --hatch-angle 45
```

**Options:**
- `--hatch-angle`: Angle of primary hatch lines in degrees (default: 45)
- `--no-cross-hatch`: Disable cross-hatching
- `--levels`: More levels = more detail in shading (default: 8)

### Lines Mode (Good for Textures)

Generate short lines perpendicular to image gradients:

```bash
python scribbletrace.py texture.jpg output.svg \
  --mode lines \
  --width 80 \
  --levels 5
```

### Curves Mode (Artistic Flow)

Generate smooth flowing curves that follow image gradients:

```bash
python scribbletrace.py portrait.jpg output.svg \
  --mode curves \
  --width 100 \
  --levels 4
```

## Command-Line Options

```
positional arguments:
  input                 Input image path
  output                Output SVG path

options:
  --mode {hatch,lines,curves}
                        Rendering mode (default: hatch)
  --width WIDTH         Output width in arbitrary units (default: 40)
  --levels LEVELS       Quantization levels - higher = more detail (default: 8)
  --hatch-angle ANGLE   Hatch angle in degrees for hatch mode (default: 45)
  --no-cross-hatch      Disable cross-hatching in hatch mode
  --stroke-width WIDTH  SVG stroke width (default: 0.5)
```

## Understanding the Parameters

### Output Width (`--width`)
- Controls the resolution of the output
- Larger values = more detail but longer plotting time
- Recommended: 40-100 units
- Units are arbitrary and scale with your plotter setup

### Quantization Levels (`--levels`)
- Number of intensity levels in the output
- More levels = smoother gradients but more paths
- Hatch mode: 8-12 works well
- Lines/Curves: 4-6 works well

### Stroke Width (`--stroke-width`)
- Line thickness in the SVG
- Adjust based on your pen width
- Default: 0.5mm

## Tips for Best Results

1. **Choose the right mode**:
   - Photos/portraits → Hatch mode
   - Detailed textures → Lines mode
   - Artistic/abstract → Curves mode

2. **Image preparation**:
   - Use high-contrast images for better results
   - Pre-crop to focus on the subject
   - Adjust brightness/contrast in photo editor if needed

3. **Optimize for plotting time**:
   - Start with lower `--width` values (40-60)
   - Use fewer `--levels` for faster plotting
   - Test with small samples first

4. **For AxiDraw**:
   - SVG files are ready to import into Inkscape
   - Use "Plot" extension in Inkscape
   - Adjust units/scale as needed

## Technical Details

### SVG Output
- Clean, optimized SVG paths
- Proper viewBox and dimensions
- Single-layer output for easy plotting
- Profile: SVG Tiny 1.2 for maximum compatibility

### Algorithm Details

**Hatch Mode**: Generates parallel lines at specified angle, tracing through areas based on image intensity. Cross-hatching adds perpendicular lines in darker regions for richer shading.

**Lines Mode**: Analyzes image gradients using Sobel filters and draws short lines perpendicular to gradients, with density based on pixel intensity.

**Curves Mode**: Traces smooth curves that flow along image gradients, creating organic, flowing patterns.

## Original Code

The original prototype scripts are preserved in the `src/` directory for reference. This modernized version provides:
- Clean Python 3 code with type hints
- Proper CLI interface
- Direct SVG output (no matplotlib dependency for output)
- Optimized path generation
- Modern dependencies

## License

See [LICENSE](LICENSE) file.

## Credits

Original concept and prototypes by Gustaf Kylberg. Modernized version maintains the core algorithms while updating to modern Python practices and adding new features like hatch shading.

Inspired by:
- Sandy Noble's Polargraph (http://www.polargraph.co.uk/)
- Maxim Barabash's ZebraTrace (https://github.com/maxim-s-barabash/ZebraTrace/)
