# Examples

This directory contains example SVG outputs generated with different rendering modes.

## Example Outputs

All examples generated from `MagrittePipe.jpg` with width=30:

- **example_hatch.svg** - Hatch mode with cross-hatching (levels=8, angle=45°)
- **example_stipple.svg** - Stipple mode with variable dot density (levels=6)
- **example_spiral.svg** - Spiral mode with Archimedean spirals (levels=5)

## Viewing the Examples

These SVG files can be:
- Opened in any modern web browser
- Imported into Inkscape for plotting with AxiDraw
- Viewed in any SVG-compatible application

## Generating Your Own

Use the main script to generate similar outputs:

```bash
# Hatch mode
python ../scribbletrace.py input.jpg output.svg --mode hatch --width 30 --levels 8

# Stipple mode
python ../scribbletrace.py input.jpg output.svg --mode stipple --width 30 --levels 6

# Spiral mode
python ../scribbletrace.py input.jpg output.svg --mode spiral --width 30 --levels 5
```

See the main README for more options and detailed usage instructions.
