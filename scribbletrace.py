#!/usr/bin/env python3
"""
ScribbleTrace - Convert images to vector drawings for pen plotters

Generates SVG files with various artistic styles suitable for AxiDraw and other pen plotters.
Supports multiple rendering modes including hatching, scribble lines, curves, and spirals.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from skimage import io, filters
from skimage.transform import resize
import svgwrite
from svgwrite.path import Path as SvgPath


# Algorithm constants
HATCH_TRACE_MULTIPLIER = 3  # Multiplier for hatch line trace length
HATCH_STEP_SIZE = 0.1  # Step size for hatch line tracing

SPIRAL_MIN_POINTS = 20  # Minimum points in a spiral
SPIRAL_POINTS_PER_TURN = 10  # Points per spiral turn
SPIRAL_INITIAL_RADIUS = 0.01  # Initial radius 'a' in r = a + b*theta
SPIRAL_PIXEL_SCALE = 0.45  # Scale factor to fit spiral within pixel

STIPPLE_JITTER = 0.4  # Position randomization for stipple dots
STIPPLE_CIRCLE_SEGMENTS = 16  # Number of segments in each stipple circle

LINE_MIN_LENGTH = 0.3  # Minimum line length in scribble lines mode
LINE_MAX_LENGTH = 1.0  # Maximum line length in scribble lines mode
LINE_GRADIENT_SCALE = 5.0  # Scale factor for gradient magnitude


class ImageToVector:
    """Base class for converting images to vector paths"""
    
    def __init__(
        self,
        image_path: str,
        output_width: float = 40.0,
        quantization_levels: int = 5
    ):
        """
        Initialize the converter
        
        Args:
            image_path: Path to input image
            output_width: Output width in arbitrary units (default: 40)
            quantization_levels: Number of intensity levels (default: 5)
        """
        self.image_path = Path(image_path)
        self.output_width = output_width
        self.quantization_levels = quantization_levels
        
        # Load and preprocess image
        self.img_orig = io.imread(str(self.image_path), as_gray=True)
        self._preprocess_image()
    
    def _preprocess_image(self) -> None:
        """Resize and quantize the image"""
        scale_factor = round(self.img_orig.shape[0] / self.output_width)
        
        new_height = round(self.img_orig.shape[0] / scale_factor)
        new_width = round(self.img_orig.shape[1] / scale_factor)
        
        self.img = resize(
            self.img_orig,
            (new_height, new_width),
            anti_aliasing=True
        )
        
        # Invert and quantize
        self.img_quantized = 1 - self.img
        self.img_quantized = np.round(
            self.img_quantized * (self.quantization_levels - 1)
        ).astype(int)
        
        self.height, self.width = self.img_quantized.shape
    
    def _compute_gradients(self) -> None:
        """Compute image gradients using Sobel filters"""
        self.img_dy = filters.sobel_h(self.img)
        self.img_dx = filters.sobel_v(self.img)
        self.grad_mag = np.sqrt(self.img_dx**2 + self.img_dy**2)
    
    def generate_paths(self) -> list:
        """Generate vector paths - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_paths()")
    
    def save_svg(
        self,
        output_path: str,
        paths: list,
        stroke_width: float = 0.5,
        units: str = 'mm'
    ) -> None:
        """
        Save paths to SVG file
        
        Args:
            output_path: Output SVG file path
            paths: List of path coordinate arrays
            stroke_width: Line width (default: 0.5)
            units: SVG units (default: 'mm')
        """
        # Calculate dimensions in the specified units
        aspect_ratio = self.width / self.height
        svg_height = self.output_width
        svg_width = svg_height * aspect_ratio
        
        dwg = svgwrite.Drawing(
            output_path,
            size=(f'{svg_width}{units}', f'{svg_height}{units}'),
            viewBox=f'0 0 {self.width} {self.height}',
            profile='tiny'
        )
        
        # Create group for all paths
        g = dwg.g(id='scribble-layer', stroke='black', fill='none', stroke_width=stroke_width)
        
        for path_coords in paths:
            if len(path_coords) > 0:
                # Create SVG path
                path_data = self._coords_to_svg_path(path_coords)
                g.add(dwg.path(d=path_data))
        
        dwg.add(g)
        dwg.save()
        print(f"SVG saved to: {output_path}")
    
    def _coords_to_svg_path(self, coords: np.ndarray) -> str:
        """Convert coordinate array to SVG path data string"""
        if len(coords) == 0:
            return ""
        
        path_parts = [f"M {coords[0][0]:.3f},{coords[0][1]:.3f}"]
        
        for i in range(1, len(coords)):
            path_parts.append(f"L {coords[i][0]:.3f},{coords[i][1]:.3f}")
        
        return " ".join(path_parts)


class HatchRenderer(ImageToVector):
    """Render image using parallel hatch lines with variable density"""
    
    def __init__(
        self,
        image_path: str,
        output_width: float = 40.0,
        quantization_levels: int = 8,
        hatch_angle: float = 45.0,
        line_spacing: float = 0.5,
        cross_hatch: bool = True,
        variable_density: bool = True
    ):
        """
        Initialize hatch renderer
        
        Args:
            image_path: Path to input image
            output_width: Output width in units
            quantization_levels: Number of intensity levels
            hatch_angle: Angle of hatch lines in degrees (default: 45)
            line_spacing: Base spacing between hatch lines (default: 0.5)
            cross_hatch: Enable cross-hatching for darker areas (default: True)
            variable_density: Vary hatch density with intensity (default: True)
        """
        super().__init__(image_path, output_width, quantization_levels)
        self.hatch_angle = np.radians(hatch_angle)
        self.line_spacing = line_spacing
        self.cross_hatch = cross_hatch
        self.variable_density = variable_density
    
    def generate_paths(self) -> list:
        """Generate hatch line paths based on image intensity"""
        paths = []
        
        # Generate primary hatch lines
        paths.extend(self._generate_hatch_lines(self.hatch_angle))
        
        # Generate cross-hatch for darker areas
        if self.cross_hatch:
            paths.extend(self._generate_hatch_lines(
                self.hatch_angle + np.pi / 2,
                threshold=self.quantization_levels // 2
            ))
        
        return paths
    
    def _generate_hatch_lines(
        self,
        angle: float,
        threshold: int = 0
    ) -> list:
        """
        Generate parallel hatch lines at specified angle
        
        Args:
            angle: Angle in radians
            threshold: Only draw in areas darker than this threshold
        
        Returns:
            List of path coordinate arrays
        """
        paths = []
        
        # Direction vector for hatch lines
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Perpendicular direction for spacing
        perp_dx = -dy
        perp_dy = dx
        
        # Determine how many lines we need
        diagonal = np.sqrt(self.width**2 + self.height**2)
        num_lines = int(diagonal / self.line_spacing) + 1
        
        for i in range(num_lines):
            # Starting point for this hatch line
            offset = (i - num_lines // 2) * self.line_spacing
            start_x = self.width / 2 + perp_dx * offset
            start_y = self.height / 2 + perp_dy * offset
            
            # Trace line segments through dark areas
            path_segments = self._trace_hatch_line(
                start_x, start_y, dx, dy, threshold
            )
            paths.extend(path_segments)
        
        return paths
    
    def _trace_hatch_line(
        self,
        start_x: float,
        start_y: float,
        dx: float,
        dy: float,
        threshold: int
    ) -> list:
        """
        Trace a single hatch line through the image, breaking at light areas
        
        Returns:
            List of continuous path segments
        """
        segments = []
        current_segment = []
        
        # Trace along the line
        max_steps = int(np.sqrt(self.width**2 + self.height**2) * HATCH_TRACE_MULTIPLIER)
        step_size = HATCH_STEP_SIZE
        
        for step in range(-max_steps // 2, max_steps // 2):
            x = start_x + dx * step * step_size
            y = start_y + dy * step * step_size
            
            # Check if within bounds
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.width and 0 <= iy < self.height:
                intensity = self.img_quantized[iy, ix]
                
                if intensity > threshold:
                    # Dark area - add to current segment
                    current_segment.append([x, y])
                else:
                    # Light area - end current segment
                    if len(current_segment) > 2:
                        segments.append(np.array(current_segment))
                    current_segment = []
            else:
                # Out of bounds - end segment
                if len(current_segment) > 2:
                    segments.append(np.array(current_segment))
                current_segment = []
        
        # Add final segment
        if len(current_segment) > 2:
            segments.append(np.array(current_segment))
        
        return segments


class SpiralRenderer(ImageToVector):
    """Render image using Archimedean spirals"""
    
    def __init__(
        self,
        image_path: str,
        output_width: float = 40.0,
        quantization_levels: int = 7,
        spiral_tightness: float = 0.05
    ):
        """
        Initialize spiral renderer
        
        Args:
            image_path: Path to input image
            output_width: Output width in units
            quantization_levels: Number of intensity levels
            spiral_tightness: Tightness of spiral (default: 0.05)
        """
        super().__init__(image_path, output_width, quantization_levels)
        self.spiral_tightness = spiral_tightness
    
    def generate_paths(self) -> list:
        """Generate Archimedean spirals with turns based on intensity"""
        paths = []
        
        for r in range(self.height):
            for c in range(self.width):
                intensity = self.img_quantized[r, c]
                
                if intensity > 0:
                    spiral_path = self._create_spiral(
                        c, r, intensity, self.spiral_tightness
                    )
                    paths.append(spiral_path)
        
        return paths
    
    def _create_spiral(
        self,
        cx: float,
        cy: float,
        turns: int,
        tightness: float
    ) -> np.ndarray:
        """
        Create an Archimedean spiral path
        
        Args:
            cx, cy: Center position
            turns: Number of turns (based on intensity)
            tightness: How tight the spiral is
        """
        max_theta = turns * np.pi
        num_points = max(SPIRAL_MIN_POINTS, turns * SPIRAL_POINTS_PER_TURN)
        
        thetas = np.linspace(0, max_theta, num_points)
        
        # Archimedean spiral equation: r = a + b*theta
        spiral_initial_radius = SPIRAL_INITIAL_RADIUS
        spiral_growth_rate = tightness
        
        r = spiral_initial_radius + spiral_growth_rate * thetas
        
        # Convert to Cartesian
        x = cx + r * np.cos(thetas)
        y = cy + r * np.sin(thetas)
        
        # Normalize to fit within pixel
        max_r = np.max(r)
        if max_r > 0:
            scale = SPIRAL_PIXEL_SCALE / max_r
            x = cx + (x - cx) * scale
            y = cy + (y - cy) * scale
        
        return np.column_stack([x, y])


class StipplingRenderer(ImageToVector):
    """Render image using stippling (dots/small circles)"""
    
    def __init__(
        self,
        image_path: str,
        output_width: float = 40.0,
        quantization_levels: int = 8,
        dot_size: float = 0.3,
        density_factor: float = 2.0
    ):
        """
        Initialize stippling renderer
        
        Args:
            image_path: Path to input image
            output_width: Output width in units
            quantization_levels: Number of intensity levels
            dot_size: Size of each stipple dot (default: 0.3)
            density_factor: Controls dot density (default: 2.0)
        """
        super().__init__(image_path, output_width, quantization_levels)
        self.dot_size = dot_size
        self.density_factor = density_factor
    
    def generate_paths(self) -> list:
        """Generate stipple dots with density based on image intensity"""
        paths = []
        
        for r in range(self.height):
            for c in range(self.width):
                intensity = self.img_quantized[r, c]
                
                # Number of dots based on intensity
                num_dots = int(intensity * self.density_factor)
                
                for _ in range(num_dots):
                    # Random position within pixel
                    x = c + np.random.uniform(-STIPPLE_JITTER, STIPPLE_JITTER)
                    y = r + np.random.uniform(-STIPPLE_JITTER, STIPPLE_JITTER)
                    
                    # Create small circle path
                    circle_points = self._create_circle(x, y, self.dot_size)
                    paths.append(circle_points)
        
        return paths
    
    def _create_circle(self, cx: float, cy: float, radius: float) -> np.ndarray:
        """Create a circle path"""
        num_points = STIPPLE_CIRCLE_SEGMENTS
        angles = np.linspace(0, 2 * np.pi, num_points + 1)
        
        x = cx + radius * np.cos(angles)
        y = cy + radius * np.sin(angles)
        
        return np.column_stack([x, y])


class ScribbleLinesRenderer(ImageToVector):
    """Render image using short lines following gradients"""
    
    def __init__(
        self,
        image_path: str,
        output_width: float = 40.0,
        quantization_levels: int = 4,
        randomness: float = 0.1
    ):
        super().__init__(image_path, output_width, quantization_levels)
        self.randomness = randomness
        self._compute_gradients()
    
    def generate_paths(self) -> list:
        """Generate scribble lines perpendicular to gradients"""
        paths = []
        
        for r in range(self.height):
            for c in range(self.width):
                intensity = self.img_quantized[r, c]
                
                # Number of lines based on intensity
                for _ in range(intensity):
                    grad_x = self.img_dx[r, c]
                    grad_y = self.img_dy[r, c]
                    
                    # Line perpendicular to gradient
                    angle = np.arctan2(grad_y, grad_x) + np.pi / 2
                    
                    # Line length based on gradient magnitude
                    length = max(LINE_MIN_LENGTH, min(LINE_MAX_LENGTH, self.grad_mag[r, c] * LINE_GRADIENT_SCALE))
                    
                    # Add randomness
                    angle += np.random.uniform(-self.randomness, self.randomness)
                    pos_offset_x = np.random.uniform(-self.randomness, self.randomness)
                    pos_offset_y = np.random.uniform(-self.randomness, self.randomness)
                    
                    # Create line
                    x1 = c + pos_offset_x - length * np.cos(angle) / 2
                    y1 = r + pos_offset_y - length * np.sin(angle) / 2
                    x2 = c + pos_offset_x + length * np.cos(angle) / 2
                    y2 = r + pos_offset_y + length * np.sin(angle) / 2
                    
                    paths.append(np.array([[x1, y1], [x2, y2]]))
        
        return paths


class ScribbleCurvesRenderer(ImageToVector):
    """Render image using smooth curves following gradients"""
    
    def __init__(
        self,
        image_path: str,
        output_width: float = 40.0,
        quantization_levels: int = 5,
        max_curve_steps: int = 4,
        step_size: float = 1.5
    ):
        super().__init__(image_path, output_width, quantization_levels)
        self.max_curve_steps = max_curve_steps
        self.step_size = step_size
        self._compute_gradients()
    
    def generate_paths(self) -> list:
        """Generate smooth curves following gradients"""
        paths = []
        
        for r in range(self.height):
            for c in range(self.width):
                intensity = self.img_quantized[r, c]
                
                for _ in range(intensity):
                    curve = self._trace_gradient_curve(c, r)
                    if len(curve) > 2:
                        paths.append(curve)
        
        return paths
    
    def _trace_gradient_curve(self, start_c: int, start_r: int) -> np.ndarray:
        """Trace a smooth curve following gradients"""
        points = [[start_c, start_r]]
        
        # Trace in both directions
        for direction in [1, -1]:
            c, r = start_c, start_r
            
            for _ in range(self.max_curve_steps):
                # Get gradient at current position
                ic, ir = int(round(c)), int(round(r))
                ic = max(0, min(ic, self.width - 1))
                ir = max(0, min(ir, self.height - 1))
                
                grad_x = self.img_dx[ir, ic]
                grad_y = self.img_dy[ir, ic]
                
                # Move perpendicular to gradient
                angle = np.arctan2(grad_y, grad_x) + np.pi / 2
                
                c += direction * self.step_size * np.cos(angle)
                r += direction * self.step_size * np.sin(angle)
                
                # Check bounds
                if not (0 <= c < self.width and 0 <= r < self.height):
                    break
                
                if direction == 1:
                    points.append([c, r])
                else:
                    points.insert(0, [c, r])
        
        return np.array(points)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='ScribbleTrace - Convert images to vector drawings for pen plotters'
    )
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('output', type=str, help='Output SVG path')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['hatch', 'lines', 'curves', 'stipple', 'spiral'],
        default='hatch',
        help='Rendering mode (default: hatch)'
    )
    parser.add_argument(
        '--width',
        type=float,
        default=40.0,
        help='Output width in arbitrary units (default: 40)'
    )
    parser.add_argument(
        '--levels',
        type=int,
        default=8,
        help='Quantization levels (default: 8)'
    )
    parser.add_argument(
        '--hatch-angle',
        type=float,
        default=45.0,
        help='Hatch angle in degrees for hatch mode (default: 45)'
    )
    parser.add_argument(
        '--no-cross-hatch',
        action='store_true',
        help='Disable cross-hatching in hatch mode'
    )
    parser.add_argument(
        '--stroke-width',
        type=float,
        default=0.5,
        help='SVG stroke width (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing {args.input} in {args.mode} mode...")
    
    # Create renderer based on mode
    if args.mode == 'hatch':
        renderer = HatchRenderer(
            args.input,
            output_width=args.width,
            quantization_levels=args.levels,
            hatch_angle=args.hatch_angle,
            cross_hatch=not args.no_cross_hatch
        )
    elif args.mode == 'lines':
        renderer = ScribbleLinesRenderer(
            args.input,
            output_width=args.width,
            quantization_levels=args.levels
        )
    elif args.mode == 'curves':
        renderer = ScribbleCurvesRenderer(
            args.input,
            output_width=args.width,
            quantization_levels=args.levels
        )
    elif args.mode == 'stipple':
        renderer = StipplingRenderer(
            args.input,
            output_width=args.width,
            quantization_levels=args.levels
        )
    elif args.mode == 'spiral':
        renderer = SpiralRenderer(
            args.input,
            output_width=args.width,
            quantization_levels=args.levels
        )
    
    # Generate paths
    print("Generating vector paths...")
    paths = renderer.generate_paths()
    print(f"Generated {len(paths)} path segments")
    
    # Save SVG
    renderer.save_svg(args.output, paths, stroke_width=args.stroke_width)
    print("Done!")


if __name__ == '__main__':
    main()
