# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Im2Oil is a stroke-based rendering (SBR) system that converts images into oil paintings. It implements the ACM MM 2022 paper "Im2Oil: Stroke-Based Oil Painting Rendering with Linearly Controllable Fineness Via Adaptive Sampling". Rather than pixel-wise approximation, it treats oil painting as an adaptive sampling problem.

## Commands

### Setup and Dependencies
```bash
uv sync          # Install dependencies (uses uv package manager)
```

### Running
```bash
# Basic usage: render an image as oil painting
uv run python Oil-Painting.py --f "./input/A2.jpg" --p 4

# Key arguments:
#   --f   input image path
#   --b   brush template path (default: ./brush/brush-0.png)
#   --p   reciprocal of max sampling rate (4, 9, 16, 25, 36) — lower = finer
#   --s   random seed
#   --SSAA  super-sampling anti-aliasing factor (default: 8)
#   --freq  save a frame every N strokes (default: 100)
#   --order 0=size order (default), 1=random order
```

### Linting and Formatting
```bash
uv run ruff check .    # Lint
uv run black .         # Format
```

## Architecture

The pipeline runs in this order, orchestrated by `Oil-Painting.py`:

1. **Adaptive Sampling** (`voronoi_sampler.py`, `voronoi.py`, `voronoi_tools.py`): Computes a probability density map from image texture complexity, then uses Voronoi/K-Means iterations to place stroke anchor points. `K_Means_Sampler` is the main entry point.

2. **Edge Tangent Flow** (`ETF/edge_tangent_flow.py`): Computes edge-aligned flow fields using PyTorch convolutions. The `ETF` class produces angle maps that guide stroke orientation. Resizes internally to 512px for computation, then maps back to original resolution.

3. **Stroke Search** (`search_and_render.py`): `Search_Stroke` determines stroke parameters (position, size, angle, color) at each anchor point by analyzing local image properties in HSV space. `Render_Stroke` composites strokes onto the canvas.

4. **Stroke Simulation** (`simulate.py`, `simulate_RGB.py`): Generates procedural brush stroke textures with gaussian noise, parallel line patterns, attenuation, and distortion. `simulate_RGB.py` extends this to color (HSV) strokes. These are purely algorithmic — no neural rendering.

5. **Brush Masking** (`brush/`): Contains brush templates and utilities for brush texture application (`brush.py`, `mask.py`, `texture.py`, `Value.py`).

6. **Support modules**: `drawpatch.py` handles coordinate transforms for rotated strokes. `quicksort.py` sorts strokes by size. `Line_Tools.py` provides geometric utilities.

After the main render pass, `Oil-Painting.py` runs a gap-filling loop that detects unpainted regions via connected components and fills them with additional strokes.

## Key Details

- All image processing uses OpenCV (BGR format internally, HSV for color matching).
- The `--p` parameter controls fineness linearly: p=4 is finest, p=36 is coarsest.
- Output goes to `./output/{filename}-p-{p}/` with subdirectories for anchors, strokes, and process frames.
- No test suite exists in this project.
