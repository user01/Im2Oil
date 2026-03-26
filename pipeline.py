"""Core pipeline logic for the Im2Oil subcommand workflow.

Provides palette extraction, brush width computation, stroke quantization,
batch sorting, and the analyze/paint/view/run command implementations.
"""

import json
import math
import os
import random
import time

import cv2
import numpy as np

from voronoi_sampler import K_Means_Sampler
from ETF.edge_tangent_flow import ETF
from simulate_RGB import Gassian_HSV
from export_viewer import build_html
from strokes import (
    generate_detail_strokes,
    generate_block_in_splines,
    render_all_strokes,
)


# ---------------------------------------------------------------------------
# Palette extraction
# ---------------------------------------------------------------------------

_HUE_NAMES = [
    (10, "red"),
    (25, "orange"),
    (34, "yellow"),
    (77, "green"),
    (99, "cyan"),
    (130, "blue"),
    (145, "purple"),
    (169, "magenta"),
    (180, "red"),
]


def _hue_name(h, s, v):
    """Human-readable color name from OpenCV HSV values."""
    if s < 30:
        if v > 200:
            return "white"
        if v < 50:
            return "black"
        return "gray"
    for threshold, name in _HUE_NAMES:
        if h <= threshold:
            if v < 80:
                return "dark " + name
            return name
    return "red"


def _palette_distance(c1, c2):
    """Perceptual distance between two HSV colors.

    Weights hue heavily (scaled to 0-255 range, circular) so that
    colors with different hues are kept separate even if S/V are similar.
    """
    h1, s1, v1 = float(c1[0]), float(c1[1]), float(c1[2])
    h2, s2, v2 = float(c2[0]), float(c2[1]), float(c2[2])

    # Circular hue distance, scaled: OpenCV H is 0-179, scale to 0-255 range
    h_diff = abs(h1 - h2)
    h_diff = min(h_diff, 180 - h_diff)
    h_dist = h_diff * (255 / 90)  # scale so 90° hue diff = 255

    # For low-saturation colors (grays), hue doesn't matter
    avg_s = (s1 + s2) / 2
    hue_weight = min(1.0, avg_s / 80)  # fade hue importance below S=80

    s_dist = abs(s1 - s2)
    v_dist = abs(v1 - v2)

    return np.sqrt((h_dist * hue_weight) ** 2 + s_dist ** 2 + v_dist ** 2)


def extract_palette(image_path, n_colors):
    """Extract n_colors perceptually distinct colors from the image.

    Strategy: over-sample with k-means (3x requested), then iteratively merge
    the two most similar colors (weighted by pixel count) until we reach n_colors.
    This deduplicates near-identical shades while preserving rare but distinct hues.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # Over-sample: extract 3x the requested colors
    n_oversample = min(n_colors * 3, 32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, n_oversample, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    centers = np.clip(np.round(centers), 0, 255).astype(int)

    # Count pixels per cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique.flatten(), counts.flatten()))

    # Build initial cluster list: (center_hsv, pixel_count)
    clusters = []
    for i in range(len(centers)):
        clusters.append({
            "hsv": centers[i].tolist(),
            "count": int(cluster_counts.get(i, 0)),
        })

    # Iteratively merge closest pair until we reach n_colors
    while len(clusters) > n_colors:
        min_dist = float("inf")
        merge_i, merge_j = 0, 1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = _palette_distance(clusters[i]["hsv"], clusters[j]["hsv"])
                if d < min_dist:
                    min_dist = d
                    merge_i, merge_j = i, j

        # Merge: weighted average by pixel count
        ci, cj = clusters[merge_i], clusters[merge_j]
        total = ci["count"] + cj["count"]
        if total == 0:
            total = 1

        # Handle hue wrapping for averaging
        h1, h2 = float(ci["hsv"][0]), float(cj["hsv"][0])
        if abs(h1 - h2) > 90:
            if h1 < h2:
                h1 += 180
            else:
                h2 += 180
        merged_h = (h1 * ci["count"] + h2 * cj["count"]) / total % 180
        merged_s = (ci["hsv"][1] * ci["count"] + cj["hsv"][1] * cj["count"]) / total
        merged_v = (ci["hsv"][2] * ci["count"] + cj["hsv"][2] * cj["count"]) / total

        clusters[merge_i] = {
            "hsv": [int(round(merged_h)), int(round(merged_s)), int(round(merged_v))],
            "count": total,
        }
        clusters.pop(merge_j)

    # Sort by brightness descending (light to dark)
    clusters.sort(key=lambda c: -c["hsv"][2])

    palette = []
    for c in clusters:
        h, s, v = c["hsv"]
        palette.append({"hsv": [h, s, v], "name": _hue_name(h, s, v)})
    return palette


def save_palette_swatch(palette, path):
    """Save a visual swatch image of the palette colors."""
    swatch_h, swatch_w = 60, 80
    n = len(palette)
    img = np.zeros((swatch_h, swatch_w * n, 3), dtype=np.uint8)
    for i, c in enumerate(palette):
        color_hsv = np.array([[c["hsv"]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        x0 = i * swatch_w
        img[:, x0:x0 + swatch_w] = color_bgr
        # Add label
        label = f"{i}:{c['name']}"
        text_color = (255, 255, 255) if c["hsv"][2] < 128 else (0, 0, 0)
        cv2.putText(img, label, (x0 + 2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
        cv2.putText(img, f"H{c['hsv'][0]}", (x0 + 2, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
        cv2.putText(img, f"S{c['hsv'][1]} V{c['hsv'][2]}", (x0 + 2, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
    cv2.imwrite(path, img)
    print(f"  Palette swatch: {path}")


# ---------------------------------------------------------------------------
# Brush width computation
# ---------------------------------------------------------------------------


def compute_brush_widths(p_max, num_brushes):
    """Compute evenly-spaced discrete brush widths for the given p_max."""
    p_max_rate = 1.0 / p_max
    p_min_rate = p_max_rate / 100
    min_width = np.sqrt(1 / p_max_rate) - 1
    max_width = np.sqrt(1 / p_min_rate)
    if num_brushes == 1:
        return [round(float((min_width + max_width) / 2), 1)]
    widths = np.linspace(min_width, max_width, num_brushes)
    return [round(float(w), 1) for w in widths]


# ---------------------------------------------------------------------------
# Stroke quantization and batching
# ---------------------------------------------------------------------------


def quantize_strokes(patch_sequence, palette, brush_widths):
    """Quantize stroke colors to palette and widths to allowed brush sizes.

    Mutates patch_sequence in-place. Adds 'palette_index' and 'brush_width' fields.
    """
    palette_hsv = np.array([p["hsv"] for p in palette], dtype=np.float32)
    brush_arr = np.array(sorted(brush_widths), dtype=np.float32)

    for stroke in patch_sequence:
        # --- Color quantization with circular hue distance ---
        s_hsv = np.array(stroke["hsv"], dtype=np.float32)
        h_diff = np.abs(palette_hsv[:, 0] - s_hsv[0])
        h_diff = np.minimum(h_diff, 180 - h_diff)
        s_diff = palette_hsv[:, 1] - s_hsv[1]
        v_diff = palette_hsv[:, 2] - s_hsv[2]
        distances = np.sqrt(h_diff**2 + s_diff**2 + v_diff**2)

        nearest_idx = int(np.argmin(distances))
        stroke["palette_index"] = nearest_idx
        stroke["hsv"] = np.array(palette[nearest_idx]["hsv"], dtype=np.uint8)

        # --- Width quantization ---
        effective_width = stroke["w1"] + stroke["w2"] + 1
        nearest_brush_idx = int(np.argmin(np.abs(brush_arr - effective_width)))
        target_width = float(brush_arr[nearest_brush_idx])
        stroke["brush_width"] = target_width

        old_sum = stroke["w1"] + stroke["w2"]
        new_sum = target_width - 1
        if old_sum > 0:
            ratio = new_sum / old_sum
            stroke["w1"] = stroke["w1"] * ratio
            stroke["w2"] = stroke["w2"] * ratio
        else:
            stroke["w1"] = new_sum / 2
            stroke["w2"] = new_sum / 2

    return patch_sequence


def _nearest_neighbor_order(strokes):
    """Reorder strokes by nearest-neighbor heuristic to minimize brush travel.

    Greedy: start from the first stroke, always jump to the closest unvisited one.
    Uses Euclidean distance between stroke center coordinates.
    """
    if len(strokes) <= 2:
        return strokes

    coords = np.array([s["coordinate"] for s in strokes], dtype=np.float32)
    n = len(strokes)
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True

    for _ in range(n - 1):
        cur = coords[order[-1]]
        dists = np.sum((coords - cur) ** 2, axis=1)
        dists[visited] = np.inf
        nearest = int(np.argmin(dists))
        order.append(nearest)
        visited[nearest] = True

    return [strokes[i] for i in order]


def batch_sort_strokes(patch_sequence):
    """Sort strokes for physical painting order.

    1. Group by brush_width (largest first) then palette_index (color batches).
    2. Within each (width, color) group, apply nearest-neighbor ordering
       to minimize brush travel distance.
    """
    from itertools import groupby

    # First, sort into groups
    patch_sequence.sort(
        key=lambda s: (-s["brush_width"], s["palette_index"], -s["importance"])
    )

    # Then reorder within each (width, color) group by spatial proximity
    result = []
    for _key, group in groupby(
        patch_sequence, key=lambda s: (s["brush_width"], s["palette_index"])
    ):
        result.extend(_nearest_neighbor_order(list(group)))

    patch_sequence[:] = result
    return patch_sequence


# ---------------------------------------------------------------------------
# Multi-pass helpers
# ---------------------------------------------------------------------------


def classify_boundary_strokes(strokes, brush_width):
    """Separate strokes into interior and boundary groups.

    A stroke is a boundary stroke if any nearby stroke (within 2*brush_width)
    has a different palette color. Uses a spatial grid for efficiency.

    Returns:
        (interior_strokes, boundary_dict)
        boundary_dict maps (palette_idx_a, palette_idx_b) -> [strokes]
    """
    from collections import defaultdict

    cell_size = max(1, int(brush_width * 2))
    grid = defaultdict(set)

    for stroke in strokes:
        y, x = stroke["coordinate"]
        gy, gx = int(y / cell_size), int(x / cell_size)
        grid[(gy, gx)].add(stroke["palette_index"])

    interior = []
    boundary = defaultdict(list)

    for stroke in strokes:
        y, x = stroke["coordinate"]
        gy, gx = int(y / cell_size), int(x / cell_size)
        my_color = stroke["palette_index"]

        neighbor_colors = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbor_colors |= grid.get((gy + dy, gx + dx), set())
        neighbor_colors.discard(my_color)

        if neighbor_colors:
            other = min(neighbor_colors)
            pair = (min(my_color, other), max(my_color, other))
            boundary[pair].append(stroke)
        else:
            interior.append(stroke)

    return interior, dict(boundary)


def save_debug_image(path, image, is_hsv=False):
    """Save a debug image, converting from HSV if needed."""
    if is_hsv:
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(path, image)
    print(f"  DEBUG: {path}")


def quantize_pixel_map(input_hsv, palette):
    """Quantize every pixel to its nearest palette color.

    Returns:
        palette_map: (H, W) array of palette indices
        color_image: (H, W, 3) BGR image showing the quantized colors
    """
    palette_hsv = np.array([p["hsv"] for p in palette], dtype=np.float32)
    (H, W, _) = input_hsv.shape
    pixels = input_hsv.reshape(-1, 3).astype(np.float32)

    # Circular hue distance + S/V
    h_diff = np.abs(pixels[:, 0:1] - palette_hsv[:, 0])  # (N, K)
    h_diff = np.minimum(h_diff, 180 - h_diff)
    s_diff = pixels[:, 1:2] - palette_hsv[:, 1]
    v_diff = pixels[:, 2:3] - palette_hsv[:, 2]
    distances = np.sqrt(h_diff**2 + s_diff**2 + v_diff**2)

    palette_map = np.argmin(distances, axis=1).reshape(H, W)

    # Build color visualization
    color_lut = np.array([p["hsv"] for p in palette], dtype=np.uint8)
    color_image_hsv = color_lut[palette_map]
    color_image = cv2.cvtColor(color_image_hsv, cv2.COLOR_HSV2BGR)

    return palette_map, color_image


def generate_block_in_strokes(input_hsv, palette, palette_map, brush_width, ratio,
                               padding, debug_dir=None):
    """Generate horizontal wash strokes for the block-in (pass 0).

    For each palette color, computes the overall bounding box across ALL pixels
    of that color (not per-component). This ensures the sky wash covers the
    full width even when fragmented by foreground objects like tree leaves.

    Colors are sorted by total pixel area (largest first = background first).
    Each color's wash fills its full bounding box. Backgrounds get painted first,
    then foreground elements paint on top.
    """
    (H, W, _) = input_hsv.shape
    half_w = max(0.5, (brush_width - 1) / 2)
    stride = max(1, int(brush_width * 0.7))

    # Compute overall bounding box and total area per palette color
    color_regions = []
    for pidx in range(len(palette)):
        color_mask = (palette_map == pidx).astype(np.uint8)
        total_area = int(color_mask.sum())

        if debug_dir:
            save_debug_image(
                os.path.join(debug_dir, f"region_{pidx}_{palette[pidx]['name']}.png"),
                color_mask * 255,
            )

        if total_area < brush_width * brush_width:
            continue

        # Overall bounding box of all pixels of this color
        ys, xs = np.where(color_mask > 0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        color_regions.append((total_area, pidx, (x0, y0, x1 - x0, y1 - y0)))

    # Sort by area descending — largest (background) colors first
    color_regions.sort(key=lambda r: -r[0])

    if debug_dir:
        # Save region ordering for inspection
        order_lines = []
        for area, pidx, (x0, y0, rw, rh) in color_regions:
            name = palette[pidx]["name"]
            order_lines.append(f"  {pidx}: {name} — {area}px, bbox=({x0},{y0},{rw}x{rh})")
        print("  Block-in region order (bg→fg):")
        for line in order_lines:
            print(line)

    strokes = []
    for area, pidx, (x0, y0, rw, rh) in color_regions:
        hsv_color = np.array(palette[pidx]["hsv"], dtype=np.uint8)

        # Generate horizontal strokes tiling the full bounding box
        for row_y in range(y0, y0 + rh, stride):
            stroke_length = rw
            x_center = x0 + rw / 2.0

            if stroke_length < 2:
                continue

            strokes.append({
                "coordinate": [float(row_y), float(x_center)],
                "w1": half_w,
                "w2": half_w,
                "l1": stroke_length / 2.0,
                "l2": stroke_length / 2.0,
                "angle_ETF": 0.0,
                "angle_hatch": 90.0,
                "hsv": hsv_color,
                "grayscale": int(hsv_color[2]),
                "gradient": 0.0,
                "density": 0.01,
                "importance": float(stroke_length * brush_width),
                "palette_index": pidx,
                "brush_width": float(brush_width),
                "type": "block_in",
                "pass": 0,
            })

    # Debug: draw stroke plan
    if debug_dir:
        plan_img = cv2.imread(
            os.path.join(os.path.dirname(debug_dir), "input_bgr.png"),
            cv2.IMREAD_COLOR,
        )
        if plan_img is not None:
            # Crop to match padded size if needed
            plan_vis = plan_img.copy()
            for s in strokes:
                sy, sx = int(s["coordinate"][0]) - padding, int(s["coordinate"][1]) - padding
                sl = int(s["l1"] + s["l2"])
                sw = int(s["w1"] + s["w2"])
                color_bgr = cv2.cvtColor(
                    np.array([[s["hsv"]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
                )[0, 0].tolist()
                x1 = max(0, int(sx - sl / 2))
                x2 = min(plan_vis.shape[1], int(sx + sl / 2))
                y1 = max(0, int(sy - sw / 2))
                y2 = min(plan_vis.shape[0], int(sy + sw / 2))
                cv2.rectangle(plan_vis, (x1, y1), (x2, y2), color_bgr, 1)
            save_debug_image(os.path.join(debug_dir, "block_in_plan.png"), plan_vis)

    print(f"  Block-in: {len(strokes)} strokes across {len(color_regions)} regions")
    return strokes


def compute_residual_error(canvas_hsv, source_hsv, kernel_size):
    """Compute per-pixel error between canvas and source in HSV space.

    Returns a blurred error map (float32, same size as inputs).
    """
    canvas_f = canvas_hsv.astype(np.float32)
    source_f = source_hsv.astype(np.float32)

    # Circular hue difference
    h_diff = np.abs(canvas_f[:, :, 0] - source_f[:, :, 0])
    h_diff = np.minimum(h_diff, 180 - h_diff)
    s_diff = np.abs(canvas_f[:, :, 1] - source_f[:, :, 1])
    v_diff = np.abs(canvas_f[:, :, 2] - source_f[:, :, 2])

    error = np.sqrt(h_diff**2 + s_diff**2 + v_diff**2)
    k = max(3, kernel_size | 1)  # ensure odd
    error = cv2.blur(error, (k, k))
    return error


def filter_anchors_by_error(points, error_map, padding, threshold=15.0):
    """Remove anchor points in low-error regions. Points use voronoi coords
    (x, y with y measured from bottom). error_map includes padding border."""
    (H, W) = error_map.shape
    kept = []
    for pt in points:
        x, y_up = pt[0], pt[1]
        y = H - y_up  # convert from bottom-up to top-down
        yi = max(0, min(int(round(y)), H - 1))
        xi = max(0, min(int(round(x)), W - 1))
        if error_map[yi, xi] >= threshold:
            kept.append(pt)
    if len(kept) == 0:
        return np.zeros((0, 2))
    return np.array(kept)


def force_brush_width(patch_sequence, brush_width):
    """Force all strokes to the given brush width."""
    half_w = max(0.5, (brush_width - 1) / 2)
    for stroke in patch_sequence:
        stroke["w1"] = half_w
        stroke["w2"] = half_w
        stroke["brush_width"] = float(brush_width)


def order_for_painting(strokes, palette):
    """Order strokes: dark colors first (value ascending), then nearest-neighbor
    within each color group."""
    from itertools import groupby

    # Sort by palette color value ascending (darks first), then palette_index
    palette_v = {i: p["hsv"][2] for i, p in enumerate(palette)}
    strokes.sort(key=lambda s: (palette_v.get(s["palette_index"], 0), s["palette_index"]))

    result = []
    for _key, group in groupby(strokes, key=lambda s: s["palette_index"]):
        result.extend(_nearest_neighbor_order(list(group)))
    return result


def _generate_masked_block_in(input_hsv, palette, masked_palette_map, brush_width,
                               padding, debug_dir=None, pass_idx=0):
    """Generate block-in strokes for regions where masked_palette_map >= 0.

    Like generate_block_in_splines but only paints where the mask is active.
    Regions are sorted by area (largest first = background first).
    """
    (H, W, _) = input_hsv.shape
    stride = max(1, int(brush_width * 0.7))
    half_w = max(0.5, (brush_width - 1) / 2)

    color_regions = []
    for pidx in range(len(palette)):
        color_mask = (masked_palette_map == pidx).astype(np.uint8)
        total_area = int(color_mask.sum())

        if total_area < brush_width * brush_width:
            continue

        # Find connected components to get individual regions
        num_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(
            color_mask * 255, connectivity=8
        )
        for comp_idx in range(1, num_comp):
            area = stats[comp_idx, cv2.CC_STAT_AREA]
            if area < brush_width * brush_width:
                continue
            x0 = stats[comp_idx, cv2.CC_STAT_LEFT]
            y0 = stats[comp_idx, cv2.CC_STAT_TOP]
            w = stats[comp_idx, cv2.CC_STAT_WIDTH]
            h = stats[comp_idx, cv2.CC_STAT_HEIGHT]
            color_regions.append((area, pidx, (x0, y0, w, h)))

    color_regions.sort(key=lambda r: -r[0])

    strokes = []
    for area, pidx, (x0, y0, rw, rh) in color_regions:
        hsv_color = palette[pidx]["hsv"]
        for row_y in range(y0, y0 + rh, stride):
            if rw < 2:
                continue
            strokes.append({
                "points": [[float(row_y), float(x0)], [float(row_y), float(x0 + rw)]],
                "width": float(brush_width),
                "hsv": [int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])],
                "palette_index": pidx,
                "type": "block_in",
                "pass": pass_idx,
                "arc_length": float(rw),
                "coordinate": [float(row_y), float(x0 + rw / 2)],
                "importance": float(rw * brush_width),
            })

    return strokes


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_analyze(args):
    """Analyze image and produce a brush kit JSON."""
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        return

    palette = extract_palette(image_path, args.palette_size)
    brushes = compute_brush_widths(args.p_max, args.num_brushes)

    # Stroke constraints: defaults based on largest brush
    max_brush = max(brushes)
    kit = {
        "image": image_path,
        "palette": palette,
        "brushes": brushes,
        "p_max": args.p_max,
        "ssaa": args.ssaa,
        "seed": args.seed,
        "ratio": args.ratio,
        "brush_template": args.brush,
        "stroke_max_length": round(max_brush * 15, 1),
        "stroke_max_curvature": round(1.0 / (max_brush * 3), 4),
    }

    if args.output:
        out_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = f"{stem}_kit.json"

    with open(out_path, "w") as f:
        json.dump(kit, f, indent=2)

    # Save swatch next to the kit JSON
    swatch_path = os.path.splitext(out_path)[0] + "_swatch.png"
    save_palette_swatch(palette, swatch_path)

    print(f"Brush kit written to: {out_path}")
    print(f"  Palette: {len(palette)} colors")
    for i, c in enumerate(palette):
        print(f"    {i}: {c['name']} (H={c['hsv'][0]} S={c['hsv'][1]} V={c['hsv'][2]})")
    print(f"  Brushes: {brushes}")
    print(f"Edit {out_path} then run: im2oil.py paint {image_path} {out_path}")


def cmd_paint(args):
    """Run multi-pass painting pipeline with spline strokes.

    Pass 0: Block-in (horizontal washes by color region)
    Pass 1..N: Detail (ETF-guided curved strokes, progressively finer)
    """
    with open(args.kit) as f:
        kit = json.load(f)

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        return

    # Load semantic layers if available
    layers_path = getattr(args, "layers", None)
    semantic_layers = None
    if layers_path and os.path.exists(layers_path):
        with open(layers_path) as f:
            layers_data = json.load(f)
        semantic_layers = sorted(layers_data["layers"], key=lambda l: l["depth"])
        print(f"Loaded {len(semantic_layers)} semantic layers from {layers_path}")

    # Parameters from kit
    p_max_int = kit["p_max"]
    SSAA = kit["ssaa"]
    seed = kit["seed"]
    ratio = kit["ratio"]
    palette = kit["palette"]
    brush_widths = sorted(kit["brushes"], reverse=True)  # largest first
    stroke_max_length = kit.get("stroke_max_length", brush_widths[0] * 15)
    stroke_max_curvature = kit.get("stroke_max_curvature", 1.0 / (brush_widths[0] * 3))
    padding = 5
    kernel_radius = 5
    freq = getattr(args, "freq", 100)
    threshold_hsv = (30, None, 15)

    # Output directory
    filename = os.path.basename(image_path)
    filestem = os.path.splitext(filename)[0]
    output_path = args.output_dir or f"./output/{filestem}-p-{p_max_int}"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + "/anchor", exist_ok=True)
    os.makedirs(output_path + "/stroke", exist_ok=True)
    os.makedirs(output_path + "/process", exist_ok=True)
    debug_dir = output_path + "/debug"
    os.makedirs(debug_dir, exist_ok=True)

    # Load and prepare images
    np.random.seed(seed)
    input_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imwrite(output_path + "/input_bgr.png", input_bgr)
    input_bgr = cv2.copyMakeBorder(
        input_bgr, padding, padding, padding, padding, cv2.BORDER_REPLICATE
    )
    input_hsv = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2HSV)
    input_gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    (H0, W0) = input_gray.shape

    # Canvas padding from largest brush
    max_brush = max(brush_widths)
    max_length = int(ratio * max_brush) + 1

    # ETF (computed once at full resolution)
    time_start = time.time()
    ETF_filter = ETF(
        img=input_gray,
        output_path=output_path + "/mask",
        kernel_radius=kernel_radius,
        iter_time=15,
        background_dir=None,
    )
    angle = ETF_filter.forward().numpy()
    print("ETF Filtering time:", int(time.time() - time_start), "seconds")

    # Initialize canvas
    Canvas = Gassian_HSV(
        (H0 * SSAA + 2 * max_length * SSAA, W0 * SSAA + 2 * max_length * SSAA, 3)
    )
    Mask = np.zeros(
        (H0 * SSAA + 2 * max_length * SSAA, W0 * SSAA + 2 * max_length * SSAA),
        dtype=np.float32,
    )

    all_stroke_metadata = []
    total_rendered = 0

    # Debug: palette quantization map
    palette_map, palette_vis = quantize_pixel_map(input_hsv, palette)
    save_debug_image(os.path.join(debug_dir, "palette_map.png"), palette_vis)
    save_palette_swatch(palette, os.path.join(debug_dir, "palette_swatch.png"))

    # ---------------------------------------------------------------------------
    # Pass 0: Block-in (horizontal washes)
    # ---------------------------------------------------------------------------
    bw_block = brush_widths[0]
    print(f"\n=== Pass 0: Block-in (brush width {bw_block}) ===")
    time_start = time.time()

    if semantic_layers:
        # Layer-aware blocking: iterate layers back-to-front
        # Each layer uses its inpainted fill image for color, its mask for extent
        all_block_strokes = []
        for layer in semantic_layers:
            layer_name = layer["name"]
            fill_path = layer.get("fill_path")
            mask_path = layer.get("mask_path")

            if fill_path and os.path.exists(fill_path):
                fill_img = cv2.imread(fill_path, cv2.IMREAD_COLOR)
                fill_hsv = cv2.cvtColor(fill_img, cv2.COLOR_BGR2HSV)
                # Resize to match padded input if needed
                if fill_hsv.shape[:2] != input_hsv.shape[:2]:
                    fill_hsv_padded = cv2.copyMakeBorder(
                        fill_hsv, padding, padding, padding, padding,
                        cv2.BORDER_REPLICATE,
                    )
                else:
                    fill_hsv_padded = fill_hsv
            else:
                fill_hsv_padded = input_hsv

            # Build palette map from the fill image for this layer
            layer_palette_map, _ = quantize_pixel_map(fill_hsv_padded, palette)

            # Determine region: use mask if available, else full image
            if mask_path and os.path.exists(mask_path):
                layer_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if layer_mask is not None:
                    # Resize mask to match padded dims
                    layer_mask = cv2.copyMakeBorder(
                        layer_mask, padding, padding, padding, padding,
                        cv2.BORDER_REPLICATE,
                    )
                    # Use bbox of mask region for blocking
                    bbox_pct = layer.get("bbox", [0, 0, 100, 100])
                    img_h, img_w = input_hsv.shape[:2]
                    x0 = int(bbox_pct[0] / 100 * img_w)
                    y0 = int(bbox_pct[1] / 100 * img_h)
                    bw_bbox = max(1, int(bbox_pct[2] / 100 * img_w))
                    bh_bbox = max(1, int(bbox_pct[3] / 100 * img_h))
                else:
                    x0, y0 = 0, 0
                    bw_bbox, bh_bbox = input_hsv.shape[1], input_hsv.shape[0]
            else:
                x0, y0 = 0, 0
                bw_bbox, bh_bbox = input_hsv.shape[1], input_hsv.shape[0]

            # Generate block-in strokes for this layer's bbox
            stride = max(1, int(bw_block * 0.7))
            half_w = max(0.5, (bw_block - 1) / 2)
            layer_strokes = []
            for row_y in range(y0, min(y0 + bh_bbox, input_hsv.shape[0]), stride):
                if bw_bbox < 2:
                    continue
                # Get the palette color for this row from the fill image
                mid_x = min(x0 + bw_bbox // 2, input_hsv.shape[1] - 1)
                mid_y = min(row_y, input_hsv.shape[0] - 1)
                pidx = int(layer_palette_map[mid_y, mid_x])
                hsv_color = palette[pidx]["hsv"]

                layer_strokes.append({
                    "points": [[float(row_y), float(x0)],
                               [float(row_y), float(x0 + bw_bbox)]],
                    "width": float(bw_block),
                    "hsv": [int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])],
                    "palette_index": pidx,
                    "type": "block_in",
                    "pass": 0,
                    "arc_length": float(bw_bbox),
                    "coordinate": [float(row_y), float(x0 + bw_bbox / 2)],
                    "importance": float(bw_bbox * bw_block),
                })

            print(f"  Layer '{layer_name}' (depth {layer['depth']}): {len(layer_strokes)} strokes")
            all_block_strokes.extend(layer_strokes)

        block_in_ordered = all_block_strokes  # already in depth order
    else:
        # Fallback: pixel-level palette blocking (no semantic layers)
        block_in_strokes = generate_block_in_splines(
            input_hsv, palette, palette_map, bw_block, padding,
            debug_dir=debug_dir,
        )
        block_in_ordered = order_for_painting(block_in_strokes, palette)

    if block_in_ordered:
        Canvas, Mask, stroke_metadata = render_all_strokes(
            block_in_ordered, input_gray, output_path, max_length,
            ssaa=SSAA, padding=padding, freq=freq,
            save_strokes=True, stroke_index_offset=total_rendered,
            canvas=Canvas, mask=Mask,
        )
        all_stroke_metadata.extend(stroke_metadata)
        total_rendered += len(stroke_metadata)

    # Save pass 0 snapshot
    result = Canvas[
        max_length * SSAA : -max_length * SSAA,
        max_length * SSAA : -max_length * SSAA,
    ]
    cv2.imwrite(
        output_path + f"/pass_0_bw{bw_block}_blockin.png",
        cv2.cvtColor(
            result[padding * SSAA : -padding * SSAA, padding * SSAA : -padding * SSAA],
            cv2.COLOR_HSV2BGR,
        ),
    )
    print(f"  Pass 0 time: {int(time.time() - time_start)} seconds")

    # ---------------------------------------------------------------------------
    # Passes 1..N-1: Recursive block-in at progressively finer brush widths
    # Pass N (last): ETF detail strokes
    # ---------------------------------------------------------------------------
    for pass_idx, bw in enumerate(brush_widths[1:], start=1):
        is_last_pass = (pass_idx == len(brush_widths) - 1)
        pass_type = "detail" if is_last_pass else "block-in"
        print(f"\n=== Pass {pass_idx}: brush width {bw} ({pass_type}) ===")
        time_start = time.time()

        # Compute residual error — what still needs painting?
        canvas_crop = Canvas[
            max_length * SSAA : -max_length * SSAA,
            max_length * SSAA : -max_length * SSAA,
        ]
        canvas_small = cv2.resize(canvas_crop, (W0, H0), interpolation=cv2.INTER_AREA)
        error = compute_residual_error(canvas_small, input_hsv, kernel_size=int(bw * 2))

        error_vis = np.clip(error * 2, 0, 255).astype(np.uint8)
        save_debug_image(os.path.join(debug_dir, f"pass_{pass_idx}_error.png"), error_vis)

        if is_last_pass:
            # --- Last pass: ETF-guided detail strokes ---
            pass_max_length = min(stroke_max_length, bw * 15)
            pass_max_curvature = max(stroke_max_curvature, 1.0 / (bw * 3))

            p_max_pass = 1.0 / max(1, bw * bw)
            p_min_pass = p_max_pass / 10

            point_num, density, gradient_magnitude, point_path = K_Means_Sampler(
                output_dir=output_path + "/anchor",
                filename=image_path,
                p_max=p_max_pass,
                p_min=p_min_pass,
                border_copy=padding,
                k_size=5, n_iter=15, figsize=6, pointsize=(8.0, 8.0),
                display=False, force=True, save=True,
            )
            points = np.load(point_path)

            points_before = len(points)
            points = filter_anchors_by_error(points, error, padding, threshold=25.0)
            print(f"  Filtered anchors: {points_before} -> {len(points)}")
            if len(points) == 0:
                print(f"  No high-error regions -- skipping pass")
                continue

            detail_strokes = generate_detail_strokes(
                points, input_gray, input_hsv, gradient_magnitude,
                angle, bw, pass_max_length, pass_max_curvature,
                threshold_hsv, palette,
            )
            print(f"  Generated {len(detail_strokes)} detail strokes")

            if not detail_strokes:
                continue

            for s in detail_strokes:
                s["pass"] = pass_idx

            pass_sequence = order_for_painting(detail_strokes, palette)

        else:
            # --- Middle passes: block-in washes on high-error regions ---
            # Build a residual palette map: re-quantize the SOURCE image but only
            # in areas where the canvas has high error
            error_mask = (error > 20).astype(np.uint8)

            # Dilate error mask to get solid regions (avoid tiny fragments)
            k = max(3, int(bw) | 1)
            error_mask = cv2.dilate(error_mask, np.ones((k, k), np.uint8), iterations=2)

            # Save debug
            save_debug_image(
                os.path.join(debug_dir, f"pass_{pass_idx}_error_mask.png"),
                error_mask * 255,
            )

            # Quantize the source image to palette, masked to error regions
            residual_palette_map = palette_map.copy()
            # Zero out regions that don't need painting (low error)
            residual_palette_map[error_mask == 0] = -1

            # Generate block-in strokes for high-error regions only
            block_strokes = _generate_masked_block_in(
                input_hsv, palette, residual_palette_map, bw, padding,
                debug_dir=debug_dir, pass_idx=pass_idx,
            )

            if not block_strokes:
                print(f"  No regions to block in -- skipping pass")
                continue

            for s in block_strokes:
                s["pass"] = pass_idx

            pass_sequence = order_for_painting(block_strokes, palette)
            print(f"  Block-in: {len(pass_sequence)} strokes")

        # Render
        Canvas, Mask, stroke_metadata = render_all_strokes(
            pass_sequence, input_gray, output_path, max_length,
            ssaa=SSAA, padding=padding, freq=freq,
            save_strokes=True, stroke_index_offset=total_rendered,
            canvas=Canvas, mask=Mask,
        )
        all_stroke_metadata.extend(stroke_metadata)
        total_rendered += len(stroke_metadata)

        # Save pass snapshot
        result = Canvas[
            max_length * SSAA : -max_length * SSAA,
            max_length * SSAA : -max_length * SSAA,
        ]
        cv2.imwrite(
            output_path + f"/pass_{pass_idx}_bw{bw}.png",
            cv2.cvtColor(
                result[padding * SSAA : -padding * SSAA, padding * SSAA : -padding * SSAA],
                cv2.COLOR_HSV2BGR,
            ),
        )
        print(f"  Pass time: {int(time.time() - time_start)} seconds")

    # Save final result
    result = Canvas[
        max_length * SSAA : -max_length * SSAA,
        max_length * SSAA : -max_length * SSAA,
    ]
    cv2.imwrite(
        output_path + "/Final_Result.png",
        cv2.cvtColor(
            result[padding * SSAA : -padding * SSAA, padding * SSAA : -padding * SSAA],
            cv2.COLOR_HSV2BGR,
        ),
    )

    # Save stroke metadata (spline format — points instead of PNGs)
    (img_h, img_w) = cv2.imread(image_path, cv2.IMREAD_COLOR).shape[:2]
    strokes_json = {
        "image_width": img_w,
        "image_height": img_h,
        "ssaa": SSAA,
        "padding": padding,
        "max_length": max_length,
        "total_strokes": len(all_stroke_metadata),
        "palette": palette,
        "brushes": kit["brushes"],
        "stroke_max_length": stroke_max_length,
        "stroke_max_curvature": stroke_max_curvature,
        "strokes": all_stroke_metadata,
    }
    with open(output_path + "/strokes.json", "w") as f:
        json.dump(strokes_json, f)
    print(f"\nSaved {len(all_stroke_metadata)} strokes to {output_path}/strokes.json")
    print(f"Output: {output_path}")


def cmd_view(args):
    """Generate HTML stroke viewer from output directory."""
    build_html(args.output_dir)


def cmd_run(args):
    """Run full pipeline: analyze -> segment -> paint -> view."""
    from types import SimpleNamespace
    from segmentation import cmd_segment

    image_path = args.image
    stem = os.path.splitext(os.path.basename(image_path))[0]

    p_max_int = getattr(args, "p_max", 4)
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        output_dir = f"./output/{stem}-p-{p_max_int}"

    kit_path = os.path.join(output_dir, f"{stem}_kit.json")
    layers_path = os.path.join(output_dir, f"{stem}_layers.json")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Analyze
    analyze_args = SimpleNamespace(
        image=image_path,
        palette_size=getattr(args, "palette_size", 12),
        num_brushes=getattr(args, "num_brushes", 5),
        p_max=p_max_int,
        ssaa=getattr(args, "ssaa", 8),
        seed=getattr(args, "seed", 0),
        ratio=getattr(args, "ratio", 3),
        brush=getattr(args, "brush", "./brush/brush-0.png"),
        output=kit_path,
    )
    print("=== Step 1: Analyze ===")
    cmd_analyze(analyze_args)

    # Step 2: Segment (requires ANTHROPIC_API_KEY)
    layers_arg = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        segment_args = SimpleNamespace(
            image=image_path,
            kit=kit_path,
            output=layers_path,
        )
        print("\n=== Step 2: Segment ===")
        cmd_segment(segment_args)
        layers_arg = layers_path
    else:
        print("\n=== Step 2: Segment (skipped — no ANTHROPIC_API_KEY) ===")

    # Step 3: Paint
    paint_args = SimpleNamespace(
        image=image_path,
        kit=kit_path,
        layers=layers_arg,
        output_dir=output_dir,
        freq=getattr(args, "freq", 100),
        force=True,
    )
    print("\n=== Step 3: Paint ===")
    cmd_paint(paint_args)

    # Step 4: View
    view_args = SimpleNamespace(output_dir=output_dir)
    print("\n=== Step 4: View ===")
    cmd_view(view_args)
