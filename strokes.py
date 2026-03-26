"""Spline-based stroke generation and rendering.

Strokes are polyline paths with a width, rendered as a round brush dragged
along the path. Replaces the PNG-based Render_Stroke for the new pipeline.
"""

import math
import os

import cv2
import numpy as np
import tqdm


# ---------------------------------------------------------------------------
# ETF spline generation
# ---------------------------------------------------------------------------


def _color_similar(input_hsv, y, x, hsv_0, threshold_hsv):
    """Check if pixel color is similar enough to continue the stroke."""
    H, W = input_hsv.shape[:2]
    yi = max(0, min(int(round(y)), H - 1))
    xi = max(0, min(int(round(x)), W - 1))
    hsv = input_hsv[yi, xi]

    h_diff = abs(float(hsv[0]) - float(hsv_0[0]))
    h_diff = min(h_diff, 180 - h_diff)
    v_diff = abs(float(hsv[2]) - float(hsv_0[2]))

    return h_diff <= threshold_hsv[0] and v_diff <= threshold_hsv[2]


def _wrap_angle(diff):
    """Wrap angle difference to [-pi, pi]."""
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


def generate_etf_spline(img_gray, input_hsv, angle_map, y, x, width,
                         max_length, max_curvature, threshold_hsv):
    """Generate a spline stroke following the ETF field from anchor (y, x).

    Walks forward and backward from the anchor along the edge tangent flow,
    constraining curvature at each step. Stops when paint runs out (max_length),
    color diverges, or image boundary is hit.

    Returns list of [y, x] points.
    """
    H, W = img_gray.shape
    hsv_0 = input_hsv[
        max(0, min(int(round(y)), H - 1)),
        max(0, min(int(round(x)), W - 1)),
    ]
    half_length = max_length / 2

    def walk(start_y, start_x, direction):
        """Walk in one direction along the ETF. direction = 1 or -1."""
        points = []
        cur_y, cur_x = float(start_y), float(start_x)
        iy = max(0, min(int(round(cur_y)), H - 1))
        ix = max(0, min(int(round(cur_x)), W - 1))
        prev_angle = angle_map[iy, ix] / 180 * math.pi * direction
        arc = 0

        while arc < half_length:
            iy = max(0, min(int(round(cur_y)), H - 1))
            ix = max(0, min(int(round(cur_x)), W - 1))
            target_angle = angle_map[iy, ix] / 180 * math.pi * direction

            # Constrain curvature
            angle_diff = _wrap_angle(target_angle - prev_angle)
            angle_diff = max(-max_curvature, min(max_curvature, angle_diff))
            new_angle = prev_angle + angle_diff

            # Step
            cur_x += math.cos(new_angle)
            cur_y -= math.sin(new_angle)
            arc += 1

            # Bounds check
            if cur_y < 0 or cur_y >= H or cur_x < 0 or cur_x >= W:
                break

            # Color check
            if not _color_similar(input_hsv, cur_y, cur_x, hsv_0, threshold_hsv):
                break

            points.append([cur_y, cur_x])
            prev_angle = new_angle

        return points

    # Walk both directions
    fwd = walk(y, x, 1)
    bwd = walk(y, x, -1)

    # Combine: backward (reversed) + anchor + forward
    path = list(reversed(bwd)) + [[float(y), float(x)]] + fwd

    # Simplify: keep every Nth point to reduce data size (but not too sparse)
    step = max(1, int(width / 2))
    if step > 1 and len(path) > 3:
        simplified = [path[0]]
        for i in range(step, len(path) - 1, step):
            simplified.append(path[i])
        simplified.append(path[-1])
        path = simplified

    return path


def generate_detail_strokes(anchors, img_gray, input_hsv, gradient_magnitude,
                             angle_map, width, max_length, max_curvature,
                             threshold_hsv, palette, palette_index_map=None):
    """Generate spline strokes at anchor points for a detail pass.

    For each anchor, generates an ETF-following spline, quantizes color to palette.
    Returns list of stroke dicts.
    """
    H, W = img_gray.shape
    palette_hsv = np.array([p["hsv"] for p in palette], dtype=np.float32)
    strokes = []

    for i in tqdm.tqdm(range(len(anchors))):
        ax, ay_up = anchors[i]
        ay = H - ay_up  # voronoi y is bottom-up
        ay = max(0, min(int(round(ay)), H - 1))
        ax = max(0, min(int(round(ax)), W - 1))

        path = generate_etf_spline(
            img_gray, input_hsv, angle_map, ay, ax,
            width, max_length, max_curvature, threshold_hsv,
        )

        if len(path) < 2:
            continue

        # Color: sample from image at anchor, quantize to palette
        hsv_pixel = input_hsv[ay, ax].astype(np.float32)
        h_diff = np.abs(palette_hsv[:, 0] - hsv_pixel[0])
        h_diff = np.minimum(h_diff, 180 - h_diff)
        s_diff = palette_hsv[:, 1] - hsv_pixel[1]
        v_diff = palette_hsv[:, 2] - hsv_pixel[2]
        distances = np.sqrt(h_diff**2 + s_diff**2 + v_diff**2)
        pidx = int(np.argmin(distances))

        # Arc length
        arc = 0
        for j in range(len(path) - 1):
            dy = path[j + 1][0] - path[j][0]
            dx = path[j + 1][1] - path[j][1]
            arc += math.sqrt(dy**2 + dx**2)

        strokes.append({
            "points": path,
            "width": float(width),
            "hsv": [int(palette[pidx]["hsv"][0]),
                    int(palette[pidx]["hsv"][1]),
                    int(palette[pidx]["hsv"][2])],
            "palette_index": pidx,
            "type": "interior",
            "pass": 0,
            "arc_length": arc,
            "coordinate": [float(ay), float(ax)],  # anchor for spatial sorting
            "importance": arc * width,
        })

    return strokes


# ---------------------------------------------------------------------------
# Block-in spline generation
# ---------------------------------------------------------------------------


def generate_block_in_splines(input_hsv, palette, palette_map, brush_width,
                               padding, debug_dir=None):
    """Generate horizontal wash strokes for block-in pass.

    Same region logic as pipeline.generate_block_in_strokes but produces
    spline stroke dicts with `points` field.
    """
    from pipeline import save_debug_image

    (H, W, _) = input_hsv.shape
    half_w = max(0.5, (brush_width - 1) / 2)
    stride = max(1, int(brush_width * 0.7))

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

        ys, xs = np.where(color_mask > 0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        color_regions.append((total_area, pidx, (x0, y0, x1 - x0, y1 - y0)))

    color_regions.sort(key=lambda r: -r[0])

    if debug_dir:
        print("  Block-in region order (bg->fg):")
        for area, pidx, (x0, y0, rw, rh) in color_regions:
            print(f"    {pidx}: {palette[pidx]['name']} -- {area}px, bbox=({x0},{y0},{rw}x{rh})")

    strokes = []
    for area, pidx, (x0, y0, rw, rh) in color_regions:
        hsv_color = palette[pidx]["hsv"]

        for row_y in range(y0, y0 + rh, stride):
            if rw < 2:
                continue
            x_left = float(x0)
            x_right = float(x0 + rw)
            strokes.append({
                "points": [[float(row_y), x_left], [float(row_y), x_right]],
                "width": float(brush_width),
                "hsv": [int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])],
                "palette_index": pidx,
                "type": "block_in",
                "pass": 0,
                "arc_length": float(rw),
                "coordinate": [float(row_y), float(x0 + rw / 2)],
                "importance": float(rw * brush_width),
            })

    # Debug: draw stroke plan
    if debug_dir:
        plan_img = cv2.imread(
            os.path.join(os.path.dirname(debug_dir), "input_bgr.png"),
            cv2.IMREAD_COLOR,
        )
        if plan_img is not None:
            plan_vis = plan_img.copy()
            for s in strokes:
                pts = s["points"]
                sy = int(pts[0][0]) - padding
                sx0 = int(pts[0][1]) - padding
                sx1 = int(pts[1][1]) - padding
                sw = int(s["width"] / 2)
                color_bgr = cv2.cvtColor(
                    np.array([[[s["hsv"][0], s["hsv"][1], s["hsv"][2]]]], dtype=np.uint8),
                    cv2.COLOR_HSV2BGR,
                )[0, 0].tolist()
                y1c = max(0, sy - sw)
                y2c = min(plan_vis.shape[0], sy + sw)
                x1c = max(0, sx0)
                x2c = min(plan_vis.shape[1], sx1)
                cv2.rectangle(plan_vis, (x1c, y1c), (x2c, y2c), color_bgr, 1)
            save_debug_image(os.path.join(debug_dir, "block_in_plan.png"), plan_vis)

    print(f"  Block-in: {len(strokes)} strokes across {len(color_regions)} regions")
    return strokes


# ---------------------------------------------------------------------------
# Spline rendering
# ---------------------------------------------------------------------------


def render_spline_stroke(canvas, mask, points, width, hsv, ssaa, offset_y=0, offset_x=0):
    """Render a single spline stroke onto the canvas.

    Draws a thick polyline with round caps. Uses a local mask approach to
    alpha-blend the stroke color onto the canvas.
    """
    if len(points) < 2:
        return

    thickness = max(1, int(round(width * ssaa)))
    color_hsv = np.array([int(hsv[0]), int(hsv[1]), int(hsv[2])], dtype=np.uint8)

    # Convert points to canvas coordinates (with SSAA and offset)
    pts_cv = []
    for p in points:
        py = int(round(p[0] * ssaa)) + offset_y
        px = int(round(p[1] * ssaa)) + offset_x
        pts_cv.append([px, py])
    pts_arr = np.array(pts_cv, dtype=np.int32)

    # Compute bounding box for efficiency
    xs = pts_arr[:, 0]
    ys = pts_arr[:, 1]
    pad = thickness + 2
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(canvas.shape[1], int(xs.max()) + pad)
    y1 = min(canvas.shape[0], int(ys.max()) + pad)

    if x1 <= x0 or y1 <= y0:
        return

    # Draw stroke mask in local region
    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local_pts = pts_arr.copy()
    local_pts[:, 0] -= x0
    local_pts[:, 1] -= y0
    cv2.polylines(local_mask, [local_pts], isClosed=False, color=255,
                  thickness=thickness, lineType=cv2.LINE_AA)

    # Alpha blend: where mask > 0, replace canvas with stroke color
    alpha = local_mask.astype(np.float32) / 255.0
    alpha3 = alpha[:, :, np.newaxis]
    region = canvas[y0:y1, x0:x1]
    stroke_color = np.full_like(region, color_hsv)
    canvas[y0:y1, x0:x1] = np.uint8((1 - alpha3) * region + alpha3 * stroke_color)

    # Update global mask
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], alpha)


def render_all_strokes(strokes, img_gray, output_path, max_length, ssaa,
                        padding, freq=100, save_strokes=True,
                        stroke_index_offset=0, canvas=None, mask=None):
    """Render all spline strokes onto the canvas.

    Replaces search_and_render.Render_Stroke for the spline pipeline.
    Returns (Canvas, Mask, stroke_metadata).
    """
    from simulate_RGB import Gassian_HSV

    (h0, w0) = img_gray.shape
    canvas_h = h0 * ssaa + 2 * max_length * ssaa
    canvas_w = w0 * ssaa + 2 * max_length * ssaa

    if canvas is None:
        canvas = Gassian_HSV((canvas_h, canvas_w, 3))
    Canvas = canvas

    if mask is None:
        Mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    else:
        Mask = mask

    # Offset for canvas padding
    offset_y = max_length * ssaa
    offset_x = max_length * ssaa

    stroke_metadata = []
    crop_offset = max_length * ssaa + padding * ssaa

    for step in tqdm.tqdm(range(len(strokes))):
        s = strokes[step]
        global_idx = stroke_index_offset + step

        render_spline_stroke(
            Canvas, Mask, s["points"], s["width"], s["hsv"], ssaa,
            offset_y=offset_y, offset_x=offset_x,
        )

        # Compute bounding box for metadata
        pts = s["points"]
        ys = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)
        half_w = s["width"] / 2

        stroke_metadata.append({
            "index": global_idx,
            "points": [[round(p[0], 1), round(p[1], 1)] for p in pts],
            "width": s["width"],
            "hsv": s["hsv"],
            "palette_index": s.get("palette_index", -1),
            "type": s.get("type", "interior"),
            "pass": s.get("pass", 0),
            "arc_length": s.get("arc_length", 0),
        })

        # Save per-stroke PNG (small crop with alpha) if requested
        if save_strokes:
            # Compute stroke bbox on canvas
            sy0 = int(round((min_y - half_w) * ssaa)) + offset_y
            sy1 = int(round((max_y + half_w) * ssaa)) + offset_y
            sx0 = int(round((min_x - half_w) * ssaa)) + offset_x
            sx1 = int(round((max_x + half_w) * ssaa)) + offset_x

            # Clamp to canvas
            sy0 = max(0, sy0)
            sy1 = min(canvas_h, sy1)
            sx0 = max(0, sx0)
            sx1 = min(canvas_w, sx1)

            if sy1 > sy0 and sx1 > sx0:
                # Create small stroke image with alpha
                stroke_region = Canvas[sy0:sy1, sx0:sx1].copy()
                stroke_bgr = cv2.cvtColor(stroke_region, cv2.COLOR_HSV2BGR)
                alpha = (Mask[sy0:sy1, sx0:sx1] * 255).astype(np.uint8)
                alpha = np.clip(alpha, 0, 255)
                stroke_bgra = cv2.merge([
                    stroke_bgr[:, :, 0], stroke_bgr[:, :, 1],
                    stroke_bgr[:, :, 2], alpha,
                ])
                cv2.imwrite(
                    output_path + f"/stroke/{global_idx:05d}.png", stroke_bgra
                )

        # Save progress snapshots
        if (step + 1) % freq == 0:
            result = Canvas[
                max_length * ssaa : -max_length * ssaa,
                max_length * ssaa : -max_length * ssaa,
            ]
            cv2.imwrite(
                output_path + f"/process/{step + 1:05d}.png",
                cv2.cvtColor(
                    result[
                        padding * ssaa : -padding * ssaa,
                        padding * ssaa : -padding * ssaa,
                    ],
                    cv2.COLOR_HSV2BGR,
                ),
            )

    Mask[Mask > 0] = 1
    return Canvas, Mask, stroke_metadata
