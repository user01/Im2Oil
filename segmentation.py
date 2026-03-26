"""Semantic layer decomposition using Claude API.

Identifies objects in a scene, orders them by depth, generates masks,
and inpaints backgrounds behind foreground objects.

All API calls are disk-cached (keyed by hash of inputs) and debug-logged.
"""

import base64
import hashlib
import json
import os
import re
import sys
import time as _time

import anthropic
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Disk cache for API calls
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.dirname(__file__) or ".", ".api_cache")


def _cache_key(*parts):
    """Generate a cache key from arbitrary inputs."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            h.update(part.encode())
        elif isinstance(part, bytes):
            h.update(part)
        else:
            h.update(json.dumps(part, sort_keys=True).encode())
    return h.hexdigest()[:24]


def _cache_get(key):
    """Retrieve cached API response. Returns None if not cached."""
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"    CACHE HIT: {key}")
        return data
    return None


def _cache_set(key, data):
    """Store API response in disk cache."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    CACHE STORE: {key}")


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------


def _log_api_call(label, model, prompt_preview, debug_dir=None):
    """Log an API call for debugging."""
    print(f"    API CALL: {label}")
    print(f"      Model: {model}")
    print(f"      Prompt: {prompt_preview[:120]}...")
    if debug_dir:
        log_path = os.path.join(debug_dir, "api_calls.log")
        with open(log_path, "a") as f:
            f.write(f"\n--- {label} ({_time.strftime('%H:%M:%S')}) ---\n")
            f.write(f"Model: {model}\n")
            f.write(f"Prompt: {prompt_preview}\n")


def _log_api_response(label, response_text, debug_dir=None):
    """Log API response for debugging."""
    preview = response_text[:200] if isinstance(response_text, str) else str(response_text)[:200]
    print(f"    API RESPONSE: {preview}")
    if debug_dir:
        log_path = os.path.join(debug_dir, "api_calls.log")
        with open(log_path, "a") as f:
            f.write(f"Response: {response_text}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_to_base64(path):
    """Read an image file and return base64-encoded data."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _get_media_type(path):
    """Get media type from file extension."""
    ext = os.path.splitext(path)[1].lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")


# ---------------------------------------------------------------------------
# Step 1: Scene analysis
# ---------------------------------------------------------------------------

SCENE_ANALYSIS_PROMPT = """Analyze this image and identify all distinct visual elements/objects/regions.

For each element, provide:
- name: short descriptive name (e.g., "sky", "main tree", "distant forest", "grass field")
- depth: integer from 0 (furthest background) to N (closest foreground)
- description: brief description of what it looks like and where it is
- behind: what would be visible behind this object if it were removed
- bbox: approximate bounding box as [x_percent, y_percent, width_percent, height_percent] where values are 0-100 representing percentage of image dimensions
- seed_points: 3-5 representative points inside this region as [[x_pct, y_pct], ...] (percentages)

Return ONLY a JSON object with a "layers" array. Order layers from background (depth 0) to foreground (highest depth).

Example response format:
{"layers": [{"name": "sky", "depth": 0, "description": "Clear blue sky across upper portion", "behind": "nothing", "bbox": [0, 0, 100, 60], "seed_points": [[50, 20], [30, 10], [70, 30]]}]}"""


def analyze_scene(image_path, client=None, debug_dir=None):
    """Use Claude Vision to identify objects and depth ordering."""
    if client is None:
        client = anthropic.Anthropic()

    image_data = _image_to_base64(image_path)
    media_type = _get_media_type(image_path)
    model = "claude-sonnet-4-20250514"

    # Check cache
    cache_key = _cache_key("analyze_scene", image_path, SCENE_ANALYSIS_PROMPT,
                            os.path.getsize(image_path))
    cached = _cache_get(cache_key)
    if cached is not None:
        layers = cached.get("layers", [])
        print(f"  Found {len(layers)} layers (from cache):")
        for layer in layers:
            print(f"    depth {layer['depth']}: {layer['name']} -- {layer['description']}")

        # Save debug even on cache hit
        if debug_dir:
            _save_debug_prompt(debug_dir, "scene_analysis", SCENE_ANALYSIS_PROMPT,
                               json.dumps(cached, indent=2))
        return layers

    print("  Analyzing scene with Claude Vision...")
    _log_api_call("scene_analysis", model, SCENE_ANALYSIS_PROMPT, debug_dir)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_data},
                },
                {"type": "text", "text": SCENE_ANALYSIS_PROMPT},
            ],
        }],
    )

    text = response.content[0].text
    _log_api_response("scene_analysis", text, debug_dir)

    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        result = json.loads(json_match.group())
    else:
        raise ValueError(f"Could not parse scene analysis response: {text[:200]}")

    # Cache the result
    _cache_set(cache_key, result)

    # Save debug files
    if debug_dir:
        _save_debug_prompt(debug_dir, "scene_analysis", SCENE_ANALYSIS_PROMPT, text)

    layers = result.get("layers", [])
    print(f"  Found {len(layers)} layers:")
    for layer in layers:
        print(f"    depth {layer['depth']}: {layer['name']} -- {layer['description']}")

    return layers


def _save_debug_prompt(debug_dir, name, prompt, response):
    """Save API prompt and response to debug directory."""
    path = os.path.join(debug_dir, f"api_{name}.txt")
    with open(path, "w") as f:
        f.write(f"=== PROMPT ===\n{prompt}\n\n=== RESPONSE ===\n{response}\n")
    print(f"  DEBUG: {path}")


# ---------------------------------------------------------------------------
# Step 2: Generate masks using GrabCut
# ---------------------------------------------------------------------------


def generate_masks(image_path, layers, debug_dir):
    """Generate binary masks for each layer using GrabCut seeded by Claude's bbox/points."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    for layer in layers:
        name = layer["name"]
        bbox_pct = layer.get("bbox", [0, 0, 100, 100])
        seed_points = layer.get("seed_points", [])

        # Convert percentage bbox to pixel coordinates
        x0 = int(bbox_pct[0] / 100 * w)
        y0 = int(bbox_pct[1] / 100 * h)
        bw = max(1, int(bbox_pct[2] / 100 * w))
        bh = max(1, int(bbox_pct[3] / 100 * h))

        # Clamp to image bounds
        x0 = max(0, min(x0, w - 2))
        y0 = max(0, min(y0, h - 2))
        bw = min(bw, w - x0)
        bh = min(bh, h - y0)

        # GrabCut
        mask_gc = np.zeros((h, w), np.uint8)
        mask_gc[:] = cv2.GC_BGD  # assume background

        # Mark bbox interior as probable foreground
        mask_gc[y0:y0+bh, x0:x0+bw] = cv2.GC_PR_FGD

        # Mark seed points as definite foreground
        for sp in seed_points:
            px = int(sp[0] / 100 * w)
            py = int(sp[1] / 100 * h)
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))
            cv2.circle(mask_gc, (px, py), max(3, int(min(w, h) * 0.02)), cv2.GC_FGD, -1)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        rect = (x0, y0, bw, bh)
        try:
            cv2.grabCut(img, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            # Fallback: use the bbox as a simple rectangle mask
            mask_gc[:] = cv2.GC_BGD
            mask_gc[y0:y0+bh, x0:x0+bw] = cv2.GC_FGD

        # Extract binary mask (foreground = 1)
        binary_mask = np.where(
            (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        mask_path = os.path.join(debug_dir, f"layer_{layer['depth']}_{name.replace(' ', '_')}_mask.png")
        cv2.imwrite(mask_path, binary_mask)
        layer["mask_path"] = mask_path
        layer["bbox_px"] = [x0, y0, bw, bh]
        print(f"    Mask for '{name}': {mask_path}")

    return layers


# ---------------------------------------------------------------------------
# Step 3: Inpainting via Claude image generation
# ---------------------------------------------------------------------------


def inpaint_layer(image_path, layer, foreground_layers, debug_dir, client=None):
    """Use Claude to generate what a layer looks like with foreground objects removed.

    For background layers (depth 0), we may not need inpainting.
    For each layer, we describe the foreground objects to remove and ask Claude
    to generate the scene without them.
    """
    if client is None:
        client = anthropic.Anthropic()

    name = layer["name"]
    depth = layer["depth"]
    model = "claude-sonnet-4-20250514"
    fill_path = os.path.join(debug_dir, f"layer_{depth}_{name.replace(' ', '_')}_fill.png")

    # If nothing is in front, just use the original image
    fg_names = [fl["name"] for fl in foreground_layers]
    if not fg_names:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cv2.imwrite(fill_path, img)
        layer["fill_path"] = fill_path
        print(f"    Fill for '{name}': original image (no foreground to remove)")
        return layer

    # Build inpainting prompt
    fg_list = ", ".join(fg_names)
    prompt = (
        f"Look at this image. I need you to generate a new version of this image "
        f"with the following objects REMOVED: {fg_list}. "
        f"Fill in what would be behind them — extend the {name} naturally to cover "
        f"where those objects were. Keep the same style, colors, and lighting. "
        f"The result should look like a complete scene showing just the {name} "
        f"and anything at the same depth or behind it, as if the foreground objects "
        f"were never there. Output ONLY the image."
    )

    # Check cache (key includes image file size + prompt)
    cache_key = _cache_key("inpaint", image_path, os.path.getsize(image_path),
                            name, fg_list)
    cached = _cache_get(cache_key)
    if cached is not None:
        # Cached response — check if it had an image
        if cached.get("image_b64"):
            img_bytes = base64.standard_b64decode(cached["image_b64"])
            with open(fill_path, "wb") as f:
                f.write(img_bytes)
            layer["fill_path"] = fill_path
            print(f"    Fill for '{name}': {fill_path} (from cache)")
        elif cached.get("text"):
            print(f"    Warning: Cached response had no image for '{name}', using original")
            _log_api_response(f"inpaint_{name}", cached["text"], debug_dir)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            cv2.imwrite(fill_path, img)
            layer["fill_path"] = fill_path
        if debug_dir:
            _save_debug_prompt(debug_dir, f"inpaint_{name.replace(' ', '_')}",
                               prompt, cached.get("text", "(image response)"))
        return layer

    print(f"    Inpainting '{name}' (removing: {fg_list})...")
    _log_api_call(f"inpaint_{name}", model, prompt, debug_dir)

    image_data = _image_to_base64(image_path)
    media_type = _get_media_type(image_path)

    response = client.messages.create(
        model=model,
        max_tokens=1600,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_data},
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    # Process response and cache it
    cache_data = {"text": "", "image_b64": None}
    saved = False
    for block in response.content:
        if block.type == "image":
            cache_data["image_b64"] = block.source.data
            img_bytes = base64.standard_b64decode(block.source.data)
            with open(fill_path, "wb") as f:
                f.write(img_bytes)
            saved = True
            print(f"    Fill for '{name}': {fill_path}")
            _log_api_response(f"inpaint_{name}", "(generated image)", debug_dir)
            break
        elif block.type == "text":
            cache_data["text"] = block.text
            _log_api_response(f"inpaint_{name}", block.text, debug_dir)

    if not saved:
        print(f"    Warning: Could not generate inpainted image for '{name}', using original")
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cv2.imwrite(fill_path, img)

    # Cache regardless of success
    _cache_set(cache_key, cache_data)

    # Save debug
    if debug_dir:
        _save_debug_prompt(debug_dir, f"inpaint_{name.replace(' ', '_')}",
                           prompt, cache_data.get("text", "(image response)"))

    layer["fill_path"] = fill_path
    return layer


def generate_fills(image_path, layers, debug_dir, client=None):
    """Generate inpainted fill images for each layer."""
    if client is None:
        client = anthropic.Anthropic()

    sorted_layers = sorted(layers, key=lambda l: l["depth"])

    for i, layer in enumerate(sorted_layers):
        foreground = [l for l in sorted_layers if l["depth"] > layer["depth"]]
        inpaint_layer(image_path, layer, foreground, debug_dir, client=client)

    return layers


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


def cmd_segment(args):
    """Run semantic segmentation pipeline."""
    image_path = args.image
    kit_path = args.kit

    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load kit for palette info
    with open(kit_path) as f:
        kit = json.load(f)

    # Determine output paths
    if args.output:
        output_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{stem}_layers.json"

    # Create debug directory next to output
    output_dir = os.path.dirname(output_path) or "."
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    print("=== Semantic Layer Decomposition ===")

    # Step 1: Analyze scene
    layers = analyze_scene(image_path, client=client, debug_dir=debug_dir)

    # Step 2: Generate masks
    print("\n  Generating masks...")
    layers = generate_masks(image_path, layers, debug_dir)

    # Step 3: Inpaint backgrounds
    print("\n  Generating inpainted fills...")
    layers = generate_fills(image_path, layers, debug_dir, client=client)

    # Save layers.json
    layers_data = {
        "image": image_path,
        "kit": kit_path,
        "layers": layers,
    }
    with open(output_path, "w") as f:
        json.dump(layers_data, f, indent=2)

    print(f"\nLayers saved to: {output_path}")
    print(f"Debug images in: {debug_dir}")
    return layers_data
