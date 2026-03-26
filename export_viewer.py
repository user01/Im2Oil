"""Generate a self-contained HTML stroke viewer from Im2Oil output.

Usage:
    python export_viewer.py ./output/S1-p-4/

Reads strokes.json (with spline point data) and produces viewer.html.
Strokes are rendered as canvas paths — no PNG embedding needed.
"""

import argparse
import base64
import json
import os
import sys


def img_to_data_uri(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_html(output_dir):
    strokes_path = os.path.join(output_dir, "strokes.json")
    if not os.path.exists(strokes_path):
        print(f"Error: {strokes_path} not found.")
        sys.exit(1)

    with open(strokes_path) as f:
        manifest = json.load(f)

    # Check if strokes have 'points' (spline format) or not (legacy PNG format)
    has_points = manifest["strokes"] and "points" in manifest["strokes"][0]

    if has_points:
        html = generate_spline_html(manifest, output_dir)
    else:
        html = generate_legacy_html(manifest, output_dir)

    out_path = os.path.join(output_dir, "viewer.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path} ({len(manifest['strokes'])} strokes)")


def generate_spline_html(manifest, output_dir):
    """Generate HTML viewer for spline-based strokes (no PNGs needed)."""
    ssaa = manifest["ssaa"]
    canvas_w = manifest["image_width"] * ssaa
    canvas_h = manifest["image_height"] * ssaa

    # Embed source image for reference overlay
    input_path = os.path.join(output_dir, "input_bgr.png")
    source_data_uri = img_to_data_uri(input_path) if os.path.exists(input_path) else ""

    strokes_json = json.dumps(manifest["strokes"])
    total = len(manifest["strokes"])
    padding = manifest.get("padding", 5)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Im2Oil Stroke Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: system-ui, -apple-system, sans-serif; background: #1a1a1a; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 20px; }}
h1 {{ font-size: 18px; margin-bottom: 12px; font-weight: 500; }}
.canvas-wrap {{ position: relative; border: 1px solid #333; margin-bottom: 16px; background: #fff; }}
canvas {{ display: block; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: center; margin-bottom: 12px; max-width: 900px; }}
button {{ background: #333; color: #e0e0e0; border: 1px solid #555; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 14px; }}
button:hover {{ background: #444; }}
button.active {{ background: #2a6; color: #fff; border-color: #2a6; }}
input[type=range] {{ width: 220px; accent-color: #2a6; }}
.info {{ font-size: 13px; color: #aaa; text-align: center; margin-bottom: 8px; }}
.stroke-info {{ background: #222; border: 1px solid #333; border-radius: 4px; padding: 10px 16px; font-size: 13px; line-height: 1.6; max-width: 700px; width: 100%; }}
.stroke-info span {{ color: #8cf; }}
.speed-group {{ display: flex; align-items: center; gap: 4px; }}
label {{ font-size: 13px; color: #aaa; }}
</style>
</head>
<body>
<h1>Im2Oil Stroke Viewer</h1>

<div class="canvas-wrap">
  <canvas id="painting" width="{canvas_w}" height="{canvas_h}"></canvas>
</div>

<div class="info" id="counter">Stroke 0 / {total}</div>

<div class="controls">
  <button id="btnReset">Reset</button>
  <button id="btnStepBack">&laquo; Back</button>
  <button id="btnPlay">Play</button>
  <button id="btnStepFwd">Fwd &raquo;</button>
  <input type="range" id="scrubber" min="0" max="{total}" value="0">
  <div class="speed-group">
    <label for="speed">Speed:</label>
    <input type="range" id="speed" min="1" max="200" value="30">
    <span id="speedLabel" style="font-size:13px; min-width:50px;">30/s</span>
  </div>
  <button id="btnSource">Source</button>
</div>

<div class="stroke-info" id="strokeInfo">
  Step forward to see stroke details.
</div>

<script>
const STROKES = {strokes_json};
const SOURCE_URI = "{source_data_uri}";
const TOTAL = STROKES.length;
const CANVAS_W = {canvas_w};
const CANVAS_H = {canvas_h};
const SSAA = {ssaa};
const PADDING = {padding};

const canvas = document.getElementById('painting');
const displayScale = Math.min(1, 800 / CANVAS_W);
canvas.style.width = (CANVAS_W * displayScale) + 'px';
canvas.style.height = (CANVAS_H * displayScale) + 'px';
const ctx = canvas.getContext('2d');

const scrubber = document.getElementById('scrubber');
const counter = document.getElementById('counter');
const speedSlider = document.getElementById('speed');
const speedLabel = document.getElementById('speedLabel');
const strokeInfo = document.getElementById('strokeInfo');
const btnPlay = document.getElementById('btnPlay');

let currentStroke = 0;
let playing = false;
let playTimer = null;

let sourceImg = null;
if (SOURCE_URI) {{
  sourceImg = new Image();
  sourceImg.src = SOURCE_URI;
}}

function hsvToRgb(h, s, v) {{
  // OpenCV HSV: H=0-179, S=0-255, V=0-255
  h = h * 2;  // to 0-358
  s = s / 255;
  v = v / 255;
  const c = v * s;
  const x = c * (1 - Math.abs((h / 60) % 2 - 1));
  const m = v - c;
  let r, g, b;
  if (h < 60)       {{ r=c; g=x; b=0; }}
  else if (h < 120) {{ r=x; g=c; b=0; }}
  else if (h < 180) {{ r=0; g=c; b=x; }}
  else if (h < 240) {{ r=0; g=x; b=c; }}
  else if (h < 300) {{ r=x; g=0; b=c; }}
  else              {{ r=c; g=0; b=x; }}
  return `rgb(${{Math.round((r+m)*255)}},${{Math.round((g+m)*255)}},${{Math.round((b+m)*255)}})`;
}}

function clearCanvas() {{
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
}}

function drawStroke(i) {{
  const s = STROKES[i];
  if (!s.points || s.points.length < 2) return;

  const color = hsvToRgb(s.hsv[0], s.hsv[1], s.hsv[2]);
  ctx.strokeStyle = color;
  ctx.lineWidth = s.width * SSAA;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.globalAlpha = 0.9;

  ctx.beginPath();
  // Points are [y, x] in image coords — offset by padding, no extra offset needed
  const ox = 0, oy = 0;
  ctx.moveTo(s.points[0][1] * SSAA + ox, s.points[0][0] * SSAA + oy);
  for (let j = 1; j < s.points.length; j++) {{
    ctx.lineTo(s.points[j][1] * SSAA + ox, s.points[j][0] * SSAA + oy);
  }}
  ctx.stroke();
  ctx.globalAlpha = 1.0;
}}

function renderUpTo(n) {{
  clearCanvas();
  for (let i = 0; i < n; i++) {{
    drawStroke(i);
  }}
}}

function goTo(n) {{
  n = Math.max(0, Math.min(n, TOTAL));
  if (n < currentStroke) {{
    renderUpTo(n);
  }} else {{
    for (let i = currentStroke; i < n; i++) {{
      drawStroke(i);
    }}
  }}
  currentStroke = n;
  scrubber.value = n;
  counter.textContent = `Stroke ${{n}} / ${{TOTAL}}`;
  updateStrokeInfo();
}}

function updateStrokeInfo() {{
  if (currentStroke === 0 || currentStroke > TOTAL) {{
    strokeInfo.textContent = 'Step forward to see stroke details.';
    return;
  }}
  const s = STROKES[currentStroke - 1];
  const h = s.hsv[0] * 2;
  const sv = Math.round(s.hsv[1] / 255 * 100);
  const v = Math.round(s.hsv[2] / 255 * 100);
  const colorCss = `hsl(${{h}}, ${{sv}}%, ${{v}}%)`;
  const arc = s.arc_length ? s.arc_length.toFixed(1) : '?';
  strokeInfo.innerHTML = `
    <b>Stroke ${{s.index}}</b> (${{s.type || 'interior'}}, pass ${{s.pass || 0}}) &mdash;
    Points: <span>${{s.points ? s.points.length : 0}}</span> |
    Width: <span>${{s.width}}</span> |
    Arc: <span>${{arc}}</span> |
    Color: <span style="display:inline-block;width:14px;height:14px;background:${{colorCss}};border:1px solid #555;vertical-align:middle;border-radius:2px;"></span>
    <span>#${{s.palette_index}}</span>
  `;
}}

function stepForward() {{ if (currentStroke < TOTAL) goTo(currentStroke + 1); }}
function stepBack() {{ if (currentStroke > 0) goTo(currentStroke - 1); }}

function togglePlay() {{
  playing = !playing;
  btnPlay.textContent = playing ? 'Pause' : 'Play';
  btnPlay.classList.toggle('active', playing);
  if (playing) scheduleNext();
  else clearTimeout(playTimer);
}}

function scheduleNext() {{
  if (!playing || currentStroke >= TOTAL) {{ if (playing) togglePlay(); return; }}
  const rate = parseInt(speedSlider.value);
  playTimer = setTimeout(() => {{ stepForward(); scheduleNext(); }}, 1000 / rate);
}}

function toggleSource() {{
  if (!sourceImg) return;
  const btn = document.getElementById('btnSource');
  const showing = btn.classList.toggle('active');
  if (showing) {{
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.drawImage(sourceImg, 0, 0, CANVAS_W, CANVAS_H);
    ctx.restore();
  }} else {{
    goTo(currentStroke);
  }}
}}

document.getElementById('btnReset').addEventListener('click', () => {{
  clearCanvas(); currentStroke = 0; scrubber.value = 0;
  counter.textContent = `Stroke 0 / ${{TOTAL}}`; updateStrokeInfo();
}});
document.getElementById('btnStepBack').addEventListener('click', stepBack);
document.getElementById('btnStepFwd').addEventListener('click', stepForward);
btnPlay.addEventListener('click', togglePlay);
scrubber.addEventListener('input', () => goTo(parseInt(scrubber.value)));
speedSlider.addEventListener('input', () => {{ speedLabel.textContent = speedSlider.value + '/s'; }});
document.getElementById('btnSource').addEventListener('click', toggleSource);

document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowRight') stepForward();
  else if (e.key === 'ArrowLeft') stepBack();
  else if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
}});

clearCanvas();
</script>
</body>
</html>"""


def generate_legacy_html(manifest, output_dir):
    """Generate HTML viewer for legacy PNG-based strokes (backward compat)."""
    # Embed stroke images as base64
    stroke_data_uris = []
    for s in manifest["strokes"]:
        idx = s["index"]
        png_path = os.path.join(output_dir, "stroke", f"{idx:05d}.png")
        if os.path.exists(png_path):
            stroke_data_uris.append(img_to_data_uri(png_path))
        else:
            stroke_data_uris.append("")

    input_path = os.path.join(output_dir, "input_bgr.png")
    source_data_uri = img_to_data_uri(input_path) if os.path.exists(input_path) else ""

    # For legacy, return a simple message directing to the new pipeline
    ssaa = manifest["ssaa"]
    canvas_w = manifest["image_width"] * ssaa
    canvas_h = manifest["image_height"] * ssaa
    strokes_json = json.dumps(manifest["strokes"])
    uris_json = json.dumps(stroke_data_uris)
    total = len(manifest["strokes"])

    return f"""<!DOCTYPE html>
<html><head><title>Im2Oil Legacy Viewer</title></head>
<body style="background:#1a1a1a;color:#e0e0e0;text-align:center;padding:40px;">
<h2>Legacy PNG-based viewer ({total} strokes)</h2>
<p>This output uses the old PNG stroke format. Re-run with the new spline pipeline for the interactive viewer.</p>
</body></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML stroke viewer from Im2Oil output")
    parser.add_argument("output_dir", help="Path to the Im2Oil output directory")
    args = parser.parse_args()
    build_html(args.output_dir)
