import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pipeline  # noqa: E402

FAKE_INKSTITCH = r'''#!{python}
"""Stand-in for the Ink/Stitch binary used by the tests: turns each filled
polygon of the SVG into a block of stitches (one thread per color) and writes
a zip of the requested formats to stdout, like `inkstitch --extension=zip`."""
import io, re, sys, zipfile, tempfile, os
import xml.etree.ElementTree as ET
import pyembroidery

args = sys.argv[1:]
svg = args[-1]
if os.environ.get("FAKE_INKSTITCH_MODE") == "garbage":
    sys.stdout.write("Traceback: something went wrong\n")
    sys.stderr.write("fake failure details\n")
    sys.exit(0)
fmts = [a[len("--format-"):].split("=")[0] for a in args if a.startswith("--format-")]
name = next((a.split("=", 1)[1] for a in args if a.startswith("--custom-file-name=")), "design")
root = ET.parse(svg).getroot()
vb = [float(v) for v in root.get("viewBox").split()]
mm_w = float(root.get("width").replace("mm", ""))
scale = mm_w * 10 / vb[2]  # user units -> 0.1 mm
num = r"[-+]?(?:\d+\.?\d*|\.\d+)"
pattern = pyembroidery.EmbPattern()
last = None
for el in root.iter():
    if not el.tag.endswith("path"):
        continue
    fill = el.get("fill")
    m = re.search(r"translate\(\s*(%s)[\s,]+(%s)" % (num, num), el.get("transform", ""))
    tx, ty = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
    n = [float(x) for x in re.findall(num, el.get("d"))]
    pts = [((n[i] + tx) * scale, (n[i + 1] + ty) * scale) for i in range(0, len(n) - 1, 2)]
    if fill != last:
        if last is not None:
            pattern.add_command(pyembroidery.COLOR_CHANGE)
        pattern.add_thread(fill)
        last = fill
    for x, y in pts:
        pattern.add_stitch_absolute(pyembroidery.STITCH, x, y)
pattern.add_command(pyembroidery.END)
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z, tempfile.TemporaryDirectory() as td:
    for f in fmts:
        if f == "threadlist":
            z.writestr(name + ".txt", "threads\n")
            continue
        p = os.path.join(td, name + "." + f)
        pyembroidery.write(pattern, p)
        z.write(p, name + "." + f)
sys.stdout.buffer.write(buf.getvalue())
'''


@pytest.fixture
def fake_inkstitch(tmp_path):
    exe = tmp_path / "inkstitch"
    exe.write_text(FAKE_INKSTITCH.replace("{python}", sys.executable))
    exe.chmod(0o755)
    return str(exe)


def rasterize_svg(svg_path, size, bg):
    """Minimal rasterizer for vtracer polygon-mode output (M/L/Z inside
    translate()), in the SVG's user units. Enough to check layering."""
    im = Image.new("RGB", size, tuple(bg))
    draw = ImageDraw.Draw(im)
    for el in ET.parse(svg_path).getroot().iter():
        if not el.tag.endswith("path"):
            continue
        m = pipeline._TRANSLATE_RE.search(el.get("transform", ""))
        tx, ty = (float(m.group(1)), float(m.group(2) or 0)) if m else (0.0, 0.0)
        nums = [float(x) for x in re.findall(pipeline._NUM, el.get("d"))]
        pts = [(nums[i] + tx, nums[i + 1] + ty) for i in range(0, len(nums) - 1, 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=el.get("fill"))
    return np.asarray(im).astype(int)


def mismatch_pct(a, b):
    return float((np.abs(a - b).sum(axis=2) > 60).mean() * 100)
