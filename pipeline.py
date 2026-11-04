#!/usr/bin/env python3
"""
image_to_embroidery pipeline
=============================
Chains: photo -> preprocess -> vectorize (vtracer) -> clean SVG ->
        Ink/Stitch headless conversion -> machine files (.dst/.pes/...) ->
        preview render -> QA report.

Usage:
    py -3.12 pipeline.py INPUT_IMAGE OUTPUT_DIR [options]

Example:
    py -3.12 pipeline.py photo.jpg out/ --max-colors 6 --resize 800 --hoop 130x180

Requires (pip install, Python 3.12):
    pillow vtracer pyembroidery

Requires on the system:
    Ink/Stitch's inkstitch.exe (ships inside the Inkscape extension folder)
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image
import numpy as np

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


# ---------------------------------------------------------------------------
# Locate the Ink/Stitch binary
# ---------------------------------------------------------------------------
def find_inkstitch_exe():
    candidates = [
        Path.home() / "AppData/Roaming/inkscape/extensions/inkstitch/inkstitch/bin/inkstitch.exe",
        Path.home() / "AppData/Local/inkstitch/inkstitch/bin/inkstitch.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # fall back to a search
    for root in [Path.home() / "AppData"]:
        for p in root.rglob("inkstitch.exe"):
            return str(p)
    raise FileNotFoundError(
        "Could not find inkstitch.exe. Pass --inkstitch-exe explicitly."
    )


# ---------------------------------------------------------------------------
# Standardized thread catalogs — Ink/Stitch ships ~75 real manufacturer
# thread charts (Brother, Madeira, Isacord, DMC, ...) as GIMP-palette (.gpl)
# files with exact RGB + thread name + spool number. Snapping a design's
# colors to one of these (instead of arbitrary photo-derived hex) means
# every image gets a consistent, reorderable thread list, AND it heads off
# the surprise re-snap that happens anyway when Ink/Stitch exports to PES
# (older PES can only store catalog thread indices, not arbitrary RGB) —
# pre-snapping to the right catalog makes that later snap a no-op.
# ---------------------------------------------------------------------------
def find_palettes_dir():
    candidates = [
        Path.home() / "AppData/Roaming/inkscape/extensions/inkstitch/inkstitch/palettes",
        Path.home() / "AppData/Local/inkstitch/inkstitch/palettes",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    for root in [Path.home() / "AppData"]:
        for p in root.rglob("palettes"):
            if (p / ".").exists() and list(p.glob("*.gpl")):
                return p
    return None


def list_thread_catalogs():
    """Return the available catalog names (without the 'InkStitch ' prefix
    or '.gpl' extension), e.g. 'Brother Embroidery', 'Madeira Polyneon'."""
    d = find_palettes_dir()
    if not d:
        return []
    names = []
    for p in sorted(d.glob("*.gpl")):
        name = p.stem
        if name.lower().startswith("inkstitch "):
            name = name[len("inkstitch "):]
        names.append(name)
    return names


def find_catalog_file(catalog_name):
    """Fuzzy-find a .gpl file by partial/case-insensitive name match."""
    d = find_palettes_dir()
    if not d:
        raise FileNotFoundError("Could not find Ink/Stitch's palettes/ folder.")
    target = catalog_name.strip().lower()
    candidates = sorted(d.glob("*.gpl"))
    # exact (ignoring prefix/case) match first
    for p in candidates:
        name = p.stem.lower()
        if name == target or name == f"inkstitch {target}":
            return p
    # substring match
    matches = [p for p in candidates if target in p.stem.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(p.stem for p in matches)
        raise ValueError(f"'{catalog_name}' matches multiple catalogs: {names}. Be more specific.")
    raise FileNotFoundError(
        f"No thread catalog matching '{catalog_name}'. Available: {', '.join(list_thread_catalogs())}"
    )


def parse_gpl(path):
    """Parse a GIMP-palette .gpl file -> [{'rgb': (r,g,b), 'name': str, 'number': str}, ...]"""
    entries = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("GIMP Palette") \
                    or line.startswith("Name:") or line.startswith("Columns:"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            rest = parts[3:]
            number = rest[-1] if rest and re.match(r"^[0-9A-Za-z.\-]+$", rest[-1]) else ""
            name = " ".join(rest[:-1]) if number and len(rest) > 1 else " ".join(rest)
            entries.append({"rgb": (r, g, b), "name": name.strip(), "number": number.strip()})
    return entries


def snap_palette_to_catalog(working_palette_hex, catalog_entries):
    """Map each color in working_palette_hex (list of '#rrggbb') to the
    nearest real thread in catalog_entries. Returns (new_hex_list, matches)
    where matches is [{'from': hex, 'to': hex, 'name':, 'number':}, ...] —
    the 'to' list is deduped (two source colors can land on the same
    thread) so the final palette may have fewer entries than the input."""
    matches = []
    seen = {}
    new_hex = []
    for hx in working_palette_hex:
        src = _hex_to_rgb(hx)
        best = min(catalog_entries, key=lambda e: _color_dist(src, e["rgb"]))
        to_hex = "#%02x%02x%02x" % best["rgb"]
        matches.append({"from": hx, "to": to_hex, "name": best["name"], "number": best["number"]})
        if to_hex not in seen:
            seen[to_hex] = True
            new_hex.append(to_hex)
    return new_hex, matches


# ---------------------------------------------------------------------------
# Stage 0 (optional): AI background removal for photos with busy/non-flat
# backgrounds (rugs, grass, clutter) that the simple corner-color heuristic
# in Stage 1 can't detect. Isolates the subject, composites it onto a plain
# color, and crops tightly so the design isn't mostly empty canvas.
# ---------------------------------------------------------------------------
def remove_background(input_path, out_path, fill_rgb=(255, 255, 255), padding_frac=0.04):
    from rembg import remove
    img = Image.open(input_path).convert("RGB")
    cut = remove(img)  # RGBA, subject opaque, background transparent

    alpha = np.array(cut.split()[-1])
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        raise RuntimeError("Background removal found no subject in this image.")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad_x = int((x1 - x0) * padding_frac)
    pad_y = int((y1 - y0) * padding_frac)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(cut.width, x1 + pad_x)
    y1 = min(cut.height, y1 + pad_y)

    bg = Image.new("RGB", cut.size, fill_rgb)
    bg.paste(cut, mask=cut.split()[-1])
    cropped = bg.crop((x0, y0, x1, y1))
    cropped.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Stage 1: preprocess the uploaded image
# ---------------------------------------------------------------------------
def snap_to_palette(img, palette_hex):
    """Remap every pixel to the nearest color in palette_hex (list of '#rrggbb')."""
    arr = np.asarray(img, dtype=np.float32)  # H x W x 3
    pal = np.array([_hex_to_rgb(h) for h in palette_hex], dtype=np.float32)  # N x 3
    flat = arr.reshape(-1, 3)
    # distance from every pixel to every palette color
    dists = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)  # (pixels, N)
    nearest = np.argmin(dists, axis=1)
    snapped = pal[nearest].reshape(arr.shape).astype(np.uint8)
    return Image.fromarray(snapped, "RGB")


def _weighted_colors(img, n_colors):
    """Quantize to n_colors and return [(rgb_tuple, pixel_count), ...]."""
    quant = img.convert("P", palette=Image.ADAPTIVE, colors=n_colors)
    pal = quant.getpalette()
    return [
        (tuple(pal[idx * 3: idx * 3 + 3]), count)
        for count, idx in quant.getcolors()
    ]


def merge_close_colors(weighted_colors, max_colors, min_distance):
    """Agglomeratively merge the closest pair of colors (population-weighted
    centroid) until at most max_colors remain AND every remaining pair is
    at least min_distance apart (best-effort — max_colors always wins if
    the two constraints conflict, so a tiny max_colors can still force
    colors closer than min_distance).

    This is what actually solves "ten shades of grey": plain adaptive
    quantization picks colors by how much screen area they cover, with no
    idea of how visually similar two picked colors are — fine detail/noise
    in a photo can easily earn itself 4-5 near-identical grays as separate
    palette entries. Merging by distance collapses those into one real
    color instead of leaving them to fragment into speckly thread-color
    changes downstream.
    """
    clusters = [
        {"rgb": np.array(rgb, dtype=np.float64), "weight": float(count)}
        for rgb, count in weighted_colors
    ]
    if min_distance <= 0:
        # No merging requested — just cap at max_colors by combining the
        # smallest/least-significant clusters first (population based).
        min_distance = -1  # never triggers the distance branch below

    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = float(np.linalg.norm(clusters[i]["rgb"] - clusters[j]["rgb"]))
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        if len(clusters) <= max_colors and d >= min_distance:
            break
        wi, wj = clusters[i]["weight"], clusters[j]["weight"]
        clusters[i]["rgb"] = (clusters[i]["rgb"] * wi + clusters[j]["rgb"] * wj) / (wi + wj)
        clusters[i]["weight"] = wi + wj
        del clusters[j]

    return ["#%02x%02x%02x" % tuple(int(round(v)) for v in c["rgb"]) for c in clusters]


def preprocess_image(input_path, out_path, resize_max=800, max_colors=8, palette=None,
                      min_color_distance=0, thread_catalog=None):
    """Resize + flatten to a limited color palette so vectorization produces
    clean flat regions instead of thousands of tiny photo-noise shapes.

    palette: optional list of hex color strings (e.g. ["#dc1e1e", "#1450c8"]).
    If given, every pixel is snapped to the NEAREST of these exact colors
    instead of an auto-picked palette — use this to match real thread colors
    you own.

    min_color_distance: in the auto-pick path, minimum Euclidean RGB
    distance (0-441) required between any two chosen colors. 0 disables
    this (old behavior: just pick max_colors by population). A positive
    value (try 40-80) merges near-duplicate colors (e.g. several shades of
    gray from fur/texture noise) into one before they ever reach vtracer.

    thread_catalog: optional name of a real manufacturer thread chart
    shipped with Ink/Stitch (e.g. "Brother Embroidery", "Madeira
    Polyneon") — see list_thread_catalogs(). When given (and `palette` is
    not), the auto-picked working palette (honoring max_colors /
    min_color_distance above) gets snapped to the nearest real thread in
    that catalog before the image is flattened, so every design ends up
    using only colors you can actually buy/own, consistently image to
    image. Returns thread_matches as a 5th tuple element (None otherwise).
    """
    img = Image.open(input_path).convert("RGB")

    # Resize (longest side -> resize_max), keep aspect ratio
    w, h = img.size
    scale = resize_max / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    # Mild smoothing to kill JPEG noise/speckle before quantizing
    img = img.filter(__import__("PIL.ImageFilter", fromlist=["ImageFilter"]).SMOOTH)

    thread_matches = None

    if palette:
        flat = snap_to_palette(img, palette)
    elif thread_catalog:
        # Build the usual working palette first (same logic as the
        # min_color_distance / plain-adaptive paths), THEN snap those
        # representative colors onto the real thread chart — snapping
        # straight from raw pixels to a 60+ entry catalog would just
        # reproduce the "many near-duplicate colors" problem one level up.
        intermediate_n = min(64, max(max_colors * 4, 16))
        weighted = _weighted_colors(img, intermediate_n)
        working_palette = merge_close_colors(weighted, max_colors, min_color_distance)
        catalog_path = find_catalog_file(thread_catalog)
        catalog_entries = parse_gpl(catalog_path)
        thread_palette, thread_matches = snap_palette_to_catalog(working_palette, catalog_entries)
        flat = snap_to_palette(img, thread_palette)
    elif min_color_distance > 0:
        # Quantize to extra colors first so fine shade variation is
        # captured, THEN merge near-duplicates down by distance (see
        # merge_close_colors). Quantizing straight to max_colors would
        # already have thrown away the information needed to tell "two
        # genuinely different colors" apart from "one color split in two".
        intermediate_n = min(64, max(max_colors * 4, 16))
        weighted = _weighted_colors(img, intermediate_n)
        merged_palette = merge_close_colors(weighted, max_colors, min_color_distance)
        flat = snap_to_palette(img, merged_palette)
    else:
        # Posterize: reduce to N dominant colors (adaptive palette)
        quant = img.convert("P", palette=Image.ADAPTIVE, colors=max_colors)
        flat = quant.convert("RGB")

    flat.save(out_path)

    # The exact set of colors actually present in the flattened image — this
    # becomes the "known good" palette that downstream vtracer output gets
    # snapped back to, since vtracer's own clustering can drift away from
    # these exact values at shape boundaries.
    arr = np.asarray(flat).reshape(-1, 3)
    unique_colors = np.unique(arr, axis=0)
    actual_palette = ["#%02x%02x%02x" % tuple(c) for c in unique_colors]

    # Sample the 4 corners to guess the "background" color
    corners = [flat.getpixel((0, 0)), flat.getpixel((flat.width - 1, 0)),
               flat.getpixel((0, flat.height - 1)), flat.getpixel((flat.width - 1, flat.height - 1))]
    bg_counts = {}
    for c in corners:
        bg_counts[c] = bg_counts.get(c, 0) + 1
    bg_color = max(bg_counts, key=bg_counts.get)  # most common corner color
    return out_path, bg_color, flat.size, actual_palette, thread_matches


# ---------------------------------------------------------------------------
# Stage 2: vectorize (raster -> SVG) with vtracer
# ---------------------------------------------------------------------------
def vectorize(image_path, svg_path, color_precision=6, filter_speckle=4, mode="polygon"):
    import vtracer
    vtracer.convert_image_to_svg_py(
        str(image_path),
        str(svg_path),
        colormode="color",
        hierarchical="stacked",
        mode=mode,
        filter_speckle=filter_speckle,
        color_precision=color_precision,
        corner_threshold=60,
        length_threshold=4.0,
        splice_threshold=45,
        path_precision=3,
    )
    return svg_path


def _hex_to_rgb(hexstr):
    hexstr = hexstr.lstrip("#")
    return tuple(int(hexstr[i:i + 2], 16) for i in (0, 2, 4))


def _color_dist(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


# ---------------------------------------------------------------------------
# Stage 3: clean the SVG — drop background fills + tiny speckle paths
# ---------------------------------------------------------------------------
def clean_svg(svg_path, cleaned_path, bg_color, bg_threshold=20, min_area_px=12,
              canvas_cover_ratio=0.9, width=None, height=None, target_palette=None,
              snap_threshold=40):
    """
    - Removes any path whose fill color is close to the detected background
      color AND whose bounding box covers most of the canvas (i.e. it's the
      backdrop rectangle vtracer always draws first).
    - Removes tiny speckle paths below a pixel-area threshold.
    - Snaps each remaining path's fill to the nearest color in
      target_palette (if given) before grouping. vtracer's internal
      layering/stacking can emit slightly drifted near-duplicate hex values
      for what is logically one color (e.g. #F9F9F3 vs #EAE8C9 at shape
      boundaries); without this snap those drifted colors each become their
      own "color" and Ink/Stitch ends up doing far more thread changes than
      the design actually needs — this is what causes the
      "too many color changes" export failure on real (non-flat) photos.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    svg_w = width or float(root.get("width", 0))
    svg_h = height or float(root.get("height", 0))
    canvas_area = svg_w * svg_h

    removed_bg = 0
    removed_speckle = 0
    kept = 0

    target_rgb_palette = [_hex_to_rgb(h) for h in target_palette] if target_palette else None

    def snap_rgb(rgb):
        """Snap an observed fill color to the nearest color in the known
        target palette, so vtracer's internal color drift collapses back
        onto the small set of colors we actually intended to use. No
        distance cutoff: every fill in this SVG originated from our own
        already-flattened source image, so there is no legitimate color
        that *isn't* one of these — always snap to the closest match."""
        if not target_rgb_palette:
            return rgb
        return min(target_rgb_palette, key=lambda p: _color_dist(rgb, p))

    # ElementTree can't easily compute path bbox from 'd' without a real
    # parser; approximate using the numbers in the 'd' string (cheap but
    # good enough for this filter's purpose: find min/max x/y).
    def path_bbox(d):
        nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", d)]
        xs = nums[0::2]
        ys = nums[1::2]
        if not xs or not ys:
            return 0, 0, 0, 0
        return min(xs), min(ys), max(xs), max(ys)

    children = list(root)
    kept_elements = []  # (fill_rgb, first_seen_order, element)
    non_path_elements = []
    order = 0
    for el in children:
        tag = el.tag.split("}")[-1]
        root.remove(el)
        if tag != "path":
            non_path_elements.append(el)
            continue
        fill = el.get("fill", "")
        d = el.get("d", "")
        if not fill.startswith("#"):
            kept_elements.append(((999, 999, 999), order, el))
            order += 1
            continue
        rgb = _hex_to_rgb(fill)
        rgb = snap_rgb(rgb)
        x0, y0, x1, y1 = path_bbox(d)
        bw, bh = (x1 - x0), (y1 - y0)
        area = bw * bh

        is_bg_colored = _color_dist(rgb, bg_color) < bg_threshold
        covers_canvas = canvas_area > 0 and area >= canvas_area * canvas_cover_ratio
        # vtracer always draws a full-canvas backdrop rectangle as the very
        # first path in the document, in hierarchical='stacked' mode — but
        # its actual fill color can drift noticeably from the true detected
        # background (observed drift of 60+ RGB units, well past
        # bg_threshold), especially once that color has also been snapped
        # to a thread catalog. Coverage alone is a reliable enough signal
        # for this one specific element (order 0): don't require color
        # closeness for it.
        is_leading_backdrop = (order == 0 and covers_canvas)

        if (is_bg_colored and covers_canvas) or is_leading_backdrop:
            removed_bg += 1
            continue
        if area < min_area_px:
            removed_speckle += 1
            continue
        el.set("fill", "#%02x%02x%02x" % rgb)
        kept_elements.append((rgb, order, el))
        order += 1
        kept += 1

    # Group same-colored shapes together so Ink/Stitch doesn't switch thread
    # color back and forth between interleaved same-color regions. Color
    # groups are ordered by the position their color first appeared, and
    # shapes within a group keep their original relative order.
    first_seen = {}
    for rgb, order_idx, _ in kept_elements:
        if rgb not in first_seen:
            first_seen[rgb] = order_idx
    kept_elements.sort(key=lambda t: (first_seen[t[0]], t[1]))

    for el in non_path_elements:
        root.append(el)
    for _, _, el in kept_elements:
        root.append(el)

    color_groups = len(first_seen)

    tree.write(cleaned_path, xml_declaration=True, encoding="UTF-8")
    return {
        "kept": kept,
        "removed_background": removed_bg,
        "removed_speckle": removed_speckle,
        "color_groups": color_groups,
    }


# ---------------------------------------------------------------------------
# Stage 4: Ink/Stitch headless — SVG -> embroidery formats (zip)
# ---------------------------------------------------------------------------
DEFAULT_FORMATS = ["dst", "pes", "threadlist"]

def run_inkstitch(svg_path, out_zip_path, inkstitch_exe, formats=DEFAULT_FORMATS, custom_file_name=None):
    args = [inkstitch_exe, "--extension=zip"]
    if custom_file_name:
        args.append(f"--custom-file-name={custom_file_name}")
    for fmt in formats:
        if fmt == "threadlist":
            args.append("--format-threadlist=True")
        else:
            args.append(f"--format-{fmt}=True")
    args.append(str(svg_path))

    with open(out_zip_path, "wb") as out_f:
        proc = subprocess.run(args, stdout=out_f, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"inkstitch.exe failed ({proc.returncode}): {proc.stderr.decode(errors='replace')}")
    if os.path.getsize(out_zip_path) == 0:
        raise RuntimeError("inkstitch.exe produced an empty file — conversion likely failed.")
    return out_zip_path


def extract_zip(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
        return z.namelist()


# ---------------------------------------------------------------------------
# Stage 5: preview + QA using pyembroidery
# ---------------------------------------------------------------------------
def render_preview_and_qa(dst_path, preview_png_path, hoop_mm=None, pes_path=None):
    import pyembroidery

    # DST carries no thread-color data at all (its threadlist comes back
    # empty), so rendering straight from the .dst forces pyembroidery to
    # guess colors from its internal catalog -- producing wrong hues (e.g. a
    # brown/cream cat rendered green). PES embeds the real per-block RGB
    # that Ink/Stitch wrote from the SVG fills, so prefer it for the preview
    # whenever it's available. Stitch geometry is identical either way.
    color_source = pes_path if (pes_path and Path(pes_path).exists()) else None
    pattern = pyembroidery.read(str(color_source or dst_path))
    pyembroidery.write_png(pattern, str(preview_png_path), settings={"background": "#ffffff"})

    stitches = pattern.stitches
    xs = [s[0] for s in stitches]
    ys = [s[1] for s in stitches]
    width_units = (max(xs) - min(xs)) if xs else 0
    height_units = (max(ys) - min(ys)) if ys else 0
    # pyembroidery stitch coords are in 0.1mm units
    width_mm = width_units / 10.0
    height_mm = height_units / 10.0

    # color count: count COLOR_CHANGE commands + 1
    color_changes = sum(1 for s in stitches if s[2] == pyembroidery.COLOR_CHANGE)
    colors = color_changes + 1

    report = {
        "stitch_count": len(stitches),
        "width_mm": round(width_mm, 1),
        "height_mm": round(height_mm, 1),
        "color_count": colors,
        "warnings": [],
    }

    if hoop_mm:
        hoop_w, hoop_h = hoop_mm
        if width_mm > hoop_w or height_mm > hoop_h:
            report["warnings"].append(
                f"Design {width_mm}x{height_mm}mm exceeds hoop {hoop_w}x{hoop_h}mm"
            )
    if len(stitches) > 60000:
        report["warnings"].append(f"Very high stitch count ({len(stitches)}) — expect a long run time")
    if colors > 15:
        report["warnings"].append(f"High color count ({colors}) — many thread changes")

    return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def sanitize_name(name):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return name or "design"


def run_pipeline(input_image, out_dir, resize_max=800, max_colors=8,
                  formats=DEFAULT_FORMATS, hoop=None, inkstitch_exe=None,
                  mode="polygon", palette=None, progress_cb=None, remove_bg=False,
                  min_color_distance=0, thread_catalog=None):
    import datetime

    base_name = sanitize_name(Path(input_image).stem)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{base_name}_{timestamp}"

    out_dir = Path(out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    inkstitch_exe = inkstitch_exe or find_inkstitch_exe()

    def step(msg):
        if progress_cb:
            progress_cb(msg)

    log = {"run_name": run_name, "run_dir": str(out_dir), "source_image": str(input_image), "stages": {}}

    # Stage 0 (optional)
    if remove_bg:
        step("Removing background (AI subject isolation)...")
        nobg_path = out_dir / "00_background_removed.png"
        remove_background(input_image, nobg_path)
        log["stages"]["remove_background"] = {"output": str(nobg_path)}
        stage1_input = nobg_path
    else:
        stage1_input = input_image

    # Stage 1
    step("Preprocessing image...")
    pre_path = out_dir / "01_preprocessed.png"
    _, bg_color, size, actual_palette, thread_matches = preprocess_image(
        stage1_input, pre_path, resize_max, max_colors, palette=palette,
        min_color_distance=min_color_distance, thread_catalog=thread_catalog,
    )
    log["stages"]["preprocess"] = {"output": str(pre_path), "bg_color_rgb": bg_color, "size": size,
                                    "palette": actual_palette, "min_color_distance": min_color_distance}
    if thread_catalog:
        log["stages"]["preprocess"]["thread_catalog"] = thread_catalog
    if thread_matches:
        log["stages"]["preprocess"]["thread_matches"] = thread_matches

    # Fit-to-hoop: SVG units map to mm at 96 DPI (1px = 0.264583mm), so if a
    # hoop size is given, shrink the image now rather than just warning
    # about an oversized design after the fact.
    if hoop:
        hoop_w, hoop_h = hoop
        px_per_mm = 96 / 25.4
        max_w_px = hoop_w * px_per_mm
        max_h_px = hoop_h * px_per_mm
        scale = min(max_w_px / size[0], max_h_px / size[1], 1.0)
        if scale < 1.0:
            step(f"Scaling design down {scale:.2f}x to fit the {hoop_w}x{hoop_h}mm hoop...")
            img = Image.open(pre_path).convert("RGB")
            new_size = (max(1, int(size[0] * scale)), max(1, int(size[1] * scale)))
            img = img.resize(new_size, Image.LANCZOS)
            img = snap_to_palette(img, actual_palette)  # LANCZOS reintroduces blended edge colors — re-snap
            img.save(pre_path)
            size = new_size
            log["stages"]["preprocess"]["fit_to_hoop_scale"] = round(scale, 3)
            log["stages"]["preprocess"]["size_after_hoop_fit"] = size

    # Stage 2
    step("Vectorizing (image -> SVG shapes)...")
    svg_raw = out_dir / "02_vectorized.svg"
    color_count = len(palette) if palette else max_colors
    vectorize(pre_path, svg_raw, color_precision=min(8, max(1, color_count.bit_length())), mode=mode)
    log["stages"]["vectorize"] = {"output": str(svg_raw)}

    # Stage 3
    step("Cleaning SVG (removing background/speckle)...")
    svg_clean = out_dir / "03_cleaned.svg"
    clean_stats = clean_svg(svg_raw, svg_clean, bg_color, width=size[0], height=size[1],
                             target_palette=actual_palette)
    log["stages"]["clean_svg"] = {"output": str(svg_clean), **clean_stats}

    # Stage 4
    step("Converting to embroidery stitches (Ink/Stitch)...")
    zip_path = out_dir / f"{base_name}_embroidery.zip"
    run_inkstitch(svg_clean, zip_path, inkstitch_exe, formats=formats, custom_file_name=base_name)
    extracted_dir = out_dir / "files"
    files = extract_zip(zip_path, extracted_dir)
    log["stages"]["inkstitch"] = {"zip": str(zip_path), "files": files}

    # Stage 5
    step("Rendering preview + running QA checks...")
    dst_candidates = list(extracted_dir.glob("*.dst"))
    pes_candidates = list(extracted_dir.glob("*.pes"))
    report = None
    preview_path = out_dir / "06_preview.png"
    if dst_candidates:
        report = render_preview_and_qa(
            dst_candidates[0], preview_path, hoop_mm=hoop,
            pes_path=pes_candidates[0] if pes_candidates else None,
        )
        if not pes_candidates:
            report.setdefault("warnings", []).append(
                "No .pes exported — preview colors are guessed, not true thread colors. "
                "Add 'pes' to --formats for an accurate preview."
            )
        log["stages"]["preview_qa"] = {"preview": str(preview_path), "report": report}
    else:
        log["stages"]["preview_qa"] = {"error": "no .dst file found to preview — add --formats dst"}

    log_path = out_dir / "pipeline_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    step("Done.")
    return log


def parse_hoop(s):
    if not s:
        return None
    m = re.match(r"(\d+(\.\d+)?)x(\d+(\.\d+)?)", s)
    if not m:
        raise argparse.ArgumentTypeError("hoop must look like 130x180")
    return float(m.group(1)), float(m.group(3))


def main():
    ap = argparse.ArgumentParser(description="Image -> embroidery design pipeline")
    ap.add_argument("input_image", nargs="?")
    ap.add_argument("output_dir", nargs="?")
    ap.add_argument("--resize", type=int, default=800, help="max dimension in px before vectorizing")
    ap.add_argument("--max-colors", type=int, default=8, help="number of flat colors to posterize to")
    ap.add_argument("--min-color-distance", type=float, default=0,
                     help="merge auto-picked colors closer than this RGB distance (0-441); "
                          "try 40-80 to avoid several near-identical shades of gray from fur/texture noise")
    ap.add_argument("--palette", nargs="+", default=None,
                     help="exact hex thread colors to snap to, e.g. --palette '#dc1e1e' '#1450c8'")
    ap.add_argument("--thread-catalog", default=None,
                     help="snap colors to a real manufacturer thread chart shipped with Ink/Stitch, "
                          "e.g. --thread-catalog 'Brother Embroidery'. See --list-thread-catalogs.")
    ap.add_argument("--list-thread-catalogs", action="store_true",
                     help="print all available --thread-catalog names and exit")
    ap.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS,
                     help="embroidery formats to export, e.g. dst pes jef vp3 threadlist")
    ap.add_argument("--remove-bg", action="store_true",
                     help="AI-remove the background first (for photos with busy/non-flat backgrounds)")
    ap.add_argument("--hoop", type=parse_hoop, default=None, help="hoop size in mm, e.g. 130x180")
    ap.add_argument("--inkstitch-exe", default=None, help="path to inkstitch.exe (auto-detected if omitted)")
    ap.add_argument("--mode", default="polygon", choices=["polygon", "spline", "pixel"],
                     help="vtracer curve fitting mode")
    args = ap.parse_args()

    if args.list_thread_catalogs:
        for name in list_thread_catalogs():
            print(name)
        return

    if not args.input_image or not args.output_dir:
        ap.error("input_image and output_dir are required (unless using --list-thread-catalogs)")

    log = run_pipeline(
        args.input_image, args.output_dir,
        resize_max=args.resize, max_colors=args.max_colors,
        formats=args.formats, hoop=args.hoop,
        inkstitch_exe=args.inkstitch_exe, mode=args.mode,
        palette=args.palette, remove_bg=args.remove_bg,
        min_color_distance=args.min_color_distance,
        thread_catalog=args.thread_catalog,
    )

    print(json.dumps(log, indent=2, default=str))

    qa = log["stages"].get("preview_qa", {})
    if "report" in qa:
        r = qa["report"]
        print(f"\n✅ Done. {r['stitch_count']} stitches, {r['color_count']} colors, "
              f"{r['width_mm']}x{r['height_mm']}mm")
        for w in r.get("warnings", []):
            print(f"⚠️  {w}")
    else:
        print("\n⚠️  No preview/QA generated:", qa.get("error"))


if __name__ == "__main__":
    main()
