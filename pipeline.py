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

Requires (pip install -r requirements.txt, Python 3.12):
    pillow numpy vtracer pyembroidery
    (optional, for --remove-bg: rembg onnxruntime — see requirements-optional.txt)

Requires on the system:
    Ink/Stitch (its `inkstitch` / `inkstitch.exe` binary ships inside the
    Inkscape extension folder). Auto-detected on Windows, macOS and Linux;
    override with --inkstitch-exe or the INKSTITCH_EXE environment variable.
"""
import argparse
import datetime
import functools
import json
import os
import re
import shutil
import subprocess
import traceback
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image, ImageColor, ImageFilter
import numpy as np

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

# SVG user units without a physical size are interpreted as CSS px (96 DPI).
MM_PER_PX_96DPI = 25.4 / 96

# vtracer hierarchical modes. "stacked" draws big shapes underneath smaller
# ones (no holes cut out, so no gaps between regions, but overlapping areas
# get stitched more than once). "cutout" produces non-overlapping shapes
# (fewer stitches / less stiffness, but fabric pull can open small gaps).
LAYERING_MODES = ("stacked", "cutout")


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
_BARE_HEX_RE = re.compile(r"^(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def parse_color(value):
    """Parse '#rrggbb', '#rgb', 'rrggbb' or a CSS color name -> (r, g, b).
    Raises ValueError with a readable message on bad input."""
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return tuple(int(v) for v in value[:3])
    text = str(value).strip()
    if _BARE_HEX_RE.match(text):
        text = "#" + text
    try:
        rgb = ImageColor.getrgb(text)
    except ValueError:
        raise ValueError(
            f"Invalid color {value!r}: use #rrggbb, #rgb or a color name like 'red'"
        ) from None
    return tuple(rgb[:3])


def normalize_hex(value):
    return "#%02x%02x%02x" % parse_color(value)


def _hex_to_rgb(hexstr):
    return parse_color(hexstr)


def _color_dist(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def _rgb_to_lab(rgb):
    """sRGB (..., 3) in 0-255 -> CIELAB (D65). Used for perceptual thread
    matching: plain RGB distance badly over-weights differences in some hues
    (e.g. greens) and under-weights others (blues/darks)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


# ---------------------------------------------------------------------------
# Locate Ink/Stitch (binary + thread palettes)
# ---------------------------------------------------------------------------
INKSTITCH_EXE_NAMES = ("inkstitch.exe", "inkstitch")


def _inkscape_extension_dirs():
    """Folders that may contain an `inkstitch` extension folder, across
    Windows, macOS (incl. app bundle) and Linux (incl. Flatpak/Snap)."""
    home = Path.home()
    dirs = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / "inkscape" / "extensions")
    dirs += [
        home / "AppData/Roaming/inkscape/extensions",
        home / "AppData/Local",  # standalone installs: AppData/Local/inkstitch/...
        home / "Library/Application Support/org.inkscape.Inkscape/config/inkscape/extensions",
        Path("/Applications/Inkscape.app/Contents/Resources/share/inkscape/extensions"),
        home / ".config/inkscape/extensions",
        home / ".var/app/org.inkscape.Inkscape/config/inkscape/extensions",
        home / "snap/inkscape/common/.config/inkscape/extensions",
        Path("/usr/share/inkscape/extensions"),
        Path("/usr/local/share/inkscape/extensions"),
    ]
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _bounded_find(root, names, max_depth=4):
    """Depth-limited search for a file named one of `names` under root.
    (A full rglob over e.g. AppData can take minutes.)"""
    root = Path(root)
    if not root.is_dir():
        return None
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base_depth
        for n in names:
            if n in filenames:
                return Path(dirpath) / n
        if depth >= max_depth:
            dirnames[:] = []
    return None


@functools.lru_cache(maxsize=None)
def find_inkstitch_exe():
    env = os.environ.get("INKSTITCH_EXE")
    if env:
        if Path(env).is_file():
            return env
        raise FileNotFoundError(f"INKSTITCH_EXE is set but does not exist: {env}")
    ext_dirs = _inkscape_extension_dirs()
    for ext in ext_dirs:
        for sub in ("inkstitch/bin", "inkstitch/inkstitch/bin"):
            for name in INKSTITCH_EXE_NAMES:
                p = ext / sub / name
                if p.is_file():
                    return str(p)
    for ext in ext_dirs:
        hit = _bounded_find(ext / "inkstitch", INKSTITCH_EXE_NAMES, max_depth=4)
        if hit:
            return str(hit)
    on_path = shutil.which("inkstitch")
    if on_path:
        return on_path
    raise FileNotFoundError(
        "Could not find the Ink/Stitch binary (inkstitch.exe / inkstitch). "
        "Pass --inkstitch-exe or set the INKSTITCH_EXE environment variable."
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
def _has_inkstitch_palettes(d):
    try:
        return d.is_dir() and any(p.name.lower().startswith("inkstitch") for p in d.glob("*.gpl"))
    except OSError:
        return False


@functools.lru_cache(maxsize=None)
def find_palettes_dir():
    env = os.environ.get("INKSTITCH_PALETTES")
    if env and Path(env).is_dir():
        return Path(env)
    candidates = []
    try:
        exe = Path(find_inkstitch_exe())
        candidates += [exe.parent.parent / "palettes", exe.parent.parent.parent / "palettes"]
    except FileNotFoundError:
        pass
    for ext in _inkscape_extension_dirs():
        candidates += [
            ext / "inkstitch" / "palettes",
            ext / "inkstitch" / "inkstitch" / "palettes",
            # Ink/Stitch's "Install thread color palettes" copies them into
            # Inkscape's own user palette folder (sibling of extensions/).
            ext.parent / "palettes",
        ]
    for c in candidates:
        if _has_inkstitch_palettes(c):
            return c
    for ext in _inkscape_extension_dirs():
        root = ext / "inkstitch"
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, _ in os.walk(root):
            if Path(dirpath).name == "palettes" and _has_inkstitch_palettes(Path(dirpath)):
                return Path(dirpath)
            if len(Path(dirpath).parts) - base_depth >= 4:
                dirnames[:] = []
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
        raise FileNotFoundError(
            "Could not find Ink/Stitch's thread palettes folder. "
            "Set INKSTITCH_PALETTES to the folder containing the InkStitch *.gpl files."
        )
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


_GPL_LINE_RE = re.compile(r"^\s*(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})(?:\s+(.*))?$")


def parse_gpl(path):
    """Parse a GIMP-palette .gpl file -> [{'rgb': (r,g,b), 'name': str, 'number': str}, ...]

    Ink/Stitch's catalogs are `R G B<TAB>Name<TAB>Number`, so tabs are the
    authoritative separator (names contain spaces: "Light Blue"). For
    space-only lines, the last word is treated as a catalog number only if
    it contains a digit — otherwise multi-word names would be split."""
    entries = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("GIMP Palette") \
                    or stripped.startswith("Name:") or stripped.startswith("Columns:"):
                continue
            m = _GPL_LINE_RE.match(line)
            if not m:
                continue
            rgb = tuple(int(m.group(i)) for i in (1, 2, 3))
            if any(v > 255 for v in rgb):
                continue
            rest = m.group(4) or ""
            if "\t" in rest:
                fields = [p.strip() for p in rest.split("\t")]
                fields = [p for p in fields if p] or [""]
                name = fields[0]
                number = fields[1] if len(fields) > 1 else ""
            else:
                words = rest.split()
                if len(words) > 1 and re.search(r"\d", words[-1]):
                    name, number = " ".join(words[:-1]), words[-1]
                else:
                    name, number = " ".join(words), ""
            entries.append({"rgb": rgb, "name": name.strip(), "number": number.strip()})
    return entries


def snap_palette_to_catalog(working_palette_hex, catalog_entries):
    """Map each color in working_palette_hex (list of '#rrggbb') to the
    perceptually nearest (CIELAB ΔE) real thread in catalog_entries.
    Returns (new_hex_list, matches) where matches is
    [{'from': hex, 'to': hex, 'name':, 'number':, 'delta_e':}, ...] — the
    'to' list is deduped (two source colors can land on the same thread) so
    the final palette may have fewer entries than the input."""
    if not catalog_entries:
        raise ValueError("Thread catalog is empty or could not be parsed.")
    cat_lab = _rgb_to_lab(np.array([e["rgb"] for e in catalog_entries], dtype=np.float64))
    matches = []
    seen = set()
    new_hex = []
    for hx in working_palette_hex:
        src_lab = _rgb_to_lab(np.array(_hex_to_rgb(hx), dtype=np.float64))
        dists = np.linalg.norm(cat_lab - src_lab, axis=1)
        idx = int(np.argmin(dists))
        best = catalog_entries[idx]
        to_hex = "#%02x%02x%02x" % best["rgb"]
        matches.append({"from": hx, "to": to_hex, "name": best["name"], "number": best["number"],
                        "delta_e": round(float(dists[idx]), 1)})
        if to_hex not in seen:
            seen.add(to_hex)
            new_hex.append(to_hex)
    return new_hex, matches


# ---------------------------------------------------------------------------
# Stage 0 (optional): AI background removal for photos with busy/non-flat
# backgrounds (rugs, grass, clutter) that the simple corner-color heuristic
# in Stage 1 can't detect. Isolates the subject, composites it onto a plain
# color, and crops tightly so the design isn't mostly empty canvas.
#
# Note: rembg downloads its segmentation model (~170 MB) on first use into
# ~/.u2net (override with the U2NET_HOME environment variable), so the first
# run needs an internet connection.
# ---------------------------------------------------------------------------
def remove_background(input_path, out_path, fill_rgb=(255, 255, 255), padding_frac=0.04):
    try:
        from rembg import remove
    except ImportError as e:
        raise RuntimeError(
            "Background removal needs the optional 'rembg' and 'onnxruntime' packages "
            "(pip install -r requirements-optional.txt)."
        ) from e
    img = Image.open(input_path).convert("RGB")
    cut = remove(img)  # RGBA, subject opaque, background transparent

    alpha = np.array(cut.split()[-1])
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        raise RuntimeError("Background removal found no subject in this image.")
    x0, x1 = int(xs.min()), int(xs.max()) + 1  # crop box right/bottom are exclusive
    y0, y1 = int(ys.min()), int(ys.max()) + 1
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
def snap_to_palette(img, palette_hex, chunk=262144):
    """Remap every pixel to the nearest color in palette_hex (list of colors).
    Processed in chunks so memory stays bounded for large palettes/images."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)  # H x W x 3
    pal = np.array([_hex_to_rgb(h) for h in palette_hex], dtype=np.float32)  # N x 3
    flat = arr.reshape(-1, 3)
    nearest = np.empty(len(flat), dtype=np.int64)
    for start in range(0, len(flat), chunk):
        block = flat[start:start + chunk]
        dists = ((block[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
        nearest[start:start + chunk] = np.argmin(dists, axis=1)
    snapped = pal[nearest].reshape(arr.shape).astype(np.uint8)
    return Image.fromarray(snapped, "RGB")


def transition_pixel_mask(img, min_edge=60.0, reach=2):
    """True for anti-aliasing / blur ramp pixels: pixels whose color lies
    part-way along the straight line between the colors `reach` pixels to
    either side of them (horizontally or vertically) across a strong edge.

    Those blended edge colors are not real colors of the subject, but plain
    adaptive quantization happily spends palette slots on them — producing
    thin halo outlines that each cost a thread change. Texture (fur, stripes)
    is not affected: a dark stripe pixel is darker than both neighbours, not
    between them."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    mask = np.zeros((h, w), dtype=bool)
    r = reach
    for axis in (0, 1):
        if arr.shape[axis] <= 2 * r:
            continue
        if axis == 1:
            lo, mid, hi = arr[:, :-2 * r], arr[:, r:-r], arr[:, 2 * r:]
        else:
            lo, mid, hi = arr[:-2 * r], arr[r:-r], arr[2 * r:]
        seg = hi - lo
        len2 = (seg ** 2).sum(axis=2)
        strong = len2 > min_edge ** 2
        safe_len2 = np.where(len2 > 0, len2, 1.0)
        t = ((mid - lo) * seg).sum(axis=2) / safe_len2
        proj = lo + t[..., None] * seg
        off = np.sqrt(((mid - proj) ** 2).sum(axis=2))
        between = strong & (t > 0.1) & (t < 0.9) & (off < 0.25 * np.sqrt(len2))
        if axis == 1:
            mask[:, r:-r] |= between
        else:
            mask[r:-r] |= between
    return mask


def _palette_sample(img):
    """Pixels to choose the palette from: everything except edge-transition
    pixels (see transition_pixel_mask). Returned as a 1xN RGB image."""
    arr = np.asarray(img.convert("RGB"))
    mask = transition_pixel_mask(img)
    keep = ~mask
    if keep.sum() < 0.25 * keep.size:  # pathological (all edges) -> use everything
        keep[:] = True
    pixels = arr[keep]
    return Image.fromarray(pixels.reshape(1, -1, 3), "RGB")


def _weighted_colors(img, n_colors):
    """Quantize to n_colors and return [(rgb_tuple, pixel_count), ...]."""
    quant = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=n_colors)
    pal = quant.getpalette()
    return [
        (tuple(pal[idx * 3: idx * 3 + 3]), count)
        for count, idx in quant.getcolors(maxcolors=256)
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
        # No distance constraint: only cap the count at max_colors, still by
        # repeatedly merging the closest pair of colors.
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

    palette: optional list of colors (e.g. ["#dc1e1e", "#1450c8"]).
    If given, every pixel is snapped to the NEAREST of these exact colors
    instead of an auto-picked palette — use this to match real thread colors
    you own. Mutually exclusive with thread_catalog.

    min_color_distance: in the auto-pick path, minimum Euclidean RGB
    distance (0-441) required between any two chosen colors. 0 disables
    this (just pick max_colors by population). A positive value (try
    40-80) merges near-duplicate colors (e.g. several shades of gray from
    fur/texture noise) into one before they ever reach vtracer.

    thread_catalog: optional name of a real manufacturer thread chart
    shipped with Ink/Stitch (e.g. "Brother Embroidery", "Madeira
    Polyneon") — see list_thread_catalogs(). When given, the auto-picked
    working palette (honoring max_colors / min_color_distance above) gets
    snapped to the perceptually nearest real thread in that catalog before
    the image is flattened, so every design ends up using only colors you
    can actually buy/own, consistently image to image. Returns
    thread_matches as a 5th tuple element (None otherwise).

    In all auto-pick paths the palette is chosen from non-edge pixels only
    (see transition_pixel_mask), so anti-aliased outlines don't steal
    palette slots; edge pixels are then snapped to the nearest real color.
    """
    if palette and thread_catalog:
        raise ValueError("Use either an exact palette or a thread catalog, not both.")
    if max_colors < 1:
        raise ValueError("max_colors must be at least 1.")

    img = Image.open(input_path).convert("RGB")

    # Resize (longest side -> resize_max), keep aspect ratio
    w, h = img.size
    scale = resize_max / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    # Mild smoothing to kill JPEG noise/speckle before quantizing
    img = img.filter(ImageFilter.SMOOTH)

    thread_matches = None

    if palette:
        flat = snap_to_palette(img, [normalize_hex(c) for c in palette])
    else:
        sample = _palette_sample(img)
        if thread_catalog or min_color_distance > 0:
            # Quantize to extra colors first so fine shade variation is
            # captured, THEN merge near-duplicates down by distance (see
            # merge_close_colors). Quantizing straight to max_colors would
            # already have thrown away the information needed to tell "two
            # genuinely different colors" apart from "one color split in two".
            intermediate_n = min(64, max(max_colors * 4, 16))
            weighted = _weighted_colors(sample, intermediate_n)
            working_palette = merge_close_colors(weighted, max_colors, min_color_distance)
        else:
            # Posterize: N dominant colors (adaptive/median-cut palette)
            working_palette = ["#%02x%02x%02x" % rgb for rgb, _ in _weighted_colors(sample, max_colors)]

        if thread_catalog:
            # Snap the representative colors (not raw pixels) onto the real
            # thread chart — snapping straight from raw pixels to a 60+ entry
            # catalog would reproduce the "many near-duplicate colors"
            # problem one level up.
            catalog_entries = parse_gpl(find_catalog_file(thread_catalog))
            working_palette, thread_matches = snap_palette_to_catalog(working_palette, catalog_entries)
        flat = snap_to_palette(img, working_palette)

    flat.save(out_path)

    # The exact set of colors actually present in the flattened image — this
    # becomes the "known good" palette that downstream vtracer output gets
    # snapped back to, since vtracer's own clustering can drift away from
    # these exact values at shape boundaries.
    arr = np.asarray(flat).reshape(-1, 3)
    unique_colors = np.unique(arr, axis=0)
    actual_palette = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in unique_colors]

    # Sample the 4 corners to guess the "background" color
    corners = [flat.getpixel((0, 0)), flat.getpixel((flat.width - 1, 0)),
               flat.getpixel((0, flat.height - 1)), flat.getpixel((flat.width - 1, flat.height - 1))]
    bg_counts = {}
    for c in corners:
        bg_counts[c] = bg_counts.get(c, 0) + 1
    bg_color = max(bg_counts, key=bg_counts.get)  # most common corner color
    return out_path, bg_color, flat.size, actual_palette, thread_matches


# ---------------------------------------------------------------------------
# Hoop fitting — done by giving the SVG a physical size (mm) rather than by
# downsampling the raster, so vectorization keeps full resolution.
# ---------------------------------------------------------------------------
def compute_hoop_fit(size_px, hoop_mm=None, margin_mm=2.0):
    """Return a dict with the mm-per-pixel to use for the SVG and fit info.

    Without a hoop, the design keeps its natural 96-DPI size. With a hoop
    (width, height in mm), it is scaled DOWN (never up) to fit inside the
    hoop minus `margin_mm` on every side. If the design would come out
    noticeably larger with the hoop rotated 90°, a hint is returned."""
    w_mm = size_px[0] * MM_PER_PX_96DPI
    h_mm = size_px[1] * MM_PER_PX_96DPI
    info = {"natural_size_mm": (round(w_mm, 1), round(h_mm, 1)), "scale": 1.0,
            "mm_per_px": MM_PER_PX_96DPI, "hint": None}
    if not hoop_mm:
        info["size_mm"] = info["natural_size_mm"]
        return info
    hoop_w, hoop_h = hoop_mm
    usable_w = max(1.0, hoop_w - 2 * margin_mm)
    usable_h = max(1.0, hoop_h - 2 * margin_mm)
    scale = min(usable_w / w_mm, usable_h / h_mm, 1.0)
    rotated = min(usable_h / w_mm, usable_w / h_mm, 1.0)
    if rotated > scale * 1.05:
        info["hint"] = (
            f"Design would be {rotated / scale:.2f}x larger with the hoop rotated "
            f"(--hoop {hoop_h:g}x{hoop_w:g})."
        )
    info["scale"] = scale
    info["mm_per_px"] = MM_PER_PX_96DPI * scale
    info["size_mm"] = (round(w_mm * scale, 1), round(h_mm * scale, 1))
    return info


# ---------------------------------------------------------------------------
# Stage 2: vectorize (raster -> SVG) with vtracer
# ---------------------------------------------------------------------------
def vectorize(image_path, svg_path, color_precision=8, filter_speckle=4, mode="polygon",
              layering="stacked"):
    """color_precision is vtracer's *bits per RGB channel* (1-8), not a color
    count. The input is already flattened to an exact palette, so full
    precision (8) keeps every palette color distinct."""
    import vtracer
    if layering not in LAYERING_MODES:
        raise ValueError(f"layering must be one of {LAYERING_MODES}")
    vtracer.convert_image_to_svg_py(
        str(image_path),
        str(svg_path),
        colormode="color",
        hierarchical=layering,
        mode=mode,
        filter_speckle=filter_speckle,
        color_precision=color_precision,
        corner_threshold=60,
        length_threshold=4.0,
        splice_threshold=45,
        path_precision=3,
    )
    return svg_path


# ---------------------------------------------------------------------------
# Stage 3: clean the SVG — drop background fills + tiny speckle paths,
# snap colors, and group same-colored shapes without breaking layering.
# ---------------------------------------------------------------------------
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_PATH_TOKEN_RE = re.compile(rf"[A-Za-z]|{_NUM}")
_TRANSLATE_RE = re.compile(rf"translate\(\s*({_NUM})(?:[\s,]+({_NUM}))?\s*\)")


def parse_path_geometry(d, transform=""):
    """Approximate geometry of an SVG path as produced by vtracer
    (absolute M/L/C/Z, optionally inside translate()).

    Returns (bbox, area, perimeter) where bbox = (x0, y0, x1, y1) in absolute
    coordinates and includes Bezier control points (so it is conservative),
    and area/perimeter come from the on-curve points of the outer subpath."""
    tx = ty = 0.0
    m = _TRANSLATE_RE.search(transform or "")
    if m:
        tx = float(m.group(1))
        ty = float(m.group(2)) if m.group(2) else 0.0

    tokens = _PATH_TOKEN_RE.findall(d or "")
    all_pts = []
    subpaths = []
    cur = []
    cmd = None
    nums = []

    def flush_nums():
        nonlocal nums
        if cmd in ("C", "c"):
            for k in range(0, len(nums) - 5, 6):
                pts = [(nums[k + i], nums[k + i + 1]) for i in (0, 2, 4)]
                all_pts.extend(pts)
                cur.append(pts[-1])
        else:  # M, L and anything unexpected: treat numbers as point pairs
            for k in range(0, len(nums) - 1, 2):
                p = (nums[k], nums[k + 1])
                all_pts.append(p)
                cur.append(p)
        nums = []

    for tok in tokens:
        if tok.isalpha():
            flush_nums()
            if tok in ("M", "m") and cur:
                subpaths.append(cur)
                cur = []
            if tok in ("Z", "z"):
                if cur:
                    subpaths.append(cur)
                cur = []
            cmd = tok
        else:
            nums.append(float(tok))
    flush_nums()
    if cur:
        subpaths.append(cur)

    if not all_pts:
        return (0.0, 0.0, 0.0, 0.0), 0.0, 0.0
    xs = [p[0] + tx for p in all_pts]
    ys = [p[1] + ty for p in all_pts]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    best_area, best_perim = 0.0, 0.0
    for sp in subpaths:
        if len(sp) < 3:
            continue
        a = np.array(sp, dtype=np.float64)
        x, y = a[:, 0], a[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perim = float(np.sum(np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))))
        if area > best_area:
            best_area, best_perim = area, perim
    return bbox, float(best_area), best_perim


def _order_by_color(items, preserve_overlaps):
    """Reorder shapes to group same-colored ones (fewer thread changes).

    items: list of (color_key, bbox) in original drawing order.
    If preserve_overlaps, a shape is never moved past a differently-colored
    shape whose bounding box overlaps it — in vtracer's 'stacked' output the
    drawing order IS the layering (later shapes sit on top), and in
    embroidery later stitching covers earlier stitching, so moving a shape
    below one that overlapped it would bury it. Returns the new order as a
    list of indices into items."""
    n = len(items)
    if n == 0:
        return []
    keys = [k for k, _ in items]
    if not preserve_overlaps:
        first_seen = {}
        for i, k in enumerate(keys):
            first_seen.setdefault(k, i)
        return sorted(range(n), key=lambda i: (first_seen[keys[i]], i))

    import heapq
    key_ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    cid = np.array([key_ids[k] for k in keys])
    boxes = np.array([b for _, b in items], dtype=np.float64)
    succ = [None] * n
    indeg = np.zeros(n, dtype=np.int64)
    for i in range(n - 1):
        b = boxes[i + 1:]
        ov = ((b[:, 0] < boxes[i, 2]) & (b[:, 2] > boxes[i, 0]) &
              (b[:, 1] < boxes[i, 3]) & (b[:, 3] > boxes[i, 1]) &
              (cid[i + 1:] != cid[i]))
        js = np.nonzero(ov)[0] + i + 1
        succ[i] = js
        indeg[js] += 1
    succ[n - 1] = np.array([], dtype=np.int64)

    ready = {}
    for i in range(n):
        if indeg[i] == 0:
            heapq.heappush(ready.setdefault(cid[i], []), i)
    out = []
    current = None
    while len(out) < n:
        if current is None or not ready.get(current):
            # Switch to the color whose earliest ready shape comes first.
            current = min((h[0], c) for c, h in ready.items() if h)[1]
        i = heapq.heappop(ready[current])
        out.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(ready.setdefault(cid[j], []), int(j))
    return out


def clean_svg(svg_path, cleaned_path, bg_color, bg_threshold=20, canvas_cover_ratio=0.9,
              width=None, height=None, target_palette=None, mm_per_px=MM_PER_PX_96DPI,
              min_area_mm2=1.0, min_width_mm=0.4, layering="stacked"):
    """
    - Removes any path whose fill color is close to the detected background
      color AND whose bounding box covers most of the canvas. If vtracer's
      leading full-canvas rectangle is a design color, it is reduced to the
      region the background leaves visible (see below) rather than deleted.
    - Removes shapes too small/thin to embroider: outline area below
      min_area_mm2, or average thickness (2*area/perimeter) below
      min_width_mm, measured at the final physical size.
    - Snaps each remaining path's fill to the nearest color in
      target_palette (if given). vtracer's internal layering/stacking can
      emit slightly drifted near-duplicate hex values for what is logically
      one color (e.g. #F9F9F3 vs #EAE8C9 at shape boundaries); without this
      snap those drifted colors each become their own "color" and Ink/Stitch
      ends up doing far more thread changes than the design needs.
    - Groups same-colored shapes to cut thread changes, without moving any
      shape past an overlapping differently-colored one (for 'stacked'
      layering — see _order_by_color).
    - Gives the SVG a physical size in mm (width/height + viewBox) so
      Ink/Stitch produces the intended stitch-out size.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    svg_w = float(width or float(root.get("width", 0) or 0))
    svg_h = float(height or float(root.get("height", 0) or 0))
    canvas_area = svg_w * svg_h

    removed_bg = 0
    removed_speckle = 0
    kept = 0

    target_rgb_palette = [_hex_to_rgb(h) for h in target_palette] if target_palette else None

    def snap_rgb(rgb):
        """Snap an observed fill color to the nearest color in the known
        target palette. No distance cutoff: every fill in this SVG
        originated from our own already-flattened source image, so there is
        no legitimate color that *isn't* one of these."""
        if not target_rgb_palette:
            return rgb
        return min(target_rgb_palette, key=lambda p: _color_dist(rgb, p))

    px_per_mm = 1.0 / mm_per_px
    min_area_px = min_area_mm2 * px_per_mm ** 2
    min_width_px = min_width_mm * px_per_mm

    # Pass 1: classify every path.
    non_path_elements = []
    paths = []  # dicts in original drawing order
    for el in list(root):
        tag = el.tag.split("}")[-1] if isinstance(el.tag, str) else ""
        root.remove(el)
        if tag != "path":
            non_path_elements.append(el)
            continue
        fill = el.get("fill", "")
        bbox, area, perim = parse_path_geometry(el.get("d", ""), el.get("transform", ""))
        x0, y0, x1, y1 = bbox
        covers_canvas = canvas_area > 0 and (x1 - x0) * (y1 - y0) >= canvas_area * canvas_cover_ratio
        rgb = snap_rgb(_hex_to_rgb(fill)) if fill.startswith("#") else None
        is_bg = rgb is not None and covers_canvas and _color_dist(rgb, bg_color) < bg_threshold
        paths.append({"el": el, "rgb": rgb, "fill": fill, "bbox": bbox, "area": area,
                      "perim": perim, "covers_canvas": covers_canvas, "is_bg": is_bg})

    # vtracer's 'stacked' output always starts with a full-canvas rectangle
    # in the color of the outermost cluster. When that is the background it
    # is simply removed (is_bg). When it is a DESIGN color (e.g. a subject
    # that fills most of the frame), the background is drawn on top of it as
    # a canvas-sized path whose holes reveal the subject — so the visible
    # part of the leading rectangle is exactly those holes. Replace its
    # geometry with them (or drop it if the background has no holes) instead
    # of either stitching a full rectangle or deleting a real region.
    if paths and paths[0]["covers_canvas"] and not paths[0]["is_bg"] and paths[0]["rgb"] is not None:
        bg_path = next((p for p in paths[1:] if p["is_bg"]), None)
        if bg_path is not None:
            subpaths = [s.strip() for s in re.split(r"(?=[Mm])", bg_path["el"].get("d", "")) if s.strip()]
            lead = paths[0]
            if len(subpaths) > 1:
                lead["el"].set("d", " ".join(subpaths[1:]))
                transform = bg_path["el"].get("transform")
                if transform:
                    lead["el"].set("transform", transform)
                elif "transform" in lead["el"].attrib:
                    del lead["el"].attrib["transform"]
                lead["bbox"], lead["area"], lead["perim"] = parse_path_geometry(
                    lead["el"].get("d", ""), lead["el"].get("transform", ""))
                lead["covers_canvas"] = False
            else:
                lead["is_bg"] = True  # fully hidden under the background

    # Pass 2: filter + snap.
    kept_items = []  # (color_key, bbox, element) in original order
    for p in paths:
        el, rgb, bbox, area, perim = p["el"], p["rgb"], p["bbox"], p["area"], p["perim"]
        if rgb is None:
            kept_items.append((("other", p["fill"]), bbox, el))
            kept += 1
            continue
        if p["is_bg"]:
            removed_bg += 1
            continue
        thickness = (2.0 * area / perim) if perim > 0 else 0.0
        if area < min_area_px or thickness < min_width_px:
            removed_speckle += 1
            continue
        el.set("fill", "#%02x%02x%02x" % rgb)
        kept_items.append((rgb, bbox, el))
        kept += 1

    order = _order_by_color([(k, b) for k, b, _ in kept_items],
                            preserve_overlaps=(layering != "cutout"))

    for el in non_path_elements:
        root.append(el)
    color_changes = 0
    prev = None
    for i in order:
        key, _, el = kept_items[i]
        if prev is not None and key != prev:
            color_changes += 1
        prev = key
        root.append(el)

    if svg_w > 0 and svg_h > 0:
        root.set("viewBox", f"0 0 {svg_w:g} {svg_h:g}")
        root.set("width", f"{svg_w * mm_per_px:.3f}mm")
        root.set("height", f"{svg_h * mm_per_px:.3f}mm")

    tree.write(cleaned_path, xml_declaration=True, encoding="UTF-8")
    return {
        "kept": kept,
        "removed_background": removed_bg,
        "removed_speckle": removed_speckle,
        "color_groups": len({k for k, _, _ in kept_items}),
        "color_changes": color_changes,
        "layering": layering,
    }


# ---------------------------------------------------------------------------
# Stage 4: Ink/Stitch headless — SVG -> embroidery formats (zip)
# ---------------------------------------------------------------------------
DEFAULT_FORMATS = ["dst", "pes", "threadlist"]
NON_STITCH_FORMATS = {"threadlist"}
# Preferred order for the preview: formats that carry real RGB thread colors
# first; DST/EXP carry no colors at all.
PREVIEW_FORMAT_PREFERENCE = ["pes", "vp3", "jef", "dst", "exp"]
FORMATS_WITHOUT_COLOR = {"dst", "exp"}


def run_inkstitch(svg_path, out_zip_path, inkstitch_exe, formats=DEFAULT_FORMATS, custom_file_name=None,
                  timeout=900):
    args = [inkstitch_exe, "--extension=zip"]
    if custom_file_name:
        args.append(f"--custom-file-name={custom_file_name}")
    for fmt in formats:
        if fmt == "threadlist":
            args.append("--format-threadlist=True")
        else:
            args.append(f"--format-{fmt}=True")
    args.append(str(svg_path))

    try:
        with open(out_zip_path, "wb") as out_f:
            proc = subprocess.run(args, stdout=out_f, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Ink/Stitch did not finish within {timeout} s — aborted.") from None
    stderr = proc.stderr.decode(errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"Ink/Stitch failed (exit code {proc.returncode}):\n{stderr}")
    if os.path.getsize(out_zip_path) == 0:
        raise RuntimeError(f"Ink/Stitch produced an empty file — conversion likely failed.\n{stderr}")
    if not zipfile.is_zipfile(out_zip_path):
        with open(out_zip_path, "rb") as f:
            head = f.read(500).decode(errors="replace")
        raise RuntimeError(
            "Ink/Stitch did not produce a zip file.\n"
            f"Output started with:\n{head}\n\nError output:\n{stderr}"
        )
    return out_zip_path


def extract_zip(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
        return z.namelist()


# ---------------------------------------------------------------------------
# Stage 5: preview + QA using pyembroidery
# ---------------------------------------------------------------------------
def render_preview_and_qa(stitch_path, preview_png_path, hoop_mm=None, pes_path=None):
    """Render a preview PNG and compute QA stats from an embroidery file.

    Colors come from the file that is rendered: DST/EXP carry no thread
    colors (pyembroidery has to guess), PES/VP3/JEF do — so pass a PES as
    stitch_path (or pes_path) whenever one exists. Stitch geometry is the
    same either way."""
    import pyembroidery

    source = pes_path if (pes_path and Path(pes_path).exists()) else stitch_path
    pattern = pyembroidery.read(str(source))
    if pattern is None:
        raise RuntimeError(f"pyembroidery could not read {source}")
    pyembroidery.write_png(pattern, str(preview_png_path), settings={"background": "#ffffff"})

    stitches = pattern.stitches
    xs = [s[0] for s in stitches]
    ys = [s[1] for s in stitches]
    width_units = (max(xs) - min(xs)) if xs else 0
    height_units = (max(ys) - min(ys)) if ys else 0
    # pyembroidery stitch coords are in 0.1mm units
    width_mm = width_units / 10.0
    height_mm = height_units / 10.0

    # Thread changes = COLOR_CHANGE commands; distinct colors come from the
    # thread list when the format stores one (DST/EXP don't).
    mask = getattr(pyembroidery, "COMMAND_MASK", 0xFF)
    color_changes = sum(1 for s in stitches if (s[2] & mask) == pyembroidery.COLOR_CHANGE)
    color_blocks = color_changes + 1
    distinct = len({t.hex_color().lower() for t in pattern.threadlist}) if pattern.threadlist else None
    colors = distinct or color_blocks

    report = {
        "stitch_count": len(stitches),
        "width_mm": round(width_mm, 1),
        "height_mm": round(height_mm, 1),
        "color_count": colors,
        "thread_changes": color_changes,
        "preview_source": Path(source).name,
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
    if color_changes >= 15:
        report["warnings"].append(f"Many thread changes ({color_changes}) for {colors} colors — "
                                  "try fewer colors, or --layering cutout (lets same-colored "
                                  "shapes be grouped freely)")

    return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def sanitize_name(name):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return name or "design"


def make_run_dir(parent, base_name, now=None):
    """Create a fresh, never-reused run folder: <base>_<timestamp>[_N]."""
    now = now or datetime.datetime.now()
    stem = f"{base_name}_{now.strftime('%Y%m%d_%H%M%S')}"
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for n in range(1000):
        name = stem if n == 0 else f"{stem}_{n + 1}"
        try:
            (parent / name).mkdir()
            return name, parent / name
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique run folder in {parent}")


def run_pipeline(input_image, out_dir, resize_max=800, max_colors=8,
                 formats=DEFAULT_FORMATS, hoop=None, inkstitch_exe=None,
                 mode="polygon", palette=None, progress_cb=None, remove_bg=False,
                 min_color_distance=0, thread_catalog=None, hoop_margin=2.0,
                 layering="stacked"):
    # --- validate up front, before any slow work ---
    if palette and thread_catalog:
        raise ValueError("Use either an exact palette or a thread catalog, not both.")
    if palette:
        palette = [normalize_hex(c) for c in palette]
    formats = [f.lower().lstrip(".") for f in formats]
    if not any(f not in NON_STITCH_FORMATS for f in formats):
        raise ValueError("Choose at least one embroidery file format (e.g. pes or dst).")
    if layering not in LAYERING_MODES:
        raise ValueError(f"layering must be one of {LAYERING_MODES}")
    inkstitch_exe = inkstitch_exe or find_inkstitch_exe()

    base_name = sanitize_name(Path(input_image).stem)
    run_name, out_dir = make_run_dir(out_dir, base_name)

    def step(msg):
        if progress_cb:
            progress_cb(msg)

    log = {"run_name": run_name, "run_dir": str(out_dir), "source_image": str(input_image),
           "stages": {}, "warnings": []}
    log_path = out_dir / "pipeline_log.json"

    def write_log():
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, default=str)

    try:
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

        # Physical size / fit-to-hoop (applied to the SVG, not the raster)
        fit = compute_hoop_fit(size, hoop, hoop_margin)
        log["stages"]["preprocess"]["hoop_fit"] = fit
        if fit["scale"] < 1.0:
            step(f"Design will be scaled {fit['scale']:.2f}x to fit the {hoop[0]:g}x{hoop[1]:g}mm hoop...")
        if fit["hint"]:
            log["warnings"].append(fit["hint"])

        # Stage 2
        step("Vectorizing (image -> SVG shapes)...")
        svg_raw = out_dir / "02_vectorized.svg"
        vectorize(pre_path, svg_raw, color_precision=8, mode=mode, layering=layering)
        log["stages"]["vectorize"] = {"output": str(svg_raw), "layering": layering}

        # Stage 3
        step("Cleaning SVG (removing background/speckle)...")
        svg_clean = out_dir / "03_cleaned.svg"
        clean_stats = clean_svg(svg_raw, svg_clean, bg_color, width=size[0], height=size[1],
                                target_palette=actual_palette, mm_per_px=fit["mm_per_px"],
                                layering=layering)
        log["stages"]["clean_svg"] = {"output": str(svg_clean), **clean_stats}
        if clean_stats["kept"] == 0:
            raise RuntimeError("Nothing left to stitch after removing the background — "
                               "try more colors or --remove-bg.")

        # Stage 4
        step("Converting to embroidery stitches (Ink/Stitch)...")
        zip_path = out_dir / f"{base_name}_embroidery.zip"
        run_inkstitch(svg_clean, zip_path, inkstitch_exe, formats=formats, custom_file_name=base_name)
        extracted_dir = out_dir / "files"
        files = extract_zip(zip_path, extracted_dir)
        log["stages"]["inkstitch"] = {"zip": str(zip_path), "files": files}

        # Stage 5
        step("Rendering preview + running QA checks...")
        preview_path = out_dir / "06_preview.png"
        source = None
        for ext in PREVIEW_FORMAT_PREFERENCE:
            hits = sorted(extracted_dir.rglob(f"*.{ext}"))
            if hits:
                source = hits[0]
                break
        if source:
            report = render_preview_and_qa(source, preview_path, hoop_mm=hoop)
            if source.suffix.lower().lstrip(".") in FORMATS_WITHOUT_COLOR:
                report["warnings"].append(
                    f"Preview rendered from .{source.suffix.lstrip('.')} which stores no thread colors — "
                    "colors are guessed. Add 'pes' to the formats for an accurate preview."
                )
            report["warnings"] = log["warnings"] + report["warnings"]
            log["stages"]["preview_qa"] = {"preview": str(preview_path), "report": report}
        else:
            log["stages"]["preview_qa"] = {
                "error": "no previewable embroidery file (pes/vp3/jef/dst/exp) was produced"
            }
    except Exception as e:
        log["error"] = f"{type(e).__name__}: {e}"
        log["traceback"] = traceback.format_exc()
        write_log()
        raise

    write_log()
    step("Done.")
    return log


def parse_hoop(s):
    if not s:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*", s)
    if not m or float(m.group(1)) <= 0 or float(m.group(2)) <= 0:
        raise argparse.ArgumentTypeError("hoop must look like 130x180 (width x height in mm)")
    return float(m.group(1)), float(m.group(2))


def _color_arg(value):
    try:
        return normalize_hex(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def _positive_int(value):
    try:
        v = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}")
    if v < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return v


def main():
    ap = argparse.ArgumentParser(description="Image -> embroidery design pipeline")
    ap.add_argument("input_image", nargs="?")
    ap.add_argument("output_dir", nargs="?")
    ap.add_argument("--resize", type=_positive_int, default=800, help="max dimension in px before vectorizing")
    ap.add_argument("--max-colors", type=_positive_int, default=8, help="number of flat colors to posterize to")
    ap.add_argument("--min-color-distance", type=float, default=0,
                    help="merge auto-picked colors closer than this RGB distance (0-441); "
                         "try 40-80 to avoid several near-identical shades of gray from fur/texture noise")
    color_src = ap.add_mutually_exclusive_group()
    color_src.add_argument("--palette", nargs="+", default=None, type=_color_arg,
                           help="exact thread colors to snap to, e.g. --palette '#dc1e1e' '#1450c8' navy")
    color_src.add_argument("--thread-catalog", default=None,
                           help="snap colors to a real manufacturer thread chart shipped with Ink/Stitch, "
                                "e.g. --thread-catalog 'Brother Embroidery'. See --list-thread-catalogs.")
    ap.add_argument("--list-thread-catalogs", action="store_true",
                    help="print all available --thread-catalog names and exit")
    ap.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS,
                    help="embroidery formats to export, e.g. dst pes jef vp3 threadlist")
    ap.add_argument("--remove-bg", action="store_true",
                    help="AI-remove the background first (for photos with busy/non-flat backgrounds)")
    ap.add_argument("--hoop", type=parse_hoop, default=None,
                    help="hoop sewing field in mm, e.g. 130x180; the design is scaled down to fit")
    ap.add_argument("--hoop-margin", type=float, default=2.0,
                    help="clearance in mm kept free on every side of the hoop (default 2)")
    ap.add_argument("--layering", default="stacked", choices=LAYERING_MODES,
                    help="stacked: shapes overlap (no gaps, more stitches); "
                         "cutout: no overlap (fewer stitches, may show small gaps)")
    ap.add_argument("--inkstitch-exe", default=None,
                    help="path to the Ink/Stitch binary (auto-detected if omitted; or set INKSTITCH_EXE)")
    ap.add_argument("--mode", default="polygon", choices=["polygon", "spline", "pixel"],
                    help="vtracer curve fitting mode")
    args = ap.parse_args()

    if args.list_thread_catalogs:
        names = list_thread_catalogs()
        if not names:
            print("No thread catalogs found (set INKSTITCH_PALETTES to Ink/Stitch's palettes folder).")
        for name in names:
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
        hoop_margin=args.hoop_margin, layering=args.layering,
    )

    print(json.dumps(log, indent=2, default=str))

    qa = log["stages"].get("preview_qa", {})
    if "report" in qa:
        r = qa["report"]
        print(f"\n✅ Done. {r['stitch_count']} stitches, {r['color_count']} colors "
              f"({r['thread_changes']} thread changes), {r['width_mm']}x{r['height_mm']}mm")
        for w in r.get("warnings", []):
            print(f"⚠️  {w}")
    else:
        print("\n⚠️  No preview/QA generated:", qa.get("error"))


if __name__ == "__main__":
    main()
