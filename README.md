# InkstitchPipeline

take da photo, make it into an embroidery thingy

A scriptable image → embroidery pipeline: photo → preprocess (resize,
color-quantize) → vectorize (`vtracer`) → clean SVG → Ink/Stitch headless
conversion → real machine embroidery files (`.dst`/`.pes`/...) → stitch
preview + QA.

## Files

- `pipeline.py` — CLI / core pipeline (stages are composable functions)
- `gui.py` — Tkinter desktop GUI front-end
- `EmbroideryPipeline.spec` — PyInstaller spec to build a standalone `.exe`
- `tests/` — pytest suite (uses a fake Ink/Stitch, so Ink/Stitch isn't needed to run it)

## Requirements

- Python 3.12
- `pip install -r requirements.txt` (pillow, numpy, vtracer, pyembroidery)
- Optional: `pip install -r requirements-optional.txt` for `--remove-bg`
  (rembg + onnxruntime) and for building the `.exe` (PyInstaller).
  rembg downloads its ~170 MB model on first use into `~/.u2net`
  (override with `U2NET_HOME`), so the first background removal needs internet.
- [Ink/Stitch](https://inkstitch.org/) installed (for its `inkstitch` binary and
  its bundled real thread-manufacturer color catalogs). It is auto-detected in
  the usual Inkscape extension folders on Windows, macOS and Linux; otherwise set
  `INKSTITCH_EXE` (path to the binary) and, if needed, `INKSTITCH_PALETTES`
  (folder containing the `InkStitch *.gpl` catalogs).

## CLI usage

```
py -3.12 pipeline.py photo.jpg out/ --max-colors 6 --hoop 130x180
```

Key options:

- `--hoop WxH` — hoop sewing field in mm. The design is scaled down to fit
  (the SVG is given a physical size; the image is *not* downsampled), keeping
  `--hoop-margin` mm (default 2) clear on every side. If the design would be
  noticeably larger with the hoop rotated, the report says so.
- `--remove-bg` — AI background isolation.
- `--min-color-distance` — merge near-duplicate auto-picked shades.
- `--palette` — snap to exact colors (`#rrggbb`, `#rgb` or color names).
- `--thread-catalog "Brother Embroidery"` — snap to a real manufacturer
  thread chart (perceptual CIELAB matching); `--list-thread-catalogs` lists
  them. Can't be combined with `--palette`.
- `--layering stacked|cutout` — `stacked` (default) layers shapes on top of
  each other so there are no gaps, and keeps overlapping shapes in their
  correct order; `cutout` makes non-overlapping shapes, which uses fewer
  stitches and lets same-colored shapes be grouped freely (fewer thread
  changes), but fabric pull can open small gaps between regions.

Every run gets its own folder (`<name>_<timestamp>`), containing the
intermediate images/SVGs, the Ink/Stitch output, a preview and
`pipeline_log.json` (which also records the error + traceback if a run fails).

## GUI usage

```
py -3.12 gui.py
```

Or build a standalone exe:

```
py -3.12 -m PyInstaller EmbroideryPipeline.spec --noconfirm
```

When a run fails, the GUI shows the error and appends the full traceback to
`EmbroideryPipeline_error.log` in the chosen output folder.

## Tests

```
py -3.12 -m pip install -r requirements-dev.txt
py -3.12 -m pytest
```
