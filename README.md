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

## Requirements

- Python 3.12
- `pip install pillow vtracer pyembroidery numpy rembg onnxruntime scikit-image`
- [Ink/Stitch](https://inkstitch.org/) installed (for `inkstitch.exe` and its
  bundled real thread-manufacturer color catalogs)

## CLI usage

```
py -3.12 pipeline.py photo.jpg out/ --max-colors 6 --hoop 130x180
```

Key options: `--remove-bg` (AI background isolation), `--min-color-distance`
(merge near-duplicate auto-picked shades), `--palette` (snap to exact hex
colors), `--thread-catalog "Brother Embroidery"` (snap to a real
manufacturer thread chart — `--list-thread-catalogs` lists all available).

## GUI usage

```
py -3.12 gui.py
```

Or build a standalone exe:

```
py -3.12 -m PyInstaller EmbroideryPipeline.spec --noconfirm
```
