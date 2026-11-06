import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import pipeline
from conftest import REPO, mismatch_pct, rasterize_svg


def _run_front_half(src, tmp_path, max_colors, layering="stacked"):
    pre = tmp_path / "pre.png"
    _, bg, size, pal, _ = pipeline.preprocess_image(src, pre, 800, max_colors)
    raw = tmp_path / "raw.svg"
    clean = tmp_path / "clean.svg"
    pipeline.vectorize(pre, raw, layering=layering)
    stats = pipeline.clean_svg(raw, clean, bg, width=size[0], height=size[1],
                               target_palette=pal, layering=layering)
    return pre, raw, clean, bg, size, pal, stats


# --- SVG cleaning -----------------------------------------------------------

def test_color_grouping_does_not_bury_overlapping_shapes(tmp_path):
    """Regression: grouping by color used to move shapes below shapes that
    covered them in vtracer's stacked output, hiding whole ellipses."""
    pre, raw, clean, bg, size, pal, stats = _run_front_half(REPO / "messy_photo.png", tmp_path, 6)
    src = np.asarray(Image.open(pre).convert("RGB")).astype(int)
    raw_err = mismatch_pct(src, rasterize_svg(raw, size, bg))
    clean_err = mismatch_pct(src, rasterize_svg(clean, size, bg))
    assert clean_err <= raw_err + 0.1, (raw_err, clean_err)


def test_order_by_color_respects_overlaps_but_still_groups():
    red, blue = (255, 0, 0), (0, 0, 255)
    items = [
        (red, (0, 0, 10, 10)),     # 0
        (blue, (5, 5, 15, 15)),    # 1 overlaps 0
        (red, (50, 50, 60, 60)),   # 2 isolated -> may join red group early
        (red, (8, 8, 12, 12)),     # 3 sits on top of blue 1 -> must stay after 1
    ]
    order = pipeline._order_by_color(items, preserve_overlaps=True)
    assert sorted(order) == [0, 1, 2, 3]
    assert order.index(0) < order.index(1) < order.index(3)
    assert order[:2] == [0, 2]  # isolated red shape grouped with the first red one


def test_leading_backdrop_removal_does_not_cascade(tmp_path):
    """Regression: after the backdrop was removed, the next canvas-sized shape
    was also treated as 'the backdrop' and deleted regardless of its color."""
    im = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((5, 5, 395, 395), fill=(30, 80, 200))
    d.ellipse((170, 170, 230, 230), fill=(200, 30, 30))
    src = tmp_path / "cascade.png"
    im.save(src)
    *_, clean, bg, size, pal, stats = _run_front_half(src, tmp_path, 3)
    fills = {el.get("fill") for el in ET.parse(clean).getroot().iter() if el.tag.endswith("path")}
    blues = [f for f in fills if pipeline.parse_color(f)[2] > 150]
    assert blues, f"big blue shape was deleted; remaining fills: {fills}"
    # ...and it must be the ellipse, not a full-canvas rectangle over the background.
    src = np.asarray(Image.open(tmp_path / "pre.png").convert("RGB")).astype(int)
    assert mismatch_pct(src, rasterize_svg(clean, size, bg)) < 2.0


def test_background_rectangle_is_removed(tmp_path):
    *_, clean, bg, size, pal, stats = _run_front_half(REPO / "sample_photo.png", tmp_path, 4)
    assert stats["removed_background"] == 1
    for el in ET.parse(clean).getroot().iter():
        if el.tag.endswith("path"):
            assert pipeline._color_dist(pipeline.parse_color(el.get("fill")), bg) > 20


def test_clean_svg_sets_physical_size(tmp_path):
    *_, clean, bg, size, pal, stats = _run_front_half(REPO / "sample_photo.png", tmp_path, 4)
    root = ET.parse(clean).getroot()
    assert root.get("viewBox") == f"0 0 {size[0]} {size[1]}"
    assert root.get("width").endswith("mm")
    assert float(root.get("width")[:-2]) == pytest.approx(size[0] * 25.4 / 96, abs=0.01)


def test_speckle_filter_uses_physical_size(tmp_path):
    im = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((100, 100, 300, 300), fill=(200, 30, 30))
    d.rectangle((20, 20, 28, 28), fill=(30, 30, 200))  # ~9 px square
    src = tmp_path / "speck.png"
    im.save(src)
    pre = tmp_path / "pre.png"
    _, bg, size, pal, _ = pipeline.preprocess_image(src, pre, 800, 3)
    pipeline.vectorize(pre, tmp_path / "raw.svg")
    # At natural size (0.26 mm/px) the 9 px square is ~2.4 mm: kept.
    s1 = pipeline.clean_svg(tmp_path / "raw.svg", tmp_path / "a.svg", bg, width=size[0], height=size[1],
                            target_palette=pal)
    # Shrunk 10x it is ~0.24 mm: too small to stitch, removed.
    s2 = pipeline.clean_svg(tmp_path / "raw.svg", tmp_path / "b.svg", bg, width=size[0], height=size[1],
                            target_palette=pal, mm_per_px=pipeline.MM_PER_PX_96DPI / 10)
    assert s1["kept"] == 2 and s2["kept"] == 1 and s2["removed_speckle"] == 1


def test_parse_path_geometry_handles_translate_and_curves():
    bbox, area, perim = pipeline.parse_path_geometry("M0,0 L10,0 L10,10 L0,10 Z", "translate(5, 7)")
    assert bbox == (5, 7, 15, 17)
    assert area == pytest.approx(100)
    assert perim == pytest.approx(40)
    bbox, area, _ = pipeline.parse_path_geometry("M0 0 C5 -2 10 0 10 0 C10 5 10 10 10 10 C5 10 0 10 0 10 Z")
    assert bbox[1] == -2  # control point included -> conservative bbox
    assert area == pytest.approx(100)  # on-curve points (0,0),(10,0),(10,10),(0,10)


# --- preprocessing ------------------------------------------------------------

def test_antialiased_edges_do_not_steal_palette_slots(tmp_path):
    """messy_photo has 5 real colors; anti-aliased rims used to claim a 6th
    pale 'halo' color that became its own thread."""
    _, bg, size, pal, _ = pipeline.preprocess_image(REPO / "messy_photo.png", tmp_path / "p.png", 800, 6)
    non_bg = [pipeline.parse_color(p) for p in pal if pipeline.parse_color(p) != tuple(bg)]
    for c in non_bg:
        assert pipeline._color_dist(c, bg) > 60, f"halo-like color {c} near background {bg}"


def test_palette_and_catalog_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError):
        pipeline.preprocess_image(REPO / "sample_photo.png", tmp_path / "p.png",
                                  palette=["#ff0000"], thread_catalog="Brother")


def test_exact_palette_accepts_short_hex_and_names(tmp_path):
    _, _, _, pal, _ = pipeline.preprocess_image(REPO / "sample_photo.png", tmp_path / "p.png",
                                                palette=["#fff", "green", "8b4513", "#e6be28"])
    assert set(pal) <= {"#ffffff", "#008000", "#8b4513", "#e6be28"}


def test_parse_color():
    assert pipeline.parse_color("#fff") == (255, 255, 255)
    assert pipeline.parse_color("ff0000") == (255, 0, 0)
    assert pipeline.parse_color("navy") == (0, 0, 128)
    with pytest.raises(ValueError, match="Invalid color"):
        pipeline.parse_color("#ggg")


def test_merge_close_colors_respects_distance_and_count():
    weighted = [((250, 250, 250), 100), ((245, 245, 245), 50), ((10, 10, 10), 80), ((200, 0, 0), 20)]
    merged = pipeline.merge_close_colors(weighted, max_colors=8, min_distance=40)
    assert len(merged) == 3
    assert len(pipeline.merge_close_colors(weighted, max_colors=2, min_distance=0)) == 2


# --- thread catalogs ----------------------------------------------------------

def test_parse_gpl_tab_separated_and_fallback(tmp_path):
    gpl = tmp_path / "t.gpl"
    gpl.write_text(
        "GIMP Palette\nName: test\nColumns: 4\n# comment\n"
        "255 0 0\tLight Blue\t\n"
        "  0   0 255\tRed\t1234\n"
        "10 20 30 Sky Blue 5678\n"
        "40 50 60 Dark Olive Green\n"
    )
    e = pipeline.parse_gpl(gpl)
    assert [(x["name"], x["number"]) for x in e] == [
        ("Light Blue", ""), ("Red", "1234"), ("Sky Blue", "5678"), ("Dark Olive Green", "")]
    assert e[1]["rgb"] == (0, 0, 255)


def test_snap_palette_to_catalog_is_perceptual():
    catalog = [{"rgb": (0, 0, 0), "name": "Black", "number": "1"},
               {"rgb": (0, 0, 255), "name": "Blue", "number": "2"},
               {"rgb": (255, 255, 255), "name": "White", "number": "3"}]
    new_hex, matches = pipeline.snap_palette_to_catalog(["#101010", "#2020e0", "#f0f0f0", "#000000"], catalog)
    assert new_hex == ["#000000", "#0000ff", "#ffffff"]
    assert [m["name"] for m in matches] == ["Black", "Blue", "White", "Black"]


def test_find_palettes_dir_env_override(tmp_path, monkeypatch):
    (tmp_path / "InkStitch Brother Embroidery.gpl").write_text("GIMP Palette\n0 0 0\tBlack\t100\n")
    monkeypatch.setenv("INKSTITCH_PALETTES", str(tmp_path))
    pipeline.find_palettes_dir.cache_clear()
    try:
        assert pipeline.list_thread_catalogs() == ["Brother Embroidery"]
        assert pipeline.find_catalog_file("brother").name.startswith("InkStitch Brother")
    finally:
        pipeline.find_palettes_dir.cache_clear()


# --- hoop fit / misc ------------------------------------------------------------

def test_compute_hoop_fit_scales_svg_not_raster():
    fit = pipeline.compute_hoop_fit((800, 600), (130, 180), margin_mm=2)
    assert fit["size_mm"][0] == pytest.approx(126, abs=0.1)  # 130 - 2*2 margin
    assert fit["mm_per_px"] == pytest.approx(126 / 800, rel=1e-6)
    assert fit["hint"] and "180x130" in fit["hint"]  # landscape design in portrait hoop
    assert pipeline.compute_hoop_fit((100, 100), (130, 180))["scale"] == 1.0  # never upscales
    assert pipeline.compute_hoop_fit((800, 600), None)["hint"] is None


def test_parse_hoop():
    assert pipeline.parse_hoop("130x180") == (130.0, 180.0)
    assert pipeline.parse_hoop(" 100 X 100.5 ") == (100.0, 100.5)
    for bad in ("130", "0x100", "130x180x5", "abc"):
        with pytest.raises(Exception):
            pipeline.parse_hoop(bad)


def test_make_run_dir_never_reuses_folder(tmp_path):
    import datetime
    now = datetime.datetime(2026, 1, 2, 3, 4, 5)
    a = pipeline.make_run_dir(tmp_path, "cat", now)[1]
    b = pipeline.make_run_dir(tmp_path, "cat", now)[1]
    assert a != b and a.is_dir() and b.is_dir()


# --- Ink/Stitch stage + end to end -------------------------------------------

def test_run_inkstitch_reports_non_zip_output(tmp_path, fake_inkstitch, monkeypatch):
    monkeypatch.setenv("FAKE_INKSTITCH_MODE", "garbage")
    svg = tmp_path / "x.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    with pytest.raises(RuntimeError, match="did not produce a zip") as ei:
        pipeline.run_inkstitch(svg, tmp_path / "o.zip", fake_inkstitch)
    assert "fake failure details" in str(ei.value)


@pytest.mark.parametrize("formats", [["dst", "pes", "threadlist"], ["pes"], ["dst"]])
def test_run_pipeline_end_to_end(tmp_path, fake_inkstitch, formats):
    log = pipeline.run_pipeline(REPO / "sample_photo.png", tmp_path / "out", max_colors=4,
                                formats=formats, hoop=(100, 100), inkstitch_exe=fake_inkstitch)
    qa = log["stages"]["preview_qa"]
    report = qa["report"]
    assert Path(qa["preview"]).is_file()
    if "pes" in formats:
        assert report["color_count"] == 3  # green, yellow, brown (background removed)
    assert report["thread_changes"] == 2
    assert report["width_mm"] <= 100 and report["height_mm"] <= 100
    assert report["preview_source"].endswith(".pes" if "pes" in formats else ".dst")
    color_warning = any("stores no thread colors" in w for w in report["warnings"])
    assert color_warning == ("pes" not in formats)
    saved = json.loads((Path(log["run_dir"]) / "pipeline_log.json").read_text())
    assert saved["run_name"] == log["run_name"]


def test_run_pipeline_writes_log_on_failure(tmp_path, fake_inkstitch, monkeypatch):
    monkeypatch.setenv("FAKE_INKSTITCH_MODE", "garbage")
    with pytest.raises(RuntimeError):
        pipeline.run_pipeline(REPO / "sample_photo.png", tmp_path / "out", max_colors=4,
                              inkstitch_exe=fake_inkstitch)
    (run_dir,) = (tmp_path / "out").iterdir()
    saved = json.loads((run_dir / "pipeline_log.json").read_text())
    assert "did not produce a zip" in saved["error"] and "Traceback" in saved["traceback"]


def test_run_pipeline_validates_inputs_before_work(tmp_path, fake_inkstitch):
    with pytest.raises(ValueError, match="embroidery file format"):
        pipeline.run_pipeline(REPO / "sample_photo.png", tmp_path / "out", formats=["threadlist"],
                              inkstitch_exe=fake_inkstitch)
    with pytest.raises(ValueError, match="not both"):
        pipeline.run_pipeline(REPO / "sample_photo.png", tmp_path / "out", palette=["#000"],
                              thread_catalog="x", inkstitch_exe=fake_inkstitch)
    assert not (tmp_path / "out").exists()
