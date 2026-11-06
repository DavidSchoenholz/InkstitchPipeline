#!/usr/bin/env python3
"""
Embroidery Pipeline GUI
========================
A simple desktop front-end for pipeline.py:
upload an image, pick colors (or let it auto-pick), set hoop size,
hit Convert, see the preview and open the output folder.

Run with:
    py -3.12 gui.py

Build a standalone .exe with (uses the checked-in spec, which bundles the
vtracer / pyembroidery / rembg / onnxruntime data PyInstaller would miss):
    py -3.12 -m PyInstaller EmbroideryPipeline.spec --noconfirm
"""
import datetime
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk, colorchooser

from PIL import Image, ImageTk

import pipeline  # the pipeline.py in this same folder

APP_TITLE = "Image → Embroidery Pipeline"
NO_CATALOG = "(none — free colors)"
LOADING_CATALOGS = "(looking for thread catalogs...)"


def _contrast_fg(hex_color):
    r, g, b = pipeline.parse_color(hex_color)
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "white"


def open_folder(path):
    """Open a folder in the platform's file manager."""
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 - Windows only
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class PaletteEditor(ttk.Frame):
    """A row of color swatches the user can add/remove, used as the
    exact thread-color palette to snap the image to."""

    def __init__(self, master):
        super().__init__(master)
        self.colors = []  # list of '#rrggbb'
        self.swatch_frame = ttk.Frame(self)
        self.swatch_frame.pack(side="left", fill="x", expand=True)
        add_btn = ttk.Button(self, text="+ Add color", command=self.add_color)
        add_btn.pack(side="left", padx=4)

    def add_color(self, initial="#808080"):
        rgb, hexcode = colorchooser.askcolor(initial=initial, title="Pick a thread color")
        if not hexcode:
            return
        self.colors.append(pipeline.normalize_hex(hexcode))
        self._redraw()

    def remove_color(self, idx):
        del self.colors[idx]
        self._redraw()

    def _redraw(self):
        for w in self.swatch_frame.winfo_children():
            w.destroy()
        for i, c in enumerate(self.colors):
            cell = tk.Frame(self.swatch_frame, bg=c, width=28, height=28,
                            highlightbackground="black", highlightthickness=1)
            cell.pack(side="left", padx=2)
            cell.bind("<Button-1>", lambda e, i=i: self.remove_color(i))
            cell.pack_propagate(False)
            lbl = tk.Label(cell, text="x", bg=c, fg=_contrast_fg(c), font=("Segoe UI", 7, "bold"))
            lbl.place(relx=1.0, rely=0.0, anchor="ne")
            lbl.bind("<Button-1>", lambda e, i=i: self.remove_color(i))

    def get_palette(self):
        return list(self.colors) if self.colors else None


class InputError(ValueError):
    pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x940")
        self.minsize(700, 600)
        self.resizable(True, True)

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.home() / "Documents" / "EmbroideryOutput"))
        self.resize_max = tk.StringVar(value="800")
        self.max_colors = tk.StringVar(value="8")
        self.min_color_distance = tk.StringVar(value="0")
        self.thread_catalog = tk.StringVar(value=LOADING_CATALOGS)
        self.hoop_w = tk.StringVar(value="130")
        self.hoop_h = tk.StringVar(value="180")
        self.hoop_margin = tk.StringVar(value="2")
        self.layering = tk.StringVar(value="stacked")
        self.fmt_dst = tk.BooleanVar(value=True)
        self.fmt_pes = tk.BooleanVar(value=True)
        self.fmt_jef = tk.BooleanVar(value=False)
        self.fmt_vp3 = tk.BooleanVar(value=False)
        self.fmt_exp = tk.BooleanVar(value=False)
        self.fmt_threadlist = tk.BooleanVar(value=True)
        self.use_palette = tk.BooleanVar(value=False)
        self.remove_bg = tk.BooleanVar(value=False)

        self.msg_queue = queue.Queue()
        self.last_run_dir = None

        self._build_ui()
        self.after(150, self._poll_queue)
        # Searching for Ink/Stitch's palettes can touch the disk a lot; do it
        # off the UI thread so the window appears immediately.
        threading.Thread(target=self._load_catalogs, daemon=True).start()

    # ---------------------------------------------------------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # --- Input image ---
        frm_in = ttk.LabelFrame(self, text="1. Image")
        frm_in.pack(fill="x", **pad)
        ttk.Entry(frm_in, textvariable=self.input_path, width=60).pack(side="left", padx=6, pady=6)
        ttk.Button(frm_in, text="Browse...", command=self.browse_input).pack(side="left", padx=6)
        ttk.Checkbutton(frm_in, text="Remove background (AI) — for photos with busy backgrounds",
                        variable=self.remove_bg).pack(side="left", padx=10)

        # --- Colors ---
        frm_color = ttk.LabelFrame(self, text="2. Colors")
        frm_color.pack(fill="x", **pad)

        mode_row = ttk.Frame(frm_color)
        mode_row.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Radiobutton(mode_row, text="Auto-pick N colors", variable=self.use_palette,
                        value=False).pack(side="left")
        ttk.Spinbox(mode_row, from_=1, to=20, textvariable=self.max_colors, width=5).pack(side="left", padx=8)
        ttk.Radiobutton(mode_row, text="Use my exact thread colors below", variable=self.use_palette,
                        value=True).pack(side="left", padx=(20, 0))

        mindist_row = ttk.Frame(frm_color)
        mindist_row.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Label(mindist_row, text="Min. color distance (0 = off; merges near-duplicate shades, "
                                    "e.g. 10 shades of gray):").pack(side="left")
        ttk.Spinbox(mindist_row, from_=0, to=441, increment=10, textvariable=self.min_color_distance,
                    width=5).pack(side="left", padx=8)

        catalog_row = ttk.Frame(frm_color)
        catalog_row.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Label(catalog_row, text="Snap auto-picked colors to thread catalog:").pack(side="left")
        self.catalog_combo = ttk.Combobox(catalog_row, textvariable=self.thread_catalog,
                                          values=[LOADING_CATALOGS], width=32, state="disabled")
        self.catalog_combo.pack(side="left", padx=8)

        self.palette_editor = PaletteEditor(frm_color)
        self.palette_editor.pack(fill="x", padx=6, pady=8)

        # --- Hoop / size ---
        frm_hoop = ttk.LabelFrame(
            self, text="3. Hoop sewing field (mm) — design is scaled down to fit (leave blank for natural size)")
        frm_hoop.pack(fill="x", **pad)
        ttk.Label(frm_hoop, text="Width").pack(side="left", padx=(6, 2))
        ttk.Entry(frm_hoop, textvariable=self.hoop_w, width=6).pack(side="left")
        ttk.Label(frm_hoop, text="x Height").pack(side="left", padx=(10, 2))
        ttk.Entry(frm_hoop, textvariable=self.hoop_h, width=6).pack(side="left")
        ttk.Label(frm_hoop, text="Margin").pack(side="left", padx=(16, 2))
        ttk.Entry(frm_hoop, textvariable=self.hoop_margin, width=4).pack(side="left")
        ttk.Label(frm_hoop, text="Shapes:").pack(side="left", padx=(16, 2))
        ttk.Combobox(frm_hoop, textvariable=self.layering, values=list(pipeline.LAYERING_MODES),
                     width=9, state="readonly").pack(side="left")
        ttk.Label(frm_hoop, text="(stacked = overlapping, no gaps; cutout = fewer stitches)",
                  foreground="#555").pack(side="left", padx=6)

        # --- Formats ---
        frm_fmt = ttk.LabelFrame(self, text="4. Output formats")
        frm_fmt.pack(fill="x", **pad)
        for text, var in [(".DST (Tajima)", self.fmt_dst), (".PES (Brother)", self.fmt_pes),
                          (".JEF (Janome)", self.fmt_jef), (".VP3 (Pfaff)", self.fmt_vp3),
                          (".EXP (Melco)", self.fmt_exp), ("Threadlist .txt", self.fmt_threadlist)]:
            ttk.Checkbutton(frm_fmt, text=text, variable=var).pack(side="left", padx=6, pady=4)

        # --- Output folder ---
        frm_out = ttk.LabelFrame(self, text="5. Output folder")
        frm_out.pack(fill="x", **pad)
        ttk.Entry(frm_out, textvariable=self.output_dir, width=60).pack(side="left", padx=6, pady=6)
        ttk.Button(frm_out, text="Browse...", command=self.browse_output).pack(side="left", padx=6)

        # --- Run ---
        frm_run = ttk.Frame(self)
        frm_run.pack(fill="x", **pad)
        self.run_btn = ttk.Button(frm_run, text="Convert ▶", command=self.run_pipeline_clicked)
        self.run_btn.pack(side="left")
        self.open_btn = ttk.Button(frm_run, text="Open output folder", command=self.open_output,
                                   state="disabled")
        self.open_btn.pack(side="left", padx=8)

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", **pad)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, foreground="#555").pack(anchor="w", padx=12)

        # --- Preview + report ---
        frm_result = ttk.LabelFrame(self, text="Result")
        frm_result.pack(fill="both", expand=True, **pad)

        frm_previews = ttk.Frame(frm_result)
        frm_previews.pack(side="left", padx=10, pady=10)
        ttk.Label(frm_previews, text="Design (intended colors)", font=("", 9, "bold")).pack(anchor="w")
        self.preview_label = ttk.Label(frm_previews, text="(preview will appear here)")
        self.preview_label.pack(anchor="w")
        ttk.Label(frm_previews, text="Machine stitch-out (approx. — colors from the embroidery file)",
                  font=("", 9, "bold")).pack(anchor="w", pady=(10, 0))
        self.machine_preview_label = ttk.Label(frm_previews, text="(preview will appear here)")
        self.machine_preview_label.pack(anchor="w")

        self.report_text = tk.Text(frm_result, width=36, height=14, wrap="word")
        self.report_text.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    # ---------------------------------------------------------------
    def _load_catalogs(self):
        try:
            names = pipeline.list_thread_catalogs()
        except Exception:  # never let catalog discovery kill the app
            names = []
        self.msg_queue.put(("catalogs", names))

    def _set_catalogs(self, names):
        self.catalog_combo.config(values=[NO_CATALOG] + names, state="readonly")
        self.thread_catalog.set(NO_CATALOG)
        if not names:
            self.status_var.set("Ready. (No Ink/Stitch thread catalogs found — catalog snapping unavailable.)")

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")])
        if path:
            self.input_path.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_dir.set(path)

    def open_output(self):
        out = self.last_run_dir or self.output_dir.get()
        if out and os.path.isdir(out):
            try:
                open_folder(out)
            except Exception as e:
                messagebox.showerror(APP_TITLE, f"Could not open the folder:\n{out}\n\n{e}")

    # ---------------------------------------------------------------
    @staticmethod
    def _parse_number(text, label, lo, hi, kind=int):
        try:
            value = kind(str(text).strip())
        except ValueError:
            raise InputError(f"{label} must be a {'whole ' if kind is int else ''}number "
                             f"(got {text!r}).") from None
        if not (lo <= value <= hi):
            raise InputError(f"{label} must be between {lo} and {hi} (got {value}).")
        return value

    def _collect_settings(self):
        """Read + validate every input. Raises InputError with a message
        for the user; nothing in the UI is changed until this succeeds."""
        img = self.input_path.get().strip()
        if not img or not os.path.isfile(img):
            raise InputError("Please choose a valid image file first.")
        out_dir = self.output_dir.get().strip()
        if not out_dir:
            raise InputError("Please choose an output folder.")

        resize_max = self._parse_number(self.resize_max.get(), "Resize", 16, 10000)
        max_colors = self._parse_number(self.max_colors.get(), "Number of colors", 1, 64)
        min_dist = self._parse_number(self.min_color_distance.get(), "Min. color distance", 0, 441, float)

        hw, hh = self.hoop_w.get().strip(), self.hoop_h.get().strip()
        if not hw and not hh:
            hoop = None
        else:
            hoop = (self._parse_number(hw, "Hoop width", 1, 2000, float),
                    self._parse_number(hh, "Hoop height", 1, 2000, float))
        margin = self._parse_number(self.hoop_margin.get() or "0", "Hoop margin", 0, 100, float)
        if hoop and (hoop[0] <= 2 * margin or hoop[1] <= 2 * margin):
            raise InputError("The hoop margin leaves no room for the design.")

        formats = [fmt for fmt, var in [("dst", self.fmt_dst), ("pes", self.fmt_pes),
                                        ("jef", self.fmt_jef), ("vp3", self.fmt_vp3),
                                        ("exp", self.fmt_exp), ("threadlist", self.fmt_threadlist)]
                   if var.get()]
        if not any(f != "threadlist" for f in formats):
            raise InputError("Choose at least one embroidery format (e.g. .PES or .DST).")

        thread_catalog = self.thread_catalog.get()
        if thread_catalog.startswith("("):
            thread_catalog = None

        palette = None
        if self.use_palette.get():
            palette = self.palette_editor.get_palette()
            if not palette:
                raise InputError("'Use my exact thread colors' is selected but no colors were added.\n"
                                 "Click '+ Add color', or switch to 'Auto-pick N colors'.")
            if thread_catalog:
                raise InputError("Exact thread colors and a thread catalog can't be combined.\n"
                                 f"Set the catalog to '{NO_CATALOG}', or switch to 'Auto-pick N colors'.")

        return dict(img=img, out_dir=out_dir, resize_max=resize_max, max_colors=max_colors,
                    formats=formats, hoop=hoop, hoop_margin=margin, palette=palette,
                    remove_bg=self.remove_bg.get(), min_color_distance=min_dist,
                    thread_catalog=thread_catalog, layering=self.layering.get())

    def run_pipeline_clicked(self):
        try:
            settings = self._collect_settings()
        except InputError as e:
            messagebox.showerror(APP_TITLE, str(e))
            return

        self.run_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Starting...")
        self.report_text.delete("1.0", "end")

        threading.Thread(target=self._run_worker, args=(settings,), daemon=True).start()

    def _run_worker(self, s):
        try:
            def progress_cb(msg):
                self.msg_queue.put(("status", msg))

            log = pipeline.run_pipeline(
                s["img"], s["out_dir"], resize_max=s["resize_max"], max_colors=s["max_colors"],
                formats=s["formats"], hoop=s["hoop"], mode="polygon", palette=s["palette"],
                progress_cb=progress_cb, remove_bg=s["remove_bg"],
                min_color_distance=s["min_color_distance"], thread_catalog=s["thread_catalog"],
                hoop_margin=s["hoop_margin"], layering=s["layering"],
            )
            self.msg_queue.put(("done", log))
        except BaseException as e:  # noqa: BLE001 - must always re-enable the UI
            self.msg_queue.put(("error", (f"{type(e).__name__}: {e}", traceback.format_exc(),
                                          s["out_dir"])))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "catalogs":
                    self._set_catalogs(payload)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._on_error(*payload)
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def _on_done(self, log):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.open_btn.config(state="normal")
        self.status_var.set("Done.")
        self.last_run_dir = log.get("run_dir")

        qa = log["stages"].get("preview_qa", {})
        report = qa.get("report")
        self.report_text.delete("1.0", "end")
        if report:
            lines = [
                f"Stitches: {report['stitch_count']}",
                f"Colors: {report['color_count']} ({report.get('thread_changes', 0)} thread changes)",
                f"Size: {report['width_mm']} x {report['height_mm']} mm",
            ]
            if report.get("warnings"):
                lines.append("")
                lines.append("Warnings:")
                for w in report["warnings"]:
                    lines.append(f"- {w}")
            else:
                lines.append("")
                lines.append("No warnings.")
            self.report_text.insert("1.0", "\n".join(lines))

            preview_path = qa.get("preview")
            if preview_path and os.path.isfile(preview_path):
                self._show_preview(preview_path, self.machine_preview_label, "machine")
            design_path = log["stages"].get("preprocess", {}).get("output")
            if design_path and os.path.isfile(design_path):
                self._show_preview(design_path, self.preview_label, "design")
        else:
            self.report_text.insert("1.0", f"No preview generated:\n{qa.get('error')}")

        messagebox.showinfo(APP_TITLE, "Conversion finished! See the report and preview.")

    def _on_error(self, message, tb, out_dir):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status_var.set("Error.")
        # The packaged .exe has no console, so keep the full traceback on disk.
        log_note = ""
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            log_file = Path(out_dir) / "EmbroideryPipeline_error.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====\n{tb}")
            log_note = f"\n\nFull details were saved to:\n{log_file}"
        except OSError:
            pass
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", tb)
        messagebox.showerror(APP_TITLE, f"Pipeline failed:\n\n{message}{log_note}")

    def _show_preview(self, path, target_label, keep_as):
        try:
            img = Image.open(path)
            # Fit within a box, preserving aspect ratio in both dimensions
            # (not just width) — a tall portrait preview otherwise stays
            # too tall for the window and gets cut off at the bottom.
            img.thumbnail((300, 380), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            setattr(self, f"_preview_ref_{keep_as}", photo)  # keep a reference!
            target_label.config(image=photo, text="")
        except Exception as e:
            target_label.config(text=f"(preview failed to load: {e})")


if __name__ == "__main__":
    app = App()
    app.mainloop()
