"""Shared build/export plumbing for MARROW parts."""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from . import params as P
from . import sdf

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "exports"
WORK = ROOT / "build"           # big intermediates (git-ignored)


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0

    def __call__(self, msg):
        now = time.time()
        print(f"  [{now - self.t0:6.1f}s +{now - self.last:5.1f}] {msg}", flush=True)
        self.last = now


def cost_usd(volume_cm3, material):
    key = "rate_316l_usd_cm3" if material == "316L" else "rate_alu_usd_cm3"
    lo, hi = P.PRINT[key]
    return [round(volume_cm3 * lo, 2), round(volume_cm3 * hi, 2)]


def density(material):
    return P.PRINT["rho_316l"] if material == "316L" else P.PRINT["rho_alu"]


def export_part(name, field, grid, material, frame, extra=None, print_faces=900_000,
                web_faces=140_000, hero_faces=None, smooth_iters=1, sl=None):
    """Mesh a finished field and write:
       exports/print/<name>.stl   - print file (binary STL, mm, part's own frame)
       build/<name>_hero.ply      - dense mesh for rendering
       build/<name>_web.ply       - light mesh for the web viewer
       exports/meta/<name>.json   - volume, mass, bbox, cost, checks
    Returns the metadata dict."""
    tm = Timer()
    (OUT / "print").mkdir(parents=True, exist_ok=True)
    (OUT / "meta").mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    vol = sdf.volume_cm3(field if sl is None else field[sl], grid.h)
    m = sdf.mesh(field, grid, sl=sl)
    tm(f"marching cubes: {len(m.faces):,} faces")
    if smooth_iters:
        sdf.smooth(m, iters=smooth_iters)
    hero = m if hero_faces is None or len(m.faces) <= hero_faces else sdf.decimate(m, hero_faces)
    hero.export(WORK / f"{name}_hero.ply")
    tm(f"hero mesh: {len(hero.faces):,} faces")
    pr = sdf.decimate(m, print_faces) if len(m.faces) > print_faces else m
    pr.export(OUT / "print" / f"{name}.stl")
    tm(f"print STL: {len(pr.faces):,} faces")
    web = sdf.decimate(pr, web_faces) if len(pr.faces) > web_faces else pr
    web.export(WORK / f"{name}_web.ply")
    tm(f"web mesh: {len(web.faces):,} faces")

    b0, b1 = m.bounds
    rho = density(material)
    meta = dict(
        name=name, material=material, frame=frame, voxel_mm=grid.h,
        volume_cm3=round(vol, 2), mass_g=round(vol * rho, 1),
        bbox_mm=[round(float(v), 1) for v in (b1 - b0)],
        bbox_lo=[round(float(v), 2) for v in b0], bbox_hi=[round(float(v), 2) for v in b1],
        fits_envelope=bool(np.all(np.sort(b1 - b0) <= np.sort(
            P.PRINT["env_316l"] if material == "316L" else P.PRINT["env_alu"]))),
        longest_mm=round(float((b1 - b0).max()), 1),
        cost_usd_jlc_pcbway=cost_usd(vol, material),
        print_stl=dict(faces=int(len(pr.faces)), watertight=bool(pr.is_watertight),
                       mesh_volume_cm3=round(float(abs(pr.volume)) / 1000.0, 2)),
        surface_area_cm2=round(float(m.area) / 100.0, 1),
    )
    if extra:
        meta.update(extra)
    (OUT / "meta" / f"{name}.json").write_text(json.dumps(meta, indent=1))
    tm("metadata written")
    return meta


def text_mask(text, height_mm, h, font_size=220, bold=True):
    """Rasterise text -> 2-D signed distance array (mm, negative inside glyphs) plus its
    physical size, for embossing/debossing part marks."""
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage as ndi

    font = None
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if pathlib.Path(cand).exists():
            font = ImageFont.truetype(cand, font_size)
            break
    if font is None:
        font = ImageFont.load_default()
    l, t, r, b = font.getbbox(text)
    img = Image.new("L", (r - l + 40, b - t + 40), 0)
    ImageDraw.Draw(img).text((20 - l, 20 - t), text, fill=255, font=font)
    a = np.asarray(img) > 127
    scale = height_mm / (b - t)                     # mm per source pixel
    # resample to the voxel pitch
    zoom = scale / h
    a = ndi.zoom(a.astype(np.float32), zoom, order=1) > 0.5
    d = (ndi.distance_transform_edt(~a) - ndi.distance_transform_edt(a)) * h
    return d.astype(np.float32)[::-1].T.copy()      # index [u, v], v up
