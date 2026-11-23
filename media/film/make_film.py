"""Cut the MARROW launch film.

Rendered stills (build/stills/<shot>.png + depth) are turned into moving shots by depth-based
parallax drift, simulation plates are animated from the solver output, supers are set in the site's
typefaces, and everything is cut to the bar grid of the score (96 BPM, 2.5 s per bar).

    /opt/venv/bin/python media/film/make_film.py [--preview]
"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[2]
STILLS = ROOT / "build" / "stills"
ORBIT = ROOT / "build" / "orbit"
SIMV = ROOT / "build" / "simviz"
FONTS = ROOT / "media" / "fonts"
OUT = ROOT / "build" / "film"
FPS = 24
W, H = 1280, 720
BAR = 2.5


def font(kind, size):
    if kind == "display":
        f = ImageFont.truetype(str(FONTS / "BodoniModa.ttf"), size)
        try:
            f.set_variation_by_axes([28, 600])
        except Exception:  # noqa: BLE001
            pass
        return f
    if kind == "italic":
        f = ImageFont.truetype(str(FONTS / "BodoniModa-Italic.ttf"), size)
        try:
            f.set_variation_by_axes([28, 500])
        except Exception:  # noqa: BLE001
            pass
        return f
    return ImageFont.truetype(str(FONTS / ("IBMPlexMono-Medium.ttf" if kind == "monob" else "IBMPlexMono-Regular.ttf")), size)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Still:
    """A rendered frame + depth that can be re-photographed with a small camera move."""

    def __init__(self, name):
        p = STILLS / f"{name}.png"
        self.ok = p.exists()
        if not self.ok:
            return
        self.img = cv2.imread(str(p), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        self.h, self.w = self.img.shape[:2]
        zs = sorted(STILLS.glob(f"{name}_z_*.png"))
        meta = json.loads((STILLS / f"{name}.json").read_text()) if (STILLS / f"{name}.json").exists() else None
        if zs and meta:
            z = cv2.imread(str(zs[-1]), cv2.IMREAD_UNCHANGED).astype(np.float32)
            z = z[..., 0] if z.ndim == 3 else z
            z /= 65535.0
            d = meta["near"] + z * (meta["far"] - meta["near"])
            d = cv2.GaussianBlur(d, (0, 0), 2.0)
            self.par = (meta["focus"] / np.maximum(d, 1e-3) - 1.0).astype(np.float32)   # 0 at the focus plane
            self.par = np.clip(self.par, -0.7, 1.2)
        else:
            self.par = np.zeros((self.h, self.w), np.float32)
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.xx, self.yy = xx, yy

    def frame(self, t, move):
        """move = dict(z0, z1, px, py, cx, cy): zoom from z0 to z1, parallax travel px/py (pixels at
        par = 1), optional pan centre cx, cy in 0..1."""
        e = ease(t)
        z = move.get("z0", 1.05) + (move.get("z1", 1.12) - move.get("z0", 1.05)) * e
        sx = move.get("px", 0.0) * (e - 0.5)
        sy = move.get("py", 0.0) * (e - 0.5)
        cx = (move.get("cx0", 0.5) + (move.get("cx1", move.get("cx0", 0.5)) - move.get("cx0", 0.5)) * e) * self.w
        cy = (move.get("cy0", 0.5) + (move.get("cy1", move.get("cy0", 0.5)) - move.get("cy0", 0.5)) * e) * self.h
        mx = cx + (self.xx - self.w / 2) / z + sx * self.par
        my = cy + (self.yy - self.h / 2) / z + sy * self.par
        out = cv2.remap(self.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)


class Sequence:
    def __init__(self, folder, pattern):
        self.files = sorted(pathlib.Path(folder).glob(pattern))
        self.ok = len(self.files) > 0
        self.cache = {}

    def frame(self, t, move=None):
        x = t * (len(self.files) - 1)
        i = int(math.floor(x))
        f = x - i
        a = self._get(i)
        if f < 1e-3 or i + 1 >= len(self.files):
            return a
        return a * (1 - f) + self._get(i + 1) * f        # cross-blend between rendered frames

    def _get(self, i):
        if i not in self.cache:
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
            im = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            self.cache[i] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA if im.shape[1] >= W else cv2.INTER_CUBIC)
        return self.cache[i]


def text_layer(lines):
    """lines: list of (kind, size, text, x, y, anchor, colour) -> RGBA float layer."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for kind, size, text, x, y, anchor, col in lines:
        fnt = font(kind, size)
        if kind in ("mono", "monob"):
            # letter-spaced caps
            cx = x
            if anchor[0] == "m":
                tot = sum(d.textlength(c, font=fnt) + size * 0.16 for c in text)
                cx = x - tot / 2
            for c in text:
                d.text((cx, y), c, font=fnt, fill=col, anchor="l" + anchor[1])
                cx += d.textlength(c, font=fnt) + size * 0.16
        else:
            d.text((x, y), text, font=fnt, fill=col, anchor=anchor)
    a = np.asarray(im).astype(np.float32) / 255.0
    return a


def comp(frame, layer, alpha):
    if alpha <= 0:
        return frame
    a = layer[..., 3:4] * alpha
    rgb = layer[..., 2::-1]
    # soft dark plate behind the type so it stays legible on bright metal
    sh = cv2.GaussianBlur(layer[..., 3], (0, 0), 14) * 0.55 * alpha
    frame = frame * (1 - sh[..., None])
    return frame * (1 - a) + rgb * a


def vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return (1.0 - 0.34 * np.clip(r - 0.35, 0, 1) ** 1.6)[..., None]


BONE = (228, 235, 240, 255)
DIMC = (150, 168, 184, 255)
RUBY = (255, 84, 112, 255)


def super_(label, title, sub=None, big=False):
    L = []
    x, y = 78, H - 150
    if big:
        L.append(("mono", 15, label, W / 2, H / 2 - 122, "mm", DIMC))
        L.append(("display", 150, title, W / 2, H / 2 - 8, "mm", BONE))
        if sub:
            L.append(("italic", 34, sub, W / 2, H / 2 + 96, "mm", BONE))
        return text_layer(L)
    if label:
        L.append(("mono", 14, label, x, y - 16, "ls", DIMC))
    L.append(("display", 54, title, x, y + 44, "ls", BONE))
    if sub:
        L.append(("mono", 15, sub, x, y + 78, "ls", DIMC))
    return text_layer(L)


def timeline(numbers):
    """Shot list on the bar grid. numbers: dict of simulated figures for the supers."""
    n = numbers
    T = []
    add = lambda start, dur, src, move=None, sup=None, fade=(0.0, 0.0), sup_in=0.35: T.append(dict(start=start, dur=dur, src=src, move=move or {}, sup=sup, fade=fade, sup_in=sup_in))
    add(0.0, 7.5, "macro_skin", dict(z0=1.16, z1=1.04, px=26, py=-8), None, fade=(2.4, 0.0))
    add(7.5, 5.0, "macro_knuckle", dict(z0=1.04, z1=1.13, px=-22), super_("316L STAINLESS STEEL", "Printed bone."))
    add(12.5, 7.5, "macro_tube", dict(z0=1.12, z1=1.03, px=20, py=10), super_("ONE LOOP", "Liquid marrow."))
    add(20.0, 5.0, "frame_only", dict(z0=1.03, z1=1.10, px=-18), super_("FIVE SEGMENTS · TEN BOLTS", "A skeleton.", f"LONGEST PART {n['longest']} MM · FITS A 300 MM BUILD PLATE"))
    add(25.0, 5.0, "organs", dict(z0=1.10, z1=1.03, px=22), super_("FOUR PRINTED ALUMINIUM PARTS", "Its organs.", "GYROID INSIDE · BONE SKIN OUTSIDE · NO GASKETS"))
    add(30.0, 4.85, "sim_flow", None, super_("RESOLVED, THEN SCALED UP", "Coolant through a gyroid.", "LATTICE-BOLTZMANN FLOW · FINITE-VOLUME HEAT"))
    add(35.0, 10.0, "orbit", dict(z0=1.0, z1=1.16, px=-44, py=8), super_("PRE-LAUNCH PREVIEW", "MARROW", "Printed bone. Liquid marrow.", big=True), sup_in=0.05)
    add(45.0, 5.0, "board", dict(z0=1.04, z1=1.12, px=-20, py=6), super_("ORGAN II", "CPU and VRM, one block."))
    add(50.0, 5.0, "gpu", dict(z0=1.12, z1=1.04, px=24), super_("ORGAN I", "A 4090, under gyroid.", f"{n['gpu_w']} W · DIE +{n['gpu_rise']} K OVER COOLANT · SIMULATED"))
    add(55.0, 5.0, "rear_ribs", dict(z0=1.03, z1=1.10, px=-24), super_("TWO 240 MM RADIATORS", "Ribs over lungs."))
    add(60.0, 5.0, "sim_heat", None, super_("FULL LOAD · SIMULATED", f"{n['gpu_hot']} °C at the hot spot.", f"{n['heat_w']} W THROUGH ONE LOOP · FANS AT 1500 RPM · 25 °C ROOM"))
    add(65.0, 5.0, "heart", dict(z0=1.10, z1=1.03, px=18, py=-6), super_("ORGAN III", "A heart with a 41 mL atrium."))
    add(70.0, 5.0, "hero_low", dict(z0=1.02, z1=1.10, px=-26), super_("EVERY PART FITS A 300 MM PRINTER", "Print it. Bolt it. Fill it."))
    add(75.0, 9.0, "hero_front", dict(z0=1.08, z1=1.02, px=14), super_("DESIGN STUDY · ALL FIGURES SIMULATED", "MARROW", "Printed bone. Liquid marrow.", big=True), fade=(0.0, 3.2), sup_in=0.05)
    return T


def load_numbers():
    p = ROOT / "site" / "data" / "site.json"
    n = dict(longest="283", gpu_w="450", gpu_rise="–", gpu_hot="–", heat_w="592")
    if p.exists():
        d = json.loads(p.read_text())
        try:
            n["longest"] = str(round(max(v["longest_mm"] for k, v in d["parts"].items() if v["material"] == "316L")))
            g = d["sims"]["macro"]["gpu_block_AlSi10Mg_3"]
            n["gpu_rise"] = f"{g['heaters']['1']['T_mean_rise_K']:.0f}"
            n["gpu_w"] = f"{g['energy_in_W']:.0f}"
            r = [x for x in d["sims"]["loop"]["sweep"] if x["material"] == "AlSi10Mg" and x["load"] == "full" and x["rpm"] == 1500][0]
            n["gpu_hot"] = f"{r['gpu_hotspot_c']:.0f}"
            n["heat_w"] = f"{r['heat_w']:.0f}"
        except Exception as ex:  # noqa: BLE001
            print("numbers incomplete:", ex)
    return n


def main(preview=False):
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = load_numbers()
    tl = timeline(numbers)
    srcs = {}
    for sh in tl:
        k = sh["src"]
        if k in srcs:
            continue
        if k == "orbit":
            srcs[k] = Sequence(ORBIT, "orbit_*.png")
        elif k.startswith("sim_"):
            srcs[k] = Sequence(SIMV / k, "*.png")
        else:
            srcs[k] = Still(k)
        if k == "orbit" and not srcs[k].ok:
            srcs[k] = Still("hero_right")          # no rendered orbit yet: drift across the three-quarter still instead
        print(k, "ok" if srcs[k].ok else "MISSING", flush=True)
    vig = vignette()
    total = int(84.0 * FPS)
    step = 6 if preview else 1
    raw = OUT / ("film_preview.mp4" if preview else "film_video.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS / step), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "17" if not preview else "26", "-pix_fmt", "yuv420p", str(raw)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rng = np.random.default_rng(1)
    for f in range(0, total, step):
        t = f / FPS
        frame = np.zeros((H, W, 3), np.float32)
        for sh in tl:
            if sh["start"] <= t < sh["start"] + sh["dur"]:
                u = (t - sh["start"]) / sh["dur"]
                s = srcs[sh["src"]]
                if s.ok:
                    frame = s.frame(u, sh["move"]).copy()
                fi, fo = sh["fade"]
                g = 1.0
                if fi > 0:
                    g *= ease((t - sh["start"]) / fi)
                if fo > 0:
                    g *= ease((sh["start"] + sh["dur"] - t) / fo)
                # short dip at each cut so hard cuts land softly on the beat
                g *= min(1.0, 0.35 + (t - sh["start"]) / 0.12) if sh["start"] > 0 and fi == 0 else 1.0
                frame = frame * vig * g
                if sh["sup"] is not None:
                    a = ease((t - sh["start"] - sh["sup_in"]) / 0.5) * ease((sh["start"] + sh["dur"] - 0.25 - t) / 0.45)
                    frame = comp(frame, sh["sup"], a * (g if fo > 0 else 1.0))
                break
        frame = frame + rng.normal(0, 0.006, frame.shape).astype(np.float32)        # fine grain against banding
        proc.stdin.write((np.clip(frame, 0, 1) * 255).astype(np.uint8).tobytes())
        if f % (FPS * 10) == 0:
            print(f"  {t:5.1f} s", flush=True)
    proc.stdin.close()
    proc.wait()
    if preview:
        print("wrote", raw)
        return
    # audio: theme at -1.5 dB (headroom for AAC), mux, plus a 15 MB web encode
    final = OUT / "marrow_film.mp4"
    wav = ROOT / "media" / "score" / "marrow_theme.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-i", str(wav), "-filter:a", "volume=-1.5dB", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(final)], check=True)
    web = OUT / "film.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(final), "-c:v", "libx264", "-preset", "slow", "-b:v", "1150k", "-maxrate", "1700k",
                    "-bufsize", "3400k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(web)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "38.5", "-i", str(final), "-frames:v", "1", "-q:v", "3", str(OUT / "film_poster.jpg")], check=True)
    print("wrote", final, round(final.stat().st_size / 1e6, 1), "MB;", web, round(web.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main(preview="--preview" in sys.argv)
