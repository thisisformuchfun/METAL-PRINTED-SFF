"""Animated simulation plates for the film, drawn straight from the solver output.

  sim_flow   tracer particles advected through the 4090 block's gyroid core (block-level mean flow)
  sim_heat   metal temperature on the underside of the block, revealed from the die outwards

    /opt/venv/bin/python media/film/simviz.py
"""
from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
from matplotlib import cm

from design import gpu_block as GB
from design import params as P

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTD = ROOT / "sim" / "out"
DST = ROOT / "build" / "simviz"
W, H = 1280, 720
BG = np.array([0.086, 0.063, 0.039], np.float32)          # BGR of #0a1016


def load(tag):
    f = np.load(OUTD / tag / "fields.npz")
    s = json.loads((OUTD / tag / "summary.json").read_text())
    return f, s


class View:
    """Maps part millimetres to frame pixels (y up)."""

    def __init__(self, x0, x1, y0, y1, margin=70):
        sc = min((W - 2 * margin) / (x1 - x0), (H - 2 * margin) / (y1 - y0))
        self.sc = sc
        self.ox = W / 2 - sc * (x0 + x1) / 2
        self.oy = H / 2 + sc * (y0 + y1) / 2

    def px(self, x, y):
        return self.ox + self.sc * np.asarray(x), self.oy - self.sc * np.asarray(y)

    def image(self, arr, xa, ya, cmap, vmin, vmax, mask):
        """Resample a (nx, ny) field onto the frame."""
        u = (np.arange(W) - self.ox) / self.sc
        v = (self.oy - np.arange(H)) / self.sc
        h = xa[1] - xa[0]
        mx = np.tile(((u - xa[0]) / h).astype(np.float32), (H, 1))
        my = np.tile(((v - ya[0]) / h).astype(np.float32)[:, None], (1, W))
        a = cv2.remap(np.nan_to_num(arr, nan=vmin).T.astype(np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=float(vmin))
        m = cv2.remap(mask.T.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        rgb = cmap(np.clip((a - vmin) / (vmax - vmin), 0, 1))[..., :3].astype(np.float32)
        return rgb[..., ::-1], np.clip(m, 0, 1)[..., None], (mx, my)


def rect(img, view, x0, y0, x1, y1, col, th=2):
    (a, b), (c, d) = view.px(x0, y0), view.px(x1, y1)
    cv2.rectangle(img, (int(a), int(d)), (int(c), int(b)), col, th, cv2.LINE_AA)


def flow(tag="gpu_block_AlSi10Mg_3", frames=118):
    f, s = load(tag)
    o, h = f["origin_mm"], float(f["dx_mm"])
    sh = f["zone"].shape
    xa, ya, za = [o[i] + h * np.arange(sh[i]) for i in range(3)]
    kz = int(np.argmin(np.abs(za - (GB.Z_FLOOR + 3.0))))
    core = f["zone"][:, :, kz] >= 2
    U = np.where(core, f["U"][:, :, kz], 0.0)
    ux = np.where(core, f["ux"][:, :, kz], 0.0).astype(np.float32)
    uy = np.where(core, f["uy"][:, :, kz], 0.0).astype(np.float32)
    B = P.GPU_BLOCK
    v = View(B["chamber_x"][0] - 6, B["chamber_x"][1] + 6, B["chamber_y"][0] - 6, B["chamber_y"][1] + 6, margin=40)
    base, m, _ = v.image(U, xa, ya, cm.bone, 0.0, float(np.percentile(U[core], 99.5)), core)
    base = base * 0.55 * m + BG * (1 - m)
    rng = np.random.default_rng(4)
    jw, jl = B["jet"]
    n = 2600

    def spawn(k):
        return np.stack([GB.CX + rng.uniform(-jw, jw, k), GB.CY + rng.uniform(-jl / 2, jl / 2, k)], 1)

    p = spawn(n)
    age = rng.uniform(0, 1, n)
    trail = np.zeros((H, W, 3), np.float32)
    umax = float(np.percentile(np.hypot(ux, uy)[core], 99))
    dt = 0.55 / umax                                            # mm per frame at peak speed ~0.55 mm... scaled below
    (DST / "sim_flow").mkdir(parents=True, exist_ok=True)
    for fr in range(-70, frames):
        for sub in range(3):
            ci = np.stack([(p[:, 0] - xa[0]) / h, (p[:, 1] - ya[0]) / h], 0).astype(np.float32)
            vx = cv2.remap(ux.T, ci[0][None], ci[1][None], cv2.INTER_LINEAR)[0]
            vy = cv2.remap(uy.T, ci[0][None], ci[1][None], cv2.INTER_LINEAR)[0]
            sp = np.hypot(vx, vy)
            # compress the speed range so slow regions still visibly move
            g = (sp / umax) ** 0.45 / np.maximum(sp, 1e-9)
            q = p + np.stack([vx, vy], 1) * g[:, None] * 0.85
            a_, b_ = v.px(p[:, 0], p[:, 1])
            c_, d_ = v.px(q[:, 0], q[:, 1])
            alive = sp > 1e-5
            col = cm.inferno(0.45 + 0.5 * np.clip(sp / umax, 0, 1))[:, :3]
            for i in np.where(alive)[0]:
                cv2.line(trail, (int(a_[i] * 4), int(b_[i] * 4)), (int(c_[i] * 4), int(d_[i] * 4)),
                         (float(col[i, 2]) * 0.5, float(col[i, 1]) * 0.5, float(col[i, 0]) * 0.5), 1, cv2.LINE_AA, 2)
            p = q
            age += 0.006
            dead = ~alive | (age > 1.0)
            k = int(dead.sum())
            if k:
                p[dead] = spawn(k)
                age[dead] = 0.0
        trail *= 0.90
        if fr < 0:
            continue
        img = base + np.clip(trail, 0, 2.5) * 1.15
        dx_, dy_ = P.GPU["die"]
        rect(img, v, GB.CX - dx_ / 2, GB.CY - dy_ / 2, GB.CX + dx_ / 2, GB.CY + dy_ / 2, (0.45, 0.3, 1.0), 2)
        for cx, cy, w_, hg in P.GPU["mem"].values():
            rect(img, v, cx - w_ / 2, cy - hg / 2, cx + w_ / 2, cy + hg / 2, (0.75, 0.75, 0.75), 1)
        z = 1.0 + 0.06 * fr / frames
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=BG.tolist())
        cv2.imwrite(str(DST / "sim_flow" / f"f_{fr:04d}.png"), (np.clip(img, 0, 1) * 255).astype(np.uint8))
    print("sim_flow frames:", frames)


def heat(tag="gpu_block_AlSi10Mg_3", frames=120):
    f, s = load(tag)
    o, h = f["origin_mm"], float(f["dx_mm"])
    sh = f["zone"].shape
    xa, ya, za = [o[i] + h * np.arange(sh[i]) for i in range(3)]
    Ts = f["Ts"]
    kb = int(np.argmin(np.abs(za - (GB.Z_UNDER + 0.6))))
    ts = np.nanmax(np.where(np.isfinite(Ts[:, :, :kb + 1]), Ts[:, :, :kb + 1], -1), axis=2)
    mask = ts >= 0
    ts = np.where(mask, ts, 0.0)
    B = P.GPU_BLOCK
    ix, iy = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    v = View(xa[ix[0]] - 4, xa[ix[-1]] + 4, ya[iy[0]] - 4, ya[iy[-1]] + 4, margin=36)
    tmax = float(ts.max())
    rgb, m, (mx, my) = v.image(ts, xa, ya, cm.inferno, 0.0, tmax, mask)
    u = (np.arange(W) - v.ox) / v.sc
    w_ = (v.oy - np.arange(H)) / v.sc
    rr = np.hypot(u[None, :] - GB.CX, w_[:, None] - GB.CY)
    (DST / "sim_heat").mkdir(parents=True, exist_ok=True)
    for fr in range(frames):
        t = fr / (frames - 1)
        R = 6 + 150 * (min(t / 0.55, 1.0)) ** 1.4               # reveal radius (mm)
        k = np.clip((R - rr) / 14.0, 0, 1)[..., None].astype(np.float32)
        grey = (0.10 + 0.10 * m) * np.ones(3, np.float32)
        img = (rgb * k + grey * (1 - k)) * m + BG * (1 - m)
        edge = np.exp(-((rr - R) / 2.2) ** 2)[..., None].astype(np.float32) * (1.0 if t < 0.55 else max(0.0, 1 - (t - 0.55) * 8))
        img = img + edge * m * np.array([0.25, 0.25, 0.9], np.float32)
        z = 1.0 + 0.05 * t
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=BG.tolist())
        cv2.imwrite(str(DST / "sim_heat" / f"f_{fr:04d}.png"), (np.clip(img, 0, 1) * 255).astype(np.uint8))
    print("sim_heat frames:", frames, "scale 0 ..", round(tmax, 1), "K")


if __name__ == "__main__":
    flow()
    heat()
