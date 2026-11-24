"""Turn simulation output into what the website and the film use: figures, per-vertex thermal
colours for the printed parts, mean-flow streamlines and one aggregated site.json.

    /opt/venv/bin/python -m sim.post.site_assets
"""
from __future__ import annotations

import base64
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import ndimage as ndi

from design import gpu_block as GB
from design import monoblock as MB
from design import params as P

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTD = ROOT / "sim" / "out"
SITE = ROOT / "site"
BUILD = ROOT / "build"

BG = "#0a1016"
INK = "#d7e0e7"
DIM = "#7f92a3"
RUBY = "#e0284a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK, "axes.labelcolor": DIM,
    "xtick.color": DIM, "ytick.color": DIM, "axes.edgecolor": "#24323f", "font.family": "DejaVu Sans Mono", "font.size": 8.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})

E = np.array([P.E_X, P.E_S, P.E_N]).T


def frame_origin(frame):
    if frame == "gpu":
        return np.array(P.gpu_to_world(0, 0, 0))
    if frame == "board":
        return np.array(P.board_to_world(0, 0, 0))
    raise ValueError(frame)


def to_world(pts, frame):
    return np.asarray(pts) @ E.T + frame_origin(frame)


def load(tag):
    d = OUTD / tag
    f = np.load(d / "fields.npz")
    s = json.loads((d / "summary.json").read_text())
    return f, s


def axes_of(f):
    o, h = f["origin_mm"], float(f["dx_mm"])
    sh = f["zone"].shape
    return [o[i] + h * np.arange(sh[i]) for i in range(3)], h


def cbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cb.set_label(label, color=DIM)
    cb.outline.set_edgecolor("#24323f")
    cb.ax.tick_params(length=2)
    return cb


def rect(ax, x0, y0, x1, y1, **kw):
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], **kw)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def fig_gpu(tag, out):
    f, s = load(tag)
    (xa, ya, za), h = axes_of(f)
    zone, U, Tf, Ts = f["zone"], f["U"], f["Tf"], f["Ts"]
    kz = int(np.argmin(np.abs(za - (GB.Z_FLOOR + 3.0))))
    G = P.GPU
    B = P.GPU_BLOCK
    ext = [xa[0], xa[-1], ya[0], ya[-1]]
    cx0, cx1 = B["chamber_x"]
    cy0, cy1 = B["chamber_y"]

    def decorate(ax, pads=True):
        dx_, dy_ = G["die"]
        rect(ax, GB.CX - dx_ / 2, GB.CY - dy_ / 2, GB.CX + dx_ / 2, GB.CY + dy_ / 2, color=RUBY, lw=1.1)
        if pads:
            for cx, cy, w, hg in G["mem"].values():
                rect(ax, cx - w / 2, cy - hg / 2, cx + w / 2, cy + hg / 2, color="#ffffff", lw=0.4, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x along the card (mm)")
        ax.set_ylabel("y (mm)")

    # (a) speed + streamlines in the core layer
    fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=200)
    core = zone[:, :, kz] >= 2
    sp_ = np.where(core, U[:, :, kz], np.nan)
    im = ax.imshow(sp_.T, origin="lower", extent=ext, cmap="bone", vmin=0, vmax=np.nanpercentile(sp_, 99.5))
    ux = np.where(core, f["ux"][:, :, kz], 0.0)
    uy = np.where(core, f["uy"][:, :, kz], 0.0)
    ax.streamplot(xa, ya, ux.T, uy.T, color="#ffffff", linewidth=0.45, density=2.3, arrowsize=0.5, arrowstyle="-|>")
    decorate(ax)
    jw, jl = B["jet"]
    rect(ax, GB.CX - jw / 2, GB.CY - jl / 2, GB.CX + jw / 2, GB.CY + jl / 2, color=RUBY, lw=0.8, ls="--")
    ax.set_xlim(cx0 - 4, cx1 + 4)
    ax.set_ylim(cy0 - 4, cy1 + 4)
    cbar(fig, im, ax, "superficial coolant speed (m/s)")
    ax.set_title(f"Gyroid core, 3 mm above the die floor  ·  {s['flow_lpm']:.1f} L/min  ·  Δp {s['dp_Pa']/1000:.1f} kPa", loc="left", fontsize=9)
    fig.savefig(out / "sim_gpu_flow.png", bbox_inches="tight")
    plt.close(fig)

    # (b) coolant temperature rise in the core
    fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=200)
    tf = np.where(core, Tf[:, :, kz], np.nan)
    im = ax.imshow(tf.T, origin="lower", extent=ext, cmap="inferno", vmin=0)
    decorate(ax)
    ax.set_xlim(cx0 - 4, cx1 + 4)
    ax.set_ylim(cy0 - 4, cy1 + 4)
    cbar(fig, im, ax, "coolant temperature rise over inlet (K)")
    ax.set_title("Coolant warming as it spreads from the nozzle", loc="left", fontsize=9)
    fig.savefig(out / "sim_gpu_coolant.png", bbox_inches="tight")
    plt.close(fig)

    # (c) metal temperature seen from below (max over the lowest few mm): die, memory, VRM lands
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=200)
    kb = int(np.argmin(np.abs(za - (GB.Z_UNDER + 0.6))))
    ts = np.nanmax(np.where(np.isfinite(Ts[:, :, :kb + 1]), Ts[:, :, :kb + 1], -1), axis=2)
    ts = np.where(ts < 0, np.nan, ts)
    im = ax.imshow(ts.T, origin="lower", extent=ext, cmap="inferno", vmin=0)
    decorate(ax, pads=False)
    cbar(fig, im, ax, "metal temperature rise over coolant inlet (K)")
    h1 = s["heaters"]["1"]
    ax.set_title(f"Underside of the block at {s['energy_in_W']:.0f} W  ·  die pad mean +{h1['T_mean_rise_K']:.1f} K, peak +{h1['T_max_rise_K']:.1f} K  ·  {s['material']}",
                 loc="left", fontsize=9)
    fig.savefig(out / "sim_gpu_metal.png", bbox_inches="tight")
    plt.close(fig)

    # (d) section through the die centre
    fig, ax = plt.subplots(figsize=(10.5, 3.3), dpi=200)
    jy = int(np.argmin(np.abs(ya - GB.CY)))
    sec = Ts[:, jy, :]
    secf = Tf[:, jy, :]
    zc = zone[:, jy, :]
    show_ = np.where(zc == 1, sec, np.where(zc >= 2, secf, np.nan))
    im = ax.imshow(show_.T, origin="lower", extent=[xa[0], xa[-1], za[0], za[-1]], cmap="inferno", vmin=0, aspect="equal")
    ax.contour(xa, za, (zc >= 3).T.astype(float), levels=[0.5], colors="#ffffff", linewidths=0.4, alpha=0.6)
    ax.set_xlim(cx0 - 14, cx1 + 14)
    ax.set_ylim(za[0], GB.Z_ROOF1 + 1.5)
    ax.set_xlabel("x along the card (mm)")
    ax.set_ylabel("z above PCB (mm)")
    cbar(fig, im, ax, "temperature rise (K)")
    ax.set_title("Section through the die centre: metal where solid, coolant inside the lattice (outlined)", loc="left", fontsize=9)
    fig.savefig(out / "sim_gpu_section.png", bbox_inches="tight")
    plt.close(fig)


def fig_mono(tag, out):
    f, s = load(tag)
    (xa, ya, za), h = axes_of(f)
    zone, U, Tf, Ts = f["zone"], f["U"], f["Tf"], f["Ts"]
    kz = int(np.argmin(np.abs(za - 13.0)))
    ext = [xa[0], xa[-1], ya[0], ya[-1]]
    fl = zone[:, :, kz] >= 2
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 6.2), dpi=200)
    sp_ = np.where(fl, U[:, :, kz], np.nan)
    im = axs[0].imshow(sp_.T, origin="lower", extent=ext, cmap="bone", vmin=0, vmax=np.nanpercentile(sp_, 99.5))
    ux = np.where(fl, f["ux"][:, :, kz], 0.0)
    uy = np.where(fl, f["uy"][:, :, kz], 0.0)
    axs[0].streamplot(xa, ya, ux.T, uy.T, color="#ffffff", linewidth=0.45, density=1.8, arrowsize=0.5)
    cbar(fig, im, axs[0], "superficial coolant speed (m/s)")
    axs[0].set_title(f"Coolant path at z = 13 mm  ·  {s['flow_lpm']:.1f} L/min  ·  Δp {s['dp_Pa']/1000:.1f} kPa", loc="left", fontsize=9)
    kb = int(np.argmin(np.abs(za - (MB.Z_UNDER + 0.6))))
    ts = np.nanmax(np.where(np.isfinite(Ts[:, :, :kb + 1]), Ts[:, :, :kb + 1], -1), axis=2)
    ts = np.where(ts < 0, np.nan, ts)
    im = axs[1].imshow(ts.T, origin="lower", extent=ext, cmap="inferno", vmin=0)
    cbar(fig, im, axs[1], "metal temperature rise over inlet (K)")
    h1 = s["heaters"]["1"]
    axs[1].set_title(f"Underside at {s['energy_in_W']:.0f} W  ·  CPU pad mean +{h1['T_mean_rise_K']:.1f} K", loc="left", fontsize=9)
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlabel("board x (mm)")
        ax.set_ylabel("board y (mm)")
    fig.savefig(out / "sim_mono.png", bbox_inches="tight")
    plt.close(fig)


def fig_cell(out):
    p = OUTD / "cell" / "core_field.npz"
    if not p.exists():
        return
    d = np.load(p)
    u, solid, dx = d["u"], d["solid"], float(d["dx"])
    sp_ = np.sqrt((u ** 2).sum(0))
    sp_ = np.where(solid, np.nan, sp_)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=200)
    n = solid.shape[0]
    k = solid.shape[2] // 2
    a = np.tile(sp_[:, :, k], (3, 2))
    im = axs[0].imshow(a.T, origin="lower", extent=[0, 3 * n * dx * 1e3, 0, 2 * n * dx * 1e3], cmap="bone", vmin=0)
    axs[0].set_title("plan view through the layer (flow left to right)", loc="left", fontsize=9)
    axs[0].set_xlabel("mm")
    b = np.tile(sp_[:, n // 3, :], (3, 1))
    im2 = axs[1].imshow(b.T, origin="lower", extent=[0, 3 * n * dx * 1e3, 0, solid.shape[2] * dx * 1e3], cmap="bone", vmin=0)
    axs[1].set_title("side view: floor below, divider above", loc="left", fontsize=9)
    axs[1].set_xlabel("mm")
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_facecolor("#2a3440")
    cbar(fig, im, axs[0], "speed (m/s)")
    cbar(fig, im2, axs[1], "speed (m/s)")
    fig.savefig(out / "sim_cell.png", bbox_inches="tight")
    plt.close(fig)


def fig_fea(out):
    p = OUTD / "fea" / "fields.npz"
    if not p.exists():
        return
    d = np.load(p)
    ec = d["ecen"]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 5.4), dpi=200)
    for ax, key, title in zip(axs, ("vm1", "vm2", "vm3"), ("standing, 1 g", "standing, 1 g + 1 g sideways", "lifted by the crown")):
        vm = d[key] / 1e6
        o = np.argsort(vm)
        side = key != "vm2"
        hx = ec[o, 1] if side else ec[o, 0]
        sc = ax.scatter(hx, ec[o, 2], c=vm[o], s=0.35, cmap="inferno", vmin=0, vmax=np.percentile(vm, 99.9), linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("depth y (mm)" if side else "width x (mm)")
        ax.set_ylabel("height z (mm)")
        cbar(fig, sc, ax, "von Mises stress (MPa)")
    fig.savefig(out / "sim_fea.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# per-vertex thermal colours + streamlines
# ----------------------------------------------------------------------------
def fill_nearest(a):
    bad = ~np.isfinite(a)
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def vertex_colours(tag, part, tmax=None):
    import trimesh

    f, s = load(tag)
    (xa, ya, za), h = axes_of(f)
    Ts = fill_nearest(np.asarray(f["Ts"], dtype=np.float32))
    m = trimesh.load(BUILD / f"{part}_web.ply", process=False)
    v = np.asarray(m.vertices)
    c = [(v[:, i] - f["origin_mm"][i]) / h for i in range(3)]
    t = ndi.map_coordinates(Ts, c, order=1, mode="nearest")
    tmax = float(np.nanmax(f["Ts"])) if tmax is None else tmax
    col = (cm.inferno(np.clip(t / tmax, 0, 1))[:, :3] * 255).astype(np.uint8)
    np.savez_compressed(BUILD / f"{part}_thermal.npz", colors=col, t=t.astype(np.float32), tmax=tmax)
    return tmax


def streamlines(tag, frame, n=140, seed=3, step=0.35, max_steps=6000, keep=110):
    f, s = load(tag)
    (xa, ya, za), h = axes_of(f)
    zone = f["zone"]
    fl = zone >= 2
    vel = [np.where(fl, f[k], 0.0).astype(np.float32) for k in ("ux", "uy", "uz")]
    Tf = fill_nearest(np.asarray(f["Tf"], dtype=np.float32))
    rng = np.random.default_rng(seed)
    seeds = np.argwhere(f["inlet"])
    seeds = seeds[rng.choice(len(seeds), size=min(n, len(seeds)), replace=False)].astype(np.float64) + rng.uniform(-0.4, 0.4, (min(n, len(seeds)), 3))
    outl = ndi.binary_dilation(f["outlet"], iterations=1)
    shape = np.array(zone.shape)

    def sample(p):
        c = p.T
        u = np.stack([ndi.map_coordinates(v, c, order=1, mode="nearest") for v in vel], axis=1)
        nrm = np.linalg.norm(u, axis=1, keepdims=True)
        return u / np.maximum(nrm, 1e-9), nrm[:, 0]

    P_ = seeds.copy()
    alive = np.ones(len(P_), dtype=bool)
    paths = [[p.copy()] for p in P_]
    speeds = [[0.0] for _ in P_]
    for it in range(max_steps):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        p = P_[idx]
        k1, sp1 = sample(p)
        k2, _ = sample(p + 0.5 * step * k1)
        k3, _ = sample(p + 0.5 * step * k2)
        k4, _ = sample(p + step * k3)
        pn = p + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        P_[idx] = pn
        ii = np.clip(np.round(pn).astype(int), 0, shape - 1)
        inside = fl[ii[:, 0], ii[:, 1], ii[:, 2]] & (sp1 > 1e-5)
        done = outl[ii[:, 0], ii[:, 1], ii[:, 2]]
        for j, g in enumerate(idx):
            if it % 3 == 0 or done[j]:
                paths[g].append(pn[j].copy())
                speeds[g].append(float(sp1[j]))
        alive[idx[~inside | done]] = False
    out = []
    for pth, spd in zip(paths, speeds):
        a = np.asarray(pth)
        if len(a) < 12:
            continue
        sel = np.unique(np.linspace(0, len(a) - 1, min(keep, len(a))).astype(int))
        a = a[sel]
        sp_ = np.asarray(spd)[sel]
        t = ndi.map_coordinates(Tf, a.T, order=1, mode="nearest")
        loc = a * h + f["origin_mm"]
        w = to_world(loc, frame)
        out.append(np.concatenate([w, t[:, None], sp_[:, None]], axis=1).astype(np.float32))
    return out


def pack_lines(lines):
    """-> dict(counts=[...], data=base64 float32 xyz,t,speed)"""
    counts = [int(len(a)) for a in lines]
    data = np.concatenate(lines, axis=0).astype(np.float32) if lines else np.zeros((0, 5), np.float32)
    return dict(counts=counts, stride=5, fields=["x", "y", "z", "dT", "speed"], b64=base64.b64encode(data.tobytes()).decode())


# ----------------------------------------------------------------------------
def main():
    (SITE / "sim").mkdir(parents=True, exist_ok=True)
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    out = SITE / "sim"
    site = dict(layout=dict(envelope=P.layout()["envelope"], tilt=P.TILT), print=P.PRINT, parts={}, sims={})
    for p in sorted((ROOT / "exports" / "meta").glob("*.json")):
        site["parts"][p.stem] = json.loads(p.read_text())
    for name in ("cell/closures", "fea/summary", "loop/loop", "m2_sink/summary", "validation/validate_cell"):
        p = OUTD / (name + ".json")
        if p.exists():
            site["sims"][name.split("/")[0]] = json.loads(p.read_text())
    macro = {}
    for d in sorted(OUTD.glob("*_*_*")):
        if (d / "summary.json").exists() and d.name.split("_")[0] in ("gpu", "monoblock"):
            macro[d.name] = json.loads((d / "summary.json").read_text())
    site["sims"]["macro"] = macro
    ref_g, ref_m = "gpu_block_AlSi10Mg_3", "monoblock_AlSi10Mg_3"
    flows = {}
    if (OUTD / ref_g / "fields.npz").exists():
        fig_gpu(ref_g, out)
        tmax = vertex_colours(ref_g, "gpu_block")
        site["sims"]["gpu_thermal_scale_K"] = tmax
        flows["gpu"] = pack_lines(streamlines(ref_g, "gpu"))
    if (OUTD / ref_m / "fields.npz").exists():
        fig_mono(ref_m, out)
        # one colour scale for both blocks, so the legend on the site reads true for either
        tmax = vertex_colours(ref_m, "monoblock", tmax=site["sims"].get("gpu_thermal_scale_K"))
        site["sims"]["mono_thermal_scale_K"] = tmax
        flows["mono"] = pack_lines(streamlines(ref_m, "board", n=110))
    fig_cell(out)
    fig_fea(out)
    (SITE / "data" / "flow.json").write_text(json.dumps(flows))
    (SITE / "data" / "site.json").write_text(json.dumps(site, indent=1))
    print("site assets written:", sorted(p.name for p in out.glob("*.png")), {k: len(v["counts"]) for k, v in flows.items()})


if __name__ == "__main__":
    main()
