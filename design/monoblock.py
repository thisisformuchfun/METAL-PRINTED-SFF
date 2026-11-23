"""ORGAN II - CPU + VRM monoblock for a mini-ITX AM5 board, AlSi10Mg, monolithic.

Coordinates: BOARD frame (see params.py), z up from the PCB top surface.

Coolant path: inlet port -> plenum -> slot nozzle over the CCD hot spot -> 5 mm sheet-gyroid
core over the CPU -> perimeter riser -> gallery -> bridge duct -> left VRM arm -> corner ->
top VRM arm -> outlet port. The arms carry a coarse 8 mm gyroid (low pressure drop, still
self-supporting). Four lugs take AM5 spring screws into the stock backplate (UNC 6-32).

The VRM layout is a REPRESENTATIVE one (board model unknown): the nine numbers under
BOARD['vrm_left'] / ['vrm_top'] / MONOBLOCK['arm_*'] must be set from the real board.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import params as P
from . import part, sdf

BD = P.BOARD
M = P.MONOBLOCK
LT = P.LATTICE

SX, SY = BD["socket"]
Z_C = BD["z_ihs"]                       # contact plane
Z_UNDER = Z_C + 0.6
Z_FLOOR = Z_C + M["floor"]
Z_CORE1 = Z_FLOOR + M["core_h"]
Z_DIV1 = Z_CORE1 + M["divider"]
Z_GAL1 = Z_DIV1 + M["gallery_h"]
Z_ROOF1 = Z_GAL1 + M["roof"]
Z_BOSS = Z_ROOF1 + M["skin_h"] + 0.3
AZ0, AZ1 = M["arm_z"]
CZ0, CZ1 = M["arm_channel_z"]
Z_ABOSS = AZ1 + M["arm_skin_h"] + 0.3
CH = M["chamber"]
W = M["wall"]
IN = (SX + M["inlet_offset"][0], SY + M["inlet_offset"][1])
HOT = (SX + BD["hotspot_offset"][0], SY + BD["hotspot_offset"][1])
OUT = M["outlet"]


def lugs():
    ax, ay = BD["am5_holes"]
    return [(SX - ax / 2, SY - ay / 2), (SX + ax / 2, SY - ay / 2), (SX - ax / 2, SY + ay / 2), (SX + ax / 2, SY + ay / 2)]


def body2(x, y, grow=0.0):
    return sdf.rect2(x, y, (SX - CH[0] / 2 - W - grow, SY - CH[1] / 2 - W - grow),
                     (SX + CH[0] / 2 + W + grow, SY + CH[1] / 2 + W + grow), r=7.0 + grow)


def chamber2(x, y, shrink=0.0):
    return sdf.rect2(x, y, (SX - CH[0] / 2 + shrink, SY - CH[1] / 2 + shrink),
                     (SX + CH[0] / 2 - shrink, SY + CH[1] / 2 - shrink), r=max(5.0 - shrink, 1.5))


def arms2(x, y, shrink=0.0):
    """L-shaped arm outline (negative inside); shrink > 0 gives the channel interior."""
    L, T = M["arm_left"], M["arm_top"]
    a = sdf.rect2(x, y, (L["x"][0] + shrink, L["y"][0] + shrink), (L["x"][1] - shrink, L["y"][1] + 6.0), r=max(4.0 - shrink, 1.2))
    b = sdf.rect2(x, y, (T["x"][0] + shrink, T["y"][0] + shrink), (T["x"][1] - shrink, T["y"][1] - shrink), r=max(4.0 - shrink, 1.2))
    return np.minimum(a, b)


def build(h=0.25, verbose=True):
    tm = part.Timer()
    grid = sdf.Grid((25.0, 43.0, 1.5), (124.0, 171.0, Z_BOSS + 1.5), h)
    if verbose:
        print(f"monoblock grid {grid.shape} = {np.prod(grid.n) / 1e6:.1f} M voxels")
    x, y, z = grid.coords()

    b2 = body2(x, y)
    a2 = arms2(x, y)
    by0, by1 = M["bridge_y"]
    L = M["arm_left"]

    # ---------------- solids ----------------
    cpu = sdf.smax(b2, np.maximum(Z_UNDER - z, z - Z_ROOF1), 1.4)
    arm = sdf.smax(a2, np.maximum(AZ0 - z, z - AZ1), 1.2)
    neck = sdf.box(x, y, z, (L["x"][1] - 2.5, by0, Z_FLOOR), (SX - CH[0] / 2 - W + 2.5, by1, AZ1), r=1.5)
    f = np.minimum(np.minimum(cpu, arm), neck)
    del cpu, arm, neck
    # contact land
    c = M["contact"]
    sdf.add_box(f, grid, (SX - c[0] / 2, SY - c[1] / 2, Z_C), (SX + c[0] / 2, SY + c[1] / 2, Z_UNDER + 0.6))
    # VRM lands: power stages (pad) - rails hanging down from the arms
    vl, vt = BD["vrm_left"], BD["vrm_top"]
    zdr = BD["z_drmos"] + 1.5
    dw, dh = BD["drmos"]
    yl = [vl["y0"] + i * vl["pitch"] for i in range(vl["n"])]
    xt = [vt["x0"] + i * vt["pitch"] for i in range(vt["n"])]
    sdf.add_box(f, grid, (vl["drmos_x"] - dw / 2 - 0.6, yl[0] - dh / 2 - 0.8, zdr),
                (vl["drmos_x"] + dw / 2 + 0.6, yl[-1] + dh / 2 + 0.8, AZ0 + 0.6))
    sdf.add_box(f, grid, (xt[0] - dh / 2 - 0.8, vt["drmos_y"] - dw / 2 - 0.6, zdr),
                (xt[-1] + dh / 2 + 0.8, vt["drmos_y"] + dw / 2 + 0.6, AZ0 + 0.6))
    tm("bodies + lands")

    # ---------------- AM5 lugs and legs ----------------
    lz0, lz1 = M["lug_z"]
    zc = 0.5 * (lz0 + lz1)
    corners = [(SX - CH[0] / 2 + 1, SY - CH[1] / 2 + 1), (SX + CH[0] / 2 - 1, SY - CH[1] / 2 + 1),
               (SX - CH[0] / 2 + 1, SY + CH[1] / 2 - 1), (SX + CH[0] / 2 - 1, SY + CH[1] / 2 - 1)]
    for (lx, ly), (cx, cy) in zip(lugs(), corners):
        sdf.add_capsule(f, grid, (cx, cy, zc + 0.6), (lx, ly, zc), 4.4, 3.6, k=2.5)
        r = M["lug_d"] / 2
        sdf.apply(f, grid, (lx - r, ly - r, lz0), (lx + r, ly + r, lz1),
                  lambda X, Y, Z, lx=lx, ly=ly, r=r: sdf.smax(sdf.circle2(X, Y, (lx, ly), r),
                                                              np.maximum(lz0 - Z, Z - lz1), 0.6), op="union", k=1.5)
    tm("lugs")

    # ---------------- CPU wet chamber ----------------
    ch_lo = (SX - CH[0] / 2, SY - CH[1] / 2, Z_FLOOR)
    ch_hi = (SX + CH[0] / 2, SY + CH[1] / 2, Z_GAL1)
    sl = grid.box(ch_lo, ch_hi, pad=1.0)
    xs, ys, zs = grid.coords(sl)
    ch2 = chamber2(xs, ys)
    f[sl] = np.maximum(f[sl], -np.maximum(ch2, np.maximum(Z_FLOOR - zs, zs - Z_GAL1)))
    a5 = 2 * np.pi / LT["wet_cell"]
    wall = LT["wet_wall"] + LT["wet_root"] * np.exp(-np.maximum(zs - Z_FLOOR, 0.0) / LT["wet_root_len"])
    core = sdf.gyroid_sheet(grid, sl, LT["wet_cell"], wall.astype(np.float32),
                            phase=(-a5 * HOT[0], -a5 * HOT[1], -a5 * Z_FLOOR + 0.6))
    z_split = Z_CORE1 + 0.5
    core = np.maximum(core, zs - z_split)
    a8 = 2 * np.pi / LT["gallery_cell"]
    gph = (-a8 * SX + 0.9, -a8 * SY, -a8 * Z_DIV1)
    gal = np.maximum(sdf.gyroid_sheet(grid, sl, LT["gallery_cell"], LT["gallery_wall"], phase=gph), z_split - zs)
    lat = np.minimum(core, gal)
    del core, gal
    lat = np.maximum(lat, np.maximum(ch2 - 0.45, np.maximum(Z_FLOOR - 0.4 - zs, zs - Z_GAL1 - 0.4)))
    f[sl] = np.minimum(f[sl], lat)
    del lat
    div = np.maximum(chamber2(xs, ys, shrink=M["riser_gap"]), np.maximum(Z_CORE1 - zs, zs - Z_DIV1))
    f[sl] = np.minimum(f[sl], div)
    # inlet plenum + slot nozzle (slot runs along x over the hot spot)
    jl, jw = M["jet"]
    plo = (min(IN[0], HOT[0]) - jl / 2 - 1.0, min(IN[1], HOT[1]) - 7.0)
    phi = (max(IN[0], HOT[0]) + jl / 2 + 1.0, max(IN[1], HOT[1]) + 7.0)
    pl_o = sdf.rect2(xs, ys, (plo[0] - 1.4, plo[1] - 1.4), (phi[0] + 1.4, phi[1] + 1.4), r=6.4)
    pl_i = sdf.rect2(xs, ys, plo, phi, r=5.0)
    f[sl] = np.minimum(f[sl], np.maximum(pl_o, np.maximum(Z_DIV1 - 0.2 - zs, zs - Z_GAL1 - 0.2)))
    f[sl] = np.maximum(f[sl], -np.maximum(pl_i, np.maximum(Z_DIV1 - zs, zs - Z_GAL1)))
    slot = sdf.rect2(xs, ys, (HOT[0] - jl / 2, HOT[1] - jw / 2), (HOT[0] + jl / 2, HOT[1] + jw / 2), r=jw / 2 - 0.05)
    f[sl] = np.maximum(f[sl], -np.maximum(slot, np.maximum(Z_CORE1 - 0.3 - zs, zs - Z_DIV1 - 0.3)))
    tm("cpu chamber")

    # ---------------- arm channel + bridge duct ----------------
    T = M["arm_top"]
    al = grid.box((L["x"][0], L["y"][0], CZ0), (T["x"][1], T["y"][1], CZ1), pad=1.0)
    xa, ya, za = grid.coords(al)
    ai = arms2(xa, ya, shrink=W)
    ch_arm = np.maximum(ai, np.maximum(CZ0 - za, za - CZ1))
    f[al] = np.maximum(f[al], -ch_arm)
    galA = sdf.gyroid_sheet(grid, al, LT["gallery_cell"], LT["gallery_wall"], phase=gph)
    galA = np.maximum(galA, np.maximum(ai - 0.45, np.maximum(CZ0 - 0.4 - za, za - CZ1 - 0.4)))
    f[al] = np.minimum(f[al], galA)
    del galA
    # bridge duct from the CPU gallery into the left arm (lattice-filled for support)
    dz0, dz1 = Z_DIV1 + 0.3, min(CZ1, Z_GAL1) - 0.1
    d_lo = (L["x"][1] - W - 0.6, by0 + 2.0, dz0)
    d_hi = (SX - CH[0] / 2 + 0.6, by1 - 2.0, dz1)
    bl = grid.box(d_lo, d_hi, pad=1.0)
    xb, yb, zb = grid.coords(bl)
    duct = sdf.box(xb, yb, zb, d_lo, d_hi, r=1.2)
    f[bl] = np.maximum(f[bl], -duct)
    galB = np.maximum(sdf.gyroid_sheet(grid, bl, LT["gallery_cell"], LT["gallery_wall"], phase=gph), duct - 0.45)
    f[bl] = np.minimum(f[bl], galB)
    # clear lattice under the outlet port
    sdf.apply(f, grid, (OUT[0] - 6.3, OUT[1] - 6.3, CZ0 + 3.0), (OUT[0] + 6.3, OUT[1] + 6.3, CZ1 + 0.1),
              lambda X, Y, Z: sdf.cyl_z(X, Y, Z, OUT, 6.3, CZ0 + 3.0, CZ1 + 0.1), op="cut")
    tm("arm channel + bridge")

    # ---------------- port bosses ----------------
    rs = P.LOOP["port_seat_d"] / 2
    for (px, py), z0, z1 in ((IN, Z_ROOF1 - 1.0, Z_BOSS), (OUT, AZ1 - 1.0, Z_ABOSS)):
        sdf.apply(f, grid, (px - rs, py - rs, z0), (px + rs, py + rs, z1),
                  lambda X, Y, Z, px=px, py=py, z0=z0, z1=z1: sdf.smax(sdf.circle2(X, Y, (px, py), rs),
                                                                       np.maximum(z0 - Z, Z - z1), 0.8), op="union", k=2.0)

    # ---------------- bone skin ----------------
    zs0 = AZ1 - 0.6
    sk = grid.box((25.0, 43.0, zs0), (124.0, 171.0, Z_BOSS), pad=0.0)
    xk, yk, zk = grid.coords(sk)
    b2k = body2(xk, yk)
    a2k = arms2(xk, yk)
    seeds = sdf.hex_seeds((25.0, 43.0), (124.0, 171.0), 24.0, seed=11)
    pillow = 0.10 + 0.90 * sdf.smoothstep(sdf.voronoi2_edge(xk, yk, seeds) / 3.8)
    top_c = Z_ROOF1 + 1.5 + (M["skin_h"] - 1.5) * sdf.smoothstep(-b2k / 9.0) * pillow
    top_a = AZ1 + 1.3 + (M["arm_skin_h"] - 1.3) * sdf.smoothstep(-a2k / 6.0) * pillow
    env_c = np.maximum(b2k + 0.8, np.maximum(Z_ROOF1 - 0.6 - zk, zk - top_c))
    env_a = np.maximum(a2k + 0.8, np.maximum(zs0 - zk, zk - top_a))
    env_a = np.maximum(env_a, -(b2k + 0.6))
    env = np.minimum(env_c, env_a)
    net = sdf.gyroid_network(grid, sk, LT["skin_cell"] * 0.9, bias=0.5, phase=(1.9, 0.4, 1.1), warp_amp=1.4, warp_len=37.0)
    skin = sdf.smax(net, env, 1.0)
    del net, env, env_c, env_a
    for (px, py), zb_ in ((IN, Z_BOSS), (OUT, Z_ABOSS)):
        keep = sdf.circle2(xk, yk, (px, py), rs + 3.0)
        skin = np.maximum(skin, -np.maximum(keep, zb_ - 3.3 - zk))
    f[sk] = sdf.smin(f[sk], skin, 0.9)
    del skin
    tm("bone skin")

    # ---------------- holes ----------------
    tap = P.LOOP["port_tap"] / 2
    sdf.apply(f, grid, (IN[0] - tap, IN[1] - tap, Z_GAL1 - 1.0), (IN[0] + tap, IN[1] + tap, Z_BOSS + 2),
              lambda X, Y, Z: sdf.cyl_z(X, Y, Z, IN, tap, Z_GAL1 - 1.0, Z_BOSS + 5), op="cut")
    sdf.apply(f, grid, (OUT[0] - tap, OUT[1] - tap, CZ1 - 1.0), (OUT[0] + tap, OUT[1] + tap, Z_ABOSS + 2),
              lambda X, Y, Z: sdf.cyl_z(X, Y, Z, OUT, tap, CZ1 - 1.0, Z_ABOSS + 5), op="cut")
    for (lx, ly) in lugs():
        sdf.apply(f, grid, (lx - 3, ly - 3, 0.0), (lx + 3, ly + 3, lz1 + 1),
                  lambda X, Y, Z, lx=lx, ly=ly: sdf.cyl_z(X, Y, Z, (lx, ly), M["lug_hole"] / 2, -5.0, 50.0), op="cut")
        rw = M["lug_well"] / 2
        sdf.apply(f, grid, (lx - rw, ly - rw, lz1), (lx + rw, ly + rw, Z_BOSS + 2),
                  lambda X, Y, Z, lx=lx, ly=ly, rw=rw: sdf.cyl_z(X, Y, Z, (lx, ly), rw, lz1, 60.0), op="cut")
    tm("ports + lug holes")

    removed = sdf.keep_largest_solid(f, h)
    info = analyse(f, grid)
    info["islands_removed_voxels"] = removed
    tm(f"clean-up: wet volume {info['wet_volume_ml']:.1f} mL, closed voids filled {info['closed_void_voxels']}")
    return f, grid, info


def _wet(f, grid, sl=None):
    from scipy import ndimage as ndi

    h = grid.h
    sub = f if sl is None else f[sl]
    o = grid.lo if sl is None else np.array(grid.origin_of(sl))
    zax = grid.ax[2] if sl is None else grid.ax[2][sl[2]]
    xk, yk, zk = grid.coords(sl)
    void = sub >= 0
    # cap each port just below its seat
    cap_in = (sdf.circle2(xk, yk, IN, 7.5) < 0) & (zk > Z_BOSS - 0.35)
    cap_out = (sdf.circle2(xk, yk, OUT, 7.5) < 0) & (zk > Z_ABOSS - 0.35)
    void = void & ~cap_in & ~cap_out
    lab, n = ndi.label(void)
    idx = lambda p: tuple(int(round((p[i] - o[i]) / h)) for i in range(3))
    li = lab[idx((IN[0], IN[1], Z_BOSS - 2.0))]
    lo_ = lab[idx((OUT[0], OUT[1], Z_ABOSS - 2.0))]
    return void, lab, li, lo_, cap_in, cap_out


def analyse(f, grid):
    h = grid.h
    void, lab, li, lo_, _, _ = _wet(f, grid)
    outside = lab[0, 0, -1]
    wet = lab == li
    closed = void & (lab != li) & (lab != outside) & (lab != lo_)
    ncl = int(closed.sum())
    if ncl:
        f[closed] = np.float32(-h)
    solid = f < 0
    area = 0
    for ax in range(3):
        a = np.swapaxes(wet, 0, ax)
        s = np.swapaxes(solid, 0, ax)
        area += int((a[1:] & s[:-1]).sum() + (a[:-1] & s[1:]).sum())
    return dict(
        inlet_outlet_connected=bool(li == lo_ and li != outside and li != 0),
        wet_leaks_to_outside=bool(li == outside),
        wet_volume_ml=float(wet.sum()) * h ** 3 / 1000.0,
        wetted_area_cm2_voxel=area * h * h / 100.0 * 0.66,
        closed_void_voxels=ncl,
        z_levels=dict(contact=Z_C, floor=Z_FLOOR, core_top=Z_CORE1, divider_top=Z_DIV1, gallery_top=Z_GAL1,
                      roof_top=Z_ROOF1, boss_top=Z_BOSS, arm=(AZ0, AZ1), arm_channel=(CZ0, CZ1)),
    )


def sim_domain(f, grid, path):
    h = grid.h
    sl = (slice(None),) * 3
    xs, ys, zs = grid.coords(sl)
    void, lab, li, lo_, cap_in, cap_out = _wet(f, grid)
    solid = f < 0
    wet = lab == li
    mat = np.full(f.shape, 250, dtype=np.uint8)
    mat[solid] = 1
    mat[wet] = 0
    inl = wet & (sdf.circle2(xs, ys, IN, 7.0) < 0) & (zs > Z_BOSS - 1.2)
    out = wet & (sdf.circle2(xs, ys, OUT, 7.0) < 0) & (zs > Z_ABOSS - 1.2)
    mat[inl] = 251
    mat[out] = 252
    heater = np.zeros(f.shape, dtype=np.uint8)
    below_void = np.zeros_like(solid)
    below_void[:, :, 1:] = solid[:, :, 1:] & ~solid[:, :, :-1]
    hx, hy = M["heater_cpu"]
    cpu2 = (np.abs(xs - HOT[0]) <= hx / 2) & (np.abs(ys - HOT[1]) <= hy / 2)
    heater[below_void & cpu2 & (zs < Z_UNDER + 0.2)] = 1
    heater[below_void & (zs < AZ0 - 0.5) & (arms2(xs, ys) < 0)] = 2
    np.savez_compressed(path, mat=mat, heater=heater, dx_m=h * 1e-3, origin_mm=np.array(grid.lo),
                        note="axes x, y (board frame), z up from PCB")
    return dict(shape=list(mat.shape), fluid_cells=int((mat == 0).sum() + (mat > 250).sum()),
                solid_cells=int((mat == 1).sum()), heater_cpu_faces=int((heater == 1).sum()),
                heater_vrm_faces=int((heater == 2).sum()), inlet_cells=int(inl.sum()), outlet_cells=int(out.sum()))


def main(h=0.25, do_export=True):
    f, grid, info = build(h=h)
    print(json.dumps(info, indent=1))
    if do_export:
        meta = part.export_part("monoblock", f, grid, "AlSi10Mg", "board", extra=dict(checks=info),
                                print_faces=900_000, web_faces=140_000, hero_faces=3_500_000)
        print(json.dumps(meta, indent=1))
        (part.ROOT / "sim" / "cases" / "monoblock").mkdir(parents=True, exist_ok=True)
        print(sim_domain(f, grid, part.ROOT / "sim" / "cases" / "monoblock" / "domain.npz"))
    return f, grid, info


if __name__ == "__main__":
    hh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    main(h=hh, do_export=("--noexport" not in sys.argv))
