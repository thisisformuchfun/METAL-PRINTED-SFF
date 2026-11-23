"""ORGAN I - full-cover liquid block for the RTX 4090 (PCB MS-V510), AlSi10Mg, monolithic.

Coordinates: GPU PCB frame (see params.py), z measured up from the PCB top surface.

Architecture (bottom -> top over the GPU):
  die pedestal      solid, lapped face on the die
  floor             2.0 mm
  CORE              6.5 mm sheet gyroid (5 mm cell) - the heat exchanger; coolant enters
                    through a slot nozzle directly above the die and spreads radially
  divider           1.0 mm plate, stops 3.5 mm short of the chamber wall all round
  GALLERY           5.0 mm coarse gyroid plenum collecting the flow to the outlet
  roof              1.6 mm
  SKIN              exposed skeletal gyroid ("bone"), dry - looks + passive area
Wings left and right are solid plates that pick up the 26 VRM power stages and
inductors through thermal pads and conduct into the wet core.
Everything inside is self-supporting lattice: no enclosed voids, no loose supports.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import params as P
from . import part, sdf

G = P.GPU
B = P.GPU_BLOCK
LT = P.LATTICE

Z_DIE = G["z_die"]
Z_UNDER = B["z_under"]
Z_FLOOR = Z_DIE + B["floor"]
Z_CORE1 = Z_FLOOR + B["core_h"]
Z_DIV1 = Z_CORE1 + B["divider"]
Z_GAL1 = Z_DIV1 + B["gallery_h"]
Z_ROOF1 = Z_GAL1 + B["roof"]
Z_BOSS = Z_ROOF1 + B["skin_h"] + 0.3
WING_TOP = B["wing_top"]
CX, CY = G["centre"]


def plan2(x, y):
    """2-D outline of the block (negative inside)."""
    near = sdf.rect2(x, y, (17.5, 1.5), (70.0, 125.5), r=9.0)
    core = sdf.rect2(x, y, (66.0, 16.0), (167.0, 109.5), r=10.0)
    far = sdf.rect2(x, y, (163.0, 6.0), (209.0, 119.5), r=9.0)
    d = sdf.smin(sdf.smin(near, core, 8.0), far, 8.0)
    mx, my = G["mec"]["MEC8"]
    d = sdf.smin(d, sdf.circle2(x, y, (mx, my), 5.6), 4.0)
    return d


def core_outer2(x, y):
    w = B["wall"]
    return sdf.rect2(x, y, (B["chamber_x"][0] - w, B["chamber_y"][0] - w),
                     (B["chamber_x"][1] + w, B["chamber_y"][1] + w), r=8.0)


def chamber2(x, y, shrink=0.0, r=6.0):
    return sdf.rect2(x, y, (B["chamber_x"][0] + shrink, B["chamber_y"][0] + shrink),
                     (B["chamber_x"][1] - shrink, B["chamber_y"][1] - shrink), r=max(r - shrink, 1.5))


def trench(x, y, z):
    """Open V-trench in the core under the nozzle: lets the jet spread before it enters the lattice."""
    wt, wb, keep = B["trench"]
    jw, jl = B["jet"]
    z0, z1 = Z_FLOOR + keep, Z_CORE1 + 0.3
    w = wb + (wt - wb) * np.clip((z - z0) / (Z_CORE1 - z0), 0.0, 1.0)
    return np.maximum(np.maximum(np.abs(x - B["inlet"][0]) - 0.5 * w, np.abs(y - B["inlet"][1]) - (0.5 * jl + 0.8)),
                      np.maximum(z0 - z, z - z1))


def mem_bars():
    """Pad pedestals: one rectangle per memory chip, slightly inset."""
    out = []
    for cx, cy, w, hgt in G["mem"].values():
        out.append((cx - w / 2 + 0.25, cy - hgt / 2 - 0.45, cx + w / 2 - 0.25, cy + hgt / 2 + 0.45))
    return out


def post_xy():
    pts = [(x, y, 6.3) for x, y in G["core_holes"]]
    pts += [(G["mec"][n][0], G["mec"][n][1], 5.5) for n in B["posts"]]
    return pts


def build(h=0.25, verbose=True):
    tm = part.Timer()
    grid = sdf.Grid((14.0, -1.0, -0.75), (229.0, 128.5, Z_BOSS + 1.5), h)
    if verbose:
        print(f"gpu_block grid {grid.shape} = {np.prod(grid.n) / 1e6:.1f} M voxels, {grid.memory_mb():.0f} MB/field")
    x, y, z = grid.coords()

    p2 = plan2(x, y)                                   # (nx, ny, 1)
    co2 = core_outer2(x, y)

    # ---------------- solid body ----------------
    wing = sdf.smax(p2, np.maximum(Z_UNDER - z, z - WING_TOP), 1.2)
    corebody = sdf.smax(co2, np.maximum(Z_UNDER - z, z - Z_ROOF1), 1.6)
    f = np.minimum(wing, corebody)          # exact union: the two share the underside plane
    del wing, corebody
    tm("body")

    # ---------------- underside lands ----------------
    pd = B["pedestal_die"]
    sdf.add_box(f, grid, (CX - pd[0] / 2, CY - pd[1] / 2, Z_DIE), (CX + pd[0] / 2, CY + pd[1] / 2, Z_UNDER + 0.6), r=0.0)
    z_mem = G["z_mem"] + B["pad_mem"]
    for (x0, y0, x1, y1) in mem_bars():
        sdf.add_box(f, grid, (x0, y0, z_mem), (x1, y1, Z_UNDER + 0.6))
    z_dr = G["z_drmos"] + B["pad_vrm"]
    near, far = G["vrm_near"], G["vrm_far"]
    yrow = lambda j: G["vrm_row_y0"] + G["vrm_row_pitch"] * j
    sdf.add_box(f, grid, (near["drmos_x"][0] - 0.45, yrow(0) - 3.2, z_dr),
                (near["drmos_x"][1] + 0.45, yrow(13) + 3.2, Z_UNDER + 0.6))
    sdf.add_box(f, grid, (far["drmos_x"][0] - 0.45, yrow(1) - 3.2, z_dr),
                (far["drmos_x"][1] + 0.45, yrow(12) + 3.2, Z_UNDER + 0.6))

    # far-side bridges that step over the capacitor column to MEC7 / MEC6
    for name, start in (("MEC7", (203.0, 9.0)), ("MEC6", (205.0, 65.47))):
        mx, my = G["mec"][name]
        sdf.add_capsule(f, grid, (start[0], start[1], 10.9), (mx, my, 10.9), 3.3, k=2.5)

    # posts down to the PCB (seat 0.15 mm proud of nominal so the die takes the preload)
    for (px, py, dia) in post_xy():
        top = 9.5 if dia > 6 else (10.9 if px > 215 else Z_UNDER + 1.0)
        sdf.apply(f, grid, (px - dia, py - dia, 0.0), (px + dia, py + dia, top + 1),
                  lambda X, Y, Z, px=px, py=py, dia=dia, top=top: sdf.cyl_z(X, Y, Z, (px, py), dia / 2, 0.15, top),
                  op="union", k=0.8)
    tm("lands + posts")

    # ---------------- pockets for tall parts ----------------
    z_ind = G["z_ind"] + B["pad_ind"]
    sdf.add_box(f, grid, (near["ind_x"][0] - 0.95, -5.0, -5.0), (near["ind_x"][1] + 0.95, 135.0, z_ind), op="cut")
    sdf.add_box(f, grid, (far["ind_x"][0] - 0.95, 8.0, -5.0), (far["ind_x"][1] + 0.95, 135.0, z_ind), op="cut")
    for (ix, iy) in G["mem_vrm_ind"]:
        sdf.add_box(f, grid, (ix - 4.0, iy - 4.0, -5.0), (ix + 4.0, iy + 4.0, G["z_mem_ind"] + 1.0), op="cut")
    sdf.add_box(f, grid, (92.0, 9.5, -5.0), (109.0, 18.4, 5.0), op="cut")       # L49 / L50
    tm("pockets")

    # ---------------- wet chamber ----------------
    ch_lo = (B["chamber_x"][0], B["chamber_y"][0], Z_FLOOR)
    ch_hi = (B["chamber_x"][1], B["chamber_y"][1], Z_GAL1)
    sl = grid.box(ch_lo, ch_hi, pad=1.0)
    xs, ys, zs = grid.coords(sl)
    ch2 = chamber2(xs, ys)
    interior = np.maximum(ch2, np.maximum(Z_FLOOR - zs, zs - Z_GAL1))
    f[sl] = np.maximum(f[sl], -interior)
    tm("chamber cut")

    a5 = 2 * np.pi / LT["wet_cell"]
    wall = LT["wet_wall"] + LT["wet_root"] * np.exp(-np.maximum(zs - Z_FLOOR, 0.0) / LT["wet_root_len"])
    core = sdf.gyroid_sheet(grid, sl, LT["wet_cell"], wall.astype(np.float32),
                            phase=(-a5 * CX, -a5 * CY, -a5 * Z_FLOOR + 0.6))
    # the two lattices never butt against each other: the core stops under the divider level, the gallery
    # starts above it, and the 1 mm band between them is open in the riser gap (clean, manifold edges)
    core = np.maximum(core, zs - (Z_CORE1 + 0.05))
    a8 = 2 * np.pi / LT["gallery_cell"]
    gal = sdf.gyroid_sheet(grid, sl, LT["gallery_cell"], LT["gallery_wall"],
                           phase=(-a8 * CX + 0.9, -a8 * CY, -a8 * Z_DIV1))
    gal = np.maximum(gal, (Z_DIV1 - 0.05) - zs)
    lat = np.minimum(core, gal)
    del core, gal
    lat = np.maximum(lat, np.maximum(ch2 - 0.45, np.maximum(Z_FLOOR - 0.4 - zs, zs - Z_GAL1 - 0.4)))
    f[sl] = np.minimum(f[sl], lat)
    del lat
    # divider plate
    div = np.maximum(chamber2(xs, ys, shrink=B["riser_gap"]), np.maximum(Z_CORE1 - zs, zs - Z_DIV1))
    f[sl] = np.minimum(f[sl], div)
    f[sl] = np.maximum(f[sl], -trench(xs, ys, zs))
    # solid pillars carrying the four GPU mounting threads up through the wet core
    for (px, py) in G["core_holes"]:
        pil = np.maximum(sdf.circle2(xs, ys, (px, py), 3.75), np.maximum(Z_UNDER - 0.3 - zs, zs - 9.6))
        f[sl] = sdf.smin(f[sl], pil, 0.8)
    tm("gyroid core + gallery + divider")

    # inlet plenum over the die
    jw, jl = B["jet"]
    ix, iy = B["inlet"]
    ox, oy = B["outlet"]
    pl_o = sdf.rect2(xs, ys, (ix - 8.6, iy - 14.6), (ix + 8.6, iy + 14.6), r=7.6)
    pl_i = sdf.rect2(xs, ys, (ix - 7.2, iy - 13.2), (ix + 7.2, iy + 13.2), r=6.2)
    zr = np.maximum(Z_DIV1 - 0.2 - zs, zs - Z_GAL1 - 0.2)
    f[sl] = np.minimum(f[sl], np.maximum(pl_o, zr))
    f[sl] = np.maximum(f[sl], -np.maximum(pl_i, np.maximum(Z_DIV1 - zs, zs - Z_GAL1)))
    slot = sdf.rect2(xs, ys, (ix - jw / 2, iy - jl / 2), (ix + jw / 2, iy + jl / 2), r=jw / 2 - 0.05)
    f[sl] = np.maximum(f[sl], -np.maximum(slot, np.maximum(Z_CORE1 - 0.3 - zs, zs - Z_DIV1 - 0.3)))
    # clear the gallery lattice under the outlet port
    outpocket = np.maximum(sdf.circle2(xs, ys, (ox, oy), 6.3), np.maximum(Z_DIV1 + 0.9 - zs, zs - Z_GAL1 - 0.1))
    f[sl] = np.maximum(f[sl], -outpocket)
    tm("plenum + nozzle")

    # ---------------- port bosses ----------------
    for (px, py) in (B["inlet"], B["outlet"]):
        r = P.LOOP["port_seat_d"] / 2
        sdf.apply(f, grid, (px - r, py - r, Z_ROOF1 - 1.0), (px + r, py + r, Z_BOSS),
                  lambda X, Y, Z, px=px, py=py, r=r: sdf.smax(sdf.circle2(X, Y, (px, py), r),
                                                              np.maximum(Z_ROOF1 - 1.0 - Z, Z - Z_BOSS), 0.8),
                  op="union", k=2.0)

    # ---------------- bone skin (dry) ----------------
    zs0 = WING_TOP - 0.6
    sk = grid.box((14.0, -1.0, zs0), (229.0, 128.5, Z_BOSS), pad=0.0)
    xk, yk, zk = grid.coords(sk)
    p2k = plan2(xk, yk)
    cok = core_outer2(xk, yk)
    smooth = sdf.smoothstep
    # Voronoi "pillowing": the skin swells inside each Voronoi cell and dips to a groove
    # along the cell boundaries -> a tessellated, shell-like relief carved from gyroid.
    seeds = sdf.hex_seeds((14.0, -1.0), (229.0, 128.5), B.get("voronoi_cell", 27.0), seed=7)
    pillow = 0.10 + 0.90 * smooth(sdf.voronoi2_edge(xk, yk, seeds) / 4.2)
    top_w = WING_TOP + 1.3 + 5.4 * smooth(-p2k / 9.0) * pillow
    top_c = Z_ROOF1 + 1.6 + (B["skin_h"] - 1.6) * smooth(-cok / 11.0) * pillow
    env_w = np.maximum(p2k + 1.0, np.maximum(zs0 - zk, zk - top_w))
    env_w = np.maximum(env_w, -(cok + 0.6))          # wing skin stays outside the wet core body
    env_c = np.maximum(cok + 0.8, np.maximum(Z_ROOF1 - 0.6 - zk, zk - top_c))
    env = np.minimum(env_w, env_c)
    del env_w, env_c
    net = sdf.gyroid_network(grid, sk, LT["skin_cell"], bias=0.72, phase=(0.7, 2.1, 0.3),
                             warp_amp=1.5, warp_len=41.0)
    skin = sdf.smax(net, env, 1.0)
    del net, env
    # keep the port seats free of lattice
    for (px, py) in (B["inlet"], B["outlet"]):
        keep = sdf.circle2(xk, yk, (px, py), P.LOOP["port_seat_d"] / 2 + 3.0)
        skin = np.maximum(skin, -np.maximum(keep, Z_BOSS - 3.3 - zk))
    f[sk] = sdf.smin(f[sk], skin, 0.9)
    del skin
    tm("bone skin")

    # ---------------- holes ----------------
    tap = P.LOOP["port_tap"] / 2
    for (px, py) in (B["inlet"], B["outlet"]):
        sdf.apply(f, grid, (px - tap, py - tap, Z_GAL1 - 1.0), (px + tap, py + tap, Z_BOSS + 2),
                  lambda X, Y, Z, px=px, py=py: sdf.cyl_z(X, Y, Z, (px, py), tap, Z_GAL1 - 1.0, Z_BOSS + 5), op="cut")
    pr = B["post_pilot"] / 2
    for (px, py, dia) in post_xy():
        sdf.apply(f, grid, (px - pr, py - pr, -1.0), (px + pr, py + pr, 7.2),
                  lambda X, Y, Z, px=px, py=py: sdf.cyl_z(X, Y, Z, (px, py), pr, -2.0, 7.0), op="cut")
    tm("ports + pilot holes")

    # ---------------- printability clean-up ----------------
    removed = sdf.keep_largest_solid(f, h)
    info = analyse(f, grid)
    info["islands_removed_voxels"] = removed
    tm(f"clean-up: wet volume {info['wet_volume_ml']:.1f} mL, closed voids filled {info['closed_void_voxels']}")
    return f, grid, info


def analyse(f, grid):
    """Wet-volume connectivity, trapped-powder check, wetted area."""
    from scipy import ndimage as ndi

    h = grid.h
    kz = int(np.searchsorted(grid.ax[2], Z_BOSS - 0.2))
    void = f >= 0
    void[:, :, kz:] = False                      # cap the ports
    lab, n = ndi.label(void)
    idx = lambda p: tuple(int(round((p[i] - grid.lo[i]) / h)) for i in range(3))
    li = lab[idx((B["inlet"][0], B["inlet"][1], Z_BOSS - 2.0))]
    lo_ = lab[idx((B["outlet"][0], B["outlet"][1], Z_BOSS - 2.0))]
    outside = lab[0, 0, 0]
    wet = lab == li
    closed = void & (lab != li) & (lab != outside) & (lab != lo_)
    ncl = int(closed.sum())
    if ncl:
        f[closed] = np.float32(-h)
    touches_edge = bool(wet[0].any() or wet[-1].any() or wet[:, 0].any() or wet[:, -1].any() or wet[:, :, 0].any())
    # wetted area ~ count of fluid/solid faces
    solid = f < 0
    area = 0
    for ax in range(3):
        a = np.swapaxes(wet, 0, ax)
        s = np.swapaxes(solid, 0, ax)
        area += int((a[1:] & s[:-1]).sum() + (a[:-1] & s[1:]).sum())
    info = dict(
        inlet_outlet_connected=bool(li == lo_ and li != outside and li != 0),
        wet_leaks_to_outside=touches_edge,
        wet_volume_ml=float(wet.sum()) * h ** 3 / 1000.0,
        wetted_area_cm2_voxel=area * h * h / 100.0 * 0.66,      # staircase correction ~2/3
        closed_void_voxels=ncl,
        die_floor_mm=B["floor"], z_levels=dict(die=Z_DIE, floor=Z_FLOOR, core_top=Z_CORE1, divider_top=Z_DIV1,
                                               gallery_top=Z_GAL1, roof_top=Z_ROOF1, boss_top=Z_BOSS),
    )
    return info


def sim_domain(f, grid, path, lo=(60.0, 9.0, 1.9), hi=(173.0, 114.0, Z_BOSS)):
    """Write the CHT case geometry: mat (0 fluid, 1 metal, 250 outside, 251 inlet, 252 outlet)
    and heater ids (1 = GPU die, 2 = memory lands) for the wet core region."""
    from scipy import ndimage as ndi

    h = grid.h
    sl = grid.box(lo, hi)
    sub = f[sl]
    xs, ys, zs = grid.coords(sl)
    solid = sub < 0
    kz = int(np.searchsorted(grid.ax[2][sl[2]], Z_BOSS - 0.2))
    void = ~solid
    capped = void.copy()
    capped[:, :, kz:] = False
    lab, n = ndi.label(capped)
    o = grid.origin_of(sl)
    idx = lambda p: tuple(int(round((p[i] - o[i]) / h)) for i in range(3))
    li = lab[idx((B["inlet"][0], B["inlet"][1], Z_BOSS - 2.0))]
    wet = lab == li
    mat = np.full(sub.shape, 250, dtype=np.uint8)
    mat[solid] = 1
    mat[wet] = 0
    top = np.zeros_like(wet)
    top[:, :, kz - 3:kz] = True
    inl = wet & top & (sdf.circle2(xs, ys, B["inlet"], 7.0) < 0)
    out = wet & top & (sdf.circle2(xs, ys, B["outlet"], 7.0) < 0)
    mat[inl] = 251
    mat[out] = 252
    heater = np.zeros(sub.shape, dtype=np.uint8)
    # die: solid cells of the pedestal face (lowest solid layer inside the die footprint)
    dx, dy = G["die"]
    die2 = (np.abs(xs - CX) <= dx / 2) & (np.abs(ys - CY) <= dy / 2)
    below_void = np.zeros_like(solid)
    below_void[:, :, 1:] = solid[:, :, 1:] & ~solid[:, :, :-1]
    heater[below_void & die2 & (zs < Z_UNDER + 0.2)] = 1
    z_mem = G["z_mem"] + B["pad_mem"]
    zb2 = zs < Z_DIE - 0.3
    for cx, cy, w, hg in G["mem"].values():
        m2 = (np.abs(xs - cx) <= w / 2 - 0.3) & (np.abs(ys - cy) <= hg / 2 - 0.3)
        heater[below_void & m2 & zb2] = 2
    np.savez_compressed(path, mat=mat, heater=heater, dx_m=h * 1e-3, origin_mm=np.array(o),
                        note="axes x (card length), y (card height), z (up from PCB); GPU frame")
    return dict(shape=list(mat.shape), fluid_cells=int((mat == 0).sum() + (mat > 250).sum()),
                solid_cells=int((mat == 1).sum()), heater_die_faces=int((heater == 1).sum()),
                heater_mem_faces=int((heater == 2).sum()), inlet_cells=int(inl.sum()), outlet_cells=int(out.sum()))


def main(h=0.25, do_export=True):
    f, grid, info = build(h=h)
    print(json.dumps(info, indent=1))
    if do_export:
        np.save(part.WORK / "gpu_block_field.npy", f) if part.WORK.mkdir(parents=True, exist_ok=True) is None else None
        meta = part.export_part("gpu_block", f, grid, "AlSi10Mg", "gpu", extra=dict(checks=info),
                                print_faces=1_100_000, web_faces=170_000, hero_faces=4_500_000)
        print(json.dumps(meta, indent=1))
        (part.ROOT / "sim" / "cases" / "gpu_block").mkdir(parents=True, exist_ok=True)
        print(sim_domain(f, grid, part.ROOT / "sim" / "cases" / "gpu_block" / "domain.npz"))
    return f, grid, info


if __name__ == "__main__":
    hh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    main(h=hh, do_export=("--noexport" not in sys.argv))
