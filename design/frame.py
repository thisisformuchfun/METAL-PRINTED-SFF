"""The steel skeleton of MARROW - five 316L segments that bolt into one closed ring:

    pelvis      keel + four feet + PSU cradle + riser shelf            (A)
    lumbar      lower front spine, ribs to the graphics card           (B1)
    thoracic    upper front spine, ribs to the motherboard + heart     (B2)
    dorsal      lower rear spine, lower/middle rib pairs for the rads  (C1)
    cervical    upper rear spine, top rib pair, crown                  (C2)

Every member is a tapered round "bone" (smooth-blended capsules), sized so the largest
segment fits a 300 x 300 x 290 mm SLM build. Joints are half-laps with two M5 bolts.
Coordinates: WORLD (mm).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import heart as HT
from . import params as P
from . import part, sdf

FR = P.FRAME
EX = np.array(P.E_X)
ES = np.array(P.E_S)
EN = np.array(P.E_N)
SN = FR["spine_n"]

R_SPINE = 6.0
R_RIB0, R_RIB1 = 4.6, 3.0
R_KEEL = 5.4
R_KNUCKLE = 8.8
K_ROOT = 8.0          # blend radius where a rib leaves a spine (webbed, fleshy junctions)
# front spine centreline as (s, n): gently S-curved, dead straight through the lap zones
FS_CTRL = [(-8, -24.0), (10, -24.0), (28, -24.0), (62, -27.0), (104, -26.5), (130, -24.3), (140, -24.0), (160, -24.0),
           (180, -24.0), (216, -26.5), (268, -28.0), (326, -26.0), (352, -24.3), (370, -24.0)]


def spine_n(s):
    ss = [c[0] for c in FS_CTRL]
    nn = [c[1] for c in FS_CTRL]
    return float(np.interp(s, ss, nn))
REAR_Y = FR["rear_spine_y"]
SPINE_TOP_S = 370.0


def S(x, s, n=SN):
    return np.array(P.show(x, s, n))


def bone(f, grid, pts, r0, r1=None, belly=0.0, k=3.0):
    return sdf.add_bone(f, grid, [np.asarray(p, float) for p in pts], r0, r1, belly=belly, k=k)


# ----------------------------------------------------------------------------
# joints
# ----------------------------------------------------------------------------
JOINTS = dict(
    J1=dict(p0=S(0, 0), d=ES, t=(140.0, 180.0), low="lumbar", high="thoracic"),
    J3=dict(p0=S(0, 0), d=ES, t=(-8.0, 28.0), low="pelvis", high="lumbar"),
    J2=dict(p0=np.array([0.0, REAR_Y, 0.0]), d=np.array([0.0, 0.0, 1.0]), t=(254.0, 290.0), low="dorsal", high="cervical"),
    J4=dict(p0=np.array([0.0, REAR_Y, 0.0]), d=np.array([0.0, 0.0, 1.0]), t=(22.0, 58.0), low="pelvis", high="dorsal"),
)


def knuckle(f, grid, J):
    a = J["p0"] + J["d"] * (J["t"][0] + 6.0)
    b = J["p0"] + J["d"] * (J["t"][1] - 6.0)
    sdf.add_capsule(f, grid, a, b, R_KNUCKLE, k=4.0)


def lap_cut(f, grid, J, role, clr=0.12):
    """Remove the mating half of a lap joint; 'low' keeps x<0 inside the zone."""
    p0, d = J["p0"], J["d"]
    t0, t1 = J["t"]
    a = p0 + d * (t0 - 14)
    b = p0 + d * (t1 + 14)
    lo = np.minimum(a, b) - 13.5
    hi = np.maximum(a, b) + 13.5
    sl = grid.box(lo, hi)
    if sl is None:
        return
    x, y, z = grid.coords(sl)
    t = (x - p0[0]) * d[0] + (y - p0[1]) * d[1] + (z - p0[2]) * d[2]
    # only act near the joint axis
    px = x - (p0[0] + t * d[0])
    py = y - (p0[1] + t * d[1])
    pz = z - (p0[2] + t * d[2])
    near = np.sqrt(px * px + py * py + pz * pz) - 13.0
    xs = x - p0[0]
    if role == "low":
        dR = np.minimum(t1 - t, np.maximum(t0 - t, -clr - xs))
    else:
        dR = np.minimum(t - t0, np.maximum(t - t1, xs - clr))
    dR = np.maximum(dR, near)                     # region to remove (negative inside)
    f[sl] = np.maximum(f[sl], -dR)
    # bolts along X
    for tb in (t0 + 10.0, t1 - 10.0):
        c = p0 + d * tb
        sdf.add_cyl(f, grid, c - EX * 20, c + EX * 20, 2.7, op="cut")
        side = -1.0 if role == "low" else 1.0
        sdf.add_cyl(f, grid, c + EX * side * 5.8, c + EX * side * 20, 5.0, op="cut")   # head / nut seat


# ----------------------------------------------------------------------------
# mounting features
# ----------------------------------------------------------------------------
def standoff(f, grid, tip, n_pad, n_top, d_boss, d_hole, cb=None):
    """Boss along e_n from the rib pad up to the PCB underside, at show point tip=(x, s)."""
    a = S(tip[0], tip[1], n_pad)
    b = S(tip[0], tip[1], n_top)
    sdf.add_cyl(f, grid, a - EN * 2.0, b, d_boss / 2, k=1.2)
    return (a, b, d_hole, cb)


def drill(f, grid, holes):
    for (a, b, d_hole, cb) in holes:
        sdf.add_cyl(f, grid, a - EN * 12, b + EN * 2, d_hole / 2, op="cut")
        if cb:
            sdf.add_cyl(f, grid, a - EN * 14, a - EN * (2.0 + 0.0) - EN * 2.2, cb / 2, op="cut")


def rib(f, grid, root_s, tip, n_tip, r0=R_RIB0, r1=R_RIB1, bow=0.0, root_x=0.0, root_n=None):
    """Rib from the front spine at s=root_s to show point tip=(x, s) at depth n_tip."""
    x1, s1 = tip
    root_n = spine_n(root_s) if root_n is None else root_n
    p0 = S(root_x, root_s, root_n)
    pm = S(root_x + 0.48 * (x1 - root_x), root_s + 0.62 * (s1 - root_s) + bow, root_n + 0.35 * (n_tip - root_n) - 1.0)
    p1 = S(x1, s1, n_tip)
    bone(f, grid, [p0, pm, p1], r0, r1, belly=-0.35, k=K_ROOT)
    sdf.add_capsule(f, grid, p1, p1, r1 + 1.6, k=2.0)          # pad swell at the tip


def gpu_points():
    out = {}
    for name in P.GPU_BLOCK["frame_holes"]:
        xg, yg = P.GPU["mec"][name]
        x, s, _ = P.gpu_to_show(xg, yg)
        out[name] = (x, s)
    return out


def board_points():
    return [P.board_to_show(x, y)[:2] for (x, y) in P.BOARD["holes"]]


def heart_ears():
    cx, cs = P.HEART["centre_show"]
    return [(cx + ex, cs + ey) for (ex, ey) in HT.EARS]


N_GPU_PAD = -P.GPU["pcb_t"] - FR["gpu_standoff"]             # top of rib pad under the GPU
N_BRD_PAD = -P.BOARD["pcb_t"] - FR["board_standoff"]
N_HEART = P.HEART.get("n0", 0.0)


# ----------------------------------------------------------------------------
# segments
# ----------------------------------------------------------------------------
def build_lumbar(h):
    lo = np.min([S(-122, -12, -34), S(124, 204, 4), S(-122, 204, -34), S(124, -12, 4)], axis=0) - 4
    hi = np.max([S(-122, -12, -34), S(124, 204, 4), S(-122, 204, -34), S(124, -12, 4)], axis=0) + 4
    grid = sdf.Grid(lo, hi, h)
    f = grid.new()
    bone(f, grid, [S(0, s_, n_) for (s_, n_) in FS_CTRL if s_ <= 180], R_SPINE, R_SPINE, belly=-0.9, k=3.0)
    knuckle(f, grid, JOINTS["J3"])
    knuckle(f, grid, JOINTS["J1"])
    gp = gpu_points()
    holes = []
    for name, root in (("MEC13", 54.0), ("MEC7", 54.0), ("MEC4", 116.0), ("MEC8", 116.0)):
        rib(f, grid, root, gp[name], N_GPU_PAD - 2.0)
        holes.append(standoff(f, grid, gp[name], N_GPU_PAD - 2.0, -P.GPU["pcb_t"], 6.0, 3.4, cb=6.4))
    # rib to the heart's lower ear, sprouting from the MEC8 pad
    e = heart_ears()[2]
    p0 = S(gp["MEC8"][0], gp["MEC8"][1], N_GPU_PAD - 2.0)
    bone(f, grid, [p0, S(96, 168, -10), S(e[0], e[1], N_HEART - 3.0)], R_RIB0 - 0.3, R_RIB1, k=3.0)
    sdf.add_cyl(f, grid, S(e[0], e[1], N_HEART - 5.0), S(e[0], e[1], N_HEART), 5.0, k=1.5)
    drill(f, grid, holes)
    sdf.add_cyl(f, grid, S(e[0], e[1], N_HEART - 9), S(e[0], e[1], N_HEART + 2), 1.65, op="cut")     # M4 pilot
    lap_cut(f, grid, JOINTS["J3"], "high")
    lap_cut(f, grid, JOINTS["J1"], "low")
    return f, grid


def build_thoracic(h):
    c = [S(-131, 136, -34), S(72, 136, 4), S(-131, 376, -34), S(72, 376, 4), S(-131, 376, 4)]
    grid = sdf.Grid(np.min(c, axis=0) - 4, np.max(c, axis=0) + 4, h)
    f = grid.new()
    bone(f, grid, [S(0, s_, n_) for (s_, n_) in FS_CTRL if s_ >= 140], R_SPINE, R_SPINE, belly=-0.9, k=3.0)
    knuckle(f, grid, JOINTS["J1"])
    sdf.add_capsule(f, grid, S(0, SPINE_TOP_S - 14), S(0, SPINE_TOP_S), R_SPINE + 1.6, k=3.0)     # head under the crown cap
    bp = board_points()                       # h1 top-left, h2 top-right, h3 bottom-left, h4 bottom-right
    holes = []
    for tip, root, bow in ((bp[0], 304.0, 6.0), (bp[1], 300.0, 2.0), (bp[2], 214.0, -2.0), (bp[3], 214.0, 2.0)):
        rib(f, grid, root, tip, N_BRD_PAD - 2.0, bow=bow)
        holes.append(standoff(f, grid, tip, N_BRD_PAD - 2.0, -P.BOARD["pcb_t"], 7.5, 2.5))
    ears = heart_ears()
    # heart ear A straight off the spine, ear B off the upper-right board rib
    rib(f, grid, 244.0, ears[0], N_HEART - 3.0, r0=R_RIB0 - 0.2)
    p0 = S(bp[1][0], bp[1][1], N_BRD_PAD - 2.0)
    bone(f, grid, [p0, S(52, 298, -9), S(ears[1][0], ears[1][1], N_HEART - 3.0)], R_RIB1 + 0.3, R_RIB1, k=3.0)
    for e in ears[:2]:
        sdf.add_cyl(f, grid, S(e[0], e[1], N_HEART - 5.0), S(e[0], e[1], N_HEART), 5.0, k=1.5)
    drill(f, grid, holes)
    for e in ears[:2]:
        sdf.add_cyl(f, grid, S(e[0], e[1], N_HEART - 9), S(e[0], e[1], N_HEART + 2), 1.65, op="cut")
    # flat top + tapped M5 for the crown cap
    top = S(0, SPINE_TOP_S)
    sl = grid.box(top - 14, top + 14)
    x, y, z = grid.coords(sl)
    f[sl] = np.maximum(f[sl], sdf.halfspace(x, y, z, top, ES))
    sdf.add_cyl(f, grid, top - ES * 14, top + ES * 2, 2.1, op="cut")
    lap_cut(f, grid, JOINTS["J1"], "high")
    return f, grid


def rear_pad(f, grid, x, z, holes):
    """Flat pad on the radiator's rear face with an M3 clearance hole along Y."""
    y0 = P.RAD["y_rear"]
    sdf.add_cyl(f, grid, (x, y0, z), (x, y0 + 4.2, z), 4.6, k=1.2)
    holes.append((x, z))


def rear_rib(f, grid, sign, z_root, tips, holes):
    y0 = P.RAD["y_rear"]
    for zt in tips:
        p0 = np.array([0.0, REAR_Y, z_root])
        pm = np.array([sign * 60.0, REAR_Y + 9.0, 0.5 * (z_root + zt) + 4.0])
        p1 = np.array([sign * 108.5, y0 + 5.2, zt + 2.0])
        bone(f, grid, [p0, pm, p1, np.array([sign * 112.5, y0 + 3.0, zt])], R_RIB0 + 0.2, R_RIB1, belly=-0.3, k=K_ROOT)
        rear_pad(f, grid, sign * 112.5, zt, holes)


def drill_rear(f, grid, holes):
    y0 = P.RAD["y_rear"]
    for (x, z) in holes:
        sdf.add_cyl(f, grid, (x, y0 - 2, z), (x, y0 + 30, z), 1.75, op="cut")           # M3 clearance
        sdf.add_cyl(f, grid, (x, y0 + 4.2, z), (x, y0 + 30, z), 3.4, op="cut")          # head access


def rad_z():
    za, zb = P.RAD["centres_z"]
    return dict(top=za + 52.5, mid_hi=za - 52.5, mid_lo=zb + 52.5, bot=zb - 52.5)


def build_dorsal(h):
    grid = sdf.Grid((-121, P.RAD["y_rear"] - 2, 18), (121, REAR_Y + 22, 294), h)
    f = grid.new()
    bone(f, grid, [(0, REAR_Y, 22), (0, REAR_Y, 40), (0, REAR_Y, 58), (0, REAR_Y + 2.5, 92), (0, REAR_Y, 126), (0, REAR_Y + 4.0, 180),
                   (0, REAR_Y, 232), (0, REAR_Y, 254), (0, REAR_Y, 272), (0, REAR_Y, 290)], R_SPINE, R_SPINE, belly=-0.8, k=3.0)
    knuckle(f, grid, JOINTS["J4"])
    knuckle(f, grid, JOINTS["J2"])
    rz = rad_z()
    holes = []
    for sgn in (-1, 1):
        rear_rib(f, grid, sgn, rz["bot"] + 20.0, [rz["bot"]], holes)
        rear_rib(f, grid, sgn, 0.5 * (rz["mid_hi"] + rz["mid_lo"]) + 4.0, [rz["mid_lo"], rz["mid_hi"]], holes)
        for zz in (rz["bot"], rz["mid_lo"], rz["mid_hi"]):
            sdf.add_capsule(f, grid, (0, REAR_Y - 2.5, zz), (sgn * 7.5, P.RAD["y_rear"] + 2.6, zz), 4.2, 3.6, k=2.0)
            rear_pad(f, grid, sgn * 7.5, zz, holes)
    drill_rear(f, grid, holes)
    lap_cut(f, grid, JOINTS["J4"], "high")
    lap_cut(f, grid, JOINTS["J2"], "low")
    return f, grid


def crown_path():
    top = S(0, SPINE_TOP_S)
    cap = top + ES * 5.0
    return [np.array([0.0, REAR_Y, 300.0]), np.array([0.0, REAR_Y, 336.0]), np.array([0.0, REAR_Y - 3.0, 356.0]),
            np.array([0.0, REAR_Y - 16.0, 367.5]), np.array([0.0, 256.0, 369.5]), np.array([0.0, 237.0, 369.0]),
            cap + np.array([0.0, 4.0, 2.5]), cap]


def build_cervical(h):
    top = S(0, SPINE_TOP_S)
    grid = sdf.Grid((-121, top[1] - 14, 250), (121, REAR_Y + 22, 382), h)
    f = grid.new()
    bone(f, grid, [(0, REAR_Y, 254), (0, REAR_Y, 280), (0, REAR_Y, 300)], R_SPINE, R_SPINE, k=3.0)
    bone(f, grid, crown_path(), R_SPINE, R_SPINE + 0.6, belly=-0.3, k=3.0)
    knuckle(f, grid, JOINTS["J2"])
    # cap that sits on the thoracic spine head
    sdf.add_capsule(f, grid, top + ES * 1.0, top + ES * 8.0, R_SPINE + 2.2, k=3.0)
    rz = rad_z()
    holes = []
    for sgn in (-1, 1):
        rear_rib(f, grid, sgn, rz["top"] - 20.0, [rz["top"]], holes)
        sdf.add_capsule(f, grid, (0, REAR_Y - 2.5, rz["top"]), (sgn * 7.5, P.RAD["y_rear"] + 2.6, rz["top"]), 4.2, 3.6, k=2.0)
        rear_pad(f, grid, sgn * 7.5, rz["top"], holes)
    drill_rear(f, grid, holes)
    # flat seat against the thoracic head + M5 clearance + counterbore
    sl = grid.box(top - 16, top + 16)
    x, y, z = grid.coords(sl)
    f[sl] = np.maximum(f[sl], -sdf.halfspace(x, y, z, top + ES * 0.12, ES))
    sdf.add_cyl(f, grid, top - ES * 4, top + ES * 30, 2.7, op="cut")
    sdf.add_cyl(f, grid, top + ES * 6.5, top + ES * 30, 5.0, op="cut")
    lap_cut(f, grid, JOINTS["J2"], "high")
    return f, grid


def build_pelvis(h):
    psu_lo = np.array(P.PSU["lo"])
    psu_sz = np.array(P.PSU["size"])
    grid = sdf.Grid((-131, 15, -0.5), (129, REAR_Y + 12, psu_lo[2] + psu_sz[2] + 3), h)
    f = grid.new()
    kz = FR["keel_z"] + 1.0
    Kf = S(0, -12)
    Kr = np.array([0.0, REAR_Y, kz])
    # keel
    bone(f, grid, [Kf, (0, 120, kz - 1.5), (0, 210, kz - 1.5), Kr], R_KEEL, R_KEEL, belly=-0.4, k=3.0)
    # stubs into the two lap joints
    bone(f, grid, [Kf, S(0, 6), S(0, 28)], R_SPINE + 0.3, R_SPINE, k=3.0)
    bone(f, grid, [Kr, (0, REAR_Y, 34), (0, REAR_Y, 58)], R_SPINE + 0.3, R_SPINE, k=3.0)
    knuckle(f, grid, JOINTS["J3"])
    knuckle(f, grid, JOINTS["J4"])
    # feet
    feet = FR["feet"]
    for (fx, fy) in feet:
        K = Kf if fy < 150 else Kr
        mid = np.array([fx * 0.48, K[1] + (fy - K[1]) * 0.55, 9.5])
        bone(f, grid, [K, mid, (fx, fy, 6.8)], R_KEEL - 0.2, 4.2, belly=-0.2, k=3.5)
        sdf.add_capsule(f, grid, (fx, fy, 6.5), (fx, fy, 6.5), 7.2, k=3.0)
    # PSU pedestals
    zp = psu_lo[2]
    peds = [(0.0, psu_lo[1] + 18.0), (0.0, psu_lo[1] + psu_sz[1] - 18.0)]
    yo = psu_lo[1] + 0.5 * psu_sz[1]
    for sgn in (-1, 1):
        bone(f, grid, [(0, yo, kz - 1.5), (sgn * 40, yo + 2, kz + 2), (sgn * 68, yo, zp - 5.0)], R_RIB0, R_RIB1 + 0.2, k=3.0)
        peds.append((sgn * 68.0, yo))
    for (px, py) in peds:
        sdf.add_cyl(f, grid, (px, py, kz - 3), (px, py, zp), 5.2, k=2.5)
    # PSU bracket ring on the AC face (x = -90, the I/O side)
    xb = psu_lo[0] - 0.4 - 3.2
    y0, y1 = psu_lo[1], psu_lo[1] + psu_sz[1]
    z0, z1 = psu_lo[2], psu_lo[2] + psu_sz[2]
    sl = grid.box((xb - 1.5, y0 - 1, z0 - 1), (xb + 4.5, y1 + 1, z1 + 1))
    x, y, z = grid.coords(sl)
    ring = np.maximum(sdf.rect2(y, z, (y0, z0), (y1, z1), r=7.0), -sdf.rect2(y, z, (y0 + 12, z0 + 12), (y1 - 12, z1 - 12), r=6.0))
    # widen the ring into gussets at the four screw positions
    scr = [(y1 - u, z0 + v) for (u, v) in P.PSU["rear_holes"]]      # pattern as seen from outside (-x)
    for (sy, sz) in scr:
        ring = np.minimum(ring, sdf.circle2(y, z, (sy, sz), 7.0))
    ring = sdf.smax(ring, np.maximum(xb - x, x - (xb + 3.2)), 0.8)
    f[sl] = np.minimum(f[sl], ring)
    for (a, b) in (((-68.0, yo, zp - 5.0), (xb + 1.6, yo, z0 + 5.0)),
                   ((feet[2][0] * 0.48, Kr[1] + (feet[2][1] - Kr[1]) * 0.55, 9.5), (xb + 1.6, y1 - 6.0, z0 + 5.0)),
                   ((feet[0][0] * 0.48, Kf[1] + (feet[0][1] - Kf[1]) * 0.55, 9.5), (xb + 1.6, y0 + 6.0, z0 + 5.0))):
        m = 0.5 * (np.array(a) + np.array(b)) + np.array([-6.0, 0.0, -2.0])
        bone(f, grid, [a, m, b], R_RIB0, R_RIB1 + 0.3, k=3.0)
    for (sy, sz) in scr:
        sdf.add_cyl(f, grid, (xb - 6, sy, sz), (xb + 6, sy, sz), 2.0, op="cut")          # 6-32 clearance
    # riser shelf (plate normal to e_s) just below the PCIe fingers
    gx0 = P.GPU_AT["x0"]
    s_sh = P.GPU_AT["s0"] + P.GPU["tongue_y"] - 8.8
    xa, xb2 = gx0 + P.GPU["tongue_x"][0] - 11.0, -1.2
    ca, cb = S(xa, s_sh, -8.0), S(xb2, s_sh, -8.0)
    lo = np.minimum(ca, cb) - 12
    hi = np.maximum(ca, cb) + 12
    sl = grid.box(lo, hi)
    x, y, z = grid.coords(sl)
    o = S(0, s_sh, -8.0)
    ds = (x - o[0]) * ES[0] + (y - o[1]) * ES[1] + (z - o[2]) * ES[2]
    dn = (x - o[0]) * EN[0] + (y - o[1]) * EN[1] + (z - o[2]) * EN[2]
    shelf = np.maximum(sdf.rect2(x, dn, (xa, -7.0), (xb2, 7.0), r=4.0) + 0 * ds, np.abs(ds) - 1.6)
    f[sl] = np.minimum(f[sl], shelf.astype(np.float32))
    for xs_ in (xa + 6.0, xb2 - 6.0):
        sdf.add_cyl(f, grid, S(xs_, s_sh - 6, -8.0), S(xs_, s_sh + 6, -8.0), 1.7, op="cut")
    bone(f, grid, [S(xa + 8, s_sh, -11.0), S(-70, s_sh - 6, -16.0), (feet[0][0] * 0.48, Kf[1] + (feet[0][1] - Kf[1]) * 0.55, 9.5)],
         R_RIB1 + 0.3, R_RIB0, k=3.0)
    bone(f, grid, [S(xb2 - 9, s_sh, -11.5), S(-7.0, s_sh - 3, -17.0), S(-4.5, s_sh, -22.0)], R_RIB1 + 0.2, R_RIB0, k=3.0)
    # flat feet (1.5 mm rubber pad underneath)
    sl = grid.box((-200, -200, -5), (200, 400, 1.5))
    f[sl] = np.maximum(f[sl], 0.25)
    lap_cut(f, grid, JOINTS["J3"], "low")
    lap_cut(f, grid, JOINTS["J4"], "low")
    return f, grid


SEGMENTS = dict(pelvis=build_pelvis, lumbar=build_lumbar, thoracic=build_thoracic,
                dorsal=build_dorsal, cervical=build_cervical)


def main(h=0.5, only=None, do_export=True):
    out = {}
    for name, fn in SEGMENTS.items():
        if only and name not in only:
            continue
        tm = part.Timer()
        f, grid = fn(h)
        removed = sdf.keep_largest_solid(f, h)
        tm(f"{name}: grid {grid.shape}, islands removed {removed}")
        if do_export:
            meta = part.export_part(f"frame_{name}", f, grid, "316L", "world", extra=dict(islands_removed_voxels=removed),
                                    print_faces=400_000, web_faces=60_000, hero_faces=1_200_000, smooth_iters=2)
            out[name] = meta
            print(name, json.dumps({k: meta[k] for k in ("volume_cm3", "mass_g", "bbox_mm", "fits_envelope", "cost_usd_jlc_pcbway")}))
        else:
            sdf.mesh(f, grid).export(part.WORK / f"frame_{name}_c.ply")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    hh = float(args[0]) if args else 0.5
    main(h=hh, only=args[1:] or None, do_export=("--noexport" not in sys.argv))
