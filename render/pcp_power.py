"""pcp_power - ATX power supply, sleeved cables and the PCIe riser."""
import math
import random

import numpy as np

import pcp_comps as C
import pcp_conn as K
from pcp_board import pcie_slot, PCIE_CAV0, PCIE_TONGUE
from pcp_cooling import fan_blades
from pcp_geom import (Soup, Part, proto, box, box2, cyl, lathe, prism, plate, rect, disc, annulus, strip,
                      text, rrect, circle, fillet_poly, tube, tube_ring, poly_face, sweep, smooth_path,
                      path_frames, hex_grid, catmull, resample, TAU)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ----------------------------------------------------------------------------
# PSU
# ----------------------------------------------------------------------------
_FACE = {
    "+z": (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), lambda s: (0, 0, s[2])),
    "-z": (np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), lambda s: (0, s[1], 0)),
    "+x": (np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), lambda s: (s[0], 0, 0)),
    "-x": (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]), lambda s: (0, s[1], 0)),
    "-y": (np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), lambda s: (0, 0, 0)),
    "+y": (np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]), lambda s: (s[0], s[1], 0)),
}


def on_face(s, face, size):
    """Map a soup built in face coords (u right, v up seen from outside, w = out of the face) onto a box face."""
    M, off = _FACE[face]
    s.V = s.V @ M.T + np.array(off(size), float)
    return s


def _minifit_socket(cols, depth=10.0, pitch=4.2, rows=2):
    """Recessed modular socket block in face coords (centred on u,v; opening at w = -0.8)."""
    W, D = cols * pitch + 1.0, rows * pitch + 1.0
    holes = []
    for i in range(cols):
        for j in range(rows):
            c = ((i - (cols - 1) / 2) * pitch, (j - (rows - 1) / 2) * pitch)
            a = (pitch - 0.9) / 2
            if (i + j) % 2 == 0:
                holes.append(rrect(2 * a, 2 * a, 0.2, 1, c))
            else:
                holes.append(np.array([(-a, -a), (a, -a), (a, a * 0.35), (a * 0.35, a), (-a * 0.35, a), (-a, a * 0.35)]) + np.array(c))
    s = plate(rrect(W, D, 0.4, 2), holes, -depth, -0.8, 0.1, "plastic_black")
    s += box(W, D, 0.6, (0, 0, -depth - 0.3), 0.0, "plastic_black")
    pin = cyl(0.6, -depth, -3.0, 8, "tin", bevel=0.2)
    s.tile(pin, [((i - (cols - 1) / 2) * pitch, (j - (rows - 1) / 2) * pitch, 0) for i in range(cols) for j in range(rows)])
    return s, W, D


def _torus(R, r, n=64, m=8, mat="steel_bright"):
    prof = [(R + r * math.cos(a), r * math.sin(a)) for a in np.linspace(0, TAU, m + 1)]
    return lathe(prof[::-1], n, mat)


def psu_rotor(r_tip=56.0):
    s = fan_blades(7, 19.0, r_tip, -1.0, 27.0, 44.0, 44.0, 24.0, 34.0, 2.2, 1.3, nr=10, nc=8)
    s += lathe([(0, -10.5), (16.5, -10.5), (18.4, -9.8), (19.0, -8.0), (19.0, 8.0), (0, 8.0)], 40, "fan_plastic", axis="y")
    st = Soup()
    st += disc(14.5, 0.0, 40, "label_silver")
    st += annulus(11.0, 11.5, 0.004, 40, "ink_grey")
    st += C.mark(["DC 12V 0.45A", "FDB 2431"], 16.0, 5.0, 0.005, "ink_black", hmax=1.5)
    s += st.rot("x", 90).move(0, -10.52, 0)
    return s.rot("x", -90)          # axis -> Z, intake side up


def psu_case(size, rear_holes, label_face="+y"):
    sx, sy, sz = size
    th, bv = 1.0, 0.35
    mat = "powdercoat_black"
    s = Soup()
    fc = (sx / 2, sy / 2)
    fr = min(sx, sy) / 2 - 16.0
    fr = min(fr, 59.0)
    # +z: fan opening
    fs = [(fc[0] + a * 52.5, fc[1] + b * 52.5) for a in (-1, 1) for b in (-1, 1)]
    top = plate(rrect(sx, sy, 0.8, 2, fc), [circle(fr, 96, fc)], -th, 0.0, bv, mat)
    s += on_face(top, "+z", size)
    for p in fs:
        s += C.screw_head(6.0, 0.5, "phillips", "screw_black").move(p[0], p[1], sz - 0.52)
    s += on_face(plate(rrect(sx, sy, 0.8, 2, fc), [], -th, 0.0, bv, mat), "-z", size)
    # +-y sides (between top and bottom)
    for face in ("-y", "+y"):
        side = plate(np.array([(0, th), (sx, th), (sx, sz - th), (0, sz - th)]), [], -th, 0.0, bv, mat)
        s += on_face(side, face, size)
    # +x: AC side with honeycomb, inlet, switch, screw holes
    iu, iv = sy - 26.0, 22.0
    su, sv = sy - 26.0, 52.0
    inlet = rrect(27.6, 20.2, 2.0, 3, (iu, iv))
    swc = rrect(13.4, 19.6, 1.0, 2, (su, sv))
    def keep(u, v):
        if u > iu - 20.0 and v < sv + 14.0:
            return False
        return all((u - h[0]) ** 2 + (v - h[1]) ** 2 > 6.0 ** 2 for h in rear_holes)
    hexes = hex_grid(7.0, 7.0, sy - 7.0, sz - 7.0, 5.3, 2.35, keep)
    holes = [inlet, swc] + [circle(1.9, 14, h) for h in rear_holes] + hexes
    rear = plate(np.array([(th, th), (sy - th, th), (sy - th, sz - th), (th, sz - th)]), holes, -th, 0.0, 0.3, mat)
    s += on_face(rear, "+x", size)
    acc = Soup()
    c13 = np.array([(-12.0, -8.2), (12.0, -8.2), (12.0, 3.4), (7.2, 8.2), (-7.2, 8.2), (-12.0, 3.4)])
    acc += plate(rrect(27.2, 19.8, 1.8, 3), [c13], -17.0, -0.1, 0.3, "plastic_black").move(iu, iv, 0)
    acc += box(27.2, 19.8, 1.0, (iu, iv, -17.5), 0.0, "plastic_black")
    for (pu, pv) in ((0.0, 3.2), (-7.0, -2.6), (7.0, -2.6)):
        acc += box(2.0, 4.0, 13.0, (iu + pu, iv + pv, -10.5), 0.3, "steel_bright")
    acc += plate(rrect(13.0, 19.2, 0.9, 2), [rrect(10.2, 16.4, 0.5, 2)], -8.0, -0.1, 0.3, "plastic_black").move(su, sv, 0)
    rk = box(10.0, 16.2, 5.0, (0, 0, -2.5), 0.6, "plastic_gloss_black")
    rk += C.mark(["I"], 3, 3, 0.02, "ink_white", c=(0, 4.2), hmax=2.6)
    rk += annulus(1.0, 1.35, 0.02, 16, "ink_white", (0, -4.2))
    acc += rk.rot("x", 9).move(su, sv, -2.0)
    for h in rear_holes:
        acc += tube_ring(2.8, 1.5, -5.0, -th, 12, "screw_black").move(h[0], h[1], 0)
    acc += C.mark(["100-240V~  10-5A  50-60HZ"], 30.0, 1.6, 0.01, "ink_white", c=(iu - 4.0, iv + 13.2), hmax=1.4)
    s += on_face(acc, "+x", size)
    # -x: modular sockets
    rows = [(sz - 26.0, [(9, "M/B"), (5, "M/B"), (6, "12V-2X6")]), (26.0, [(4, "CPU/PCIE"), (4, "CPU/PCIE"), (4, "CPU/PCIE"), (3, "PERIF"), (3, "PERIF"), (3, "SATA")])]
    cut, mod = [], Soup()
    for v, items in rows:
        tot = sum(c * 4.2 + 1.0 for c, _ in items)
        gap = (sy - 2 * th - 12.0 - tot) / (len(items) - 1)
        u = th + 6.0
        for cols, lab in items:
            blk, W, D = _minifit_socket(cols)
            cut.append(rrect(W + 0.5, D + 0.5, 0.5, 2, (u + W / 2, v)))
            mod += blk.move(u + W / 2, v, 0)
            mod += C.mark([lab], W, 2.2, 0.01, "ink_white", c=(u + W / 2, v - D / 2 - 2.6), hmax=1.7)
            u += W + gap
    front = plate(np.array([(th, th), (sy - th, th), (sy - th, sz - th), (th, sz - th)]), cut, -th, 0.0, 0.3, mat)
    s += on_face(front, "-x", size)
    s += on_face(mod, "-x", size)
    # dark interior + fan well
    s += box2(th + 0.6, th + 0.6, th, sx - th - 0.6, sy - th - 0.6, sz - 27.5, 0.0, "hole_dark")
    # wire grille
    g = Soup()
    for R in (9.0, 18.5, 28.0, 37.5, 47.0, fr - 1.6):
        g += _torus(R, 0.85, 72, 6)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        for off in (-2.2, 2.2):
            rod = cyl(0.95, 7.0, fr + 12.0, 8, "steel_bright", axis="x").move(0, off, 0.9).rot("z", math.degrees(a))
            g += rod
    s += g.move(fc[0], fc[1], sz - 1.9)
    # rating label (generic text only)
    lw, lh = min(120.0, sx - 40.0), min(58.0, sz - 20.0)
    lab = Soup()
    lab += poly_face(rrect(lw, lh, 2.0, 3), 0.012, "label_silver")
    lab += strip([(-lw / 2 + 3, -lh / 2 + 3), (lw / 2 - 3, -lh / 2 + 3), (lw / 2 - 3, lh / 2 - 3), (-lw / 2 + 3, lh / 2 - 3)], 0.35, 0.02, "ink_black", closed=True)
    lab += C.mark(["SWITCHING POWER SUPPLY  ATX 650W"], lw - 12, 4.5, 0.02, "ink_black", c=(0, lh / 2 - 8.5), hmax=3.2)
    ty0 = lh / 2 - 15.0
    for i in range(5):
        lab += strip([(-lw / 2 + 6, ty0 - i * 5.2), (lw / 2 - 6, ty0 - i * 5.2)], 0.25, 0.02, "ink_black")
    cols = [-lw / 2 + 6, -lw / 2 + 30] + [-lw / 2 + 30 + (lw - 36) / 5 * (k + 1) for k in range(5)]
    for x in cols:
        lab += strip([(x, ty0), (x, ty0 - 20.8)], 0.25, 0.02, "ink_black")
    table = [("AC INPUT", ["100-240V~", "10-5A", "50-60HZ", "", ""]), ("DC OUTPUT", ["+3.3V", "+5V", "+12V", "-12V", "+5VSB"]),
             ("MAX CURRENT", ["20A", "20A", "54.1A", "0.3A", "3A"]), ("MAX POWER", ["100W", "", "649.2W", "3.6W", "15W"])]
    for r, (hd, vals) in enumerate(table):
        y = ty0 - 2.6 - r * 5.2
        lab += C.mark([hd], 21.0, 2.6, 0.02, "ink_black", c=((cols[0] + cols[1]) / 2, y), hmax=1.9)
        for k, v in enumerate(vals):
            if v:
                lab += C.mark([v], (lw - 36) / 5 - 2, 2.6, 0.02, "ink_black", c=((cols[1 + k] + cols[2 + k]) / 2, y), hmax=1.9)
    lab += C.mark(["TOTAL POWER 650W    80+ EFFICIENCY    S/N 2431 0000 4127", "CAUTION: HAZARDOUS AREA - DO NOT REMOVE THIS COVER"], lw - 14, 6.5, 0.02,
                  "ink_black", c=(0, -lh / 2 + 8.0), hmax=1.9)
    if label_face in ("+y", "-y"):
        lab.move(sx / 2, sz / 2, 0)
    else:
        lab.move(sx / 2, sy / 2, 0)
    s += on_face(lab, label_face, size)
    return s


def build_psu_atx(L, name="psu_atx", label_face="+y"):
    """ATX PSU. Origin at the box min corner; fan + grille on +z, AC side +x, modular side -x.
    rear_holes are taken as (u, v) = (y, z) on the +x face. The fan rotor is '<root>_fan'."""
    Pp = L["psu"]
    size = tuple(Pp["size"])
    P = Part(name)
    key = "psu_%g_%g_%g_%s" % (size + (label_face,))
    P.obj("case", proto(key, lambda: psu_case(size, [tuple(h) for h in Pp["rear_holes"]], label_face)))
    P.obj("fan", proto("psu_rotor", psu_rotor, smooth=50), (size[0] / 2, size[1] / 2, size[2] - 15.0))
    return P.root


# ----------------------------------------------------------------------------
# cables
# ----------------------------------------------------------------------------
CABLES = {
    # cols, rows, pin pitch, wire radius, plug body length, extra thin wires
    "atx24": dict(cols=12, rows=2, pitch=4.2, r=1.7, plug=19.5, sense=0),
    "eps8": dict(cols=4, rows=2, pitch=4.2, r=1.7, plug=19.5, sense=0),
    "12vhpwr": dict(cols=6, rows=2, pitch=3.0, r=1.35, plug=15.0, sense=4),
    "fan": dict(cols=1, rows=1, pitch=0.0, r=1.9, plug=12.5, sense=0),
}


def cable_plug(kind):
    """Plug body: local z = 0 is the mating face, body extends to +z; columns along x, latch on +y."""
    c = CABLES[kind]
    cols, rows, p, Lp = c["cols"], c["rows"], c["pitch"], c["plug"]
    if kind == "fan":
        s = plate(rrect(10.3, 5.6, 0.5, 2), [rrect(1.3, 1.3, 0.1, 1, ((i - 1.5) * 2.54, -0.3)) for i in range(4)], 0.0, Lp, 0.25, "plastic_black")
        s += box(7.0, 0.9, Lp - 3.0, (-1.27, 3.0, 1.0), 0.2, "plastic_black", base=True)
        return s
    W, D = cols * p + 0.9, rows * p + 0.9
    holes = []
    a = (p - 1.0) / 2
    for i in range(cols):
        for j in range(rows):
            cc = ((i - (cols - 1) / 2) * p, (j - (rows - 1) / 2) * p)
            if (i + j) % 2 == 0:
                holes.append(rrect(2 * a, 2 * a, 0.15, 1, cc))
            else:
                holes.append(np.array([(-a, -a), (a, -a), (a, a * 0.35), (a * 0.35, a), (-a * 0.35, a), (-a, a * 0.35)]) + np.array(cc))
    nose = Lp * 0.56
    s = plate(rrect(W - 0.5, D - 0.5, 0.4, 2), holes, 0.0, nose, 0.12, "plastic_black")
    s += plate(rrect(W + 0.6, D + 0.6, 0.6, 2), [circle(c["r"] * 0.9, 8, ((i - (cols - 1) / 2) * p, (j - (rows - 1) / 2) * p)) for i in range(cols) for j in range(rows)],
               nose - 0.2, Lp, 0.25, "plastic_black")
    lw = 3.4 if kind != "12vhpwr" else 5.0
    s += box(lw, 1.1, Lp * 0.78, (0, D / 2 + 1.25, Lp * 0.2), 0.3, "plastic_black", base=True)
    s += box(lw, 1.4, 2.0, (0, D / 2 + 0.6, Lp * 0.62), 0.2, "plastic_black", base=True)
    s += box(lw + 0.6, 1.0, 1.6, (0, D / 2 + 0.9, Lp * 0.2), 0.3, "plastic_black", base=True)
    return s


def build_cable(points, kind="atx24", name=None, up=None, up_end=None, sleeve="sleeve_black", plugs=(True, True),
                wobble=0.3, seed=1):
    """Sleeved cable through WORLD points. Path ends = seated mating faces of the two plugs; `up` (and
    optionally `up_end`) = direction of the latch / row-stacking side. Returns the root Empty."""
    c = CABLES[kind]
    P = Part(name or ("cable_" + kind))
    path = smooth_path(points, 2.5 if kind != "fan" else 1.5)
    T, N, B = path_frames(path, up)
    n = len(path)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    u = arc / arc[-1]
    if up_end is not None:
        ue = np.asarray(up_end, float)
        ue = ue - (ue @ T[-1]) * T[-1]
        if np.linalg.norm(ue) > 1e-6:
            ue /= np.linalg.norm(ue)
            ang = math.atan2(float(np.cross(N[-1], ue) @ T[-1]), float(N[-1] @ ue))
            tw = ang * (3 * u ** 2 - 2 * u ** 3)
            cs, sn = np.cos(tw)[:, None], np.sin(tw)[:, None]
            N, B = N * cs + B * sn, B * cs - N * sn
    rnd = random.Random(seed)
    wires = Soup()
    win = np.sin(np.pi * u) ** 0.7
    cols, rows, p = c["cols"], c["rows"], c["pitch"]
    offs = [((i - (cols - 1) / 2) * p, (j - (rows - 1) / 2) * p, c["r"]) for i in range(cols) for j in range(rows)]
    for k in range(c["sense"]):
        offs.append(((k - (c["sense"] - 1) / 2) * 1.5, (rows / 2) * p + 0.75, 0.62))
    for (b, a, r) in offs:
        ph1, ph2 = rnd.uniform(0, TAU), rnd.uniform(0, TAU)
        wl = rnd.uniform(60, 110)
        wa = wobble * win * np.sin(arc * TAU / wl + ph1) if kind != "fan" else 0.0
        wb = wobble * win * np.sin(arc * TAU / (wl * 0.8) + ph2) if kind != "fan" else 0.0
        Pw = path + N * (a + wa)[:, None] + B * (b + wb)[:, None] if kind != "fan" else path
        wires += sweep(Pw, circle(r, 10 if r > 1 else 6), sleeve if r > 1 else "wire_black", caps=True, frames=(T, N, B), uv=True)
    P.obj("wires", wires.mesh(P.name + "_wires", smooth=60, weighted=False))
    plug = proto("plug_" + kind, lambda: cable_plug(kind))
    import bpy
    from mathutils import Matrix
    for end, on in ((0, plugs[0]), (1, plugs[1])):
        if not on:
            continue
        i = 0 if end == 0 else n - 1
        zc = T[i] if end == 0 else -T[i]
        xc = -B[i] if end == 0 else B[i]
        yc = N[i]
        M = Matrix(((xc[0], yc[0], zc[0], path[i][0]), (xc[1], yc[1], zc[1], path[i][1]), (xc[2], yc[2], zc[2], path[i][2]), (0, 0, 0, 1)))
        ob = P.obj("plug_" + "ab"[end], plug)
        ob.matrix_local = M
    P.root["length_mm"] = float(arc[-1])
    return P.root


# ----------------------------------------------------------------------------
# PCIe riser
# ----------------------------------------------------------------------------
def riser_male(w_pcb=88.0, h_up=16.0):
    """Male edge card. Local frame: x = width, z = out of the slot, origin = slot opening centre
    (tongue centre); PCB mid-plane is y = 0."""
    half = PCIE_TONGUE / 2
    t = 1.57
    kx = -half + 12.15
    out = [(-half, -8.4), (kx - 0.95, -8.4), (kx - 0.95, -0.6), (kx + 0.95, -0.6), (kx + 0.95, -8.4), (half, -8.4), (half, 0.5),
           (w_pcb / 2, 2.0), (w_pcb / 2, h_up), (-w_pcb / 2, h_up), (-w_pcb / 2, 2.0), (-half, 0.5)]
    out = fillet_poly(np.array(out), [0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.3, 0.5, 1.5, 1.5, 0.5, 0.3], 3)
    s = K.pcb_plate(out, [], t, "pcb_black_gpu").move(0, 0, t / 2)
    xs = [-half + 0.65 + i for i in range(11)] + [-half + 13.65 + i for i in range(71)]
    f = rect(0.7, 4.3, 0.0, "gold_pin")
    s.tile(f.copy().move(0, 0, t / 2 + 0.007), [(x, -8.4 + 0.5 + 2.15, 0) for x in xs])
    s.tile(f.copy().rot("y", 180).move(0, 0, -t / 2 - 0.007), [(x, -8.4 + 0.5 + 2.15, 0) for x in xs])
    s += K.silk_label("PCIE 4.0 X16 RISER  REV 1.1", 0, h_up - 1.6, 0, 1.0).move(0, 0, t / 2)
    s.rot("x", 90)                       # plate (x, y, z) -> (x, -z, y): PCB plane = x-z, thickness along y
    return s


def riser_female(w_pcb=106.0, d_pcb=22.0, zs=11.25):
    """Female slot on its small PCB. Local frame: x = width, z = slot opening direction, origin = slot
    opening centre (tongue centre). PCB top surface at z = -zs."""
    Ls = PCIE_CAV0 + 0.15 + PCIE_TONGUE + 0.15 + PCIE_CAV0 + 4.6
    xc = PCIE_CAV0 + 0.15 + PCIE_TONGUE / 2
    s = pcie_slot(Ls, zs).move(-xc, 0, -zs)
    cxp = Ls / 2 - xc
    holes = [circle(1.7, 16, (cxp + sg * (w_pcb / 2 - 4.0), 0)) for sg in (-1, 1)]
    s += K.pcb_plate(rrect(w_pcb, d_pcb, 2.0, 4, (cxp, 0)), holes, 1.6, "pcb_black_gpu").move(0, 0, -zs)
    for sg in (-1, 1):
        s += annulus(1.7, 3.1, 0.006, 20, "enig_gold", (cxp + sg * (w_pcb / 2 - 4.0), 0)).move(0, 0, -zs)
    c4 = C.chip(1.0, 0.5, 0.5)
    s.tile(c4, [(-38.0 + i * 2.0, sg * 6.6, -zs) for i in range(40) for sg in (-1, 1)], [90] * 80)
    s += K.silk_label("PCIE 4.0 X16", cxp - w_pcb / 2 + 17.0, -d_pcb / 2 + 2.0, 0, 1.0).move(0, 0, -zs)
    return s


def build_riser(points, width_dir=(1.0, 0.0, 0.0), male_dir=None, female_dir=None, name="riser", ribbon_w=83.0,
                segments=5):
    """PCIe 4.0 x16 riser through WORLD points.

    points[0]  = centre of the motherboard slot opening (tongue centre, slot top plane); the male edge card
                 plugs in along -male_dir (default: direction points[0] -> points[1]).
    points[-1] = centre of the female slot opening where the graphics-card tongue enters; the slot opens
                 towards female_dir (default: direction points[-2] -> points[-1]).
    points[1:-1] shape the flat ribbon (5 black shielded segments). width_dir = long axis of both slots."""
    from mathutils import Matrix
    pts = [np.asarray(p, float) for p in points]
    W = _unit(width_dir)
    D0 = _unit(pts[1] - pts[0]) if male_dir is None else _unit(male_dir)
    F = _unit(pts[-1] - pts[-2]) if female_dir is None else _unit(female_dir)
    D0 = _unit(D0 - (D0 @ W) * W)
    F = _unit(F - (F @ W) * W)
    Nm = np.cross(W, D0)                     # male card normal
    P = Part(name)
    # male card: local (x, y, z) -> (W, Nm... ) keep right-handed: x=W, z=D0, y = z x x = D0 x W
    ym = np.cross(D0, W)
    M = Matrix(((W[0], ym[0], D0[0], pts[0][0]), (W[1], ym[1], D0[1], pts[0][1]), (W[2], ym[2], D0[2], pts[0][2]), (0, 0, 0, 1)))
    ob = P.obj("male", proto("riser_male", riser_male))
    ob.matrix_local = M
    side = 1.0 if (pts[1] - pts[0]) @ ym >= -1e-6 else -1.0
    A0 = pts[0] + D0 * 10.5 + ym * side * (0.785 + 0.75)
    # female slot
    yf = np.cross(F, W)
    Mf = Matrix(((W[0], yf[0], F[0], pts[-1][0]), (W[1], yf[1], F[1], pts[-1][1]), (W[2], yf[2], F[2], pts[-1][2]), (0, 0, 0, 1)))
    ob = P.obj("female", proto("riser_female", riser_female))
    ob.matrix_local = Mf
    zs = 11.25
    back = pts[-2] - pts[-1]
    sidef = 1.0 if back @ yf >= 0 else -1.0
    A1 = pts[-1] - F * (zs + 1.6 + 0.75) + yf * sidef * 2.0
    ctrl = [A0, A0 + D0 * 5.0] + pts[1:-1] + [A1 + yf * sidef * 7.0, A1]
    path = resample(catmull(ctrl, 14), 1.5)
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Nw = W[None, :] - (T @ W)[:, None] * T
    Nw /= np.linalg.norm(Nw, axis=1, keepdims=True)
    Bn = np.cross(T, Nw)
    gap = 1.0
    sw = (ribbon_w - (segments - 1) * gap) / segments
    rib = Soup()
    for k in range(segments):
        off = (k - (segments - 1) / 2) * (sw + gap)
        sec = rrect(sw, 1.15, 0.5, 2, (off, 0.0))
        rib += sweep(path, sec, "ribbon_black", caps=True, frames=(T, Nw, Bn), uv=True)
    P.obj("ribbon", rib.mesh(P.name + "_ribbon", smooth=50, weighted=False))
    # over-moulded strain reliefs where the ribbon is soldered
    blk = proto("riser_block", lambda: box(ribbon_w + 3.0, 3.0, 8.0, (0, 0, 0), 0.6, "plastic_black"))
    o = P.obj("male_block", blk)
    c0 = A0 + D0 * 1.0 + ym * side * 0.3
    o.matrix_local = Matrix(((W[0], ym[0], D0[0], c0[0]), (W[1], ym[1], D0[1], c0[1]), (W[2], ym[2], D0[2], c0[2]), (0, 0, 0, 1)))
    o = P.obj("female_block", blk)
    c1 = A1 + yf * sidef * 2.0 - F * 0.2
    o.matrix_local = Matrix(((W[0], F[0], -yf[0], c1[0]), (W[1], F[1], -yf[1], c1[1]), (W[2], F[2], -yf[2], c1[2]), (0, 0, 0, 1)))
    P.root["length_mm"] = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return P.root
