"""pcp_gpu - bare RTX-4090-class graphics card PCB (no cooler), built from the measured layout.

GPU frame of design/data/gpu_pcb_layout.json: origin = bracket-end / PCIe-edge corner, +x along
the card, +y to the top edge, +z out of the component side, PCB top z = 0 (back side z < -pcb_t).
Positions come from layout["gpu"] (params.py) first, extra detail from the measured JSON.
"""
import math
import random

import numpy as np

import pcp_comps as C
import pcp_conn as K
from pcp_geom import (Soup, Part, proto, box, box2, cyl, lathe, prism, plate, rect, disc, annulus, strip,
                      text, rrect, circle, fillet_poly, loft, tube_ring, poly_face, hex_grid, TAU)

EPS = C.EPS


def _outline(g, G):
    P = np.array(G["pcb_outline"]["polygon_xy"], float)
    R = float(G["pcb_outline"]["corner_radius"]["value"])
    Lx, Hy = g["pcb_l"], g["pcb_h"]
    rx, ry = g["hpwr"]["recess_x"], g["hpwr"]["recess_y"]
    ty = g["tongue_y"]
    big = [(0.0, Hy), (rx, Hy), (Lx, ry), (Lx, 0.0)]
    radii = []
    for x, y in P:
        r = 0.0
        if any(abs(x - bx) < 1e-6 and abs(y - by) < 1e-6 for bx, by in big):
            r = R
        elif abs(x - rx) < 1e-6 and abs(y - ry) < 1e-6:
            r = 1.0
        elif abs(y - ty) < 1e-6:
            r = 0.6
        elif y < -0.01:
            r = 0.9 if abs(y + 4.3) < 0.05 else 0.5
        radii.append(r)
    return fillet_poly(P, radii, 6)


def hpwr_header(w, d, h):
    """Right-angle 16-pin (12+4) power header. Mating face at y = 0 looking towards +y, body to -y,
    x centred, z = 0 on the PCB."""
    holes = []
    for i in range(6):
        for j in range(2):
            c = ((i - 2.5) * 3.0, 2.6 + j * 3.0 + 1.3)
            a = 1.15
            if (i + j) % 2:
                holes.append(rrect(2 * a, 2 * a, 0.25, 1, c))
            else:
                holes.append(np.array([(-a, -a), (a, -a), (a, a * 0.3), (a * 0.3, a), (-a * 0.3, a), (-a, a * 0.3)]) + np.array(c))
    for i in range(4):
        holes.append(rrect(1.2, 1.0, 0.1, 1, ((i - 1.5) * 2.0, 1.25)))
    face = plate(rrect(w, h - 0.3, 0.6, 2, (0, (h - 0.3) / 2 + 0.3)), holes, 0.0, 7.5, 0.12, "plastic_black")
    s = face.rot("x", 90)
    s += box(w, d - 7.4, h - 0.3, (0, -7.4 - (d - 7.4) / 2, 0.3), 0.15, "plastic_black", base=True)
    pin = cyl(0.55, 0.0, 5.5, 8, "tin", bevel=0.2, axis="y")
    s.tile(pin.move(0, -7.0, 0), [((i - 2.5) * 3.0, 0, 2.6 + j * 3.0 + 1.3) for i in range(6) for j in range(2)])
    spin = box(0.35, 5.0, 0.35, (0, -4.6, 1.25), 0.0, "gold_pin")
    s.tile(spin, [((i - 1.5) * 2.0, 0, 0) for i in range(4)])
    s += box(6.0, 5.0, 0.9, (0, -3.2, h - 0.35), 0.2, "plastic_black", base=True)          # latch ramp
    s += box(4.0, 1.2, 1.3, (0, -1.4, h + 0.4), 0.2, "plastic_black", base=True)
    for sx in (-1, 1):
        s += box(1.6, 3.0, 2.0, (sx * (w / 2 + 0.3), -d + 4.0, 0), 0.1, "steel_sheet", base=True)
    return s


def slide_switch(w=6.7, d=2.8, h=1.4):
    s = box(w, d, h, (0, 0, 0), 0.12, "steel_sheet", base=True)
    s += box(1.5, 2.0, 0.9, (-1.1, d / 2 + 0.9, 0.25), 0.15, "plastic_black", base=True)
    for sx in (-1, 0, 1):
        s += box(0.5, 0.8, 0.2, (sx * 2.0, -d / 2 - 0.3, 0), 0.0, "tin", base=True)
    return s


def shunt(l=6.4, w=3.2, h=0.7):
    s = box(l - 2.0, w * 0.96, h * 0.9, (0, 0, 0.03), 0.06, "res_black", base=True)
    for sg in (-1, 1):
        s += box(1.1, w, h, (sg * (l / 2 - 0.55), 0, 0.02), 0.06, "tin", base=True)
        s += C.wedge(sg * l / 2, sg * (l / 2 + 0.4), w * 0.9, w, h * 0.5)
    s += C.mark(["R001"], l * 0.45, w * 0.5, h * 0.93 + 0.01, "silk_white")
    return s


def io_bracket(openings, z_open, y0=-10.5, y1=103.0, z0=-3.4, z1=35.3, x_face=-4.0, th=0.9):
    """Plain black 2-slot bracket. openings = list of (kind, centre_y). Built directly in the GPU frame."""
    out = [(y0, z0 + 2.0), (y0 - 7.5, z0 + 5.0), (y0 - 7.5, z0 + 13.5), (y0, z0 + 16.5), (y0, z0 + 22.3),
           (y0 - 7.5, z0 + 25.3), (y0 - 7.5, z0 + 33.8), (y0, z0 + 36.8), (y0, z1), (y1, z1), (y1, z0), (y0, z0)]
    out = fillet_poly(np.array(out), [0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.0, 0.3, 0.3, 1.0], 3)
    holes = []
    for kind, cy in openings:
        if kind == "dp":
            holes.append(np.array([(-8.9, -2.9), (6.7, -2.9), (8.9, -0.9), (8.9, 2.9), (-8.9, 2.9)]) + np.array((cy, z_open)))
        else:
            holes.append(np.array([(-7.8, 2.8), (-7.8, -0.8), (-5.6, -2.8), (5.6, -2.8), (7.8, -0.8), (7.8, 2.8)]) + np.array((cy, z_open)))
    holes += hex_grid(y0 + 6.0, z_open + 6.5, y1 - 5.0, z1 - 2.0, 5.4, 2.1)
    s = plate(out, holes, 0.0, th, 0.2, "paint_satin_black")
    s.V = np.stack([x_face - s.V[:, 2], s.V[:, 0], s.V[:, 1]], 1)       # plate (y, z, depth) -> card frame
    # top mounting flange with screw notches
    fl = [(0.0, z0), (-11.4, z0), (-11.4, z0 + 5.5), (-7.0, z0 + 5.5), (-7.0, z0 + 12.0), (-11.4, z0 + 12.0),
          (-11.4, z0 + 25.8), (-7.0, z0 + 25.8), (-7.0, z0 + 32.3), (-11.4, z0 + 32.3), (-11.4, z1), (0.0, z1)]
    f = prism(fillet_poly(np.array(fl), [0, 1, 0.8, 1.5, 1.5, 0.8, 0.8, 1.5, 1.5, 0.8, 1, 0], 3), 0.0, th, "paint_satin_black", bevel=0.2,
              bevel_bottom=True)
    f.V = np.stack([x_face + f.V[:, 0], y1 - th + f.V[:, 2], f.V[:, 1]], 1)
    f.flip()
    s += f
    return s


def _datamatrix(n=12, cell=0.22, z=0.0, mat="laser_mark_dim", seed=3):
    rnd = random.Random(seed)
    s = Soup()
    V, F = [], []
    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0 or rnd.random() < 0.5:
                k = len(V)
                x, y = i * cell, j * cell
                V += [(x, y, z), (x + cell, y, z), (x + cell, y + cell, z), (x, y + cell, z)]
                F.append((k, k + 1, k + 2, k + 3))
    return s.add(V, np.array(F), mat).move(-n * cell / 2, -n * cell / 2, 0)


def gpu_package(g, G):
    """GPU FCBGA centred on the package centre."""
    pw, pl = g["package"]
    dw, dl = g["die"]
    zs, zd = g["z_stiffener"], g["z_die"]
    dc = G["gpu"]["die"]["centre"]
    cx, cy = g["centre"]
    ball_h = 0.42
    sub_t = zs - 0.7 - ball_h + 0.03
    s = C.fcbga(pw, pl, zd, die=(dw, dl), sub_t=sub_t, ball_h=ball_h, stiffener=g["stiffener_open"], z_stiff=zs,
                die_c=(dc["x"] - cx, dc["y"] - cy), ball_pitch=1.0)
    z = zd + 0.004
    s += C.mark(["GX102-300-A1", "2243A1 PKY7W.001", "TWN N4"], dw * 0.7, 5.0, z, "laser_mark_dim", c=(dc["x"] - cx, dc["y"] - cy + 2.0), hmax=1.0)
    s += _datamatrix(12, 0.2, z).move(dc["x"] - cx + dw / 2 - 3.2, dc["y"] - cy - dl / 2 + 3.2, 0)
    s += poly_face([(-pw / 2 + 1.2, -pl / 2 + 1.2), (-pw / 2 + 3.4, -pl / 2 + 1.2), (-pw / 2 + 1.2, -pl / 2 + 3.4)], zs + 0.004, "ihs_mark")
    return s


def build_gpu_4090(L, G, name="gpu_4090", seed=5, bracket=True):
    g = L["gpu"]
    t = g["pcb_t"]
    Lx, Hy = g["pcb_l"], g["pcb_h"]
    P = Part(name)
    rnd = random.Random(seed)
    occ = K.Occ(-5, -14, Lx + 1, Hy + 1, 0.2)
    bocc = K.Occ(-5, -14, Lx + 1, Hy + 1, 0.2)
    ov, bov = Soup(), Soup()              # top overlay; back overlay (built with y mirrored, then flipped)
    cnt = {}

    def inst(key, fn, nm, x, y, rot=0.0, z=0.0, back=False):
        cnt[nm] = cnt.get(nm, -1) + 1
        if back:
            return P.obj("%s.%03d" % (nm, cnt[nm]), proto(key, fn), (x, y, -t), (180, 0, rot))
        return P.obj("%s.%03d" % (nm, cnt[nm]), proto(key, fn), (x, y, z), (0, 0, rot))

    # ------------------------------------------------------------------ PCB
    MH = G["mounting_holes"]
    core = [tuple(c) for c in g["core_holes"]]
    mec = {k: tuple(v) for k, v in g["mec"].items()}
    for k, v in MH["other_holes"].items():
        mec.setdefault(k, (v["x"], v["y"]))
    c_dr = MH["gpu_cooler_holes"]["TL"]["drill_dia_approx"] / 2
    m_dr = MH["other_holes"]["MEC1"]["drill_dia_approx"] / 2
    c_fp, c_bp = MH["gpu_cooler_holes"]["TL"]["front_pad_dia"] / 2, MH["gpu_cooler_holes"]["TL"]["back_pad_dia"] / 2
    m_fp, m_bp = MH["other_holes"]["MEC1"]["front_pad_dia"] / 2, MH["other_holes"]["MEC1"]["back_pad_dia"] / 2
    hd = G["headers"]["front"]
    small_th = [tuple(p) for p in hd["J11"]["holes"]] + [(x, y) for x in hd["J6_PJP1"]["pins_x"] for y in hd["J6_PJP1"]["y"].values()]
    hp = G["power_connector_12vhpwr"]
    hp_pins = [(x, y) for x in hp["power_pins"]["x"] for y in hp["power_pins"]["rows_y"]] + [(x, hp["sense_pins"]["y"]) for x in hp["sense_pins"]["x"]]
    holes = [circle(c_dr, 24, c) for c in core] + [circle(m_dr, 20, c) for c in mec.values()] + [circle(0.45, 10, c) for c in small_th]
    pcb = K.pcb_plate(_outline(g, G), holes, t, "pcb_black_gpu")
    P.obj("pcb", pcb.mesh(P.name + "_pcb", smooth=40, weighted=False))
    for c in core:
        ov += annulus(c_dr, c_fp, 0.006, 28, "enig_gold", c)
        bov += annulus(c_dr, c_bp, 0.006, 32, "enig_gold", (c[0], -c[1]))
        ov += tube_ring(c_dr + 0.02, c_dr - 0.08, -t, 0.0, 24, "enig_gold").move(c[0], c[1], 0)
        occ.mark_disc(c[0], c[1], c_fp + 0.3)
        bocc.mark_disc(c[0], c[1], c_bp + 0.3)
    for c in mec.values():
        ov += annulus(m_dr, m_fp, 0.006, 24, "enig_gold", c)
        bov += annulus(m_dr, m_bp, 0.006, 24, "enig_gold", (c[0], -c[1]))
        ov += tube_ring(m_dr + 0.02, m_dr - 0.08, -t, 0.0, 20, "enig_gold").move(c[0], c[1], 0)
        occ.mark_disc(c[0], c[1], m_fp + 0.3)
        bocc.mark_disc(c[0], c[1], m_bp + 0.3)
    for c in small_th:
        ov += annulus(0.45, 0.9, 0.006, 12, "enig_gold", c)
        bov += annulus(0.45, 0.9, 0.006, 12, "enig_gold", (c[0], -c[1]))
        occ.mark_disc(c[0], c[1], 1.1)
    ob = hd["J11"]["outline_bbox"]
    ov += K.silk_box((ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2 - 1.0, ob[2] - ob[0], ob[3] - ob[1] - 2.4, pin1=False)

    # ------------------------------------------------------------------ gold fingers (both sides)
    E = G["pcie_edge_connector"]["contact_centres_x"]
    xs = [E["B1"]["value"] + i for i in range(11)] + [E["B12"]["value"] + i for i in range(71)]
    ty = g["tongue_y"]
    f_long, f_short = rect(0.7, 4.3, 0.007, "gold_pin"), rect(0.7, 3.3, 0.007, "gold_pin")
    fl = [(x, ty + 0.55 + 2.15, 0) for i, x in enumerate(xs) if i % 9 != 4]
    fs = [(x, ty + 0.55 + 1.65 + 1.0, 0) for i, x in enumerate(xs) if i % 9 == 4]
    ov.tile(f_long, fl)
    ov.tile(f_short, fs)
    bov.tile(f_long, [(x, -y, 0) for x, y, _ in fl])
    bov.tile(f_short, [(x, -y, 0) for x, y, _ in fs])
    occ.mark(g["tongue_x"][0], ty, g["tongue_x"][1], -3.0)
    bocc.mark(g["tongue_x"][0], ty, g["tongue_x"][1], -3.0)

    # ------------------------------------------------------------------ GPU package + memory
    cx, cy = g["centre"]
    pw, pl = g["package"]
    P.obj("gpu_package", proto("gpu_pkg_%g_%g_%g" % (pw, g["z_die"], g["z_stiffener"]), lambda: gpu_package(g, G)), (cx, cy, 0))
    occ.mark_c(cx, cy, pw + 1.2, pl + 1.2)
    ov += K.silk_corners(cx, cy, pw + 1.8, pl + 1.8, 3.0, 0.15)
    zm = g["z_mem"]
    for nm, (mx, my, mw, mh) in g["mem"].items():
        long_x = mw >= mh
        a, b = (mw, mh) if long_x else (mh, mw)
        inst("gddr_%g_%g_%g" % (a, b, zm), lambda: C.bga(a, b, zm, ("2QA47", "D8BZC"), ball_pitch=0.75, hmax=1.6), "mem", mx, my, 0 if long_x else 90)
        occ.mark_c(mx, my, mw + 0.5, mh + 0.5)
        ov += K.silk_box(mx, my, mw + 0.28, mh + 0.26, 0.1, pin1=False)
        ov += K.silk_label(nm, mx - mw / 2 + 1.2, my + mh / 2 + 0.75, 0, 0.6)

    # ------------------------------------------------------------------ VRM banks (row by row)
    y0, pitch = g["vrm_row_y0"], g["vrm_row_pitch"]
    zd, zi = g["z_drmos"], g["z_ind"]
    V = G["vrm"]
    stage = lambda: C.qfn(6.0, 5.0, zd, (9, 5), ("MP86957", "2431 H7K"), pitch=0.5)
    near, far = g["vrm_near"], g["vrm_far"]
    nb, fb = V["bracket_side_bank"]["inductor_row"]["body_x"], V["far_side_bank"]["inductor_row"]["body_x"]
    ind_w = 7.1

    def bank(rows, drmos_x, ind_x, body_x, rot, tag):
        sxc = (drmos_x[0] + drmos_x[1]) / 2
        ixc, il = (body_x[0] + body_x[1]) / 2, body_x[1] - body_x[0]
        for r in rows:
            y = y0 + r * pitch
            inst("gstage_%g" % zd, stage, "stage", sxc, y, rot)
            inst("gind_%g_%g" % (il, zi), lambda: C.inductor(il, ind_w, zi, "R15", term_len=1.5), "ind", ixc, y, rot)
            occ.mark_c(sxc, y, drmos_x[1] - drmos_x[0] + 0.3, 5.5)
            occ.mark_c((ind_x[0] + ind_x[1]) / 2, y, ind_x[1] - ind_x[0] + 0.2, 7.9)
            ov.__iadd__(K.silk_box(sxc, y, drmos_x[1] - drmos_x[0], 5.36, 0.1, pin1=False))
            ov.__iadd__(K.silk_corners((ind_x[0] + ind_x[1]) / 2, y, ind_x[1] - ind_x[0], 7.8, 1.0, 0.1))

    bank(near["rows"], near["drmos_x"], near["ind_x"], nb, 0, "n")
    bank(far["rows"], far["drmos_x"], far["ind_x"], fb, 180, "f")
    zc = g["z_cap"]
    for y in far["cap_y"]:
        inst("pcan_%g_%g" % (far["cap_d"], zc), lambda: C.polymer_can(far["cap_d"], zc, ("180", "16V")), "cap", far["cap_x"], y, 0)
        occ.mark_c(far["cap_x"], y, far["cap_d"] + 0.4, far["cap_d"] + 0.4)
        ov += annulus(far["cap_d"] / 2 + 0.25, far["cap_d"] / 2 + 0.37, EPS, 28, "silk_white", (far["cap_x"], y))
    # unpopulated polymer footprints on the bracket-side bank (tinned pads only)
    pad = rect(1.7, 3.0, 0.007, "tin")
    pp = []
    for r in near["rows"]:
        y = y0 + r * pitch
        for xa, xb in ((22.0, 26.0), (52.5, 56.5)):
            pp += [(xa + 0.85, y, 0), (xb - 0.85, y, 0)]
            occ.mark(xa, y - 1.6, xb, y + 1.6)
    ov.tile(pad, pp)

    # ------------------------------------------------------------------ memory VRM + other inductors / caps
    zmi = g["z_mem_ind"]
    for (x, y) in g["mem_vrm_ind"]:
        inst("mind_%g" % zmi, lambda: C.inductor(7.0, 7.0, zmi, "R15", term_len=1.3), "ind_mem", x, y, 90)
        occ.mark_c(x, y, 8.4, 13.4)
        ov += K.silk_corners(x, y, 8.4, 13.4, 1.0, 0.1)
    for (x, y) in g["mem_vrm_stage"]:
        inst("mstage_%g" % zd, lambda: C.qfn(5.0, 5.0, zd, (7, 7), ("MP86956", "2431"), pitch=0.5), "stage_mem", x, y, 0)
        occ.mark_c(x, y, 5.6, 5.6)
        ov += K.silk_box(x, y, 5.5, 5.5, 0.1)
    names = list(V.get("other_inductors_bbox_x0_y0_x1_y1", {}).keys())
    for i, bb in enumerate(g["small_ind"]):
        x0, y0b, x1, y1 = bb
        w, h = x1 - x0, y1 - y0b
        lab = names[i].split("_")[-1] if i < len(names) and names[i].split("_")[-1][0] in "0123456789R" else "R22"
        if h > 1.6 * w:                               # two stacked input-filter chokes
            for k in (0, 1):
                yc = y0b + h * (0.25 + 0.5 * k)
                inst("gind_f_%g" % zi, lambda: C.inductor(w - 1.4, h / 2 - 1.1, zi, "R22", term_len=1.5), "ind_misc", (x0 + x1) / 2, yc, 0)
        else:
            hh = zmi if max(w, h) > 6.5 else 2.6
            inst("sind_%g_%g_%g_%s" % (w, h, hh, lab), lambda: C.inductor(w - 1.2, h - 0.3, hh, lab, term_len=1.0), "ind_misc", (x0 + x1) / 2, (y0b + y1) / 2, 0)
        occ.mark(x0 - 0.2, y0b - 0.2, x1 + 0.2, y1 + 0.2)
    for nm, bb in V.get("other_front_capacitors_approx_bbox_x0_y0_x1_y1", {}).items():
        if not isinstance(bb, list):
            continue
        x0, y0b, x1, y1 = bb
        w, h = x1 - x0, y1 - y0b
        code = "470" if "470" in nm else "330"
        rot = 0 if w >= h else 90
        ll, ww = max(w, h), min(w, h)
        inst("tant_%g_%g_%s" % (round(ll, 1), round(ww, 1), code), lambda: C.tant(ll - 0.9, ww - 0.2, 1.9, (code, "2R5")), "cap_chip", (x0 + x1) / 2, (y0b + y1) / 2, rot)
        occ.mark(x0, y0b, x1, y1)

    # ------------------------------------------------------------------ power header + small headers
    h = g["hpwr"]
    hw, hdp = h["x"][1] - h["x"][0], h["y"][1] - h["y"][0]
    P.obj("hpwr", proto("hpwr_%g_%g_%g" % (hw, hdp, g["z_hpwr"]), lambda: hpwr_header(hw, hdp, g["z_hpwr"])), ((h["x"][0] + h["x"][1]) / 2, h["y"][1], 0))
    occ.mark(h["x"][0] - 1.5, h["y"][0] - 0.5, h["x"][1] + 1.5, h["y"][1])
    ht, hb = g["headers_top"], g["headers_bot"]
    b = ht[0]
    P.obj("hdr_j10", proto("jst4_red", lambda: C.jst_header(4, 2.5, "plastic_red")), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 0))
    occ.mark(*b)
    b = ht[1]                                           # J8: not populated on the liquid card -> pads only
    ov.tile(rect(1.0, 3.2, 0.007, "tin"), [(b[0] + 2.2 + i * 2.5, (b[1] + b[3]) / 2 - 0.6, 0) for i in range(4)])
    ov += K.silk_box((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1], 0.1)
    occ.mark(*b)
    for i, (b, col) in enumerate(zip(hb, ("plastic_white", "plastic_black"))):
        P.obj("hdr_bot.%d" % i, proto("jst4_side_" + col, lambda: C.jst_header(4, 2.0, col, side_entry=True)), ((b[0] + b[2]) / 2, b[1] + 2.9, 0))
        occ.mark(*b)

    # ------------------------------------------------------------------ display outputs + bracket
    dc = g["display_cage"]
    xf = dc["x"][0]
    dd = dc["x"][1] - xf - 1.6
    zo = 0.6 + 3.0
    for i, y in enumerate(dc["dp_y"]):
        P.obj("dp.%d" % i, proto("dp_port_%g" % dd, lambda: K.dp_port(dd)), (xf, y, 0))
    P.obj("hdmi", proto("hdmi_port_%g" % (dd - 1.5), lambda: K.hdmi_port(dd - 1.5)), (xf, dc["hdmi_y"], 0))
    leads = box(1.4, 0.3, 0.2, (dc["x"][1] - 0.8, 0, 0), 0.0, "tin", base=True)
    ov.tile(leads, [(0, y - 4.75 + k * 0.5, 0) for y in list(dc["dp_y"]) + [dc["hdmi_y"]] for k in range(20)])
    occ.mark(dc["x"][0], dc["y"][0], dc["x"][1] + 0.3, dc["y"][1])
    if bracket:
        opens = [("dp", y) for y in dc["dp_y"]] + [("hdmi", dc["hdmi_y"])]
        P.obj("bracket", proto("io_bracket_%g" % xf, lambda: io_bracket(opens, zo + 0.02, x_face=xf)))
        for nm in ("MEC12", "MEC5"):
            if nm in mec:
                mx, my = mec[nm]
                P.obj("bracket_ear_" + nm, proto("br_ear", lambda: box2(xf - 0.2, -4.5, -t - 1.0, 4.5, 4.5, -t - 0.1, 0.2, "paint_satin_black").__iadd__(
                    C.screw_head(5.0, 1.4, "phillips", "screw_black").rot("x", 180).move(0, 0, -t - 1.0))), (mx, my, 0))

    # ------------------------------------------------------------------ ICs + shunts
    def place(key, fn, nm, x, y, w, hh, rot=0.0, lab=None):
        ww, h2 = (w, hh) if rot % 180 == 0 else (hh, w)
        if not occ.take(x, y, ww, h2, 0.25):
            return None
        o = inst(key, fn, nm, x, y, rot)
        if lab:
            ov.__iadd__(K.silk_label(lab, x - ww / 2 + 0.3, y + h2 / 2 + 0.7, 0, 0.6, align="l"))
        return o

    place("pwm7", lambda: C.qfn(7.0, 7.0, 0.9, (14, 14), ("MP2891", "GQKT", "2431"), pitch=0.4), "ic", 226.0, 92.0, 7.6, 7.6, lab="U1")
    place("pwm5", lambda: C.qfn(5.0, 5.0, 0.9, (10, 10), ("UP9512", "R4K"), pitch=0.4), "ic", 68.0, 110.0, 5.6, 5.6, lab="U2")
    place("mcu", lambda: C.qfn(6.0, 6.0, 0.85, (12, 12), ("ITE 8915", "FN-BX", "2431"), pitch=0.4), "ic", 232.0, 40.0, 6.6, 6.6, lab="U9")
    place("flash", lambda: C.gull_ic(4, 1.27, 5.2, 5.2, 1.45, 8.0, ("25Q80", "EWSIG")), "ic", 20.5, 120.0, 8.2, 5.6, 0, lab="BIOS1")
    for i in range(3):
        place("shunt", shunt, "shunt", 227.5 + i * 4.6, 108.5, 3.6, 7.4, 90)
    place("fuse", lambda: C.chip(3.2, 1.6, 0.8, body="alumina"), "fuse", 246.0, 99.5, 4.4, 2.0)
    for (x, y) in ((240.0, 70.0), (240.0, 62.0), (236.0, 22.0), (60.0, 118.5), (176.0, 118.0), (172.0, 62.0)):
        place("soic8g", lambda: C.gull_ic(4, 1.27, 3.9, 4.9, 1.45, 6.0, ("INA3221", "431")), "ic", x, y, 6.4, 5.2)
    for (x, y) in ((244.0, 46.0), (60.0, 60.0), (176.0, 40.0), (224.0, 20.0), (60.0, 12.0), (10.0, 100.0)):
        place("qfn4g", lambda: C.qfn(4.0, 4.0, 0.85, (6, 6), ("NCP45491", "2431"), pitch=0.5), "ic", x, y, 4.6, 4.6)

    # ------------------------------------------------------------------ merged SMD clutter (top)
    protos = K.smd_protos()
    cl = K.Clutter(protos)

    def group(x, y, kind, nx, ny, rot=0, o=occ, c=cl, ysign=1):
        fw, fh = protos[kind][1], protos[kind][2]
        w, hh = (fw, fh) if rot == 0 else (fh, fw)
        n = 0
        for i in range(nx):
            for j in range(ny):
                qx, qy = x + i * (w + 0.25), y + j * (hh + 0.25)
                if o.take(qx, qy, w, hh, 0.08):
                    c.put(kind, qx, ysign * qy, ysign * rot)
                    n += 1
        return n

    # ring of decoupling caps hugging the package
    for xx in np.arange(cx - pw / 2 + 1.0, cx + pw / 2 - 0.5, 1.85):
        group(xx, cy + pl / 2 + 1.25, "c0402", 1, 1, 90)
        group(xx, cy - pl / 2 - 1.25, "c0402", 1, 1, 90)
    for yy in np.arange(cy - pl / 2 + 1.0, cy + pl / 2 - 0.5, 1.85):
        group(cx - pw / 2 - 1.25, yy, "c0402", 1, 1, 0)
        group(cx + pw / 2 + 1.25, yy, "c0402", 1, 1, 0)
    # per-phase passives: bootstrap / gate parts by each stage, MLCC columns on the GPU side of the inductors
    for rows, bk, sgn in ((near["rows"], near, 1), (far["rows"], far, -1)):
        sxc = (bk["drmos_x"][0] + bk["drmos_x"][1]) / 2
        gx = bk["ind_x"][1] + 6.6 if sgn > 0 else bk["ind_x"][0] - 6.6
        ox = bk["drmos_x"][0] - 1.4 if sgn > 0 else bk["drmos_x"][1] + 1.4
        for r in rows:
            y = y0 + r * pitch
            group(sxc - 2.2, y + 3.45, "c0402", 3, 1, 0)
            group(sxc - 2.2, y - 3.45, "r0402", 2, 1, 0)
            group(ox, y - 1.2, "c0603", 1, 2, 0)
            group(gx, y - 2.6, "c0805", 2, 3, 0)
    # memory decoupling: short rows beside every memory chip
    for nm, (mx, my, mw, mh) in g["mem"].items():
        if mw >= mh:
            sg = -1 if mx < cx else 1
            group(mx + sg * (mw / 2 + 1.1) - (0.9 if sg < 0 else 0.0), my - 4.6, "c0402", 1, 8, 0)
        else:
            group(mx - 4.6, my + mh / 2 + 1.0, "c0402", 8, 1, 90)
    # random organised groups
    kinds = ["c0402"] * 5 + ["r0402"] * 4 + ["c0603"] * 3 + ["r0603"] * 2 + ["c0402g"] * 2 + ["c0805"] * 2
    tries = 0
    while cl.count() < 1500 and tries < 12000:
        tries += 1
        x, y = round(rnd.uniform(1.5, Lx - 1.5) * 4) / 4, round(rnd.uniform(1.5, Hy - 1.5) * 4) / 4
        if x > g["hpwr"]["recess_x"] - 1.0 and y > g["hpwr"]["recess_y"] - 1.5:
            continue
        hz = rnd.random() < 0.5
        k = rnd.randint(2, 6)
        group(x, y, rnd.choice(kinds), k if hz else 1, 1 if hz else k, 90 if hz else 0)
    for i in range(40):
        x, y = rnd.uniform(3, Lx - 3), rnd.uniform(3, Hy - 3)
        if x > g["hpwr"]["recess_x"] - 3.0 and y > g["hpwr"]["recess_y"] - 3.0:
            continue
        if occ.take(x, y, 3.0, 3.2, 0.2):
            inst("sot23_3", lambda: C.sot23(3), "q", x, y, rnd.choice((0, 90, 180, 270)))
    tp = []
    for i in range(40):
        x, y = rnd.uniform(3, Lx - 3), rnd.uniform(3, Hy - 3)
        for j in range(rnd.randint(1, 5)):
            if occ.take(x + j * 1.5, y, 1.0, 1.0, 0.15) and not (x + j * 1.5 > g["hpwr"]["recess_x"] - 1 and y > g["hpwr"]["recess_y"] - 1):
                tp.append((x + j * 1.5, y, 0))
    ov.tile(disc(0.45, 0.008, 12, "enig_gold"), tp)
    ov += K.silk_label("GFX-4090 PCB  REV 2.2", 196.0, 3.2, 0, 1.3)
    ov += K.silk_label("94V-0  2431  E123456", 196.0, 1.2, 0, 0.7)
    P.obj("smd_top", cl.soup().mesh(P.name + "_smd_top"))
    P.obj("silk", ov.mesh(P.name + "_silk", smooth=None))

    # ------------------------------------------------------------------ back side
    bcl = K.Clutter(protos)
    # dense capacitor array behind the GPU
    for i in range(30):
        for j in range(27):
            x, y = cx - 21.8 + i * 1.5, cy - 21.5 + j * 1.65
            if ((i // 5) + (j // 4)) % 3 == 2 and (i % 5 == 4 or j % 4 == 3):
                continue
            if (i % 5 == 4) or bocc.g[0, 0]:
                continue
            bcl.put("c0402" if (i + j) % 7 else "c0402g", x, -y, 90)
    bocc.mark_c(cx, cy, 46.5, 46.5)
    for i in range(2):
        for j in range(6):
            bcl.put("c0805", cx - 24.8 + i * 49.6, -(cy - 14.0 + j * 5.6), 90)
    # polymer capacitor fields under both VRM banks
    F = V["back_side_capacitor_fields_approx"]
    f1, f2 = F["under_bracket_side_bank"], F["under_far_side_bank"]
    tant_p = lambda: C.tant(7.3, 4.3, 1.9, ("330", "2R5"))
    ncol = int((f1["x"][1] - f1["x"][0] + 1.0) / 9.0)
    for r in near["rows"]:
        y = y0 + r * pitch
        for k in range(ncol):
            x = f1["x"][0] + 4.3 + k * (f1["x"][1] - f1["x"][0] - 8.6) / max(ncol - 1, 1)
            if bocc.take(x, y, 8.3, 4.6, 0.1):
                inst("tant_back", tant_p, "bcap", x, y, 0, back=True)
    for r in far["rows"]:
        y = y0 + r * pitch
        if f2["y"][0] <= y <= f2["y"][1] + 6 and bocc.take((f2["x"][0] + f2["x"][1]) / 2, y, 8.3, 4.6, 0.1):
            inst("tant_back", tant_p, "bcap", (f2["x"][0] + f2["x"][1]) / 2, y, 0, back=True)
        group(f2["x"][1] + 2.5, y - 2.0, "c0805", 2, 3, 0, bocc, bcl, -1)
        group(f2["x"][0] - 6.5, y - 2.0, "c0805", 2, 3, 0, bocc, bcl, -1)
    # BIOS switch, LEDs, unpopulated footprints
    sw = G["bios_switch"]["body_bbox"]
    P.obj("bios_switch", proto("slide_sw", slide_switch), ((sw["x"][0] + sw["x"][1]) / 2, (sw["y"][0] + sw["y"][1]) / 2, -t), (180, 0, 180))
    bocc.mark(sw["x"][0] - 1, sw["y"][0] - 1, sw["x"][1] + 1, Hy)
    bov += K.silk_label("SILENT  GAMING", (sw["x"][0] + sw["x"][1]) / 2, -(sw["y"][0] - 1.5), 0, 0.7)
    for k, v in G["headers"]["back"].items():
        bb = v["bbox"]
        if k.startswith("LED"):
            inst("led_back", lambda: C.led(3.2, 1.6, 0.8), "bled", (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, 90, back=True)
        else:
            bov.tile(rect(1.2, 3.0, 0.007, "tin"), [((bb[0] + bb[2]) / 2 - 2.0 + 4.0 * (i % 2), -(bb[1] + 2.0 + (i // 2) * 2.4), 0) for i in range(8)])
            bov += K.silk_box((bb[0] + bb[2]) / 2, -(bb[1] + bb[3]) / 2, bb[2] - bb[0], bb[3] - bb[1], 0.1, pin1=False)
        bocc.mark(*bb)
    # through-hole joints: power header, polymer cans, display connector legs
    jp = [(x, -y, 0) for x, y in hp_pins] + [(x, -y, 0) for x, y in hp["hold_down_slots_approx"]]
    jp += [(far["cap_x"] + sg * 1.25, -y, 0) for y in far["cap_y"] for sg in (-1, 1)]
    sl = G["display_outputs"]["shell_tab_slots_xy"]
    for k, v in sl.items():
        if k != "x":
            jp += [(x, -y, 0) for x in sl["x"] for y in v]
    back = C.tht_joints(jp, 0.7, 0.8)
    for x, y, _ in jp:
        bocc.mark_disc(x, -y, 1.0)
    tries = 0
    while bcl.count() < 2300 and tries < 12000:
        tries += 1
        x, y = round(rnd.uniform(1.5, Lx - 1.5) * 4) / 4, round(rnd.uniform(1.5, Hy - 1.5) * 4) / 4
        if x > g["hpwr"]["recess_x"] - 1.0 and y > g["hpwr"]["recess_y"] - 1.5:
            continue
        hz = rnd.random() < 0.5
        k = rnd.randint(2, 7)
        group(x, y, rnd.choice(kinds), k if hz else 1, 1 if hz else k, 90 if hz else 0, bocc, bcl, -1)
    bov += K.silk_label("GFX-4090 PCB  REV 2.2", 196.0, -3.0, 0, 1.4)
    back += bcl.soup()
    back += bov
    back.rot("x", 180).move(0, 0, -t)
    P.obj("back", back.mesh(P.name + "_back"))
    return P.root
