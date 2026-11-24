"""pcp_board - generic mini-ITX AM5 motherboard with all stock heatsinks removed.

BOARD frame of design/params.py: origin = corner shared by the rear-I/O edge and the
PCIe-slot edge, +x along the PCIe edge away from the I/O, +y to the top edge, +z out of
the component side, PCB top surface z = 0. Every position is read from the layout dict.
"""
import math
import random

import numpy as np

import pcp_comps as C
import pcp_conn as K
from pcp_geom import (Soup, Part, proto, box, box2, cyl, lathe, prism, plate, rect, disc, annulus, strip,
                      text, rrect, circle, fillet_poly, loft, tube, fillet_path, tube_ring, poly_face, rbox,
                      TAU)

EPS = C.EPS
PCIE_CAV0 = 1.9          # slot body start -> start of the card cavity (mm)
PCIE_TONGUE = 84.3       # x16 edge-finger tongue length


# ----------------------------------------------------------------------------
# keep-outs
# ----------------------------------------------------------------------------
def keepouts(L):
    """Rectangles (x0, y0, x1, y1) in the BOARD frame where nothing taller than 1.0 mm may sit."""
    B, MB, MS = L["board"], L["monoblock"], L["m2_sink"]
    sx, sy = B["socket"]
    hx, hy = B["am5_holes"][0] / 2, B["am5_holes"][1] / 2
    ex = max(B["sam"][0] / 2 + 4.0, hx + 5.0)
    ey = max(B["sam"][1] / 2 + 4.0, hy + 5.0)
    return {
        "socket": (sx - ex, sy - ey, sx + ex, sy + ey),
        "arm_left": (MB["arm_left"]["x"][0], MB["arm_left"]["y"][0], MB["arm_left"]["x"][1], MB["arm_left"]["y"][1]),
        "arm_top": (MB["arm_top"]["x"][0], MB["arm_top"]["y"][0], MB["arm_top"]["x"][1], MB["arm_top"]["y"][1]),
        "m2_sink": (MS["x"][0], MS["y"][0], MS["x"][1], MS["y"][1]),
    }


# ----------------------------------------------------------------------------
# AM5 socket, CPU, backplate
# ----------------------------------------------------------------------------
def _notched_square(S, nc, nw, nd):
    h = S / 2
    side = [(-h, -h), (-nc - nw / 2, -h), (-nc - nw / 2, -h + nd), (-nc + nw / 2, -h + nd), (-nc + nw / 2, -h),
            (nc - nw / 2, -h), (nc - nw / 2, -h + nd), (nc + nw / 2, -h + nd), (nc + nw / 2, -h)]
    pts = []
    for k in range(4):
        c, s = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
        pts += [(x * c - y * s, x * s + y * c) for x, y in side]
    return np.array(pts)


def _ihs_poly(S, bottom, r_out, n=4):
    """Octopus outline: notch floors stay at `bottom` from the centre whatever S is."""
    P = _notched_square(S, 8.4, 6.4, S / 2 - bottom)
    radii = [r_out, 0.45, 0.9, 0.9, 0.45, 0.45, 0.9, 0.9, 0.45] * 4
    return fillet_poly(P, radii, n)


def am5_cpu(zi, ihs=40.0):
    """Installed CPU (socket-centred): substrate + octopus IHS, top at z = zi."""
    z_sub0, z_sub1, z_fl = zi - 3.8, zi - 2.7, zi - 1.5
    s = box(ihs, ihs, z_sub1 - z_sub0, (0, 0, z_sub0), 0.08, "pcb_darkgreen", base=True)
    t = 2.2
    c = ihs / 2 - 0.5
    s += poly_face([(-c, -c), (-c + t, -c), (-c, -c + t)], z_sub1 + 0.006, "enig_gold")
    # capacitors visible in the IHS notches
    cap = C.chip(1.0, 0.5, 0.5, pad=False)
    pos, rots = [], []
    for k in range(4):
        a = k * math.pi / 2
        for nx in (-8.4, 8.4):
            for i in range(4):
                for j in range(2):
                    x, y = nx - 1.9 + i * 1.25, -ihs / 2 + 1.5 + j * 1.5
                    pos.append((x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a), z_sub1))
                    rots.append(90 + k * 90)
    s.tile(cap, pos, rots)
    S_f, S_t, bottom = ihs - 1.6, ihs - 7.0, ihs / 2 - 6.6
    s += prism(_ihs_poly(S_f, bottom, 2.6), z_sub1, z_fl, "nickel_ihs", bevel=0.2)
    s += loft([(_ihs_poly(S_t + 2.2, bottom - 0.5, 2.4), z_fl - 0.05), (_ihs_poly(S_t + 0.5, bottom - 0.1, 2.2), zi - 0.3),
               (_ihs_poly(S_t, bottom, 2.0), zi)], "nickel_ihs", cap0=False)
    z = zi + 0.004
    s += C.mark(["8C16T 4.2GHZ 120W", "100-000000910", "BG 2431PGY", "9MX5082S40017"], S_t * 0.72, 9.0, z, "ihs_mark",
                c=(0, 3.2), hmax=1.25)
    s += C.mark(["DIFFUSED 2026 - ASSEMBLED 2026"], S_t * 0.72, 1.2, z, "ihs_mark", c=(0, -5.0), hmax=0.8)
    tt = 1.7
    cc = S_t / 2 - 2.1
    s += poly_face([(-cc, -cc), (-cc + tt, -cc), (-cc, -cc + tt)], z, "ihs_mark")
    return s


def am5_socket(samx, samy, zi):
    """Socket housing + retention frame (SAM) + lever, socket-centred."""
    z_sub0, z_fl = zi - 3.8, zi - 1.5
    s = Soup()
    s += plate(rrect(samx, samy, 3.0, 5), [rrect(47.6, 47.6, 1.0, 3)], 0.02, 0.22, 0.0, "plastic_gloss_black")
    s += plate(rrect(samx - 1.2, samy - 1.2, 2.6, 5), [rrect(48.6, 48.6, 1.2, 3)], 0.2, 1.3, 0.16, "steel_sheet")
    for sx in (-1, 1):
        s += box(1.0, samy - 24.0, 3.4, (sx * (samx / 2 - 1.15), 0, 1.2), 0.25, "steel_sheet", base=True)
    # LGA housing: wall around the package + contact floor
    s += plate(rrect(46.8, 46.8, 1.3, 3), [rrect(40.6, 40.6, 0.5, 2)], 0.05, z_sub0 + 0.75, 0.2, "plastic_black")
    s += box(40.8, 40.8, z_sub0 - 0.35, (0, 0, 0.05), 0.0, "plastic_black", base=True)
    for sx in (-1, 1):
        for sy in (-1, 1):
            s += C.screw_head(5.6, 1.9, "torx", "steel_bright").move(sx * (samx / 2 - 5.5), sy * (samy / 2 - 4.6), 1.3)
            s += cyl(3.6, 1.25, 1.75, 20, "steel_sheet", bevel=0.2).move(sx * (samx / 2 - 5.5), sy * (samy / 2 - 4.6), 0)
    # load plate with IHS window, side skirts, hinge and front tongue
    lp_y0, lp_y1 = -samy / 2 + 7.0, samy / 2 - 7.5
    lp_w = samx - 6.0
    outline = rrect(lp_w, lp_y1 - lp_y0, 3.0, 5, (0, (lp_y0 + lp_y1) / 2))
    s += plate(outline, [rrect(34.0, 40.6, 1.6, 4)], z_fl, z_fl + 0.8, 0.22, "steel_sheet")
    for sx in (-1, 1):
        s += box(0.8, lp_y1 - lp_y0 - 16.0, 2.6, (sx * (lp_w / 2 - 0.4), 0, z_fl - 2.2), 0.2, "steel_sheet", base=True)
    s += cyl(1.7, -15.5, 15.5, 16, "steel_bright", bevel=0.3, axis="x").move(0, lp_y1 + 0.6, z_fl - 0.3)
    for sx in (-1, 1):
        s += box(7.0, 4.4, 0.8, (sx * 9.0, lp_y1 - 0.4, z_fl + 0.0), 0.2, "steel_sheet", base=True)
        s += tube_ring(2.5, 1.7, -3.5, 3.5, 16, "steel_sheet", bevel=0.2, axis="x").move(sx * 9.0, lp_y1 + 0.6, z_fl - 0.3)
        s += box(5.0, 3.0, z_fl - 1.5, (sx * 9.0, lp_y1 + 2.2, 1.2), 0.2, "steel_sheet", base=True)
    s += box(13.0, 5.5, 0.8, (0, lp_y0 - 2.0, z_fl - 0.9), 0.2, "steel_sheet", base=True)
    s += box(13.0, 0.8, 1.7, (0, lp_y0 + 0.35, z_fl - 0.9), 0.2, "steel_sheet", base=True)
    z = z_fl + 0.8 + 0.004
    s += C.mark(["LGA1718"], 12.0, 1.5, z, "ihs_mark", c=(0, lp_y1 - 3.2))
    s += C.mark(["< OPEN"], 9.0, 1.2, z, "ihs_mark", c=(lp_w / 2 - 3.6, -8.0), rot=90)
    s += C.mark(["PRESS"], 9.0, 1.2, z, "ihs_mark", c=(-lp_w / 2 + 3.6, 8.0), rot=90)
    # lever: axle with cam under the tongue, arm along +x side, handle
    ya = lp_y0 - 3.4
    xa = samx / 2 - 3.3
    pts = [(-samx / 2 + 7.0, ya, 2.7), (-7.5, ya, 2.7), (-6.0, ya + 0.9, z_fl - 1.5), (6.0, ya + 0.9, z_fl - 1.5),
           (7.5, ya, 2.7), (xa, ya, 2.7), (xa, ya + 6.0, 3.3), (xa, lp_y1 - 6.0, z_fl - 0.9),
           (xa, lp_y1 - 1.5, z_fl - 0.9), (xa + 1.9, lp_y1 + 1.8, z_fl - 0.6), (xa + 1.9, lp_y1 + 5.0, z_fl - 0.6)]
    s += tube(fillet_path(pts, 1.6, 5), 1.25, 10, "steel_bright")
    s += box(3.0, 5.0, z_fl - 1.0, (xa + 0.9, lp_y1 - 14.0, 1.2), 0.25, "steel_sheet", base=True)
    s += box(4.6, 5.0, 0.8, (xa - 0.2, lp_y1 - 14.0, z_fl + 0.45), 0.25, "steel_sheet", base=True)
    for sx in (-1, 1):
        s += box(3.0, 6.0, 3.2, (sx * (samx / 2 - 8.0), ya, 1.2), 0.25, "steel_sheet", base=True)
    return s


def am5_post(z_top=4.0, t=1.6):
    """Backplate threaded post poking through the board (origin on the PCB top, hole centre)."""
    prof = [(3.0, -t - 0.2), (3.0, z_top - 0.4), (2.6, z_top), (1.8, z_top), (1.5, z_top - 0.3)]
    z = z_top - 0.3
    while z > 0.9:
        prof += [(1.28, z - 0.2), (1.5, z - 0.4)]
        z -= 0.4
    prof += [(1.5, 0.6), (0, 0.6)]
    return lathe(prof, 20, "steel_bright")


def am5_backplate(hx, hy, t):
    """Steel backplate with insulator, socket-centred, on the back side of the PCB."""
    w = hx + 7.5
    pts = [(-w, -hy - 7), (w, -hy - 7), (w, -hy + 9), (w - 6, -hy + 15), (w - 6, hy - 15), (w, hy - 9), (w, hy + 7),
           (-w, hy + 7), (-w, hy - 9), (-w + 6, hy - 15), (-w + 6, -hy + 15), (-w, -hy + 9)]
    out = fillet_poly(pts, [5, 5, 3, 6, 6, 3, 5, 5, 3, 6, 6, 3], 5)
    holes = [rrect(34.0, 34.0, 3.0, 4)] + [circle(1.9, 16, (sx * 20.0, sy * 31.0)) for sx in (-1, 1) for sy in (-1, 1)]
    s = plate(out, [rrect(35.0, 35.0, 3.0, 4)], -t - 0.3, -t - 0.02, 0.0, "plastic_gloss_black")
    s += plate(out, holes, -t - 2.5, -t - 0.3, 0.3, "screw_black")
    for sx in (-1, 1):
        for sy in (-1, 1):
            s += cyl(4.6, -t - 3.1, -t - 2.4, 20, "screw_black", bevel=0.3).move(sx * hx, sy * hy, 0)
    return s


# ----------------------------------------------------------------------------
# memory, PCIe, M.2
# ----------------------------------------------------------------------------
def dimm_slot(Ld, w, h):
    """DDR5 slot centred on the origin, long axis along y, single latch at +y."""
    s = Soup()
    Lb = Ld - 9.0
    s += plate(rrect(w, Lb, 0.5, 2), [rrect(1.55, Lb - 5.0, 0.2, 1)], 1.0, h, 0.15, "plastic_black")
    s += box(w, Lb, 1.1, (0, 0, 0), 0.1, "plastic_black", base=True)
    yf = -(Ld / 2 - 2.6)
    for sx in (-1, 1):
        s += box(2.5, 5.2, 13.0, (sx * (w / 2 - 1.25), yf, 0), 0.3, "plastic_black", base=True)
    s += box(w, 1.5, 13.0, (0, -(Ld / 2 - 0.75), 0), 0.3, "plastic_black", base=True)
    yl = Ld / 2 - 4.5
    for sx in (-1, 1):
        s += box(1.6, 5.6, 9.5, (sx * (w / 2 - 0.8), yl + 1.5, 0), 0.3, "plastic_black", base=True)
    prof = [(0.0, 3.0), (4.4, 3.0), (5.6, 5.5), (5.6, 18.5), (3.8, 20.8), (0.9, 20.8), (0.9, 15.5), (2.3, 14.2),
            (2.3, 6.2), (0.0, 6.2)]
    lat = prism(prof, -1.7, 1.7, "slot_grey", bevel=0.25, bevel_bottom=True)
    lat.V = lat.V[:, [2, 0, 1]]
    s += lat.move(0, yl - 1.0, 0)
    n = int((Lb - 6) / 0.85)
    ys = (np.arange(n) - (n - 1) / 2) * 0.85
    tail = box(1.0, 0.36, 0.22, (0, 0, 0), 0.0, "tin", base=True)
    s.tile(tail, [(sx * (w / 2 + 0.3), y, 0) for sx in (-1, 1) for y in ys])
    return s


def dimm_module(z_top, z_slot=6.0, Lm=133.35):
    """Plain low-profile black heat-spreader DDR5 module, seated (slot-centred frame)."""
    zb = 2.6
    s = box(1.27, Lm, z_top - 0.15 - zb, (0, 0, zb), 0.05, "pcb_black", base=True)
    z0 = z_slot + 2.2
    h = Lm / 2
    prof = [(-h + 0.4, z0), (h - 0.4, z0), (h - 0.4, z_top - 4.0), (h - 4.5, z_top), (-h + 4.5, z_top), (-h + 0.4, z_top - 4.0)]
    inner = [(-h + 5.0, z0 + 3.0), (h - 5.0, z0 + 3.0), (h - 5.0, z_top - 6.5), (h - 7.5, z_top - 4.0), (-h + 7.5, z_top - 4.0),
             (-h + 5.0, z_top - 6.5)]
    for sx in (-1, 1):
        a = prism(prof, 1.45, 2.55, "anodized_black", bevel=0.3)
        b = prism(inner, 2.5, 2.9, "anodized_black", bevel=0.3)
        for q in (a, b):
            q.V = q.V[:, [2, 0, 1]]
            if sx < 0:
                q.mirror(0)
            s += q
    s += box(3.4, Lm - 10.0, 1.1, (0, 0, z_top - 1.25), 0.3, "anodized_black", base=True)
    s += box(2.9, Lm - 6.0, z_top - 4.0 - z0, (0, 0, z0 + 0.6), 0.0, "rubber_grey", base=True)
    lab = Soup()
    lab += rect(46.0, 9.0, 0.0, "label_silver")
    lab += C.mark(["DDR5 UDIMM 16GB 6000 CL30 1.35V", "PC5-48000  SN 2431-0077-1A5C"], 42.0, 6.0, 0.006, "ink_black")
    lab.rot("x", 90).rot("z", 90).move(2.91, -28.0, z0 + 9.5)
    s += lab
    return s


def pcie_slot(Ls, zs, wy=7.5):
    """Reinforced PCIe x16 slot. x = 0 at the slot start, centred on y = 0."""
    Lb = Ls - 4.6
    c0, key = PCIE_CAV0, PCIE_CAV0 + 0.15 + 12.15
    c1 = c0 + PCIE_TONGUE + 0.3
    out = rrect(Lb, wy - 0.7, 0.5, 2, (Lb / 2, 0))
    g1 = rrect(key - 0.95 - c0, 1.95, 0.15, 1, ((c0 + key - 0.95) / 2, 0))
    g2 = rrect(c1 - key - 0.95, 1.95, 0.15, 1, ((c1 + key + 0.95) / 2, 0))
    s = plate(out, [g1, g2], 1.0, zs - 0.32, 0.12, "plastic_black")
    s += box(Lb, wy - 0.7, 1.1, (Lb / 2, 0, 0), 0.1, "plastic_black", base=True)
    w1 = rrect(key - 0.6 - c0 + 1.4, 3.5, 0.3, 2, ((c0 + key) / 2 - 0.4, 0))
    w2 = rrect(c1 - key + 0.2, 3.5, 0.3, 2, ((c1 + key) / 2 + 0.5, 0))
    s += plate(rrect(Lb + 0.5, wy, 0.6, 2, (Lb / 2, 0)), [w1, w2], zs - 0.34, zs, 0.1, "steel_sheet")
    hs = zs - 1.3
    wins = [rrect(7.0, hs - 3.6, 0.6, 2, (9.0 + i * 10.2, hs / 2 + 0.2)) for i in range(int((Lb - 12) / 10.2) + 1)]
    for sy in (-1, 1):
        side = plate(rrect(Lb + 0.5, hs, 0.3, 1, (Lb / 2, hs / 2)), wins, 0.0, 0.32, 0.08, "steel_sheet")
        side.rot("x", 90).move(0, sy * (wy / 2 - 0.16) + 0.16, 1.2)
        s += side
        for i in range(5):
            s += box(2.6, 0.32, 1.4, (6.0 + i * (Lb - 12.0) / 4, sy * (wy / 2 - 0.16), 0), 0.0, "steel_sheet", base=True)
            s += box(3.4, 1.3, 0.25, (6.0 + i * (Lb - 12.0) / 4, sy * (wy / 2 + 0.5), 0), 0.0, "solder", base=True)
    pin = box(0.55, 0.2, zs - 4.2, (0, 0, 2.6), 0.0, "gold_pin", base=True)
    xs = [c0 + 0.8 + i for i in range(11)] + [c0 + 0.15 + 13.65 + i for i in range(71)]
    s.tile(pin, [(x, sy * 0.86, 0) for x in xs for sy in (-1, 1)])
    prof = [(0.0, 1.0), (4.4, 1.0), (4.4, zs - 2.5), (3.4, zs + 1.4), (1.5, zs + 1.4), (1.5, zs - 4.5), (0.0, zs - 4.5)]
    lat = prism(prof, -2.6, 2.6, "slot_grey", bevel=0.3).rot("x", 90).move(Lb, 0, 0)
    s += lat
    return s


def m2_connector(zb=1.45):
    """M.2 M-key socket. x = 0 is the SSD's connector-end edge (SSD towards +x), y centred."""
    s = box2(-5.4, -11.0, 0.0, 0.2, 11.0, 3.2, 0.2, "plastic_black")
    s += box2(0.2, -11.0, 0.0, 3.3, 11.0, zb - 0.08, 0.0, "plastic_black")
    s += box2(0.2, -10.2, zb + 0.9, 1.9, 10.2, 3.2, 0.12, "plastic_black")
    for sy in (-1, 1):
        s += box2(0.2, sy * 11.0 - 0.6 * (sy > 0), 0.0, 3.3, sy * 11.0 + 0.6 * (sy < 0), 3.2, 0.1, "plastic_black")
        s += box(2.2, 1.6, 0.3, (-2.5, sy * 11.6, 0), 0.0, "steel_sheet", base=True)
    tail = box(0.9, 0.24, 0.16, (0, 0, 0), 0.0, "tin", base=True)
    s.tile(tail, [(-5.75, -8.25 + i * 0.5, 0) for i in range(34)])
    return s


def m2_ssd(Ls, W, zb, z_top):
    """Bare M.2 2280 SSD. x = 0 connector end, y centred, PCB bottom at zb, tallest part at z_top."""
    zt = zb + 0.8
    a = np.linspace(math.pi / 2, 3 * math.pi / 2, 9)
    notch = [(Ls + 1.75 * math.cos(t), 1.75 * math.sin(t)) for t in a]
    out = [(0, -W / 2), (Ls, -W / 2)] + notch[::-1] + [(Ls, W / 2), (0, W / 2)]
    s = K.pcb_plate(np.array(out), [], 0.8, "pcb_black_gpu").move(0, 0, zt)
    ring = [(Ls + 3.0 * math.cos(t), 3.0 * math.sin(t)) for t in a]
    V = [(p[0], p[1], zt + 0.006) for p in notch + ring]
    n = len(notch)
    s.add(V, [(i, n + i, n + i + 1, i + 1) for i in range(n - 1)], "enig_gold")
    h = z_top - zt
    def put(q, x, y, r=0):
        return q.rotz(r).move(x, y, zt)
    s += put(C.bga(11.0, 12.0, h, ("PS5021-E21T", "2431 TWN", "B7K204.1"), balls=False, sub_t=0.25, ball_h=0.12), 13.5, 0.5)
    s += put(C.bga(7.5, 10.5, h - 0.25, ("D9XPG", "4KA77"), balls=False, sub_t=0.2, ball_h=0.1), 26.5, 0.0)
    for x in (43.5, 61.5):
        s += put(C.bga(12.0, 18.0, h, ("TH58LKT2T", "25BA8H", "2431 JPN"), balls=False, sub_t=0.25, ball_h=0.12), x, 0.0)
    s += put(C.qfn(3.0, 3.0, 0.8, (5, 5), ("P8A", "431")), 72.5, 5.0)
    s += put(C.inductor(2.5, 2.0, 1.0, None, dot=False), 72.5, -1.0)
    s += put(C.inductor(2.5, 2.0, 1.0, None, dot=False), 72.5, -5.5)
    c4, c6, r4 = C.chip(1.0, 0.5, 0.5), C.chip(1.6, 0.8, 0.8), C.chip(1.0, 0.5, 0.35, body="res_black", tl=0.2)
    rnd = random.Random(7)
    pos4, rot4, pos6, rot6 = [], [], [], []
    for x in (5.2, 6.4, 20.4, 21.6, 33.0, 34.2, 35.4, 52.3, 53.5, 69.0, 70.2, 76.0):
        for y in np.arange(-9.5, 9.6, 1.9):
            if rnd.random() < 0.55 and not (x > 74 and abs(y) < 4.5):
                pos4.append((x, y, zt))
                rot4.append(90)
    for y in (-9.6, 9.6):
        for x in np.arange(8.5, 31.0, 2.6):
            if rnd.random() < 0.6:
                pos6.append((x, y, zt))
                rot6.append(0)
    s.tile(c4, pos4, rot4)
    s.tile(c6, pos6, rot6)
    s.tile(r4, [(66.0 + (i % 2) * 1.4, -8.0 + i * 1.1, zt) for i in range(6)], [0] * 6)
    s += C.mark(["M2-2280 S3-M  94V-0  2431"], 30.0, 0.9, zt + EPS, "silk_white", c=(52.5, -10.2))
    s += C.mark(["U1"], 3, 0.8, zt + EPS, "silk_white", c=(13.5, 7.6))
    return s


def m2_standoff(zb, hex_r=3.0):
    s = prism(circle(hex_r, 6), 0.0, zb, "brass", bevel=0.2)
    s += C.screw_head(5.6, 0.95, "phillips", "screw_zinc").move(0, 0, zb + 0.8)
    return s


def coin_cell_vertical(d=20.0):
    """CR2032 standing in a vertical holder (long axis along x)."""
    s = box(d + 3.0, 5.6, 6.5, (0, 0, 0), 0.4, "plastic_black", base=True)
    s += box(3.0, 5.6, 14.0, (-(d / 2 + 0.2), 0, 0), 0.4, "plastic_black", base=True)
    cell = lathe([(0, -1.6), (d / 2 - 0.5, -1.6), (d / 2, -1.2), (d / 2, 1.1), (d / 2 - 0.8, 1.6), (0, 1.6)], 48, "steel_bright", axis="y")
    s += cell.move(0, 0, d / 2 + 2.2)
    s += C.mark(["CR2032", "3V"], d * 0.55, d * 0.3, 0, "ihs_mark", hmax=2.0).rot("x", 90).move(0, -1.61, d / 2 + 4.5)
    s += C.mark(["+"], 4, 4, 0, "ihs_mark", hmax=3.5).rot("x", 90).move(0, -1.61, d / 2 - 2.5)
    s += box(1.2, 0.5, 9.0, (d / 2 - 2.0, -1.9, 6.0), 0.1, "steel_bright", base=True)
    return s


def smd_can(d=5.0, h=5.8, code=("270", "16V")):
    """SMD aluminium polymer capacitor on its plastic base."""
    b = d + 0.3
    P = [(-b / 2, -b / 2), (b / 2 - 1.0, -b / 2), (b / 2, -b / 2 + 1.0), (b / 2, b / 2 - 1.0), (b / 2 - 1.0, b / 2), (-b / 2, b / 2)]
    s = prism(P, 0.0, 1.0, "plastic_black", bevel=0.15)
    r = d / 2
    s += lathe([(r - 0.3, 0.9), (r, 1.2), (r, h - 0.6), (r - 0.25, h - 0.15), (r - 0.7, h), (0, h)], 28, "alu_can")
    z = h + EPS
    a = np.linspace(-0.6 * math.pi, 0.6 * math.pi, 12)
    s += poly_face([(-(r - 0.8) * math.cos(t), (r - 0.8) * math.sin(t)) for t in a], z, "cap_print")
    s += C.mark(list(code), d * 0.5, d * 0.5, z, "ink_black", c=(d * 0.13, 0), hmax=1.1)
    for sg in (-1, 1):
        s += box(1.2, 0.9, 0.2, (sg * (b / 2 + 0.25), 0, 0), 0.0, "tin", base=True)
    return s


# ----------------------------------------------------------------------------
# the board
# ----------------------------------------------------------------------------
def build_itx_board(L, name="itx_board", seed=11):
    B = L["board"]
    W, H = B["size"]
    t = B["pcb_t"]
    sx, sy = B["socket"]
    hx, hy = B["am5_holes"][0] / 2, B["am5_holes"][1] / 2
    zi = B["z_ihs"]
    KO = keepouts(L)
    P = Part(name)
    rnd = random.Random(seed)
    occ = K.Occ(0, 0, W, H, 0.2)          # anything placed
    tall = K.Occ(0, 0, W, H, 0.2)         # keep-outs for parts taller than 1 mm
    for r in KO.values():
        tall.mark(*r)
    ov = Soup()                            # silk / pads overlay (top)
    cnt = {}

    def inst(key, fn, nm, x, y, rot=0.0, z=0.0, rx=0.0):
        cnt[nm] = cnt.get(nm, -1) + 1
        return P.obj("%s.%03d" % (nm, cnt[nm]), proto(key, fn), (x, y, z), (rx, 0, rot))

    def label(s, x, y, rot=0.0, h=0.9, reserve=True):
        nonlocal ov
        ov += K.silk_label(s, x, y, rot, h)
        if reserve:
            w = len(s) * h * 0.92
            ww, hh = (w, h * 1.3) if rot % 180 == 0 else (h * 1.3, w)
            occ.mark_c(x, y, ww, hh)

    # ------------------------------------------------------------------ PCB
    cool = [(sx + a * hx, sy + b * hy) for a in (-1, 1) for b in (-1, 1)]
    holes = [circle(2.0, 24, h) for h in B["holes"]] + [circle(3.35, 28, c) for c in cool]
    pcb = K.pcb_plate(rrect(W, H, 1.0, 4, (W / 2, H / 2)), holes, t, "pcb_black")
    for h in B["holes"]:
        for z in (0.006, -t - 0.006):
            ov += annulus(2.0, 3.9, z, 28, "enig_gold", h) if z > 0 else annulus(2.0, 3.9, 0.006, 28, "enig_gold", (h[0], -h[1])).rot("x", 180).move(0, 0, -t)
        ov += tube_ring(2.02, 1.9, -t, 0.0, 24, "enig_gold").move(h[0], h[1], 0)
        ov.tile(disc(0.28, 0.008, 8, "enig_gold"), [(h[0] + 3.0 * math.cos(k * TAU / 8), h[1] + 3.0 * math.sin(k * TAU / 8), 0) for k in range(8)])
        occ.mark_disc(h[0], h[1], 4.6)
    for c in cool:
        ov += annulus(3.35, 4.6, 0.006, 28, "enig_gold", c)
        occ.mark_disc(c[0], c[1], 5.0)
    P.obj("pcb", pcb.mesh(P.name + "_pcb", smooth=40, weighted=False))

    # ------------------------------------------------------------------ socket + CPU
    samx, samy = B["sam"]
    P.obj("socket", proto("am5_socket_%g_%g_%g" % (samx, samy, zi), lambda: am5_socket(samx, samy, zi)), (sx, sy, 0))
    P.obj("cpu", proto("am5_cpu_%g_%g" % (zi, B["ihs"][0]), lambda: am5_cpu(zi, B["ihs"][0])), (sx, sy, 0))
    for c in cool:
        inst("am5_post_%g" % t, lambda: am5_post(4.0, t), "post", c[0], c[1])
    P.obj("backplate", proto("am5_backplate_%g_%g_%g" % (hx, hy, t), lambda: am5_backplate(hx, hy, t)), (sx, sy, 0))
    occ.mark_c(sx, sy, samx + 0.6, samy + 0.6)
    ov += K.silk_corners(sx, sy, samx + 1.6, samy + 1.6, 4.0, 0.15)
    label("SOCKET LGA1718", sx, sy - samy / 2 - 2.0, 0, 1.0)

    # ------------------------------------------------------------------ VRM (exact layout positions)
    dx, dy = B["drmos"]
    ix, iy = B["inductor"]
    zd, zl = B["z_drmos"], B["z_ind"]
    stage = lambda: C.qfn(dx, dy, zd, (6, 8), ("ZX8695", "2431K"), pitch=0.5)
    ind = lambda: C.inductor(ix, iy, zl, "R15")
    vl, vt = B["vrm_left"], B["vrm_top"]
    for i in range(vl["n"]):
        y = vl["y0"] + i * vl["pitch"]
        inst("stage_%g_%g_%g" % (dx, dy, zd), stage, "vrm_stage", vl["drmos_x"], y, 0)
        inst("ind_%g_%g_%g" % (ix, iy, zl), ind, "vrm_ind", vl["ind_x"], y, 0)
        occ.mark_c(vl["drmos_x"], y, dx + 0.6, dy + 0.6)
        occ.mark_c(vl["ind_x"], y, ix + 2.2, iy + 0.6)
        ov += K.silk_corners(vl["ind_x"], y, ix + 1.0, iy + 1.0, 1.2)
        ov += K.silk_label("L%d" % (i + 1), vl["ind_x"], y + iy / 2 + 1.0, 0, 0.7)
    for i in range(vt["n"]):
        x = vt["x0"] + i * vt["pitch"]
        inst("stage_%g_%g_%g" % (dx, dy, zd), stage, "vrm_stage", x, vt["drmos_y"], 90)
        inst("ind_%g_%g_%g" % (ix, iy, zl), ind, "vrm_ind", x, vt["ind_y"], 90)
        occ.mark_c(x, vt["drmos_y"], dy + 0.6, dx + 0.6)
        occ.mark_c(x, vt["ind_y"], iy + 0.6, ix + 2.2)
        ov += K.silk_corners(x, vt["ind_y"], iy + 1.0, ix + 1.0, 1.2)
        ov += K.silk_label("L%d" % (i + 7), x + iy / 2 + 1.2, vt["ind_y"], 90, 0.7)

    # ------------------------------------------------------------------ power connectors
    e = B["eps8"]
    ex, ey = (e["x"][0] + e["x"][1]) / 2, (e["y"][0] + e["y"][1]) / 2
    P.obj("eps8", proto("minifit4", lambda: C.minifit_header(4, "plastic_black")), (ex, ey - 1.0, 0), (0, 0, 180))
    occ.mark(e["x"][0], e["y"][0], e["x"][1], e["y"][1])
    label("ATX_12V", ex, e["y"][0] - 1.2, 0, 0.9)
    a = B["atx24"]
    ax, ay = (a["x"][0] + a["x"][1]) / 2, (a["y"][0] + a["y"][1]) / 2
    P.obj("atx24", proto("minifit12", lambda: C.minifit_header(12, "plastic_black")), (ax - 0.4, ay, 0), (0, 0, 90))
    occ.mark(a["x"][0] - 0.5, a["y"][0], a["x"][1] + 1, a["y"][1])
    label("ATX_PWR", a["x"][0] - 1.6, ay, 90, 0.9)

    # ------------------------------------------------------------------ DIMM
    d = B["dimm"]
    Ld = d["y"][1] - d["y"][0]
    yc = (d["y"][0] + d["y"][1]) / 2
    for i, x in enumerate(d["x"]):
        inst("dimm_slot_%g_%g_%g" % (Ld, d["w"], d["z_slot"]), lambda: dimm_slot(Ld, d["w"], d["z_slot"]), "dimm_slot", x, yc)
        inst("dimm_mod_%g_%g" % (d["z_module"], d["z_slot"]), lambda: dimm_module(d["z_module"], d["z_slot"]), "dimm", x, yc)
        occ.mark_c(x, yc, d["w"] + 1.6, Ld + 0.5)
        ov += K.silk_label("DDR5_%s1" % "AB"[i], x, d["y"][0] - 1.6, 0, 0.8)

    # ------------------------------------------------------------------ PCIe slot
    pc = B["pcie"]
    Ls = pc["x"][1] - pc["x"][0]
    P.obj("pcie_slot", proto("pcie_slot_%g_%g" % (Ls, pc["z"]), lambda: pcie_slot(Ls, pc["z"])), (pc["x"][0], pc["y"], 0))
    occ.mark(pc["x"][0] - 0.5, pc["y"] - 4.6, pc["x"][1] + 0.5, pc["y"] + 4.6)
    label("PCIEX16_1", pc["x"][0] + 14.0, pc["y"] + 5.6, 0, 0.9)

    # ------------------------------------------------------------------ M.2 + chipset
    m = B["m2"]
    mW, mL = m["y"][1] - m["y"][0], m["x"][1] - m["x"][0]
    my = (m["y"][0] + m["y"][1]) / 2
    zb = m["z_top"] - 1.35 - 0.8
    P.obj("m2_conn", proto("m2_conn_%g" % zb, lambda: m2_connector(zb)), (m["x"][0], my, 0))
    P.obj("m2_ssd", proto("m2_ssd_%g_%g_%g_%g" % (mL, mW, zb, m["z_top"]), lambda: m2_ssd(mL, mW, zb, m["z_top"])), (m["x"][0], my, 0))
    P.obj("m2_screw", proto("m2_standoff_%g" % zb, lambda: m2_standoff(zb)), (m["x"][1], my, 0))
    occ.mark(m["x"][0] - 6.6, m["y"][0] - 1.0, m["x"][0] + 3.6, m["y"][1] + 1.0)
    occ.mark_disc(m["x"][1], my, 3.4)
    label("M2_1", m["x"][0] + 9.0, m["y"][1] + 1.0, 0, 0.8, reserve=False)
    ch = B["chipset"]
    chw, chl = ch["size"]
    P.obj("chipset", proto("chipset_%g_%g_%g" % (chw, chl, ch["z_top"]),
                           lambda: C.fcbga(chw, chl, ch["z_top"], die=(9.2, 7.8), sub_t=0.75, lines=("PCH B6X", "2431 A1"))),
          (ch["c"][0], ch["c"][1], 0))
    occ.mark_c(ch["c"][0], ch["c"][1], chw + 0.8, chl + 0.8)
    ov += K.silk_corners(ch["c"][0], ch["c"][1], chw + 1.0, chl + 1.0, 1.5)

    # ------------------------------------------------------------------ rear I/O (port frame: front face at io x0)
    io = B["io"]
    x_io = io["x"][0]
    io_items = [("audio", lambda: K.audio_stack(2), 12.4, 13.0), ("wifi", lambda: K.wifi_can(), 24.0, 17.5),
                ("lan_usb", lambda: K.rj45_usb_combo(), 17.3, 21.3), ("usb_a", lambda: K.usb_a_stack(2, "plastic_blue"), 13.4, 17.4),
                ("usb_a", lambda: K.usb_a_stack(2, "plastic_blue"), 13.4, 17.4), ("usb_a2", lambda: K.usb_a_stack(2, "plastic_teal"), 13.4, 17.4),
                ("hdmi_dp", lambda: K.hdmi_dp_stack(), 18.6, 18.0)]
    span = io["y"][1] - io["y"][0]
    gap = (span - sum(i[2] for i in io_items)) / (len(io_items) + 1)
    y = io["y"][0] + gap
    for nm, fn, wy, dd in io_items:
        inst("io_" + nm, fn, "io_" + nm, x_io, y + wy / 2)
        occ.mark(0, y - 0.4, x_io + dd + 1.0, y + wy + 0.4)
        y += wy + gap
    occ.mark(0, io["y"][0], 3.0, io["y"][1])

    # ------------------------------------------------------------------ larger parts outside the keep-outs
    def place_tall(key, fn, nm, x, y, w, h, rot=0.0, lab=None, lab_dy=None):
        """Place a tall part if it is clear of keep-outs and other parts."""
        ww, hh = (w, h) if rot % 180 == 0 else (h, w)
        x0, y0, x1, y1 = x - ww / 2, y - hh / 2, x + ww / 2, y + hh / 2
        if not (tall.free(x0, y0, x1, y1) and occ.free(x0 - 0.2, y0 - 0.2, x1 + 0.2, y1 + 0.2)):
            return None
        occ.mark(x0, y0, x1, y1)
        ob = inst(key, fn, nm, x, y, rot)
        if lab:
            label(lab, x, y + (lab_dy if lab_dy is not None else -(hh / 2 + 0.9)), 0, 0.75)
        return ob

    can5 = lambda: smd_can(5.0, 5.8, ("270", "16V"))
    can6 = lambda: smd_can(6.3, 5.8, ("560", "6.3"))
    kx1 = KO["arm_top"][2]                # right end of the top arm
    kl0 = KO["arm_left"][0]               # left edge of the left arm
    dimm_x0 = d["x"][0] - d["w"] / 2
    # VRM input / output bulk capacitors: right of the top arm, behind the I/O for the left bank
    n = 0
    for j in range(3):
        for i in range(2):
            n += place_tall("can6", can6, "cap", kx1 + 4.6 + i * 7.2, H - 6.0 - j * 7.2, 6.8, 6.8) is not None
    for i in range(vl["n"]):
        place_tall("can5", can5, "cap", kl0 - 3.4, vl["y0"] + i * vl["pitch"] + 3.0, 5.6, 5.6)
    ko_s = KO["socket"]
    for i in range(3):
        place_tall("can5", can5, "cap", kl0 + 3.5 + i * 6.3, ko_s[1] + 9.0, 5.6, 5.6)
    # PWM controller + drivers near the top bank
    place_tall("pwm_ctrl", lambda: C.qfn(6.0, 6.0, 0.9, (12, 12), ("ZX2891", "GQJ-P", "2431"), pitch=0.4), "ic",
               kx1 + 8.0, H - 30.5, 6.6, 6.6, lab="PU1")
    # headers along the top / right edges
    place_tall("fan_hdr", lambda: C.fan_header(4), "hdr_fan", kx1 + 8.0, H - 37.5, 10.4, 6.0, lab="CPU_FAN", lab_dy=-4.0)
    place_tall("fan_hdr", lambda: C.fan_header(4), "hdr_fan", W - 6.5, a["y"][1] + 22.5, 10.4, 6.0, 90, lab="PUMP", lab_dy=-6.6)
    place_tall("fan_hdr", lambda: C.fan_header(4), "hdr_fan", W - 6.5, a["y"][1] + 35.0, 10.4, 6.0, 90, lab="SYS_FAN", lab_dy=6.4)
    place_tall("rgb_hdr", lambda: C.pin_header(3, 1), "hdr_rgb", W - 5.0, a["y"][1] + 4.5, 8.0, 2.8, 0, lab="ARGB")
    # front-panel / USB headers on the right edge below the 24-pin
    yb = a["y"][0]
    place_tall("usb3_hdr", lambda: C.usb3_header(), "hdr_usb3", W - 5.2, yb - 19.0, 30.4, 6.8, 90, lab=None)
    label("USB3_1", W - 10.4, yb - 19.0, 90, 0.8)
    place_tall("fp_hdr", lambda: C.pin_header(5, 2, missing={(4, 1)}), "hdr_fp", W - 5.2, yb - 46.0, 12.9, 5.2, 90)
    label("JFP1", W - 9.6, yb - 46.0, 90, 0.8)
    place_tall("usb2_hdr", lambda: C.pin_header(5, 2, missing={(4, 0)}), "hdr_usb2", W - 5.2, yb - 61.0, 12.9, 5.2, 90)
    label("USB2_1", W - 9.6, yb - 61.0, 90, 0.8)
    # column between the socket keep-out and the DIMM slots
    cx = (ko_s[2] + dimm_x0) / 2
    place_tall("coin_cell", lambda: C.coin_cell(20.0), "battery", cx + 0.4, ko_s[1] + 14.5, 23.4, 24.0, lab="BAT1", lab_dy=13.2)
    place_tall("usbc_hdr", lambda: C.usbc_header(), "hdr_usbc", cx, ko_s[1] + 35.0, 15.6, 6.6, lab="USB_C1")
    place_tall("sio", lambda: C.gull_ic(12, 0.5, 7.0, 7.0, 1.0, 9.0, ("SIO 6687", "DHX-A2", "2431"), quad=True), "ic",
               cx - 4.0, ko_s[1] + 49.0, 9.4, 9.4, lab="U12")
    place_tall("spi", lambda: C.gull_ic(4, 1.27, 5.2, 5.2, 1.45, 8.0, ("25Q256", "JVEQ")), "ic", cx + 7.0, ko_s[1] + 50.0, 8.2, 5.6, 90, lab="BIOS")
    place_tall("xtal_can", lambda: C.crystal_can(), "xtal", cx - 3.0, ko_s[1] + 58.5, 12.2, 5.6, lab="Y2")
    place_tall("ind4", lambda: C.inductor(4.0, 4.0, 2.0, "2R2"), "ind", cx - 6.5, ko_s[1] + 68.0, 5.6, 4.4, lab="L21")
    place_tall("buck", lambda: C.qfn(3.0, 3.0, 0.85, (4, 4), ("NB6", "71K")), "ic", cx + 0.2, ko_s[1] + 68.0, 3.6, 3.6)
    place_tall("can5", can5, "cap", cx + 7.0, ko_s[1] + 68.0, 5.6, 5.6)
    place_tall("ind4", lambda: C.inductor(4.0, 4.0, 2.0, "2R2"), "ind", cx - 6.5, ko_s[1] + 77.0, 5.6, 4.4, lab="L22")
    place_tall("buck", lambda: C.qfn(3.0, 3.0, 0.85, (4, 4), ("NB6", "71K")), "ic", cx + 0.2, ko_s[1] + 77.0, 3.6, 3.6)
    place_tall("can5", can5, "cap", cx + 7.0, ko_s[1] + 77.0, 5.6, 5.6)
    for i in range(4):
        place_tall("led0603", lambda: C.led(), "led", dimm_x0 - 2.6, ko_s[1] + 86.0 + i * 2.4, 2.4, 1.2)
    label("CPU DRAM VGA BOOT", dimm_x0 - 5.2, ko_s[1] + 89.6, 90, 0.6)
    # audio corner (below the I/O, left of the PCIe slot)
    ax0 = 9.5
    place_tall("codec", lambda: C.qfn(6.0, 6.0, 0.85, (12, 12), ("ALC 4080", "K3B77", "2431"), pitch=0.4), "ic", 22.0, 11.0, 6.6, 6.6, lab="U20")
    for i in range(4):
        place_tall("can4", lambda: smd_can(4.0, 5.4, ("100", "16V")), "cap", ax0 + 2.6 + (i % 2) * 5.2 + 22.0, 3.6 + (i // 2) * 5.2 + 6.4, 4.6, 4.6)
    place_tall("aud_hdr", lambda: C.pin_header(5, 2, missing={(3, 1)}), "hdr_audio", 22.0, 3.6, 12.9, 5.2, lab=None)
    label("JAUD1", 12.6, 2.4, 0, 0.8)
    # lower-right corner: clock, SATA, misc
    place_tall("sata", lambda: C.sata_port(), "sata", W - 28.0, 6.0, 17.4, 7.0, lab="SATA1", lab_dy=4.6)
    place_tall("xtal3225", lambda: C.crystal(), "xtal", pc["x"][1] + 5.0, 14.5, 3.6, 2.9)
    place_tall("clr_hdr", lambda: C.pin_header(2, 1), "hdr_clr", W - 12.0, 13.5, 5.4, 2.8, lab="JBAT1")
    # LAN / USB controllers behind the I/O stack
    xi = KO["m2_sink"][0] - 2.0
    place_tall("lan_ic", lambda: C.qfn(6.0, 6.0, 0.85, (12, 12), ("RTL 8125BG", "N4K21", "2431"), pitch=0.4), "ic", 35.0, ko_s[1] + 1.5, 6.6, 6.6)
    place_tall("xtal3225", lambda: C.crystal(), "xtal", 42.0, ko_s[1] + 2.0, 3.6, 2.9)
    place_tall("soic8", lambda: C.gull_ic(4, 1.27, 3.9, 4.9, 1.45, 6.0, ("24C02", "431")), "ic", 34.0, ko_s[1] + 15.0, 6.4, 5.2)
    place_tall("ind4", lambda: C.inductor(4.0, 4.0, 2.0, "4R7"), "ind", 42.0, ko_s[1] + 16.0, 5.6, 4.4)

    # ------------------------------------------------------------------ merged SMD clutter (<= 1 mm tall)
    protos = K.smd_protos()
    cl = K.Clutter(protos)
    ndes = [0]

    def group(x, y, kind, k, horiz, des=True):
        """Row of k identical chips starting at (x, y); returns number placed."""
        fw, fh = protos[kind][1], protos[kind][2]
        rot = 90 if horiz else 0       # chips sit side by side: long axis perpendicular to the row
        w, h = (fh, fw) if horiz else (fw, fh)
        pitch = (w if horiz else h) + 0.28
        placed = 0
        for i in range(k):
            px, py = (x + i * pitch, y) if horiz else (x, y + i * pitch)
            if occ.take(px, py, w, h, 0.1):
                cl.put(kind, px, py, rot)
                placed += 1
            else:
                break
        if des and placed >= 2 and rnd.random() < 0.6:
            ndes[0] += 1
            s = ("C%d" if kind[0] == "c" else "R%d") % (100 + ndes[0] * 3 % 880)
            lx, ly = (x - 0.4 + 1.0, y - h / 2 - 0.6) if horiz else (x + w / 2 + 0.5, y + 0.4)
            ww = len(s) * 0.5
            if occ.free(lx - 0.2, ly - 0.4, lx + ww, ly + 0.4):
                ov.__iadd__(K.silk_label(s, lx, ly, 0, 0.5, align="l"))
                occ.mark(lx - 0.2, ly - 0.4, lx + ww, ly + 0.4)
        return placed

    # decoupling rows hugging the socket frame (inside the keep-out, all <= 1 mm)
    fx0, fx1, fy0, fy1 = sx - samx / 2, sx + samx / 2, sy - samy / 2, sy + samy / 2
    for yy in np.arange(fy0 + 3.0, fy1 - 2.0, 3.1):
        group(fx0 - 3.3, yy, "c0603", 2, True, des=False)
        group(fx1 + 1.4, yy, "c0603", 2, True, des=False)
    for xx in np.arange(fx0 + 12.0, fx1 - 12.0, 2.6):
        group(xx, fy1 + 1.4, "c0603", 2, False, des=False)
        group(xx, fy0 - 5.4, "c0603", 2, False, des=False)
    # output MLCC banks between inductors and socket, gate resistors by each stage
    for i in range(vl["n"]):
        y = vl["y0"] + i * vl["pitch"]
        group(vl["drmos_x"] - 2.0, y + dy / 2 + 1.0, "c0402", 4, True, des=False)
        group(vl["drmos_x"] + dx / 2 + 0.9, y - 2.0, "r0402", 3, False, des=False)
        group(vl["ind_x"] - 4.0, y + iy / 2 + 0.75, "c0805", 3, True, des=False)
    for i in range(vt["n"]):
        x = vt["x0"] + i * vt["pitch"]
        group(x + dy / 2 + 1.0, vt["drmos_y"] - 2.0, "c0402", 4, False, des=False)
        group(x - 2.2, vt["drmos_y"] - dx / 2 - 1.0, "r0402", 3, True, des=False)
        group(x + iy / 2 + 1.0, vt["ind_y"] - 3.4, "c0805", 3, False, des=False)
    # AC-coupling caps between the PCIe slot and the M.2 area, DIMM termination columns
    for xx in np.arange(pc["x"][0] + 16.0, pc["x"][1] - 6.0, 3.4):
        group(xx, pc["y"] + 5.6, "c0402", 2, False, des=False)
    for i, x in enumerate(d["x"]):
        for yy in np.arange(d["y"][0] + 8.0, d["y"][1] - 14.0, 6.2):
            group(x - d["w"] / 2 - 1.2, yy, "r0402" if (int(yy) % 2) else "c0402", 4, False, des=False)
    # passives around every IC / chipset / M.2 edges
    for (cx_, cy_, hw, hh) in [(ch["c"][0], ch["c"][1], chw / 2, chl / 2)]:
        for xx in np.arange(cx_ - hw + 1.0, cx_ + hw - 0.5, 1.9):
            group(xx, cy_ + hh + 1.0, "c0402", 2, False, des=False)
            group(xx, cy_ - hh - 2.0, "c0402", 2, False, des=False)
        for yy in np.arange(cy_ - hh + 1.0, cy_ + hh - 0.5, 1.9):
            group(cx_ + hw + 1.0, yy, "c0402", 2, True, des=False)
    # random organised groups everywhere else
    kinds = ["c0402"] * 5 + ["r0402"] * 4 + ["c0603"] * 3 + ["r0603"] * 2 + ["c0402g"] * 2 + ["c0805"]
    target = 950
    tries = 0
    while cl.count() < target and tries < 9000:
        tries += 1
        x, y = rnd.uniform(3, W - 3), rnd.uniform(3, H - 3)
        if x < x_io + 23 and io["y"][0] - 1 < y < io["y"][1] + 1:
            continue
        x, y = round(x * 4) / 4, round(y * 4) / 4
        group(x, y, rnd.choice(kinds), rnd.randint(2, 6), rnd.random() < 0.5)
    # small actives + test pads sprinkled
    for i in range(46):
        x, y = rnd.uniform(4, W - 4), rnd.uniform(4, H - 4)
        if x < x_io + 23 and io["y"][0] - 1 < y < io["y"][1] + 1:
            continue
        r = rnd.choice((0, 90, 180, 270))
        k = rnd.random()
        if k < 0.5:
            if occ.take(x, y, 3.0, 3.2, 0.2):
                inst("sot23_%d" % (3 if k < 0.3 else 6), lambda: C.sot23(3 if k < 0.3 else 6), "q", x, y, r)
        elif k < 0.75:
            if occ.take(x, y, 3.4, 3.4, 0.25):
                inst("qfn3", lambda: C.qfn(3.0, 3.0, 0.8, (4, 4), ("T7A", "K31")), "ic", x, y, r)
                ov += K.silk_corners(x, y, 3.6, 3.6, 0.6, 0.1)
        else:
            if occ.take(x, y, 2.4, 2.4, 0.25):
                inst("qfn2", lambda: C.qfn(2.0, 2.0, 0.75, (3, 3), ("4K",), dot=True), "ic", x, y, r)
    tp = []
    for i in range(34):
        x, y = rnd.uniform(4, W - 4), rnd.uniform(4, H - 4)
        k = rnd.randint(1, 5)
        hz = rnd.random() < 0.5
        for j in range(k):
            px, py = (x + j * 1.6, y) if hz else (x, y + j * 1.6)
            if occ.take(px, py, 1.1, 1.1, 0.15):
                tp.append((px, py, 0))
    ov.tile(disc(0.5, 0.008, 12, "enig_gold"), tp)
    ov.tile(annulus(0.5, 0.62, 0.009, 12, "silk_white"), tp[::3])
    # fiducials + board legend
    for fx, fy in ((W - 14.0, H - 3.0), (36.0, 2.2), (W - 3.0, 12.0)):
        if occ.take(fx, fy, 2.2, 2.2, 0.1):
            ov += disc(0.5, 0.008, 12, "enig_gold", (fx, fy))
            ov += annulus(0.95, 1.05, EPS, 20, "silk_white", (fx, fy))
    for s_, x, y, r, h in (("MINI-ITX AM5  REV 1.02", ko_s[2] + 3.2, sy - 6.0, 90, 1.5), ("94V-0  E123456  2431", ko_s[2] + 5.6, sy - 6.0, 90, 0.8)):
        ww = len(s_) * h * 0.95
        if occ.free(x - h, y - ww / 2, x + h, y + ww / 2):
            label(s_, x, y, r, h)
    P.obj("smd_top", cl.soup().mesh(P.name + "_smd_top"))
    P.obj("silk", ov.mesh(P.name + "_silk", smooth=None))

    # ------------------------------------------------------------------ back side
    back = Soup()
    bcl = K.Clutter(protos)
    bocc = K.Occ(0, 0, W, H, 0.25)
    bocc.mark_c(sx, sy, 2 * hx + 16, 2 * hy + 15)
    for h in B["holes"]:
        bocc.mark_disc(h[0], h[1], 5.0)

    def bgroup(x, y, kind, nx, ny, px, py, rot=0):
        fw, fh = protos[kind][1], protos[kind][2]
        w, h = (fw, fh) if rot == 0 else (fh, fw)
        for i in range(nx):
            for j in range(ny):
                qx, qy = x + i * px, y + j * py
                if bocc.take(qx, qy, w, h, 0.08):
                    bcl.put(kind, qx, -qy, -rot)

    # socket decoupling array seen through the backplate window
    bocc2 = K.Occ(0, 0, W, H, 0.25)
    for i in range(12):
        for j in range(12):
            if (i + j) % 5 != 4:
                bcl.put("c0402" if (i * 7 + j) % 3 else "c0603", sx - 14.3 + i * 2.6, -(sy - 14.3 + j * 2.6), 90 if (i + j) % 2 else 0)
    for i in range(vl["n"]):
        y = vl["y0"] + i * vl["pitch"]
        bgroup(vl["drmos_x"] - 2.0, y - 3.0, "c0805", 2, 4, 3.2, 2.0)
        bgroup(vl["ind_x"] - 2.0, y - 3.0, "c0603", 3, 3, 2.6, 2.4)
    for i in range(vt["n"]):
        x = vt["x0"] + i * vt["pitch"]
        bgroup(x - 3.0, vt["drmos_y"] - 2.0, "c0805", 4, 2, 2.0, 3.2, 90)
        bgroup(x - 3.0, vt["ind_y"] - 2.5, "c0603", 3, 3, 2.4, 2.6, 90)
    bgroup(ch["c"][0] - 7.5, ch["c"][1] - 7.5, "c0402", 7, 9, 2.4, 1.8)
    for xx in np.arange(d["x"][0] - 5.0, d["x"][1] + 5.0, 2.2):
        bgroup(xx, d["y"][0] + 10.0, "c0402", 1, 40, 0, 3.1, 90)
    tr = 0
    while bcl.count() < 1100 and tr < 5000:
        tr += 1
        x, y = round(rnd.uniform(4, W - 4) * 4) / 4, round(rnd.uniform(4, H - 4) * 4) / 4
        kind = rnd.choice(kinds)
        hz = rnd.random() < 0.5
        k = rnd.randint(2, 6)
        fw, fh = protos[kind][1], protos[kind][2]
        bgroup(x, y, kind, k if hz else 1, 1 if hz else k, fh + 0.28, fh + 0.28, 90 if hz else 0)
    back += bcl.soup().rot("x", 180).move(0, 0, -t)
    # through-hole joints of the big connectors
    jp = []
    for i in range(12):
        for j in range(2):
            jp.append((ax - 0.4 + (j - 0.5) * 4.2, ay + (i - 5.5) * 4.2))
    for i in range(4):
        for j in range(2):
            jp.append((ex + (i - 1.5) * 4.2, ey - 1.0 + (j - 0.5) * 4.2))
    y = io["y"][0] + gap
    for nm, fn, wy, dd in io_items:
        for k in range(int(wy / 2.2)):
            jp.append((x_io + 6.0, y + 1.5 + k * 2.2))
            jp.append((x_io + dd - 3.0, y + 1.5 + k * 2.2))
        for sx_ in (4.0, dd - 2.0):
            for sy_ in (0.4, wy - 0.4):
                jp.append((x_io + sx_, y + sy_))
        y += wy + gap
    for x in d["x"]:
        jp += [(x, d["y"][0] + 3.0), (x, d["y"][1] - 3.0), (x, yc)]
    for i in range(5):
        for sy_ in (-1, 1):
            jp.append((pc["x"][0] + 6.0 + i * (Ls - 16.6) / 4, pc["y"] + sy_ * 4.2))
    jp += [(kx1 + 8.0 + (i - 1.5) * 2.54, H - 37.9) for i in range(4)]
    jp += [(W - 5.2 + (j - 0.5) * 2.0, yb - 19.0 + (i - 4.5) * 2.0) for i in range(10) for j in range(2)]
    jp += [(W - 5.2 + (j - 0.5) * 2.54, yy + (i - 2) * 2.54) for yy in (yb - 46.0, yb - 61.0) for i in range(5) for j in range(2)]
    back += C.tht_joints([(x, -y, 0) for x, y in jp]).rot("x", 180).move(0, 0, -t)
    bl = Soup()
    bl += K.silk_label("MINI-ITX AM5 REV 1.02", W / 2, -(sy - hy - 16.0), 0, 2.0)
    bl += K.silk_label("LEAD FREE  ROHS  2431", W / 2, -(sy - hy - 19.5), 0, 1.1)
    for h in B["holes"]:
        bl += annulus(2.0, 3.9, 0.006, 28, "enig_gold", (h[0], -h[1]))
    back += bl.rot("x", 180).move(0, 0, -t)
    P.obj("back", back.mesh(P.name + "_back"))
    P.root["keepouts"] = {k: list(v) for k, v in KO.items()}
    P.root["pcie_tongue_x0"] = pc["x"][0] + PCIE_CAV0 + 0.15
    return P.root
