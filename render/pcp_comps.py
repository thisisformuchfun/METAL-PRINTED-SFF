"""pcp_comps - electronic component prototypes for the PCB builders.

Every function returns a `Soup` whose origin is the footprint centre on the PCB top
surface (z = 0, +z up out of the component side). Markings are generic codes only.
"""
import math

import numpy as np

from pcp_geom import (Soup, box, box2, cyl, lathe, prism, plate, rect, disc, annulus, strip, rect_outline, loft,
                      text, rrect, circle, bevel, rbox, tube_ring, poly_face, TAU)

EPS = 0.012          # height of flat overlays (silk, laser marks) above their surface


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def mark(lines, max_w, max_h, z, mat="laser_mark", c=(0.0, 0.0), rot=0.0, hmax=1.3, w=None):
    """Stacked stroke-text lines fitted into a max_w x max_h window centred on c."""
    lines = [ln for ln in lines if ln]
    if not lines:
        return Soup()
    n = len(lines)
    longest = max(len(ln) for ln in lines)
    h = min(hmax, max_h / (n * 1.5), max_w / (longest * 5.5 / 6.0))
    s = Soup()
    for i, ln in enumerate(lines):
        s += text(ln, h, w, mat, z).move(0, ((n - 1) / 2 - i) * 1.5 * h, 0)
    if rot:
        s.rotz(rot)
    return s.move(c[0], c[1], 0)


def extrude_x(sec, x0, x1, mat, cap0=True, cap1=True, skip=()):
    """Extrude a (y,z) section (CCW seen from +x) from x0 to x1; skip = section edge indices to omit."""
    sec = np.asarray(sec, float)
    k = len(sec)
    V = np.array([(x, y, z) for x in (x0, x1) for (y, z) in sec])
    F = [(j, (j + 1) % k, k + (j + 1) % k, k + j) for j in range(k) if j not in skip]
    if cap0:
        F.append(tuple(range(k - 1, -1, -1)))
    if cap1:
        F.append(tuple(range(k, 2 * k)))
    return Soup().add(V, F, mat)


def wedge(x0, x1, w0, w1, h, mat="solder"):
    """Solder fillet: full height h at x0, zero at x1 (extends along x)."""
    V = [(x0, -w0 / 2, h), (x0, w0 / 2, h), (x1, w1 / 2, 0.01), (x1, -w1 / 2, 0.01),
         (x0, -w0 / 2, 0.01), (x0, w0 / 2, 0.01)]
    F = [(0, 1, 2, 3), (1, 5, 2), (0, 3, 4)] if x1 > x0 else [(1, 0, 3, 2), (1, 2, 5), (0, 4, 3)]
    return Soup().add(V, F, mat)


# ----------------------------------------------------------------------------
# passives
# ----------------------------------------------------------------------------
def chip(l, w, h, body="mlcc_tan", term="tin", tl=None, pad=True):
    """Two-terminal chip (MLCC / resistor), length along x. ~50 triangles."""
    tl = tl or max(0.15, 0.24 * l)
    b = min(0.05, 0.2 * h)
    sec = [(-w / 2, 0.02), (w / 2, 0.02), (w / 2, h - b), (w / 2 - b, h), (-w / 2 + b, h), (-w / 2, h - b)]
    s = Soup()
    s += extrude_x(sec, -l / 2, -l / 2 + tl, term, True, False, skip=(0,))
    s += extrude_x(sec, -l / 2 + tl, l / 2 - tl, body, False, False, skip=(0,))
    s += extrude_x(sec, l / 2 - tl, l / 2, term, False, True, skip=(0,))
    f = 0.45 * h + 0.08
    for sg in (-1, 1):
        s += wedge(sg * l / 2, sg * (l / 2 + f), w * 0.86, w * 1.02, h * 0.62)
        if pad:
            s += rect(tl + f + 0.06, w + 0.12, 0.008, "tin", (sg * (l / 2 - tl / 2 + f / 2 + 0.03), 0))
    return s


def resistor(l, w, h, code=None):
    s = chip(l, w, h, body="res_black", tl=max(0.15, 0.2 * l))
    if code:
        s += mark([code], (l * 0.56), w * 0.8, h + 0.004, "silk_white", hmax=w * 0.62)
    return s


def tant(l=7.3, w=4.3, h=1.9, code=("330", "2R5")):
    """Polymer / tantalum chip capacitor (7343 by default)."""
    s = box(l - 0.5, w, h - 0.05, (0, 0, 0.05), 0.12, "tant_body", base=True)
    tw = w * 0.56
    for sg in (-1, 1):
        s += box(1.3, tw, h * 0.55, (sg * (l / 2 - 0.62), 0, 0.02), 0.05, "tin", base=True)
        s += wedge(sg * l / 2, sg * (l / 2 + 0.45), tw, tw + 0.3, h * 0.35)
        s += rect(1.9, tw + 0.5, 0.008, "tin", (sg * (l / 2 - 0.35), 0))
    s += rect(0.7, w * 0.9, h + EPS, "silk_grey", (-(l / 2 - 1.0), 0))
    s += mark(list(code), l * 0.42, w * 0.7, h + EPS, "silk_grey", c=(0.6, 0))
    return s


def polymer_can(d=6.3, h=6.5, code=("180", "16V"), tht=True):
    """Aluminium solid-polymer can capacitor with printed top."""
    r = d / 2
    s = Soup()
    s += cyl(r - 0.15, 0.0, 0.45, 28, "rubber")
    prof = [(r - 0.5, 0.35), (r, 0.5), (r, 1.05), (r - 0.28, 1.3), (r, 1.55), (r, h - 0.7),
            (r - 0.08, h - 0.35), (r - 0.3, h - 0.1), (r - 0.75, h), (0, h)]
    s += lathe(prof, 28, "alu_can")
    z = h + EPS
    a = np.linspace(-0.62 * math.pi, 0.62 * math.pi, 14)
    seg = [(-(r - 0.9) * math.cos(t), (r - 0.9) * math.sin(t)) for t in a]
    s += poly_face(seg, z, "cap_print")
    s += mark(list(code), d * 0.5, d * 0.52, z, "ink_black", c=(d * 0.13, 0), hmax=1.25)
    return s


def inductor(l, w, h, label="R15", term_len=None, flat_wire=True, dot=True):
    """Molded power inductor, terminals on the two short (x) ends."""
    s = rbox(l, w, h - 0.06, (0, 0, 0.06), r=min(0.45, 0.2 * h), seg=2, mat="inductor_grey", base=True)
    tl = term_len or min(1.6, 0.14 * l)
    tw = w * (0.62 if flat_wire else 0.5)
    for sg in (-1, 1):
        s += box(tl, tw, h * 0.42, (sg * (l / 2 - tl / 2 + 0.07), 0, 0.02), 0.07, "tin", base=True)
        s += wedge(sg * (l / 2 + 0.07), sg * (l / 2 + 0.6), tw, tw + 0.5, h * 0.2)
        s += rect(tl + 0.9, tw + 0.8, 0.008, "tin", (sg * (l / 2 - tl / 2 + 0.4), 0))
    z = h + EPS
    if label:
        s += mark([label], l * 0.5, w * 0.42, z, "silk_white", hmax=2.4)
    if dot:
        s += disc(min(l, w) * 0.045, z, 10, "silk_white", (-l * 0.36, w * 0.30))
    return s


def crystal(l=3.2, w=2.5, h=0.75):
    s = box(l, w, h * 0.55, (0, 0, 0.02), 0.04, "mlcc_grey", base=True)
    s += box(l - 0.3, w - 0.3, h * 0.47, (0, 0, h * 0.53), 0.08, "steel_bright", base=True)
    s += mark(["25.0"], l * 0.6, w * 0.4, h + EPS, "ink_black")
    return s


def crystal_can(l=11.0, w=4.6, h=3.4):
    """HC-49S style metal can crystal."""
    s = prism(rrect(l, w, w / 2 - 0.05, 6), 0.35, h, "steel_bright", bevel=0.35)
    s += prism(rrect(l + 0.9, w + 0.9, w / 2, 6), 0.0, 0.4, "steel_bright", bevel=0.1)
    s += mark(["24.000"], l * 0.6, w * 0.42, h + EPS, "ink_black")
    return s


def led(l=1.6, w=0.8, h=0.6):
    s = chip(l, w, h * 0.5, body="alumina")
    s += box(l * 0.55, w * 0.9, h * 0.55, (0, 0, h * 0.45), 0.08, "led_lens", base=True)
    return s


# ----------------------------------------------------------------------------
# ICs
# ----------------------------------------------------------------------------
def _side_pads(w, l, n, pitch, pw=None, ph=0.22, out=0.03):
    """Tin lands peeking out along the 4 sides of a leadless package."""
    s = Soup()
    pw = pw or pitch * 0.5
    one = box(pw, 0.3, ph, (0, 0, 0.0), 0.0, "tin", base=True)
    for (cnt, length, half, rz) in ((n[0], w, l / 2, 0), (n[1], l, w / 2, 90)):
        if cnt <= 0:
            continue
        pt = min(pitch, (length - 0.6) / max(cnt, 1))
        xs = (np.arange(cnt) - (cnt - 1) / 2) * pt
        for sg in (-1, 1):
            if rz == 0:
                pos = np.column_stack([xs, np.full(cnt, sg * (half - 0.15 + out)), np.zeros(cnt)])
            else:
                pos = np.column_stack([np.full(cnt, sg * (half - 0.15 + out)), xs, np.zeros(cnt)])
            s.tile(one, pos, np.full(cnt, rz))
    return s


def qfn(w, l, h=0.85, n=(8, 8), lines=("A7X2", "2431"), pitch=0.5, dot=True, body="ic_epoxy",
        mk="laser_mark", bev=0.07):
    """Leadless IC (QFN / LGA / DrMOS)."""
    s = box(w, l, h - 0.04, (0, 0, 0.04), bev, body, base=True)
    s += _side_pads(w, l, n, pitch)
    z = h + EPS * 0.5
    s += mark(list(lines), w * 0.74, l * 0.56, z, mk)
    if dot:
        s += disc(max(0.12, min(w, l) * 0.045), z, 10, mk, (-w / 2 + 0.16 * w, -l / 2 + 0.16 * l))
    return s


def gull_ic(n_side=4, pitch=1.27, bw=3.9, bl=4.9, h=1.45, span=6.0, lines=("LM358", "2431"), quad=False):
    """Gull-wing leaded IC (SOIC / TSSOP / QFP when quad=True). Leads leave the +-x sides."""
    s = box(bw, bl, h - 0.12, (0, 0, 0.12), min(0.16, 0.2 * h), "ic_epoxy", base=True)
    lw = pitch * 0.42
    t = 0.15
    zt = min(h * 0.55, 0.75)
    ext = (span - bw) / 2
    foot = min(0.55, ext * 0.5)
    prof = [(0, zt), (ext - foot - 0.12, zt), (ext - foot, t), (ext, t), (ext, 0.0), (ext - foot - 0.1, 0.0),
            (ext - foot - 0.25, zt - t), (0, zt - t)]
    lead = prism([(x, z) for x, z in prof], -lw / 2, lw / 2, "tin").rot("x", 90)
    sides = [(n_side, bl, bw / 2, 0), (n_side, bl, bw / 2, 180)]
    if quad:
        sides += [(n_side, bw, bl / 2, 90), (n_side, bw, bl / 2, 270)]
    for cnt, length, half, rz in sides:
        ys = (np.arange(cnt) - (cnt - 1) / 2) * pitch
        a = math.radians(rz)
        c, sn = math.cos(a), math.sin(a)
        pos = np.column_stack([half * c - ys * sn, half * sn + ys * c, np.zeros(cnt)])
        s.tile(lead, pos, np.full(cnt, rz))
    z = h + EPS * 0.5
    s += mark(list(lines), bw * 0.74, bl * 0.56, z, "laser_mark", rot=90 if (bl > bw * 1.25) else 0)
    s += disc(0.16, z, 10, "laser_mark", (-bw / 2 + 0.45, bl / 2 - 0.45))
    return s


def sot23(n=3):
    """SOT-23 (3 or 6 leads), 0.95 mm tall."""
    s = box(1.3, 2.9, 0.83, (0, 0, 0.12), 0.1, "ic_epoxy", base=True)
    lead = box(0.6, 0.4, 0.14, (0, 0, 0.0), 0.0, "tin", base=True)
    ys = [-0.95, 0.0, 0.95]
    pos = [(-0.95, y, 0) for y in (ys if n == 6 else (ys[0], ys[2]))] + \
          [(0.95, y, 0) for y in (ys if n == 6 else (ys[1],))]
    s.tile(lead, pos)
    s += mark(["A1"], 0.9, 1.2, 0.95 + EPS * 0.5, "laser_mark_dim", rot=90)
    return s


def bga(w, l, h, lines=("D8BZC", "2KA17"), ball_pitch=0.8, sub="pcb_darkgreen", sub_t=0.2, ball_h=0.22,
        dot=True, balls=True, hmax=1.3):
    """Moulded BGA package: perimeter solder balls, thin substrate, mould cap, laser mark."""
    s = Soup()
    if balls:
        ball = lathe([(0, 0), (0.16, 0.02), (0.22, ball_h / 2), (0.16, ball_h), (0, ball_h)], 8, "solder")
        nx, ny = int((w - 0.9) / ball_pitch), int((l - 0.9) / ball_pitch)
        xs = (np.arange(nx + 1) - nx / 2) * ball_pitch
        ys = (np.arange(ny + 1) - ny / 2) * ball_pitch
        pos = [(x, sg * ys[-1], 0) for x in xs for sg in (-1, 1)] + [(sg * xs[-1], y, 0) for y in ys[1:-1] for sg in (-1, 1)]
        s.tile(ball, pos)
    s += box(w, l, sub_t, (0, 0, ball_h - 0.02), 0.03, sub, base=True)
    s += box(w - 0.1, l - 0.1, h - ball_h - sub_t + 0.02, (0, 0, ball_h + sub_t - 0.02), 0.09, "ic_epoxy", base=True)
    z = h + EPS * 0.5
    rot = 90 if l > w * 1.1 else 0
    ww, ll = (l, w) if rot else (w, l)
    s += mark(list(lines), ww * 0.62, ll * 0.42, z, "laser_mark", rot=rot, hmax=hmax)
    if dot:
        s += annulus(0.28, 0.4, z, 14, "laser_mark_dim", (-w / 2 + 1.1, -l / 2 + 1.1))
    return s


def fcbga(w, l, z_top, die=(9.0, 8.0), sub_t=0.8, ball_h=0.32, sub="pcb_green", stiffener=None,
          z_stiff=None, caps=True, die_c=(0.0, 0.0), ball_pitch=1.0, lines=None):
    """Bare flip-chip BGA: perimeter balls, organic substrate, exposed silicon die (+ optional stiffener ring).

    stiffener = inner opening size (square) of a metal frame covering the substrate rim; z_stiff its top.
    """
    s = Soup()
    ball = lathe([(0, 0), (0.22, 0.03), (0.3, ball_h / 2), (0.22, ball_h), (0, ball_h)], 8, "solder")
    nx, ny = int((w - 1.2) / ball_pitch), int((l - 1.2) / ball_pitch)
    xs = (np.arange(nx + 1) - nx / 2) * ball_pitch
    ys = (np.arange(ny + 1) - ny / 2) * ball_pitch
    pos = [(x, sg * ys[-1], 0) for x in xs for sg in (-1, 1)] + [(sg * xs[-1], y, 0) for y in ys[1:-1] for sg in (-1, 1)]
    s.tile(ball, pos)
    zs = ball_h - 0.03 + sub_t
    s += box(w, l, sub_t, (0, 0, ball_h - 0.03), 0.05, sub, base=True)
    dx, dy = die
    cx, cy = die_c
    s += box(dx, dy, z_top - zs, (cx, cy, zs), 0.03, "silicon_die", base=True)
    s += loft([(rrect(dx + 1.3, dy + 1.3, 0.5, 3, (cx, cy)), zs + 0.005), (rrect(dx + 0.02, dy + 0.02, 0.02, 3, (cx, cy)), zs + (z_top - zs) * 0.55)],
              "underfill", cap0=False, cap1=False)
    inner = max(dx, dy) / 2 + 1.6
    lim = (stiffener / 2 - 0.8) if stiffener else (min(w, l) / 2 - 0.9)
    if caps and lim > inner + 0.6:
        c = chip(0.6, 0.3, 0.3, pad=False)
        pos, rots = [], []
        nrow = max(1, int((lim - inner) / 0.75))
        for r in range(min(nrow, 3)):
            off = inner + 0.3 + r * 0.8
            for t in np.arange(-inner + 0.6, inner - 0.5, 1.15):
                for sg in (-1, 1):
                    pos.append((cx + t, cy + sg * off, zs))
                    rots.append(90)
                    if r < 2:
                        pos.append((cx + sg * off, cy + t, zs))
                        rots.append(0)
        s.tile(c, pos, rots)
    if stiffener:
        zt = z_stiff if z_stiff is not None else zs + 0.7
        s += plate(rrect(w - 0.1, l - 0.1, 1.2, 4), [rrect(stiffener, stiffener, 1.0, 4)], zs + 0.04, zt, 0.1, "stiffener")
        s += plate(rrect(w + 0.3, l + 0.3, 1.3, 4), [rrect(w - 0.6, l - 0.6, 1.0, 4)], zs - 0.3, zs + 0.12, 0.04, "underfill")
    else:
        t = 1.6
        s += poly_face([(-w / 2 + 0.5, -l / 2 + 0.5), (-w / 2 + 0.5 + t, -l / 2 + 0.5), (-w / 2 + 0.5, -l / 2 + 0.5 + t)], zs + 0.006, "enig_gold")
    if lines:
        s += mark(list(lines), dx * 0.7, dy * 0.3, z_top + 0.004, "laser_mark_dim", c=(cx, cy - dy * 0.22))
    return s


# ----------------------------------------------------------------------------
# pin headers and board connectors
# ----------------------------------------------------------------------------
def gold_pin(h, t=0.64, z0=0.0):
    b = t * 0.3
    return prism(rrect(t, t, 0.0), z0, z0 + h, "gold_pin", bevel=b)


def pin_header(nx, ny=1, pitch=2.54, base_h=2.5, pin_h=8.5, shroud=False, missing=(), mat="plastic_black",
               wall_h=None):
    """Straight pin header; pins along x (nx) by y (ny). missing = set of (ix, iy) keyed positions."""
    s = Soup()
    W, L = nx * pitch, ny * pitch
    if shroud:
        wh = wall_h or 9.0
        outline = rrect(W + 2.6, L + 2.6, 0.5, 3)
        s += plate(outline, [rrect(W + 0.5, L + 0.5, 0.2, 2)], 1.6, wh, 0.12, mat)
        s += box(W + 2.6, L + 2.6, 1.7, (0, 0, 0), 0.1, mat, base=True)
        s += box(4.5, 0.8, wh - 4.0, (0, -(L / 2 + 0.9), 4.0), 0.0, "hole_dark", base=True)   # key notch look
    else:
        seg = prism(rrect(pitch - 0.08, pitch - 0.08, 0.45, 1), 0.0, base_h, mat, bevel=0.12)
        pos = [((i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch, 0) for i in range(nx) for j in range(ny)]
        s.tile(seg, pos)
    pin = gold_pin(pin_h - 0.2, 0.64 if pitch > 2.2 else 0.5, 0.2)
    pos = [((i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch, 0) for i in range(nx) for j in range(ny)
           if (i, j) not in missing]
    s.tile(pin, pos)
    return s


def fan_header(n=4, mat="plastic_black"):
    """KK-254 style fan header with the back friction ramp."""
    p = 2.54
    W = n * p
    s = box(W, 5.8, 3.2, (0, 0, 0), 0.15, mat, base=True)
    s += box((n - 1) * p - 0.2, 0.9, 8.0, (-p / 2 + 0.1, 2.4, 3.0), 0.12, mat, base=True)
    s.tile(gold_pin(11.0, 0.64, 0.2), [((i - (n - 1) / 2) * p, -0.4, 0) for i in range(n)])
    return s


def jst_header(n=4, pitch=2.0, mat="plastic_white", side_entry=False):
    """Small shrouded wire-to-board header (PH style). side_entry: opening towards -y."""
    W = (n - 1) * pitch + 4.0
    if side_entry:
        s = plate(rrect(W, 6.0, 0.25, 2), [rrect(W - 1.4, 4.6, 0.1, 1)], -3.8, 0.0, 0.08, mat)
        s.rot("x", -90).move(0, 0.9, 3.0)
        s += box(W, 1.0, 6.0, (0, 3.3, 0), 0.1, mat, base=True)
        s += box(W, 4.6, 0.7, (0, 1.0, 0), 0.0, mat, base=True)
        pin = box(0.5, 4.2, 0.5, (0, 0.9, 2.7), 0.1, "gold_pin")
        s.tile(pin, [((i - (n - 1) / 2) * pitch, 0, 0) for i in range(n)])
    else:
        s = plate(rrect(W, 4.6, 0.25, 2), [rrect(W - 1.2, 3.3, 0.1, 1)], 0.9, 6.0, 0.08, mat)
        s += box(W, 4.6, 1.0, (0, 0, 0), 0.08, mat, base=True)
        s.tile(gold_pin(4.6, 0.5, 0.4), [((i - (n - 1) / 2) * pitch, 0.2, 0) for i in range(n)])
    return s


def minifit_header(cols, mat="pa66_natural", h=12.8, pitch=4.2, latch=True):
    """Mini-Fit Jr style 2 x cols vertical power header (ATX-24: cols=12, EPS-8: cols=4)."""
    W = cols * pitch + 1.2
    D = 2 * pitch + 1.2
    holes = []
    for i in range(cols):
        for j in range(2):
            c = ((i - (cols - 1) / 2) * pitch, (j - 0.5) * pitch)
            sq = (i + j) % 2 == 0
            if sq:
                holes.append(rrect(3.45, 3.45, 0.15, 1, c))
            else:
                a = 3.45 / 2
                holes.append(np.array([(-a, -a), (a, -a), (a, a * 0.35), (a * 0.35, a), (-a * 0.35, a), (-a, a * 0.35)]) + np.array(c))
    s = plate(rrect(W, D, 0.5, 2), holes, 2.0, h, 0.12, mat)
    s += box(W, D, 2.1, (0, 0, 0), 0.1, mat, base=True)
    pin = cyl(0.55, 1.9, h - 3.2, 8, "tin", bevel=0.2)
    s.tile(pin, [((i - (cols - 1) / 2) * pitch, (j - 0.5) * pitch, 0) for i in range(cols) for j in range(2)])
    if latch:
        s += box(3.6, 1.5, 3.0, (0, -(D / 2 + 0.6), h - 6.5), 0.25, mat, base=True)
        s += prism([(-1.8, 0), (1.8, 0), (1.8, 1.5), (-1.8, 1.5)], h - 3.5, h - 2.2, mat).move(0, -(D / 2 + 1.35), 0)
    return s


def usb3_header(mat="plastic_black"):
    """Shrouded 2 x 10 (19-pin) 2.0 mm front-panel USB header."""
    s = plate(rrect(30.0, 6.4, 0.5, 2), [rrect(27.6, 4.4, 0.25, 2)], 1.6, 11.2, 0.12, mat)
    s += box(30.0, 6.4, 1.7, (0, 0, 0), 0.1, mat, base=True)
    s += box(27.0, 3.6, 0.9, (0, 0, 1.6), 0.0, "plastic_blue", base=True)
    pos = [((i - 4.5) * 2.0, (j - 0.5) * 2.0, 0) for i in range(10) for j in range(2) if not (i == 9 and j == 1)]
    s.tile(gold_pin(6.0, 0.5, 2.3), pos)
    s += box(4.6, 0.7, 6.0, (0, -3.0, 5.0), 0.0, "hole_dark", base=True)
    return s


def usbc_header(mat="plastic_black"):
    """Front-panel USB Type-E (key-A) header in a steel shell."""
    s = plate(rrect(15.2, 6.3, 1.0, 3), [rrect(13.4, 4.4, 0.7, 3)], 0.4, 7.6, 0.1, "steel_sheet")
    s += box(13.6, 4.6, 1.5, (0, 0, 0.3), 0.0, mat, base=True)
    s += box(11.6, 1.1, 5.2, (0, 0, 1.6), 0.15, mat, base=True)
    return s


def sata_port(mat="plastic_black"):
    """Vertical SATA data port."""
    s = plate(rrect(17.0, 6.6, 0.5, 2), [rrect(14.6, 4.6, 0.2, 1)], 1.2, 8.6, 0.12, mat)
    s += box(17.0, 6.6, 1.3, (0, 0, 0), 0.1, mat, base=True)
    s += box(10.0, 1.2, 6.0, (-1.4, -0.9, 1.2), 0.1, mat, base=True)       # L-shaped key
    s += box(1.2, 2.6, 6.0, (-6.0, -0.2, 1.2), 0.1, mat, base=True)
    s.tile(rect(0.55, 3.2, 0, "gold_pin").rot("x", 90).move(0, -0.28, 4.3), [(-5.0 + i * 1.27, 0, 0) for i in range(7)])
    return s


def coin_cell(d=20.0):
    """CR2032 in a flat SMD holder."""
    s = Soup()
    s += plate(circle(d / 2 + 1.6, 40), [circle(d / 2 + 0.15, 40)], 0.0, 3.4, 0.2, "plastic_black")
    s += box(d + 3.2, d + 3.2, 0.6, (0, 0, 0), 0.1, "plastic_black", base=True)
    s += lathe([(0, 0.6), (d / 2 - 0.8, 0.6), (d / 2, 0.9), (d / 2, 3.3), (d / 2 - 0.35, 3.75), (0, 3.75)], 48, "steel_bright")
    z = 3.75 + EPS
    s += mark(["CR2032", "3V"], d * 0.6, d * 0.3, z, "ihs_mark", c=(0, 3.0), hmax=2.2)
    s += mark(["+"], 4, 4, z, "ihs_mark", c=(0, -4.5), hmax=4.0)
    s += box(5.0, 3.0, 0.5, (0, d / 2 + 0.2, 3.9), 0.1, "steel_bright", base=True)      # spring clip
    s += box(5.0, 0.5, 4.3, (0, d / 2 + 1.85, 0.1), 0.1, "steel_bright", base=True)
    return s


def push_button(mat="plastic_black"):
    s = box(6.0, 6.0, 3.4, (0, 0, 0), 0.2, "steel_sheet", base=True)
    s += cyl(1.7, 3.3, 4.6, 16, mat, bevel=0.25)
    return s


def test_point(d=1.0):
    return disc(d / 2, 0.006, 12, "enig_gold")


def screw_head(d=5.5, h=1.6, kind="phillips", mat="screw_black", n=24):
    """Pan head screw head sitting on z = 0."""
    r = d / 2
    s = lathe([(0, 0), (r, 0), (r, h * 0.45), (r * 0.82, h * 0.9), (r * 0.55, h), (0, h)], n, mat)
    z = h + 0.004
    if kind == "phillips":
        s += rect(d * 0.5, d * 0.1, z, "hole_dark")
        s += rect(d * 0.1, d * 0.5, z, "hole_dark")
    elif kind == "torx":
        P = [(d * (0.26 if i % 2 == 0 else 0.15) * math.cos(i * TAU / 12), d * (0.26 if i % 2 == 0 else 0.15) * math.sin(i * TAU / 12)) for i in range(12)]
        s += poly_face(P, z, "hole_dark")
    elif kind == "hex":
        s += poly_face(circle(d * 0.27, 6), z, "hole_dark")
    return s


def tht_joints(pos, r=0.75, h=0.9):
    """Solder cones for through-hole pins seen from the BACK side (built pointing +z; caller flips)."""
    cone = lathe([(0, 0), (r, 0), (r * 0.55, h * 0.3), (0.3, h * 0.75), (0.22, h), (0, h)], 10, "solder")
    return Soup().tile(cone, pos)
