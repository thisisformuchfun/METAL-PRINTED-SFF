"""pcp_cooling - radiator, fan, DDC pump body, G1/4 fittings and soft tubing."""
import math

import numpy as np

import pcp_comps as C
from pcp_geom import (Soup, Part, proto, box, box2, cyl, lathe, prism, plate, rect, disc, annulus, strip,
                      text, rrect, circle, fillet_poly, loft, tube, tube_ring, poly_face, sweep, smooth_path,
                      path_frames, fillet_path, finish, TAU)


def _fix_closed(s):
    """Flip a closed shell so its signed volume is positive (outward normals)."""
    vol, k = 0.0, 0
    V = s.V
    for sz in s.S:
        idx = s.L[k:k + sz]
        for j in range(1, sz - 1):
            vol += float(np.dot(V[idx[0]], np.cross(V[idx[j]], V[idx[j + 1]])))
        k += sz
    if vol < 0:
        s.flip()
    return s


def thread_bore(r, depth, mat="black_nickel", n=20, pitch=1.337, axis="z"):
    """Internal thread look: bore opening at z = 0 going to -depth (profile travels downwards)."""
    prof = [(r + 0.9, 0.0), (r + 0.25, -0.5)]
    z = -0.5
    while z - pitch > -depth:
        prof += [(r - 0.38, z - pitch / 2), (r + 0.25, z - pitch)]
        z -= pitch
    prof += [(r + 0.25, -depth), (0, -depth)]
    return lathe(prof, n, mat, axis)


def thread_male(r, length, mat="black_nickel", n=20, pitch=1.337):
    """External thread from z = 0 down to -length."""
    prof = [(0, -length), (r - 0.9, -length), (r - 0.2, -length + 0.6)]
    z = -length + 0.6
    while z + pitch < -0.2:
        prof += [(r - 0.85, z + pitch / 2), (r - 0.2, z + pitch)]
        z += pitch
    prof += [(r - 0.2, 0.0), (0, 0.0)]
    return lathe(prof, n, mat)


# ----------------------------------------------------------------------------
# radiator
# ----------------------------------------------------------------------------
def _fin_row(length, hf, depth, pitch=1.45, slats=10):
    """One louvered fin strip (corrugated along x, height hf along z, depth along y), centred."""
    V, F = [], []
    m = 1.2                                    # un-louvered margin front/back
    yl = np.linspace(-depth / 2 + m, depth / 2 - m, slats + 1)
    ang = math.radians(27)

    def wall(xb, xt):
        """wall from bottom (xb, -hf/2) to top (xt, +hf/2)."""
        for ya, yb in ((-depth / 2, -depth / 2 + m), (depth / 2 - m, depth / 2)):
            i = len(V)
            V.extend([(xb, ya, -hf / 2), (xb, yb, -hf / 2), (xt, yb, hf / 2), (xt, ya, hf / 2)])
            F.append((i, i + 1, i + 2, i + 3))
        for k in range(slats):
            yc, hw = (yl[k] + yl[k + 1]) / 2, (yl[k + 1] - yl[k]) / 2 * 0.96
            sg = 1 if k < slats // 2 else -1
            dx, dy = sg * hw * math.sin(ang), hw * math.cos(ang)
            i = len(V)
            zb, zt = -hf / 2 + 0.5, hf / 2 - 0.5
            xb2 = xb + (xt - xb) * 0.5 / hf
            xt2 = xt - (xt - xb) * 0.5 / hf
            V.extend([(xb2 - dx, yc - dy, zb), (xb2 + dx, yc + dy, zb), (xt2 + dx, yc + dy, zt), (xt2 - dx, yc - dy, zt)])
            F.append((i, i + 1, i + 2, i + 3))

    def crest(x0, x1, z):
        i = len(V)
        V.extend([(x0, -depth / 2, z), (x1, -depth / 2, z), (x1, depth / 2, z), (x0, depth / 2, z)])
        F.append((i, i + 1, i + 2, i + 3))

    p = pitch
    wall(0.0, 0.14 * p)
    crest(0.14 * p, 0.5 * p, hf / 2)
    wall(0.64 * p, 0.5 * p)
    crest(0.64 * p, p, -hf / 2)
    one = Soup().add(V, np.array(F), "paint_fin_black")
    n = int(length / p)
    s = Soup().tile(one, [(-(n * p) / 2 + i * p, 0, 0) for i in range(n)])
    return s


def radiator_body(Lx, Ty, Hz, tank, port_dz, fan_pitch, hole, port_r=5.9):
    """Tanks + rails + flanges (everything except fins/tubes). Origin at the core centre."""
    s = Soup()
    mat = "paint_satin_black"
    xt = Lx / 2 - tank / 2
    out = rrect(tank, Hz, 3.0, 4)
    bev = 1.2
    # +X tank: front slab with the two port holes, body behind it
    slab = plate(out, [circle(port_r + 0.9, 28, (0, sg * port_dz)) for sg in (-1, 1)], -bev, bev, bev, mat)
    slab.rot("x", 90).move(xt, -Ty / 2 + bev, 0)
    s += slab
    body = prism(out, 0.0, Ty - bev, mat, bevel=bev, cap_bottom=False).rot("x", -90)
    body.move(xt, -Ty / 2 + bev, 0)
    s += body
    for sg in (-1, 1):
        b = thread_bore(port_r, 8.5, "black_nickel", 24).rot("x", 90).move(xt, -Ty / 2 + 0.02, sg * port_dz)
        s += b
        s += annulus(port_r + 0.9, min(10.3, tank / 2 - 0.5), 0.0, 32, "black_nickel").rot("x", 90).move(xt, -Ty / 2 - 0.015, sg * port_dz)
    # -X tank
    s += prism(out, -Ty / 2, Ty / 2, mat, bevel=bev, bevel_bottom=True).rot("x", -90).move(-xt, 0, 0)
    # top / bottom rails and the four face flanges with threaded fan holes
    x0, x1 = -Lx / 2 + tank - 0.3, Lx / 2 - tank + 0.3
    fw = 11.5
    hx = [fx + sg * hole / 2 for fx in (-fan_pitch / 2, fan_pitch / 2) for sg in (-1, 1)]
    for sz in (-1, 1):
        s += box2(x0, -Ty / 2 + 0.6, sz * Hz / 2 - (1.6 if sz > 0 else 0), x1, Ty / 2 - 0.6, sz * Hz / 2 + (1.6 if sz < 0 else 0), 0.3, mat)
        for sy in (-1, 1):
            holes = [circle(1.45, 12, (x, sz * hole / 2)) for x in hx]
            fl = plate(np.array([(x0, sz * Hz / 2 - sz * fw), (x1, sz * Hz / 2 - sz * fw), (x1, sz * Hz / 2), (x0, sz * Hz / 2)]),
                       holes, -0.6, 0.6, 0.25, mat)
            fl.rot("x", 90).move(0, sy * (Ty / 2 - 0.6), 0)
            s += fl
            for x in hx:
                nut = tube_ring(2.6, 1.3, 0.0, 3.2, 12, "black_nickel", axis="y")
                s += nut.move(x, (-Ty / 2 + 1.1) if sy < 0 else (Ty / 2 - 4.3), sz * hole / 2)
    return s


def build_radiator_240(L, name="radiator_240"):
    """240 mm slim radiator. Origin at the core centre, X = length, Y = thickness, Z = height.
    Ports: +X end tank, -Y face, z = +-port_dz (seat plane y = -thickness/2)."""
    R = L["rad"]
    Lx, Ty, Hz = R["core"]
    tank, pdz = R["tank_len"], R["port_dz"]
    fan_pitch, hole = R["fan"][0], R["fan_hole"]
    P = Part(name)
    key = "rad_%g_%g_%g_%g_%g" % (Lx, Ty, Hz, tank, pdz)
    P.obj("body", proto(key + "_body", lambda: radiator_body(Lx, Ty, Hz, tank, pdz, fan_pitch, hole)))
    core_h = Hz - 3.4
    n_tubes = 12
    tt = 1.8
    hf = (core_h - n_tubes * tt) / (n_tubes + 1)
    depth = Ty - 6.0
    core_l = Lx - 2 * tank
    fin = proto(key + "_fin", lambda: _fin_row(core_l, hf, depth), smooth=None)
    tb = proto(key + "_tube", lambda: prism(rrect(depth - 1.5, tt, tt / 2 - 0.02, 3), -core_l / 2 - 0.5, core_l / 2 + 0.5, "paint_fin_black").rot("y", 90))
    z = -core_h / 2
    for i in range(n_tubes + 1):
        P.obj("fin.%02d" % i, fin, (0, 0, z + hf / 2))
        z += hf
        if i < n_tubes:
            P.obj("tube.%02d" % i, tb, (0, 0, z + tt / 2))
            z += tt
    xt = Lx / 2 - tank / 2
    for i, sg in enumerate((1, -1)):
        e = P.empty("port.%d" % i, (xt, -Ty / 2, sg * pdz), (90, 0, 0))     # +Z of the empty = out of the port (-Y)
    P.root["ports"] = [[xt, -Ty / 2, pdz], [xt, -Ty / 2, -pdz]]
    P.root["port_dir"] = [0.0, -1.0, 0.0]
    return P.root


# ----------------------------------------------------------------------------
# fan
# ----------------------------------------------------------------------------
def fan_blades(n_blades=9, r_hub=20.5, r_tip=56.4, yc=-2.2, c_hub=25.0, c_tip=41.0, b_hub=47.0, b_tip=25.0,
               sweep_deg=30.0, t_hub=2.3, t_tip=1.3, camber=0.07, nr=12, nc=10, mat="fan_blade"):
    """Swept, cambered blades around the Y axis. Leading edge at -Y, rotor turns towards +theta
    (counter-clockwise seen from the intake side, -Y), pushing air to +Y."""
    one = Soup()
    V = []
    ring = 2 * nc
    for i in range(nr + 1):
        u = i / nr
        r = r_hub - 1.0 + (r_tip - r_hub + 1.0) * u
        c = c_hub + (c_tip - c_hub) * u ** 0.8
        if u > 0.9:                                   # rounded tip
            c *= 1.0 - 0.35 * ((u - 0.9) / 0.1) ** 2
        b = math.radians(b_hub + (b_tip - b_hub) * u ** 0.75)
        th0 = math.radians(sweep_deg) * u ** 1.6
        tmax = t_hub + (t_tip - t_hub) * u
        pts = []
        for k in range(ring):
            if k <= nc:
                sp, side = k / nc, 1.0
            else:
                sp, side = (ring - k) / nc, -1.0
            sp2 = 0.5 - 0.5 * math.cos(sp * math.pi)                  # cluster points at LE / TE
            th = 0.12 + 0.88 * (4 * sp2 * (1 - sp2)) ** 0.6           # thickness distribution
            cam = -camber * c * (1 - (2 * sp2 - 1) ** 2)
            off = cam + side * tmax * th / 2
            tang = (0.5 - sp2) * c * math.cos(b) + off * math.sin(b)
            axial = (sp2 - 0.5) * c * math.sin(b) + off * math.cos(b)
            a = th0 + tang / r
            pts.append((r * math.cos(a), yc + axial, r * math.sin(a)))
        V.append(pts)
    V = np.array(V).reshape(-1, 3)
    F = []
    for i in range(nr):
        for k in range(ring):
            k2 = (k + 1) % ring
            F.append((i * ring + k, i * ring + k2, (i + 1) * ring + k2, (i + 1) * ring + k))
    F.append(tuple(range(ring)))
    F.append(tuple(nr * ring + k for k in range(ring - 1, -1, -1)))
    one.add(V, F, mat)
    _fix_closed(one)
    s = Soup()
    for j in range(n_blades):
        s += one.copy().rot("y", -360.0 * j / n_blades)
    return s


def fan_rotor():
    s = fan_blades()
    hub = lathe([(0, -11.6), (17.5, -11.6), (19.6, -10.9), (20.5, -9.0), (20.5, 6.0), (19.0, 6.6), (0, 6.6)], 48, "fan_plastic", axis="y")
    s += hub.flip() if False else hub
    st = Soup()
    st += disc(15.5, 0.0, 48, "sticker_black")
    st += annulus(13.6, 14.2, 0.004, 48, "label_silver")
    st += annulus(4.0, 4.3, 0.004, 24, "label_silver")
    s += st.rot("x", 90).move(0, -11.62, 0)
    return s


def fan_frame(size=120.0, thick=25.0, hole=105.0):
    """Static frame: two flanges, duct ring, corner posts, exhaust-side struts + stator hub, rubber pads."""
    s = Soup()
    mat = "fan_plastic"
    h = size / 2
    pad = 0.3
    yf = thick / 2 - pad
    ft = 3.6
    rd = 57.6
    holes = [circle(rd, 96)] + [circle(2.2, 16, (sx * hole / 2, sz * hole / 2)) for sx in (-1, 1) for sz in (-1, 1)]
    fl = plate(rrect(size, size, 7.0, 6), holes, -ft / 2, ft / 2, 0.5, mat).rot("x", 90)
    s += fl.copy().move(0, -yf + ft / 2, 0)
    s += fl.copy().move(0, yf - ft / 2, 0)
    # duct ring with a rounded inlet lip (profile travels: inner wall from exhaust to intake, then outer back)
    prof = [(rd, yf - 0.4), (rd, -yf + 2.6), (rd + 0.5, -yf + 1.0), (rd + 1.3, -yf + 0.3), (rd + 1.9, -yf + 0.3),
            (rd + 1.9, yf - 0.4)]
    s += lathe(prof, 96, mat, axis="y")
    for sx in (-1, 1):
        for sz in (-1, 1):
            cx, cz = sx * hole / 2, sz * hole / 2
            s += tube_ring(4.4, 2.2, -yf + ft - 0.2, yf - ft + 0.2, 20, mat, axis="y").move(cx, 0, cz)
            pd = plate(rrect(15.0, 15.0, 4.5, 4), [circle(2.5, 16)], 0.0, pad + 0.25, 0.2, "rubber_grey").rot("x", 90)
            px, pz = sx * (h - 8.2), sz * (h - 8.2)
            s += pd.copy().move(px, -yf + 0.25, pz)
            s += pd.copy().move(px, yf + pad, pz)
    # stator hub + curved struts on the exhaust side
    ys0, ys1 = yf - 4.6, yf - 0.2
    s += lathe([(0, ys0), (19.3, ys0), (19.8, ys0 + 0.6), (19.8, ys1 - 0.5), (19.3, ys1), (0, ys1)], 48, mat, axis="y").flip()
    st = Soup()
    st += disc(16.5, 0.0, 48, "sticker_black")
    st += annulus(14.8, 15.4, 0.004, 48, "label_silver")
    st += C.mark(["DC 12V  0.20A", "FAN 120 PWM  2431"], 20.0, 6.0, 0.005, "ink_grey", c=(0, -2.0), hmax=1.6)
    s += st.rot("x", -90).move(0, ys1 + 0.02, 0)
    sec = rrect(3.6, 3.2, 1.0, 2)
    for k in range(4):
        a0 = math.radians(45 + k * 90)
        pts = []
        for u in np.linspace(0, 1, 9):
            r = 19.0 + (rd + 1.0 - 19.0) * u
            a = a0 - math.radians(34) * u ** 1.3
            pts.append((r * math.cos(a), (ys0 + ys1) / 2 + 0.2, r * math.sin(a)))
        wide = 1.9 if k == 3 else 1.0
        s += sweep(np.array(pts), sec * np.array([1.0, wide]), mat, caps=True, up=(0, 1, 0))
    return s


def build_fan_120(L=None, name="fan_120"):
    """120 x 25 fan. Origin at the centre, rotation axis = Y, air moves towards +Y.
    The rotor is the object '<root>_rotor' (spin it about its local Y; negative angles = real direction)."""
    size, thick, hole = 120.0, 25.0, 105.0
    if L is not None:
        size, thick = L["rad"]["fan"]
        hole = L["rad"]["fan_hole"]
    P = Part(name)
    P.obj("frame", proto("fan_frame_%g_%g_%g" % (size, thick, hole), lambda: fan_frame(size, thick, hole)))
    rot = P.obj("rotor", proto("fan_rotor", fan_rotor, smooth=50))
    P.root["rotor"] = rot.name
    return P.root


# ----------------------------------------------------------------------------
# DDC pump body (no top)
# ----------------------------------------------------------------------------
def ddc_body(w=62.0, h=38.0, chamber_d=37.0, pattern=50.0):
    s = Soup()
    mat = "plastic_black"
    rc = chamber_d / 2
    hp = pattern / 2
    screws = [circle(2.25, 16, (sx * hp, sy * hp)) for sx in (-1, 1) for sy in (-1, 1)]
    out = rrect(w, w, 7.0, 6)
    g0, g1 = rc + 1.9, rc + 4.3
    s += plate(out, [circle(g1, 64), circle(rc, 64)] + screws, -1.7, 0.0, 0.35, mat)
    s += plate(circle(g0, 64), [circle(rc, 64)], -1.7, 0.0, 0.15, mat)
    s += plate(out, [circle(rc, 64)] + screws, -13.0, -1.7, 0.35, mat)
    s += lathe([(g0 + 0.25, -0.85), (g0 + 0.5, -0.25), (g0 + 1.2, 0.0), (g1 - 1.2, 0.0), (g1 - 0.5, -0.25), (g1 - 0.25, -0.85),
                (g1 - 0.5, -1.45), (g0 + 0.5, -1.45), (g0 + 0.25, -0.85)], 64, "oring")
    # chamber floor with the bearing post
    s += lathe([(rc, -10.6), (6.0, -10.6), (6.0, -11.4), (0, -11.4)], 64, mat).flip()
    for sx in (-1, 1):
        for sy in (-1, 1):
            s += tube_ring(3.6, 2.25, -4.0, -0.02, 16, "brass", bevel=0.2).move(sx * hp, sy * hp, 0)
            s += cyl(6.0, -h + 2.0, -13.0, 20, mat, bevel=0.6, bevel_bottom=0.6).move(sx * hp, sy * hp, 0)
            s += C.screw_head(5.5, 1.5, "hex", "screw_black").rot("x", 180).move(sx * hp, sy * hp, -h + 2.0)
    s += prism(rrect(w - 9.0, w - 9.0, 9.0, 6), -h + 2.0, -13.0, mat, bevel=0.8, bevel_bottom=True)
    s += prism(rrect(w - 14.0, w - 14.0, 8.0, 6), -h, -h + 2.0, "screw_black", bevel=0.5, bevel_bottom=True)
    lab = Soup()
    lab += poly_face(rrect(30.0, 18.0, 2.0, 3), 0.0, "label_silver")
    lab += C.mark(["DC PUMP 12V", "18W  PWM  4-PIN", "SN 2431-00417"], 26.0, 12.0, 0.005, "ink_black")
    s += lab.rot("x", 180).move(0, 0, -h - 0.02)
    # cable gland on the -Y side
    s += cyl(3.4, 0.0, 3.0, 16, "rubber", bevel=0.5, axis="y").rot("z", 180).move(8.0, -(w - 9.0) / 2, -h + 9.0)
    return s


def ddc_impeller(r=16.6):
    s = Soup()
    mat = "plastic_white"
    s += lathe([(0, -10.3), (r, -10.3), (r, -8.6), (7.0, -8.2), (5.5, -5.0), (4.2, -3.6), (2.6, -3.6), (2.6, -9.0), (0, -9.0)], 48, mat)
    s += lathe([(0, -9.0), (1.9, -9.0), (1.9, -3.2), (1.3, -2.4), (0, -2.2)], 16, "ceramic")
    sec = np.array([(-0.55, 0.0), (0.55, 0.0), (0.55, 5.6), (-0.55, 5.6)])
    for k in range(7):
        a0 = k * TAU / 7
        pts = []
        for u in np.linspace(0, 1, 9):
            rr = 6.2 + (r - 0.4 - 6.2) * u
            a = a0 - math.radians(62) * u
            pts.append((rr * math.cos(a), rr * math.sin(a), -8.5))
        T, N, B = path_frames(np.array(pts), up=(0, 0, 1))
        s += sweep(np.array(pts), sec[:, ::-1] * np.array([1, 1]), mat, caps=True, frames=(T, N, B))
    s += tube_ring(r, r - 3.2, -3.4, -2.6, 48, mat, bevel=0.2)
    return s


def build_ddc_pump(L=None, name="ddc_pump"):
    """DDC-class pump body without a top. Origin = centre of the mating (top) face, +Z out of that face."""
    w, _, h = (62.0, 62.0, 38.0) if L is None else L["heart"]["pump_body"]
    P = Part(name)
    P.obj("body", proto("ddc_body_%g_%g" % (w, h), lambda: ddc_body(w, h)))
    imp = P.obj("impeller", proto("ddc_impeller", ddc_impeller))
    # cable tail
    y0 = -(w - 9.0) / 2 - 2.0
    pts = [(8.0, y0, -h + 9.0), (8.0, y0 - 12.0, -h + 9.0), (9.0, y0 - 26.0, -h + 5.0), (13.0, y0 - 44.0, -h + 3.0), (20.0, y0 - 62.0, -h + 2.2)]
    path = smooth_path(pts, 1.5)
    tail = tube(path, 2.3, 12, "sleeve_black", uv=True)
    tail += cyl(2.6, 0.0, 6.0, 12, "rubber", bevel=0.4, axis="y").rot("z", 180).move(8.0, y0 - 0.5, -h + 9.0)
    P.obj("cable", tail.mesh(P.name + "_cable", smooth=50, weighted=False))
    P.root["impeller"] = imp.name
    return P.root


# ----------------------------------------------------------------------------
# fittings
# ----------------------------------------------------------------------------
def knurl_ring(r, z0, z1, n=44, depth=0.34, mat="black_nickel"):
    a = np.arange(2 * n) * TAU / (2 * n)
    rr = np.where(np.arange(2 * n) % 2 == 0, r, r - depth)
    P = np.stack([rr * np.cos(a), rr * np.sin(a)], 1)
    return prism(P, z0, z1, mat)


def _compression(mat="black_nickel", od=13.0, idd=10.0):
    """Compression stage along +z starting at z = 0 (shoulder); returns (soup, exit_z)."""
    s = Soup()
    rc = od / 2 + 3.3
    s += lathe([(idd / 2 - 1.2, 0.0), (idd / 2 - 1.2, 15.0), (idd / 2 - 0.4, 15.0), (idd / 2 + 0.25, 13.6), (idd / 2, 13.4),
                (idd / 2, 9.4), (idd / 2 + 0.25, 9.2), (idd / 2, 9.0), (idd / 2, 0.0)][::-1], 28, mat)
    s += lathe([(od / 2 + 0.2, 0.6), (rc - 0.5, 0.6), (rc, 1.1), (rc, 2.4)], 44, mat)
    s += knurl_ring(rc, 2.4, 6.6, mat=mat)
    s += lathe([(rc, 6.6), (rc - 0.5, 6.9), (rc - 0.5, 7.5), (rc, 7.8)], 44, mat)
    s += knurl_ring(rc, 7.8, 12.0, mat=mat)
    s += lathe([(rc, 12.0), (rc, 13.4), (rc - 1.6, 15.4), (od / 2 + 0.9, 15.4), (od / 2 + 0.35, 14.9), (od / 2 + 0.35, 0.6)], 44, mat)
    return s, 15.4


def fitting(kind="straight", od=13.0, idd=10.0):
    """Returns (soup, exit_point, exit_dir, tube_seat_point)."""
    mat = "black_nickel"
    s = Soup()
    s += thread_male(6.58, 5.0, mat, 24)
    s += lathe([(7.4, -0.75), (8.0, -1.35), (8.6, -0.75), (8.0, -0.02), (7.4, -0.75)], 28, "oring")
    if kind == "plug":
        s += lathe([(0, 0), (9.2, 0), (9.6, 0.4), (9.6, 0.9)], 44, mat)
        s += knurl_ring(9.6, 0.9, 3.4, 40, mat=mat)
        s += lathe([(9.6, 3.4), (9.0, 4.0), (4.2, 4.0)], 44, mat)
        s += lathe([(4.2, 4.0), (3.9, 1.6), (0, 1.6)], 6, mat, phase=math.pi / 6)
        return s, None, None, None
    if kind == "straight":
        s += lathe([(0, 0), (9.4, 0), (9.9, 0.5), (9.9, 2.6), (9.4, 3.1), (6.0, 3.1)], 44, mat)
        s += prism(circle(9.2, 6), 3.1, 5.0, mat, bevel=0.3)
        c, ez = _compression(mat, od, idd)
        s += c.move(0, 0, 5.0)
        return s, (0.0, 0.0, 5.0 + ez), (0.0, 0.0, 1.0), (0.0, 0.0, 5.6)
    if kind == "elbow90":
        zc = 15.0
        s += lathe([(0, 0), (9.4, 0), (9.9, 0.5), (9.9, 3.0), (9.4, 3.5), (8.3, 3.5), (8.3, 4.1), (8.9, 4.1), (9.3, 4.5), (9.3, 6.0),
                    (8.6, 6.6)], 44, mat)
        s += lathe([(8.6, 6.6), (8.6, zc), (8.4, zc + 2.4), (7.7, zc + 4.7), (6.4, zc + 6.5), (4.6, zc + 7.8), (2.4, zc + 8.5), (0, zc + 8.6)], 40, mat)
        arm = lathe([(8.6, 0.0), (8.6, 9.5), (8.0, 10.1), (8.0, 10.7), (8.9, 10.7), (9.3, 11.1), (9.3, 12.6), (8.8, 13.1), (6.0, 13.1)], 40, mat)
        c, ez = _compression(mat, od, idd)
        arm += c.move(0, 0, 13.1)
        s += arm.rot("y", 90).move(0, 0, zc)
        return s, (13.1 + ez, 0.0, zc), (1.0, 0.0, 0.0), (13.7, 0.0, zc)
    raise ValueError("fitting kind must be straight / elbow90 / plug")


def build_fitting(kind="straight", name=None, od=13.0, idd=10.0):
    """G1/4 fitting. Origin = centre of the port seat plane, +Z away from the part it screws into.
    Returns (root, exit_point, exit_dir) in the fitting's local frame (None, None for a plug)."""
    P = Part(name or ("fitting_" + kind))
    res = {}

    def mk():
        s, p, d, seat = fitting(kind, od, idd)
        return s

    _, p, d, seat = fitting(kind, od, idd) if kind != "_" else (None, None, None, None)
    P.obj("body", proto("fitting_%s_%g_%g" % (kind, od, idd), mk))
    if p is not None:
        P.root["tube_exit"] = list(p)
        P.root["tube_dir"] = list(d)
        P.root["tube_seat"] = list(seat)
    return P.root, p, d


# ----------------------------------------------------------------------------
# tubing
# ----------------------------------------------------------------------------
def build_tube(points, od=13.0, id=10.0, coolant=True, name="tube", liquid="coolant_ruby", step=None):
    """Soft tube through WORLD points (Catmull-Rom). Objects: <root>_wall (clear PVC, inner + outer
    surface, annular end caps) and <root>_liquid (slightly smaller core)."""
    P = Part(name)
    path = smooth_path(points, step or max(1.0, od / 6.0))
    fr = path_frames(path)
    n = 28
    wall = sweep(path, circle(od / 2, n), "pvc_clear", caps=False, frames=fr)
    inner = sweep(path, circle(id / 2, n), "pvc_clear", caps=False, frames=fr).flip()
    k = len(path)
    wall += inner
    nv = k * n
    for end, base in ((0, 0), (1, (k - 1) * n)):
        F = []
        for j in range(n):
            j2 = (j + 1) % n
            q = (base + j, base + j2, nv + base + j2, nv + base + j)
            F.append(q if end == 1 else q[::-1])
        wall.add(wall.V[:0], [], "pvc_clear")
        wall.L = np.concatenate([wall.L, np.array(F).reshape(-1)])
        wall.S = np.concatenate([wall.S, np.full(len(F), 4, np.int32)])
        wall.M = np.concatenate([wall.M, np.zeros(len(F), np.int32)])
    P.obj("wall", wall.mesh(P.name + "_wall", smooth=60, weighted=False))
    if coolant:
        core = sweep(path, circle(id / 2 - 0.06, 24), liquid, caps=True, frames=fr)
        P.obj("liquid", core.mesh(P.name + "_liquid", smooth=60, weighted=False))
    P.root["length_mm"] = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return P.root
