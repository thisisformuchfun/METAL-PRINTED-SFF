"""pcp_conn - I/O connectors, PCB plate helper and placement bookkeeping shared by the board and GPU.

I/O connector soups use a "port" frame: x = 0 is the outside (mating) face, the body extends
towards +x, y is centred, z = 0 is the PCB top surface.
"""
import math

import numpy as np

from pcp_geom import (Soup, box, box2, cyl, lathe, prism, plate, rect, disc, annulus, strip, text, rrect,
                      circle, tube_ring, poly_face, TAU)
from pcp_comps import EPS, mark, gold_pin, chip, resistor


# ----------------------------------------------------------------------------
# PCB plate + occupancy bookkeeping
# ----------------------------------------------------------------------------
def pcb_plate(outline, holes, t, mat="pcb_black", edge="pcb_edge", bevel=0.06):
    """PCB body: top surface at z = 0, thickness t downwards; side walls get the `edge` material."""
    s = plate(outline, holes, -t, 0.0, bevel, mat)
    nf = len(s.S)
    starts = np.zeros(nf, np.int64)
    starts[1:] = np.cumsum(s.S)[:-1]
    a, b, c = s.V[s.L[starts]], s.V[s.L[starts + 1]], s.V[s.L[starts + 2]]
    n = np.cross(b - a, c - a)
    nz = np.abs(n[:, 2]) / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    s.mats = [mat, edge]
    s.M = np.where(nz < 0.5, 1, 0).astype(np.int32)
    return s


class Occ:
    """Coarse 2D occupancy grid for placing components without overlaps."""

    def __init__(self, x0, y0, x1, y1, res=0.2):
        self.x0, self.y0, self.res = x0, y0, res
        self.nx = int(math.ceil((x1 - x0) / res))
        self.ny = int(math.ceil((y1 - y0) / res))
        self.g = np.zeros((self.nx, self.ny), bool)

    def _ix(self, x0, y0, x1, y1):
        i0 = max(0, int(math.floor((x0 - self.x0) / self.res)))
        i1 = min(self.nx, int(math.ceil((x1 - self.x0) / self.res)))
        j0 = max(0, int(math.floor((y0 - self.y0) / self.res)))
        j1 = min(self.ny, int(math.ceil((y1 - self.y0) / self.res)))
        return i0, i1, j0, j1

    def mark(self, x0, y0, x1, y1, m=0.0):
        i0, i1, j0, j1 = self._ix(x0 - m, y0 - m, x1 + m, y1 + m)
        self.g[i0:i1, j0:j1] = True

    def mark_c(self, cx, cy, w, h, m=0.0):
        self.mark(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, m)

    def mark_disc(self, cx, cy, r):
        self.mark(cx - r, cy - r, cx + r, cy + r)

    def free(self, x0, y0, x1, y1):
        if x0 < self.x0 or y0 < self.y0 or x1 > self.x0 + self.nx * self.res or y1 > self.y0 + self.ny * self.res:
            return False
        i0, i1, j0, j1 = self._ix(x0, y0, x1, y1)
        return not self.g[i0:i1, j0:j1].any()

    def take(self, cx, cy, w, h, m=0.12):
        """Reserve a w x h rectangle centred on (cx, cy) if it is free."""
        if self.free(cx - w / 2 - m, cy - h / 2 - m, cx + w / 2 + m, cy + h / 2 + m):
            self.mark(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            return True
        return False

    def copy(self):
        o = Occ.__new__(Occ)
        o.__dict__.update(self.__dict__)
        o.g = self.g.copy()
        return o


class Clutter:
    """Collects merged small-part placements: kind -> list of (x, y, z, rot)."""

    def __init__(self, protos):
        self.protos = protos          # name -> (soup, footprint_w, footprint_h, height)
        self.items = {k: [] for k in protos}

    def put(self, kind, x, y, rot=0.0, z=0.0):
        self.items[kind].append((x, y, z, rot))

    def count(self):
        return sum(len(v) for v in self.items.values())

    def soup(self, flip=False):
        s = Soup()
        for k, lst in self.items.items():
            if not lst:
                continue
            a = np.array(lst, float)
            s.tile(self.protos[k][0], a[:, :3], a[:, 3])
        return s


def smd_protos():
    """Low-poly merged-clutter prototypes: name -> (soup, w, h, height)."""
    return {
        "c0402": (chip(1.0, 0.5, 0.5), 1.5, 0.6, 0.5),
        "c0402g": (chip(1.0, 0.5, 0.5, body="mlcc_grey"), 1.5, 0.6, 0.5),
        "c0603": (chip(1.6, 0.8, 0.8), 2.3, 0.95, 0.8),
        "c0805": (chip(2.0, 1.25, 0.95), 2.8, 1.4, 0.95),
        "r0402": (chip(1.0, 0.5, 0.35, body="res_black", tl=0.2), 1.5, 0.6, 0.35),
        "r0603": (resistor(1.6, 0.8, 0.45, None), 2.3, 0.95, 0.45),
        "c0201": (chip(0.6, 0.3, 0.3, pad=False), 0.9, 0.4, 0.3),
    }


# ----------------------------------------------------------------------------
# rear I/O connectors (port frame)
# ----------------------------------------------------------------------------
def shell_block(w, h, d, holes, mat="steel_sheet", r=0.5, z0=0.0, bev=0.18):
    """Block w (y) x h (z) x d (x) with tunnels; holes are polygons in (y, z)."""
    out = rrect(w, h, r, 3, (0, z0 + h / 2))
    s = plate(out, holes, 0.0, d, bev, mat)
    s.V = s.V[:, [2, 0, 1]]
    return s


def _contacts(n, pitch, y0, z, x0, length, wdt=None, mat="gold_pin"):
    wdt = wdt or pitch * 0.55
    one = box(length, wdt, 0.12, (x0 + length / 2, 0, z), 0.0, mat)
    return Soup().tile(one, [(0, y0 + i * pitch, 0) for i in range(n)])


USB_A_HOLE = (12.4, 5.0)


def usb_a_stack(n=2, color="plastic_blue", d=17.4):
    w, pitch = 13.4, 7.3
    h = 1.2 + n * pitch + 0.3
    zc = [1.5 + USB_A_HOLE[1] / 2 + i * pitch for i in range(n)]
    s = shell_block(w, h, d, [rrect(USB_A_HOLE[0], USB_A_HOLE[1], 0.35, 2, (0, z)) for z in zc], z0=0.15)
    for z in zc:
        s += box(9.0, 11.4, 1.85, (0.9 + 4.5, 0, z - 0.75), 0.15, color)
        s += _contacts(4, 2.3, -3.45, z - 0.75 + 0.95, 1.6, 6.0, 1.0)
        s += box(0.6, USB_A_HOLE[0] + 0.2, USB_A_HOLE[1] + 0.2, (10.2, 0, z), 0.0, color)
    # stamped side dimples / seam
    s += box(6.0, 0.25, h * 0.55, (d * 0.55, w / 2 + 0.02, h * 0.5), 0.1, "steel_sheet")
    s += box(6.0, 0.25, h * 0.55, (d * 0.55, -w / 2 - 0.02, h * 0.5), 0.1, "steel_sheet")
    return s


def _hdmi_hole(c):
    return np.array([(-7.0, 2.275), (-7.0, -0.6), (-5.2, -2.275), (5.2, -2.275), (7.0, -0.6), (7.0, 2.275)]) + np.array(c)


def _dp_hole(c):
    return np.array([(-8.05, -2.38), (6.3, -2.38), (8.05, -0.7), (8.05, 2.38), (-8.05, 2.38)]) + np.array(c)


def hdmi_port(d=12.0, z0=0.6):
    """Single HDMI type-A receptacle lying on the PCB."""
    h = 5.85
    s = shell_block(15.3, h, d, [_hdmi_hole((0, z0 + h / 2))], z0=z0, r=0.8)
    zc = z0 + h / 2
    s += box(6.5, 11.0, 1.0, (0.8 + 3.25, 0, zc + 0.3), 0.1, "plastic_black")
    s += _contacts(10, 1.0, -4.5, zc + 0.3 + 0.52, 1.2, 4.6, 0.45)
    s += box(0.6, 14.4, 5.0, (7.6, 0, zc), 0.0, "plastic_black")
    for sy in (-1, 1):                                    # mounting legs
        s += box(2.2, 0.35, z0 + 2.0, (2.5, sy * 7.75, 0), 0.0, "steel_sheet", base=True)
        s += box(2.2, 0.35, z0 + 2.0, (d - 2.5, sy * 7.75, 0), 0.0, "steel_sheet", base=True)
    return s


def dp_port(d=13.5, z0=0.6):
    """Single DisplayPort receptacle lying on the PCB."""
    h = 6.0
    s = shell_block(17.4, h, d, [_dp_hole((0, z0 + h / 2))], z0=z0, r=0.8)
    zc = z0 + h / 2
    s += box(7.0, 13.4, 1.0, (0.8 + 3.5, 0, zc), 0.1, "plastic_black")
    s += _contacts(10, 1.0, -4.5, zc + 0.52, 1.2, 5.0, 0.45)
    s += box(0.6, 16.4, 5.0, (8.2, 0, zc), 0.0, "plastic_black")
    for sy in (-1, 1):
        s += box(2.2, 0.35, z0 + 2.0, (2.6, sy * 8.8, 0), 0.0, "steel_sheet", base=True)
        s += box(2.2, 0.35, z0 + 2.0, (d - 2.6, sy * 8.8, 0), 0.0, "steel_sheet", base=True)
    return s


def hdmi_dp_stack(d=18.0):
    """HDMI over DisplayPort stacked rear-I/O connector."""
    w, h = 18.6, 17.6
    z1, z2 = 4.4, 12.6
    s = shell_block(w, h, d, [_dp_hole((0, z1)), _hdmi_hole((0, z2))], z0=0.2, r=0.7)
    s += box(7.0, 13.4, 1.0, (4.3, 0, z1), 0.1, "plastic_black")
    s += _contacts(10, 1.0, -4.5, z1 + 0.52, 1.2, 5.0, 0.45)
    s += box(6.5, 11.0, 1.0, (4.05, 0, z2 + 0.3), 0.1, "plastic_black")
    s += _contacts(10, 1.0, -4.5, z2 + 0.82, 1.2, 4.6, 0.45)
    s += box(0.6, 17.0, 16.0, (8.6, 0, h / 2 + 0.2), 0.0, "plastic_black")
    return s


def rj45_usb_combo(d=21.3):
    """RJ45 over two USB-A ports."""
    w, h = 17.3, 29.3
    zu = [1.5 + 2.5, 1.5 + 2.5 + 7.3]
    zr = 22.3
    rj = np.array([(-5.85, -3.6), (5.85, -3.6), (5.85, 2.2), (3.1, 2.2), (3.1, 3.4), (1.6, 3.4), (1.6, 4.3),
                   (-1.6, 4.3), (-1.6, 3.4), (-3.1, 3.4), (-3.1, 2.2), (-5.85, 2.2)]) + np.array((0, zr))
    holes = [rrect(USB_A_HOLE[0], USB_A_HOLE[1], 0.35, 2, (0, z)) for z in zu] + [rj]
    s = shell_block(w, h, d, holes, z0=0.15)
    for z in zu:
        s += box(9.0, 11.4, 1.85, (5.4, 0, z - 0.75), 0.15, "plastic_red")
        s += _contacts(4, 2.3, -3.45, z + 0.2, 1.6, 6.0, 1.0)
        s += box(0.6, 12.6, 5.2, (10.2, 0, z), 0.0, "plastic_red")
    s += box(0.6, 12.0, 8.2, (12.0, 0, zr + 0.3), 0.0, "plastic_black")
    s += box(11.0, 11.5, 1.2, (6.5, 0, zr - 3.3), 0.0, "plastic_black")
    pin = box(7.0, 0.42, 0.35, (5.0, 0, zr - 2.5), 0.0, "gold_pin")
    s.tile(pin.rot("y", -14), [(0, -3.57 + i * 1.02, 0) for i in range(8)])
    for sy, m in ((-1, "led_lens"), (1, "led_lens")):
        s += box(1.0, 2.6, 1.6, (0.3, sy * 6.6, zr + 5.2), 0.2, m)
    return s


def audio_stack(n=2, d=13.0):
    """Stacked 3.5 mm audio jacks."""
    w, pitch = 12.4, 12.0
    h = n * pitch + 0.6
    zc = [6.3 + i * pitch for i in range(n)]
    s = shell_block(w, h, d, [circle(1.8, 20, (0, z)) for z in zc], mat="plastic_black", r=0.8)
    for z in zc:
        s += tube_ring(3.0, 1.8, -0.9, 0.3, 24, "steel_bright", bevel=0.2, axis="x").move(0, 0, z)
        s += tube_ring(4.2, 3.0, -0.15, 0.25, 24, "gold_pin", bevel=0.1, axis="x").move(0, 0, z)
    return s


def sma_jack(length=7.0):
    """RP-SMA bulkhead jack, axis along -x, base flange at x = length."""
    s = lathe([(0, 0), (1.9, 0), (1.9, 0.05), (3.1, 0.05), (3.1, length - 2.2), (4.6, length - 2.2),
               (4.6, length), (0, length)], 20, "gold_pin", axis="x")
    s += lathe([(0, 0.4), (0.45, 0.4), (0.45, 0.05), (1.9, 0.06)], 12, "plastic_white", axis="x").move(-0.0, 0, 0)
    nut = prism(circle(4.9, 6), length - 3.6, length - 2.0, "gold_pin", bevel=0.25)
    nut.V = nut.V[:, [2, 0, 1]]
    s += nut
    # thread rings
    for i in range(6):
        s += tube_ring(3.22, 3.0, 0.5 + i * 0.6, 0.8 + i * 0.6, 20, "gold_pin", axis="x")
    return s


def wifi_can(wy=22.0, d=11.0, h=28.5, jack_len=6.5):
    """Wi-Fi module shield with two antenna jacks; front bracket at x = jack_len."""
    s = box(d, wy, h - 0.3, (jack_len + d / 2, 0, 0.3), 0.5, "steel_sheet", base=True)
    s += box(0.8, wy + 2.0, h, (jack_len + 0.4, 0, 0), 0.15, "steel_sheet", base=True)
    for sy in (-1, 1):
        s += sma_jack(jack_len + 0.2).move(0, sy * 5.5, h - 8.5)
    s += mark(["WLAN 2X2", "RF MODULE"], wy * 0.7, 6.0, 0, "ihs_mark").rot("x", 90).rot("z", 90).move(jack_len + d + 0.02, 0, h * 0.5)
    return s


# ----------------------------------------------------------------------------
# small silkscreen helpers
# ----------------------------------------------------------------------------
def silk_label(s, x, y, rot=0.0, h=0.9, z=EPS, mat="silk_white", align="c"):
    return text(s, h, None, mat, z, align).rotz(rot).move(x, y, 0)


def silk_box(cx, cy, w, h, lw=0.12, z=EPS, mat="silk_white", pin1=True):
    s = Soup()
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    s += strip([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], lw, z, mat, closed=True)
    if pin1:
        s += disc(0.18, z, 8, mat, (x0 - 0.3, y0 - 0.3))
    return s


def silk_corners(cx, cy, w, h, ln=0.8, lw=0.12, z=EPS, mat="silk_white"):
    """Four corner brackets around a footprint."""
    s = Soup()
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * w / 2, cy + sy * h / 2
            s += strip([(x - sx * ln, y), (x, y), (x, y - sy * ln)], lw, z, mat)
    return s
