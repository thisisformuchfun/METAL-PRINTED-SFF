"""Interference check: printed frame segments against the keep-out volumes of the hardware they carry.
    /opt/venv/bin/python tools/check_interference.py
Reports metal volume (mm3) of each segment found inside each keep-out box (0 = clear)."""
import json

import numpy as np

from design import frame as FR
from design import params as P

ES, EN, EX = np.array(P.E_S), np.array(P.E_N), np.array(P.E_X)
O = np.array(P.SHOW_O)


def show_box(x0, x1, s0, s1, n0, n1):
    """Box given in SHOW coordinates -> predicate on world coords."""
    def inside(x, y, z):
        xs = x - O[0]
        s = (x - O[0]) * ES[0] + (y - O[1]) * ES[1] + (z - O[2]) * ES[2]
        n = (x - O[0]) * EN[0] + (y - O[1]) * EN[1] + (z - O[2]) * EN[2]
        return (xs > x0) & (xs < x1) & (s > s0) & (s < s1) & (n > n0) & (n < n1)
    return inside


def world_box(lo, hi):
    return lambda x, y, z: (x > lo[0]) & (x < hi[0]) & (y > lo[1]) & (y < hi[1]) & (z > lo[2]) & (z < hi[2])


psu_lo = np.array(P.PSU["lo"])
psu_hi = psu_lo + np.array(P.PSU["size"])
R = P.RAD
bx, bs = P.BOARD_AT["x0"], P.BOARD_AT["s0"]
gx, gs = P.GPU_AT["x0"], P.GPU_AT["s0"]
hc = P.HEART["centre_show"]
KEEP = {
    "psu body": world_box(psu_lo + 0.3, psu_hi - 0.3),
    "radiators + fans": world_box((-137.5, R["y_fan0"] + 0.3, R["centres_z"][1] - 60), (137.5, R["y_rear"] - 0.3, R["centres_z"][0] + 60)),
    "board PCB + back clearance 4 mm": show_box(bx + 0.5, bx + 169.5, bs + 0.5, bs + 169.5, -1.6 - 4.0, 0.0),
    "gpu PCB + back clearance 3.5 mm": show_box(gx + 0.5, gx + 266.1, gs + 0.5, gs + 126.7, -1.57 - 3.5, 0.0),
    "pump body": show_box(hc[0] - 31, hc[0] + 31, hc[1] - 31, hc[1] + 31, -38.0, -0.2),
}
# intended contacts: the standoff bosses under each PCB mounting hole
BOSS = [np.array(FR.S(x, s, 0.0)) for (x, s) in list(FR.gpu_points().values()) + list(FR.board_points())]


def near_boss(x, y, z):
    m = np.zeros(np.broadcast(x, y, z).shape, dtype=bool)
    for b in BOSS:
        t = (x - b[0]) * EN[0] + (y - b[1]) * EN[1] + (z - b[2]) * EN[2]
        r2 = (x - b[0] - t * EN[0]) ** 2 + (y - b[1] - t * EN[1]) ** 2 + (z - b[2] - t * EN[2]) ** 2
        m |= r2 < 5.5 ** 2
    return m


out = {}
h = 1.0
for name, fn in FR.SEGMENTS.items():
    f, g = fn(h)
    x, y, z = g.coords()
    solid = (f < -0.2) & ~near_boss(x, y, z)
    out[name] = {k: float((solid & pred(x, y, z)).sum()) * h ** 3 for k, pred in KEEP.items()}
print(json.dumps(out, indent=1))
bad = {(a, b): v for a, d in out.items() for b, v in d.items() if v > 0}
print("INTERFERENCES:" if bad else "no interference", bad if bad else "")
