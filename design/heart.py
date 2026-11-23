"""ORGAN IV - "the heart": printed top for a DDC-class pump with an integrated micro
reservoir (air trap + fill port), AlSi10Mg, monolithic.

Local HEART frame: origin at the centre of the pump's mating face, +z toward the viewer
(= SHOW e_n), x = SHOW e_x, y = SHOW e_s (up the slope). The pump motor hangs behind (z<0).

  volute      Ø42 x 11 chamber on the mating face, tangential discharge -> OUT port (+z)
  atrium      ~43 mL capsule reservoir above the volute, feeds the impeller eye axially
  IN port     on the back of the atrium (-z), return from radiator B
  FILL port   at the top of the atrium (+y), sits at the high point of the loop
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import params as P
from . import part, sdf

H = P.HEART
LT = P.LATTICE

VOL_R = H["volute_d"] / 2
VOL_H = H["volute_h"]
RES_R = 15.0
RES_A = (0.0, 9.0, 13.5 + RES_R)          # capsule axis start
RES_B = (0.0, 47.0, 13.5 + RES_R)
WALL = 2.2
OUT = (-21.5, -22.0)                       # discharge port (faces +z)
OUT_Z = 19.0
IN_BACK = (0.0, 47.0)                      # return port on the back of the atrium (faces -z)
FILL_Y = RES_B[1] + RES_R + WALL + 4.0     # top face of the fill boss
EARS = [(-37.0, -6.0), (-27.0, 55.0), (23.0, -39.5)]
SCREW = [(-25.0, -25.0), (25.0, -25.0), (-25.0, 25.0), (25.0, 25.0)]


def build(h=0.25, verbose=True):
    tm = part.Timer()
    grid = sdf.Grid((-45.0, -47.0, -1.0), (37.0, FILL_Y + 1.5, RES_A[2] + RES_R + WALL + 6.5), h)
    if verbose:
        print(f"heart grid {grid.shape} = {np.prod(grid.n) / 1e6:.1f} M voxels")
    x, y, z = grid.coords()

    # ---------------- outer body ----------------
    flange = sdf.smax(sdf.rect2(x, y, (-32, -32), (32, 32), r=9.0), np.maximum(-z, z - 6.0), 1.2)
    housing = sdf.smax(sdf.circle2(x, y, (0, 0), VOL_R + 4.0), np.maximum(-z, z - (VOL_H + 3.2)), 2.0)
    res = sdf.capsule(x, y, z, RES_A, RES_B, RES_R + WALL)
    f = sdf.smin(sdf.smin(flange, housing, 5.0), res, 7.0)
    f = np.maximum(f, -z)                                       # flat mating face
    del flange, housing, res
    # ears for the frame
    for (ex, ey) in EARS:
        sdf.add_capsule(f, grid, (ex * 0.55, ey * 0.55, 3.0), (ex, ey, 3.0), 5.2, 5.0, k=4.0)
    f = np.maximum(f, -z)
    # port bosses
    rs = P.LOOP["port_seat_d"] / 2
    sdf.apply(f, grid, (OUT[0] - rs, OUT[1] - rs, 0), (OUT[0] + rs, OUT[1] + rs, OUT_Z),
              lambda X, Y, Z: sdf.smax(sdf.circle2(X, Y, OUT, rs), np.maximum(-Z, Z - OUT_Z), 0.8), op="union", k=3.0)
    zb = RES_A[2] - RES_R - WALL - 2.0                           # back boss bottom face
    sdf.apply(f, grid, (IN_BACK[0] - rs, IN_BACK[1] - rs, zb), (IN_BACK[0] + rs, IN_BACK[1] + rs, RES_A[2]),
              lambda X, Y, Z: sdf.smax(sdf.circle2(X, Y, IN_BACK, rs), np.maximum(zb - Z, Z - RES_A[2]), 0.8), op="union", k=3.0)
    fa, fb = (0.0, RES_B[1], RES_B[2]), (0.0, FILL_Y, RES_B[2])
    sdf.apply(f, grid, (-rs, RES_B[1], RES_B[2] - rs), (rs, FILL_Y, RES_B[2] + rs),
              lambda X, Y, Z: sdf.cylinder(X, Y, Z, fa, fb, rs), op="union", k=3.0)
    tm("outer body")

    # ---------------- bone skin on the atrium ----------------
    sk = grid.box((-30, -6, 13.0), (30, FILL_Y - 5.0, 60.0))
    xk, yk, zk = grid.coords(sk)
    r_res = sdf.capsule(xk, yk, zk, RES_A, RES_B, RES_R + WALL)
    env = np.maximum(r_res - 4.6, -(r_res + 0.6))               # 4.6 mm thick skin shell
    env = np.maximum(env, 16.0 - zk)                            # front + sides only
    net = sdf.gyroid_network(grid, sk, LT["skin_cell"] * 0.82, bias=0.45, phase=(0.3, 1.2, 2.2), warp_amp=1.2, warp_len=29.0)
    skin = sdf.smax(net, env, 0.9)
    skin = np.maximum(skin, -(sdf.cylinder(xk, yk, zk, fa, fb, rs + 2.5)))
    f[sk] = sdf.smin(f[sk], skin, 0.9)
    del net, env, skin
    tm("bone skin")

    # ---------------- wet volumes ----------------
    # volute, open to the pump
    sdf.apply(f, grid, (-VOL_R, -VOL_R, -2), (VOL_R, VOL_R, VOL_H),
              lambda X, Y, Z: sdf.smax(sdf.circle2(X, Y, (0, 0), VOL_R), Z - VOL_H, 2.5), op="cut")
    # atrium
    sdf.apply(f, grid, (-RES_R, RES_A[1] - RES_R, RES_A[2] - RES_R), (RES_R, RES_B[1] + RES_R, RES_A[2] + RES_R),
              lambda X, Y, Z: sdf.capsule(X, Y, Z, RES_A, RES_B, RES_R), op="cut")
    # eye: atrium -> impeller centre
    sdf.add_cyl(f, grid, (0, 0, VOL_H - 1.0), (0, 6.0, RES_A[2] - RES_R + 4.0), 5.0, op="cut")
    # discharge: tangential run along the bottom of the volute, then up to the port
    sdf.add_capsule(f, grid, (4.0, -(VOL_R - 4.6), 5.6), (OUT[0], -(VOL_R - 3.4), 5.8), 4.4, op="cut")
    sdf.add_capsule(f, grid, (OUT[0], -(VOL_R - 3.4), 5.8), (OUT[0], OUT[1], 9.5), 4.4, op="cut")
    tap = P.LOOP["port_tap"] / 2
    sdf.add_cyl(f, grid, (OUT[0], OUT[1], 8.5), (OUT[0], OUT[1], OUT_Z + 3), tap, op="cut")
    sdf.add_cyl(f, grid, (IN_BACK[0], IN_BACK[1], zb - 3), (IN_BACK[0], IN_BACK[1], RES_A[2] - RES_R + 3.0), tap, op="cut")
    sdf.add_cyl(f, grid, (0, RES_B[1] + RES_R - 3.0, RES_B[2]), (0, FILL_Y + 3, RES_B[2]), tap, op="cut")
    # anti-vortex vane across the atrium floor
    sdf.add_box(f, grid, (-0.6, 2.0, RES_A[2] - RES_R - 0.5), (0.6, 30.0, RES_A[2] - 2.0), r=0.0)
    sdf.add_cyl(f, grid, (0, 0, VOL_H - 1.0), (0, 6.0, RES_A[2] - RES_R + 4.0), 3.6, op="cut")
    tm("wet volumes")

    # ---------------- seals + fasteners ----------------
    sdf.apply(f, grid, (-27, -27, -1), (27, 27, 2.2),
              lambda X, Y, Z: np.maximum(np.abs(sdf.circle2(X, Y, (0, 0), 24.3)) - 1.15, Z - 1.9), op="cut")   # O-ring groove
    for (sx, sy) in SCREW:
        sdf.add_cyl(f, grid, (sx, sy, -2), (sx, sy, 5.0), 1.65, op="cut")                                    # M4 pilot
    for (ex, ey) in EARS:
        sdf.add_cyl(f, grid, (ex, ey, -2), (ex, ey, 12), 2.15, op="cut")
    removed = sdf.keep_largest_solid(f, h)
    wet_ml = float(((sdf.capsule(x, y, z, RES_A, RES_B, RES_R) < 0)).sum()) * h ** 3 / 1000.0
    info = dict(reservoir_ml=round(wet_ml, 1), islands_removed_voxels=removed,
                ports=dict(out=(OUT[0], OUT[1], OUT_Z), inlet_back=(IN_BACK[0], IN_BACK[1], zb), fill=(0.0, FILL_Y, RES_B[2])),
                ears=EARS, screws=SCREW)
    tm("clean-up")
    return f, grid, info


def main(h=0.25, do_export=True):
    f, grid, info = build(h=h)
    print(json.dumps(info, indent=1))
    if do_export:
        meta = part.export_part("heart", f, grid, "AlSi10Mg", "heart", extra=dict(checks=info),
                                print_faces=500_000, web_faces=90_000, hero_faces=2_000_000)
        print(json.dumps(meta, indent=1))
    return f, grid, info


if __name__ == "__main__":
    hh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    main(h=hh, do_export=("--noexport" not in sys.argv))
