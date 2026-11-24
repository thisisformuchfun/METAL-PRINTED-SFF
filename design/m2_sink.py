"""ORGAN IV - M.2 + chipset sink, AlSi10Mg. A solid spreader plate carrying a true 3-D Voronoi
strut lattice (trabecular-bone style). On its own in still air the lattice only holds about 3 W, so a
raised land along its upper edge takes the monoblock's conduction tail (0.5 mm gap pad, 2x M3): the
heat goes into the loop, and the lattice is the second, passive path.

Coordinates: BOARD frame, z up from the PCB top surface.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import params as P
from . import part, sdf

BD = P.BOARD
S = P.M2_SINK
LT = P.LATTICE

X0, X1 = S["x"]
Y0, Y1 = S["y"]
Z_PL0 = BD["m2"]["z_top"] + 1.0            # 1.0 mm pad on the SSD
Z_PL1 = Z_PL0 + S["base"]
Z_TOP = Z_PL1 + S["lattice_h"]
TL = P.MONOBLOCK["tail"]
LAND = S["land"]
Z_LAND = TL["z"][0] - TL["pad"]       # land top: tail underside minus the gap pad
SCREWS = [(BD["m2"]["x"][1] + 2.5, 0.5 * (BD["m2"]["y"][0] + BD["m2"]["y"][1])), (X0 + 5.0, Y1 - 5.0)]


def plan2(x, y, grow=0.0):
    return sdf.rect2(x, y, (X0 - grow, Y0 - grow), (X1 + grow, Y1 + grow), r=6.0 + grow)


def build(h=0.25, verbose=True):
    tm = part.Timer()
    grid = sdf.Grid((X0 - 2, Y0 - 2, 1.5), (X1 + 2, Y1 + 2, Z_TOP + 1.5), h)
    x, y, z = grid.coords()
    p2 = plan2(x, y)
    f = sdf.smax(p2, np.maximum(Z_PL0 - z, z - Z_PL1), 0.8)
    # chipset pedestal (0.5 mm pad)
    cx, cy = BD["chipset"]["c"]
    zc = BD["chipset"]["z_top"] + 0.5
    sdf.add_box(f, grid, (cx - 8.5, cy - 8.5, zc), (cx + 8.5, cy + 8.5, Z_PL0 + 0.6), r=0.0)
    # screw bosses
    for (sx, sy) in SCREWS:
        sdf.apply(f, grid, (sx - 4.5, sy - 4.5, 2.4), (sx + 4.5, sy + 4.5, Z_PL1 + 2.0),
                  lambda X, Y, Z, sx=sx, sy=sy: sdf.cyl_z(X, Y, Z, (sx, sy), 4.2, 3.4, Z_PL1 + 2.0), op="union", k=1.0)
    # land under the monoblock's conduction tail, running on as a thick root to the chipset pedestal
    l_lo, l_hi = (LAND["x"][0], LAND["y0"], Z_PL1 - 0.6), (LAND["x"][1], Y1, Z_LAND)
    sdf.apply(f, grid, l_lo, l_hi, lambda X, Y, Z: np.maximum(sdf.box(X, Y, Z, l_lo, l_hi, r=0.3), plan2(X, Y) + 0.15), op="union")
    tm("plate")
    # Voronoi strut lattice inside a pillowed envelope
    seeds = sdf.hex_seeds((X0, Y0), (X1, Y1), 24.0, seed=5)
    xk, yk, zk = grid.coords()
    pillow = 0.5 + 0.5 * sdf.smoothstep(sdf.voronoi2_edge(xk, yk, seeds) / 3.5)
    top = Z_PL1 + 2.0 + (S["lattice_h"] - 2.0) * sdf.smoothstep(-p2 / 7.0) * pillow
    env = np.maximum(p2 + 0.6, np.maximum(Z_PL1 - 0.6 - zk, zk - top))
    lat = grid.new(sdf.BIG)
    n = 0
    for a, b in sdf.voronoi_edges3((X0, Y0, Z_PL1 - 2.0), (X1, Y1, Z_TOP + 1.0), LT["voronoi_cell"] * 0.8, seed=3):
        sdf.add_capsule(lat, grid, a, b, LT["voronoi_strut"] / 2, k=0.9)
        n += 1
    lat = sdf.smax(lat, env, 0.9)
    # nothing of the lattice may stand where the tail and its screw heads come down
    clear = np.maximum(np.maximum(TL["x"][0] - 1.5 - xk, xk - (TL["x"][1] + 1.5)),
                       np.maximum(TL["y0"] - 1.5 - yk, (Z_LAND - 0.3) - zk))
    lat = np.maximum(lat, -clear)
    del clear
    for (sx, sy) in SCREWS:                      # keep screw heads reachable
        lat = np.maximum(lat, -sdf.circle2(xk, yk, (sx, sy), 4.6))
    f = sdf.smin(f, lat, 0.9)
    del lat
    tm(f"voronoi lattice ({n} struts)")
    for (sx, sy) in SCREWS:
        sdf.apply(f, grid, (sx - 2, sy - 2, 0), (sx + 2, sy + 2, Z_TOP + 2),
                  lambda X, Y, Z, sx=sx, sy=sy: sdf.cyl_z(X, Y, Z, (sx, sy), 1.7, -5, 60), op="cut")
    for (tx, ty) in TL["screws"]:                # blind M3 pilots for the tail screws (0.7 mm skin left over the SSD)
        sdf.apply(f, grid, (tx - 2, ty - 2, Z_PL0), (tx + 2, ty + 2, Z_LAND + 1),
                  lambda X, Y, Z, tx=tx, ty=ty: sdf.cyl_z(X, Y, Z, (tx, ty), 1.25, Z_PL0 + 0.7, 60), op="cut")
    removed = sdf.keep_largest_solid(f, h)
    closed = sdf.fill_closed_voids(f, h)
    info = dict(struts=n, islands_removed_voxels=removed, closed_void_voxels=closed,
                z_levels=dict(plate=(Z_PL0, Z_PL1), land=Z_LAND, top=Z_TOP), screws=SCREWS, tail_screws=TL["screws"])
    tm("clean-up")
    return f, grid, info


def main(h=0.25, do_export=True):
    f, grid, info = build(h=h)
    if do_export:
        meta = part.export_part("m2_sink", f, grid, "AlSi10Mg", "board", extra=dict(checks=info),
                                print_faces=500_000, web_faces=90_000, hero_faces=2_000_000)
        print(json.dumps(meta, indent=1))
    return f, grid, info


if __name__ == "__main__":
    hh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    main(h=hh, do_export=("--noexport" not in sys.argv))
