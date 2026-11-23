"""Passive M.2 + chipset sink: steady conduction in the printed aluminium with a natural-convection +
radiation film on every exposed face (finite volume on the CAD voxels, AMG-CG).

    /opt/venv/bin/python -m sim.macro.passive_sink
The film coefficient is the big unknown for a lattice in still air, so the result is reported as a
band over h = 5 ... 10 W/m2K rather than a single number.
"""
import json
import pathlib

import numpy as np
import scipy.sparse as sp

from design import m2_sink as MS
from design import params as P

OUT = pathlib.Path(__file__).resolve().parents[2] / "sim" / "out" / "m2_sink"
K_ALU = 130.0


def run(h_mm=0.5):
    f, grid, info = MS.build(h=h_mm, verbose=False)
    x, y, z = grid.coords()
    solid = f < 0
    h = h_mm * 1e-3
    N = solid.size
    sid = -np.ones(N, dtype=np.int64)
    sf = solid.ravel()
    sid[sf] = np.arange(int(sf.sum()))
    n = int(sf.sum())
    idx = np.arange(N).reshape(solid.shape)
    rows, cols, vals = [], [], []
    diag_c = np.zeros(n)
    exposed = np.zeros(n)                       # count of exposed faces per solid cell
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax] = slice(0, -1)
        b[ax] = slice(1, None)
        i, j = idx[tuple(a)].ravel(), idx[tuple(b)].ravel()
        m = sf[i] & sf[j]
        ia, ib = sid[i[m]], sid[j[m]]
        g = np.full(len(ia), K_ALU * h)
        rows += [ia, ib]
        cols += [ib, ia]
        vals += [-g, -g]
        np.add.at(diag_c, ia, g)
        np.add.at(diag_c, ib, g)
        np.add.at(exposed, sid[i[sf[i] & ~sf[j]]], 1)
        np.add.at(exposed, sid[j[sf[j] & ~sf[i]]], 1)
    # boundary cells of the grid count as exposed too (cheap: ignore - the part does not touch the box)
    BD = P.BOARD
    below_void = np.zeros_like(solid)
    below_void[:, :, 1:] = solid[:, :, 1:] & ~solid[:, :, :-1]
    m2 = BD["m2"]
    ssd = below_void & (x >= m2["x"][0]) & (x <= m2["x"][1]) & (y >= m2["y"][0]) & (y <= m2["y"][1]) & (z < MS.Z_PL0 + 0.6)
    cx, cy = BD["chipset"]["c"]
    chip = below_void & (np.abs(x - cx) <= 8.5) & (np.abs(y - cy) <= 8.5) & (z < MS.Z_PL0 - 0.5)
    ssd &= ~chip
    q = np.zeros(n)
    q[sid[ssd.ravel()]] += BD["m2_power_w"] / ssd.sum()
    q[sid[chip.ravel()]] += BD["chipset_power_w"] / chip.sum()
    pad = np.zeros(n, dtype=bool)
    pad[sid[ssd.ravel()]] = True
    pad[sid[chip.ravel()]] = True
    area_exposed = float((exposed[~pad].sum() + (exposed[pad] - 1).clip(0).sum()) * h * h) * 0.68   # staircase correction
    import pyamg
    res = dict(power_w=dict(ssd=BD["m2_power_w"], chipset=BD["chipset_power_w"]), exposed_area_cm2=area_exposed * 1e4,
               k_alu=K_ALU, voxel_mm=h_mm, cases=[])
    Tsave = None
    for hf in (5.0, 7.0, 10.0, 30.0):      # 5-10: still air (natural convection + radiation); 30: a light draught, ~1.5 m/s
        film = hf * exposed * h * h * 0.68
        film[pad] = hf * (exposed[pad] - 1).clip(0) * h * h * 0.68       # pad faces are covered by the chip
        A = sp.coo_matrix((np.concatenate(vals + [diag_c + film]), (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))),
                          shape=(n, n)).tocsr()
        ml = pyamg.smoothed_aggregation_solver(A)
        T = ml.solve(q, tol=1e-10, accel="cg")
        c = dict(h_W_m2K=hf, rise_ssd_K=float(T[sid[ssd.ravel()]].mean()), rise_chipset_K=float(T[sid[chip.ravel()]].mean()),
                 rise_max_K=float(T.max()), rise_min_K=float(T.min()),
                 balance_pct=float(100 * ((film * T).sum() / q.sum() - 1)))
        res["cases"].append(c)
        print(c, flush=True)
        if hf == 7.0:
            Tsave = np.full(N, np.nan, dtype=np.float32)
            Tsave[sf] = T
    # how much power keeps the SSD pad at +45 K (70 C in a 25 C room)?
    mid = res["cases"][1]
    res["power_for_45K_W"] = 45.0 / (mid["rise_ssd_K"] / sum(res["power_w"].values()))
    res["power_for_45K_with_airflow_W"] = 45.0 / (res["cases"][3]["rise_ssd_K"] / sum(res["power_w"].values()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / "fields.npz", Ts=Tsave.reshape(solid.shape), origin_mm=np.array(grid.lo), dx_mm=h_mm)
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
