"""M.2 + chipset sink: steady conduction in the printed aluminium (finite volume on the CAD voxels, AMG-CG).

    /opt/venv/bin/python -m sim.macro.passive_sink

Two questions, two sets of boundary conditions:

1. "alone"   - the lattice by itself in room air, a natural-convection + radiation film on every exposed
               face. The film coefficient is the big unknown for a lattice in still air, so this is a band
               over h = 5 ... 10 W/m2K (plus one light-draught case). This is the result that showed the
               passive sink was not enough.
2. "coupled" - the same sink with its land clamped under the monoblock's conduction tail through a gap
               pad. The tail-face temperature comes from the monoblock conjugate run (heater 3, the full
               sink power entering the tail) on top of the coolant temperature from the loop model.

Run order: macro_cht monoblock -> loop_model -> this.
"""
import json
import pathlib

import numpy as np
import scipy.sparse as sp

from design import m2_sink as MS
from design import monoblock as MB
from design import params as P

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "sim" / "out" / "m2_sink"
K_ALU = 130.0
AREA_FACTOR = 0.68          # staircase correction for the film area of a voxelised lattice
ASSUME = dict(
    ambient_c=25.0, h_air=7.0,
    pad_k=P.MONOBLOCK["tail"]["pad_k"], pad_t_mm=P.MONOBLOCK["tail"]["pad"],     # tail <-> land gap pad  [ASM]
    ssd_pad_t_mm=1.0, chipset_pad_t_mm=0.5, comp_pad_k=6.0,                        # component pads         [ASM]
    ssd_controller=dict(share=0.6, area_mm2=144.0),                                # 60 % of SSD power in a 12 x 12 mm controller [ASM]
    reference=dict(material="AlSi10Mg", load="full", rpm=1500),
)


def solve(K_off, diag, rhs):
    import pyamg
    n = len(diag)
    rows, cols, vals = K_off
    A = sp.coo_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))),
                      shape=(n, n)).tocsr()
    return pyamg.smoothed_aggregation_solver(A).solve(rhs, tol=1e-10, accel="cg")


def run(h_mm=0.5):
    f, grid, info = MS.build(h=h_mm, verbose=False)
    x, y, z = grid.coords()
    solid = f < 0
    h = h_mm * 1e-3
    N = solid.size
    sid = -np.ones(N, dtype=np.int64)
    sf = solid.ravel()
    n = int(sf.sum())
    sid[sf] = np.arange(n)
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
    K_off = (rows, cols, vals)
    BD = P.BOARD
    below_void = np.zeros_like(solid)
    below_void[:, :, 1:] = solid[:, :, 1:] & ~solid[:, :, :-1]
    above_void = np.zeros_like(solid)
    above_void[:, :, :-1] = solid[:, :, :-1] & ~solid[:, :, 1:]
    m2 = BD["m2"]
    ssd = below_void & (x >= m2["x"][0]) & (x <= m2["x"][1]) & (y >= m2["y"][0]) & (y <= m2["y"][1]) & (z < MS.Z_PL0 + 0.6)
    cx, cy = BD["chipset"]["c"]
    chip = below_void & (np.abs(x - cx) <= 8.5) & (np.abs(y - cy) <= 8.5) & (z < MS.Z_PL0 - 0.5)
    ssd &= ~chip
    lap = above_void & MB.lap_zone(x, y) & (z > MS.Z_LAND - 0.8) & (z < MS.Z_LAND + 0.3)
    i_ssd, i_chip, i_lap = sid[ssd.ravel()], sid[chip.ravel()], sid[lap.ravel()]
    p_ssd, p_chip = BD["m2_power_w"], BD["chipset_power_w"]
    p_all = p_ssd + p_chip
    q = np.zeros(n)
    q[i_ssd] += p_ssd / len(i_ssd)
    q[i_chip] += p_chip / len(i_chip)
    covered = np.zeros(n)                       # faces that are not in air (component pads, lap joint)
    covered[i_ssd] += 1
    covered[i_chip] += 1
    faces_air_alone = (exposed - covered).clip(0)
    covered[i_lap] += 1
    faces_air_coupled = (exposed - covered).clip(0)
    res = dict(power_w=dict(ssd=p_ssd, chipset=p_chip), exposed_area_cm2=float(faces_air_alone.sum() * h * h * AREA_FACTOR * 1e4),
               k_alu=K_ALU, voxel_mm=h_mm, assumptions=ASSUME, cases=[])

    # ---- 1. the lattice alone in room air (temperatures are rises over the room) ----
    for hf in (5.0, 7.0, 10.0, 30.0):      # 5-10: still air (natural convection + radiation); 30: a light draught, ~1.5 m/s
        film = hf * faces_air_alone * h * h * AREA_FACTOR
        T = solve(K_off, diag_c + film, q)
        c = dict(h_W_m2K=hf, rise_ssd_K=float(T[i_ssd].mean()), rise_chipset_K=float(T[i_chip].mean()),
                 rise_max_K=float(T.max()), rise_min_K=float(T.min()),
                 balance_pct=float(100 * ((film * T).sum() / q.sum() - 1)))
        res["cases"].append(c)
        print("alone", c, flush=True)
    mid = res["cases"][1]
    res["power_for_45K_W"] = 45.0 / (mid["rise_ssd_K"] / p_all)
    res["power_for_45K_with_airflow_W"] = 45.0 / (res["cases"][3]["rise_ssd_K"] / p_all)

    # ---- 2. clamped under the monoblock tail (absolute temperatures) ----
    A_ = ASSUME
    loop_p = ROOT / "sim" / "out" / "loop" / "loop.json"
    runs = sorted((ROOT / "sim" / "out").glob("monoblock_AlSi10Mg_*/summary.json"))
    if loop_p.exists() and runs:
        loop = json.loads(loop_p.read_text())
        ref = A_["reference"]
        op = next(r for r in loop["sweep"] if r["material"] == ref["material"] and r["load"] == ref["load"] and r["rpm"] == ref["rpm"])
        chts = [json.loads(p.read_text()) for p in runs]
        cht = min((c for c in chts if "3" in c["heaters"]), key=lambda c: abs(c["flow_lpm"] - op["flow_lpm"]), default=None)
    else:
        cht = None
    if cht is not None:
        tail_rise = cht["heaters"]["3"]["T_mean_rise_K"]
        t_tail = op["coolant_mono_in_c"] + tail_rise
        g_pad = np.zeros(n)
        np.add.at(g_pad, i_lap, A_["pad_k"] * h * h / (A_["pad_t_mm"] * 1e-3))
        film = A_["h_air"] * faces_air_coupled * h * h * AREA_FACTOR
        T = solve(K_off, diag_c + film + g_pad, q + film * A_["ambient_c"] + g_pad * t_tail)
        q_loop = float((g_pad * (T - t_tail)).sum())
        q_air = float((film * (T - A_["ambient_c"])).sum())
        kp = A_["comp_pad_k"]
        ctrl = A_["ssd_controller"]
        d_ssd = ctrl["share"] * p_ssd * A_["ssd_pad_t_mm"] * 1e-3 / (kp * ctrl["area_mm2"] * 1e-6)
        d_chip = p_chip * A_["chipset_pad_t_mm"] * 1e-3 / (kp * 17.0e-3 ** 2)
        res["coupled"] = dict(
            reference=dict(ref, flow_lpm=op["flow_lpm"], cht_flow_lpm=cht["flow_lpm"]),
            ambient_c=A_["ambient_c"], coolant_mono_in_c=op["coolant_mono_in_c"],
            tail_face_rise_K=float(tail_rise), tail_face_c=float(t_tail),
            lap_area_mm2=float(len(i_lap) * h * h * 1e6), pad_drop_K=float(q_loop / g_pad.sum()),
            sink_at_ssd_c=float(T[i_ssd].mean()), sink_at_chipset_c=float(T[i_chip].mean()),
            ssd_pad_drop_K=float(d_ssd), chipset_pad_drop_K=float(d_chip),
            ssd_c=float(T[i_ssd].mean() + d_ssd), chipset_c=float(T[i_chip].mean() + d_chip),
            sink_max_c=float(T.max()), sink_min_c=float(T.min()),
            q_to_loop_W=q_loop, q_to_air_W=q_air, balance_pct=float(100 * ((q_loop + q_air) / p_all - 1)),
            rise_ssd_over_coolant_K=float(T[i_ssd].mean() + d_ssd - op["coolant_mono_in_c"]),
        )
        print("coupled", res["coupled"], flush=True)
        Tsave = np.full(N, np.nan, dtype=np.float32)
        Tsave[sf] = T
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "fields.npz", Ts=Tsave.reshape(solid.shape), origin_mm=np.array(grid.lo), dx_mm=h_mm)
    else:
        print("no monoblock run with the tail heater / no loop model yet: coupled case skipped")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    run()
