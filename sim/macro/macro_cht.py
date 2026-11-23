"""Block-level conjugate heat transfer of the MARROW cold plates (scale 2 of 2).

The lattice-filled passages are treated as porous zones whose flow resistance, volumetric
heat-transfer coefficient and effective metal conductivity come from the resolved unit-cell
simulations (sim/cell/unit_cell.py -> closures.json). Everything else - floor, walls, pedestals,
wings, divider, plenum, ports - is resolved directly on a 0.5 mm voxel grid built from the same
CAD fields that generate the print files.

  flow     Darcy-Forchheimer:  -grad p = (A + B|U|) U,  div U = 0      (Picard / AMG-CG)
  energy   two-temperature model in the lattice (metal + coolant), conduction in solid metal,
           upwind advection in the coolant, heater flux on the pads   (monolithic, AMG-GMRES)

    /opt/venv/bin/python -m sim.macro.macro_cht gpu_block AlSi10Mg 3.0
Outputs sim/out/<case>_<material>_<flow>/summary.json + fields.npz
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage as ndi

from design import gpu_block as GB
from design import monoblock as MB
from design import params as P
from design import sdf

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTD = ROOT / "sim" / "out"

MATERIALS = {
    # k of SLM AlSi10Mg depends on heat treatment (as-built ~110-130, stress-relieved ~150-170) - we take
    # a conservative 130; 316L from the JLC3DP datasheet.
    "AlSi10Mg": dict(k=130.0, note="SLM AlSi10Mg, conservative as-built value"),
    "316L": dict(k=15.1, note="SLM 316L, JLC3DP datasheet"),
}
VOID, SOLID, OPEN = 0, 1, 2          # zone ids; lattice zones are >= 3
H_OPEN = 600.0                       # W/m2K, wall film in open (un-latticed) coolant passages  [ASM]


def closures():
    c = json.loads((OUTD / "cell" / "closures.json").read_text())
    return c["fluid"], c["lattices"]


# ----------------------------------------------------------------------------
# case geometry -> zone map
# ----------------------------------------------------------------------------
def gpu_case(h=0.5):
    f, grid, info = GB.build(h=h, verbose=False)
    x, y, z = grid.coords()
    B = P.GPU_BLOCK
    solid = f < 0
    kz = int(np.searchsorted(grid.ax[2], GB.Z_BOSS - 0.2))
    void = ~solid
    void[:, :, kz:] = False
    lab, _ = ndi.label(void)
    idx = lambda p: tuple(int(round((p[i] - grid.lo[i]) / h)) for i in range(3))
    wet = lab == lab[idx((B["inlet"][0], B["inlet"][1], GB.Z_BOSS - 2.0))]
    zone = np.where(solid, SOLID, VOID).astype(np.uint8)
    zone[wet] = OPEN
    ch = GB.chamber2(x, y) < 0
    div = GB.chamber2(x, y, shrink=B["riser_gap"]) < 0
    ix, iy = B["inlet"]
    pl_o = sdf.rect2(x, y, (ix - 8.6, iy - 14.6), (ix + 8.6, iy + 14.6), r=7.6) < 0
    pocket = sdf.circle2(x, y, B["outlet"], 6.3) < 0
    pill = np.zeros_like(ch)
    for (px, py) in P.GPU["core_holes"]:
        pill |= sdf.circle2(x, y, (px, py), 3.75) < 0
    core = ch & (z >= GB.Z_FLOOR) & (z < GB.Z_CORE1) & ~(pill & (z < 9.6)) & ~(GB.trench(x, y, z) < 0)
    gal = ch & (z < GB.Z_GAL1) & (z >= GB.Z_DIV1) & ~pl_o & ~(pocket & (z >= GB.Z_DIV1 + 0.9))
    zone[core] = 3
    zone[gal] = 4
    lattice = {3: "core", 4: "gallery"}
    # slot nozzle cells (for the jet loss)
    jw, jl = B["jet"]
    slot = (np.abs(x - ix) <= jw / 2 + 0.01) & (np.abs(y - iy) <= jl / 2) & (z >= GB.Z_CORE1 - 0.3) & (z <= GB.Z_DIV1 + 0.3) & (zone == OPEN)
    top = (z > GB.Z_BOSS - 1.6) & (zone == OPEN)
    inlet = top & (sdf.circle2(x, y, B["inlet"], 7.0) < 0)
    outlet = top & (sdf.circle2(x, y, B["outlet"], 7.0) < 0)
    # heaters: bottom faces of the pads
    below_void = np.zeros_like(solid)
    below_void[:, :, 1:] = (zone[:, :, 1:] == SOLID) & (zone[:, :, :-1] == VOID)
    G = P.GPU
    heater = np.zeros(zone.shape, dtype=np.uint8)
    dxx, dyy = G["die"]
    heater[below_void & (np.abs(x - GB.CX) <= dxx / 2) & (np.abs(y - GB.CY) <= dyy / 2) & (z < GB.Z_UNDER + 0.3)] = 1
    for cx, cy, w, hg in G["mem"].values():
        heater[below_void & (np.abs(x - cx) <= w / 2 - 0.3) & (np.abs(y - cy) <= hg / 2 - 0.3) & (z < GB.Z_DIE - 0.3)] = 2
    zdr = G["z_drmos"] + B["pad_vrm"]
    for dxr in (G["vrm_near"]["drmos_x"], G["vrm_far"]["drmos_x"]):
        heater[below_void & (x >= dxr[0] - 0.4) & (x <= dxr[1] + 0.4) & (z < zdr + 0.6) & (z > zdr - 0.6)] = 3
    powers = {1: G["power_split"]["die"], 2: G["power_split"]["mem"], 3: G["power_split"]["vrm"]}
    names = {1: "GPU die", 2: "memory (12 x GDDR6X)", 3: "VRM power stages"}
    return dict(name="gpu_block", grid=grid, zone=zone, lattice=lattice, slot=slot, inlet=inlet, outlet=outlet,
                heater=heater, powers=powers, heater_names=names, frame="gpu")


def mono_case(h=0.5):
    f, grid, info = MB.build(h=h, verbose=False)
    x, y, z = grid.coords()
    M = P.MONOBLOCK
    solid = f < 0
    void, lab, li, lo_, cap_in, cap_out = MB._wet(f, grid)
    wet = lab == li
    zone = np.where(solid, SOLID, VOID).astype(np.uint8)
    zone[wet] = OPEN
    ch = MB.chamber2(x, y) < 0
    div = MB.chamber2(x, y, shrink=M["riser_gap"]) < 0
    jl, jw = M["jet"]
    IN, HOT, OUT = MB.IN, MB.HOT, MB.OUT
    plo = (min(IN[0], HOT[0]) - jl / 2 - 1.0, min(IN[1], HOT[1]) - 7.0)
    phi = (max(IN[0], HOT[0]) + jl / 2 + 1.0, max(IN[1], HOT[1]) + 7.0)
    pl_o = sdf.rect2(x, y, (plo[0] - 1.4, plo[1] - 1.4), (phi[0] + 1.4, phi[1] + 1.4), r=6.4) < 0
    core = ch & (z >= MB.Z_FLOOR) & (z < MB.Z_CORE1) & ~(MB.trench(x, y, z) < 0)
    gal = ch & (z < MB.Z_GAL1) & (z >= MB.Z_DIV1) & ~pl_o
    W = M["wall"]
    ai = MB.arms2(x, y, shrink=W) < 0
    pocket = (sdf.circle2(x, y, OUT, 6.3) < 0) & (z >= MB.CZ0 + 3.0)
    arm = ai & (z >= MB.CZ0) & (z < MB.CZ1) & ~pocket
    L = M["arm_left"]
    by0, by1 = M["bridge_y"]
    dz0, dz1 = MB.Z_DIV1 + 0.3, min(MB.CZ1, MB.Z_GAL1) - 0.1
    bridge = (x >= L["x"][1] - W - 0.6) & (x <= MB.SX - MB.CH[0] / 2 + 0.6) & (y >= by0 + 2.0) & (y <= by1 - 2.0) & (z >= dz0) & (z <= dz1)
    zone[core] = 3
    zone[gal] = 4
    zone[arm] = 5
    zone[bridge & ~ch] = 5
    lattice = {3: "core", 4: "gallery", 5: "arm"}
    slot = (np.abs(x - HOT[0]) <= jl / 2) & (np.abs(y - HOT[1]) <= jw / 2 + 0.01) & (z >= MB.Z_CORE1 - 0.3) & (z <= MB.Z_DIV1 + 0.3) & (zone == OPEN)
    inlet = (zone == OPEN) & (sdf.circle2(x, y, IN, 7.0) < 0) & (z > MB.Z_BOSS - 1.9)
    outlet = (zone == OPEN) & (sdf.circle2(x, y, OUT, 7.0) < 0) & (z > MB.Z_ABOSS - 1.9)
    below_void = np.zeros_like(solid)
    below_void[:, :, 1:] = (zone[:, :, 1:] == SOLID) & (zone[:, :, :-1] == VOID)
    heater = np.zeros(zone.shape, dtype=np.uint8)
    hx, hy = M["heater_cpu"]
    heater[below_void & (np.abs(x - HOT[0]) <= hx / 2) & (np.abs(y - HOT[1]) <= hy / 2) & (z < MB.Z_UNDER + 0.3)] = 1
    heater[below_void & (z < MB.AZ0 - 0.5) & (MB.arms2(x, y) < 0)] = 2
    BD = P.BOARD
    powers = {1: BD["cpu_power_w"], 2: BD["vrm_power_w"]}
    names = {1: "CPU (IHS hot patch)", 2: "VRM power stages"}
    return dict(name="monoblock", grid=grid, zone=zone, lattice=lattice, slot=slot, inlet=inlet, outlet=outlet,
                heater=heater, powers=powers, heater_names=names, frame="board")


CASES = dict(gpu_block=gpu_case, monoblock=mono_case)


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------
def neighbours(shape, axis):
    """Flat indices (i, j) of every pair of cells adjacent along axis (j = i + 1 along axis)."""
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    sl_a = [slice(None)] * 3
    sl_b = [slice(None)] * 3
    sl_a[axis] = slice(0, -1)
    sl_b[axis] = slice(1, None)
    return idx[tuple(sl_a)].ravel(), idx[tuple(sl_b)].ravel()


def solve_flow(case, fluid, lat, q_lpm, verbose=True):
    zone = case["zone"]
    h = case["grid"].h * 1e-3
    shape = zone.shape
    N = zone.size
    zf = zone.ravel()
    isf = zf >= OPEN
    mu = fluid["nu"] * fluid["rho"]
    A = np.zeros(N)
    Bc = np.zeros(N)
    A[zf == OPEN] = 12.0 * mu / (3.0e-3) ** 2
    Bc[zf == OPEN] = fluid["rho"] * 0.5 / (2 * 5e-3)
    Bc[case["slot"].ravel()] = fluid["rho"] * 1.5 / (2 * P.GPU_BLOCK["divider"] * 1e-3)
    for zid, name in case["lattice"].items():
        fo = lat[name]["forchheimer"]
        A[zf == zid] = fo["A_Pa_s_m2"]
        Bc[zf == zid] = fo["B_Pa_s2_m3"]
    Q = q_lpm / 60000.0
    inlet = case["inlet"].ravel()
    outlet = case["outlet"].ravel()
    fid = -np.ones(N, dtype=np.int64)
    unk = isf & ~outlet
    fid[unk] = np.arange(int(unk.sum()))
    nu_ = int(unk.sum())
    pairs = [neighbours(shape, ax) for ax in range(3)]
    pairs = [(i[isf[i] & isf[j]], j[isf[i] & isf[j]]) for (i, j) in pairs]
    U = np.zeros(N)
    p = np.zeros(N)
    import pyamg

    hist = []
    flux = None
    for it in range(40):
        sig = np.zeros(N)
        sig[isf] = 1.0 / (A[isf] + Bc[isf] * U[isf])
        rows, cols, vals = [], [], []
        diag = np.zeros(nu_)
        rhs = np.zeros(nu_)
        gs = []
        for (i, j) in pairs:
            g = h * 2.0 * sig[i] * sig[j] / (sig[i] + sig[j])
            gs.append(g)
            a, b = fid[i], fid[j]
            both = (a >= 0) & (b >= 0)
            rows += [a[both], b[both]]
            cols += [b[both], a[both]]
            vals += [-g[both], -g[both]]
            np.add.at(diag, a[a >= 0], g[a >= 0])
            np.add.at(diag, b[b >= 0], g[b >= 0])
        rhs[fid[inlet]] += Q / inlet.sum()
        K = sp.coo_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(nu_)]), np.concatenate(cols + [np.arange(nu_)]))),
                          shape=(nu_, nu_)).tocsr()
        ml = pyamg.smoothed_aggregation_solver(K, max_coarse=500)
        x0 = p[unk] if it else None
        sol = ml.solve(rhs, x0=x0, tol=1e-10, accel="cg", maxiter=300)
        p_new = np.zeros(N)
        p_new[unk] = sol
        # face fluxes (m3/s), positive from i to j
        flux = [g * (p_new[i] - p_new[j]) for g, (i, j) in zip(gs, pairs)]
        # cell-centred superficial speed
        comp = []
        for ax, ((i, j), fl) in enumerate(zip(pairs, flux)):
            acc = np.zeros(N)
            cnt = np.zeros(N)
            np.add.at(acc, i, fl)
            np.add.at(acc, j, fl)
            np.add.at(cnt, i, 1)
            np.add.at(cnt, j, 1)
            comp.append(acc / np.maximum(cnt, 1) / (h * h))
        U_new = np.sqrt(comp[0] ** 2 + comp[1] ** 2 + comp[2] ** 2)
        dp = float(p_new[inlet].mean())
        hist.append(dp)
        rel = abs(dp - hist[-2]) / dp if it else 1.0
        U = 0.5 * U + 0.5 * U_new if it else U_new
        p = p_new
        if verbose:
            print(f"    flow it {it:2d}: dp {dp/1000:7.3f} kPa", flush=True)
        if it > 3 and rel < 2e-3:
            break
    qout = 0.0
    for (i, j), fl in zip(pairs, flux):
        qout += fl[outlet[j] & ~outlet[i]].sum() - fl[outlet[i] & ~outlet[j]].sum()
    return dict(p=p.reshape(shape), U=U.reshape(shape), comp=[c.reshape(shape) for c in comp], pairs=pairs, flux=flux,
                dp=dp, q_out=float(qout), iters=it + 1)


def solve_heat(case, fluid, lat, flow, k_metal, t_in=0.0, verbose=True):
    zone = case["zone"]
    h = case["grid"].h * 1e-3
    shape = zone.shape
    N = zone.size
    zf = zone.ravel()
    has_s = (zf == SOLID) | (zf >= 3)
    has_f = zf >= OPEN
    sid = -np.ones(N, dtype=np.int64)
    sid[has_s] = np.arange(int(has_s.sum()))
    ns = int(has_s.sum())
    fid = -np.ones(N, dtype=np.int64)
    fid[has_f] = ns + np.arange(int(has_f.sum()))
    nt = ns + int(has_f.sum())
    rho_cp = fluid["rho"] * fluid["cp"]
    # conductivities per direction
    kx = np.zeros(N)
    kzz = np.zeros(N)
    kx[zf == SOLID] = k_metal
    kzz[zf == SOLID] = k_metal
    ha = np.zeros(N)
    Uc = flow["U"].ravel()
    for zid, name in case["lattice"].items():
        m = zf == zid
        kx[m] = k_metal * lat[name]["k_solid_ratio_x"]
        kzz[m] = k_metal * lat[name]["k_solid_ratio_z"]
        hf = lat[name]["heat_fit"]
        ha[m] = hf["c"] * np.maximum(Uc[m], 1e-4) ** hf["n"]
    rows, cols, vals = [], [], []
    diag = np.zeros(nt)
    rhs = np.zeros(nt)

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for ax in range(3):
        i, j = neighbours(shape, ax)
        kk = kzz if ax == 2 else kx
        # metal - metal conduction
        m = has_s[i] & has_s[j]
        a, b = sid[i[m]], sid[j[m]]
        g = h * 2.0 * kk[i[m]] * kk[j[m]] / (kk[i[m]] + kk[j[m]])
        add(a, b, -g)
        add(b, a, -g)
        np.add.at(diag, a, g)
        np.add.at(diag, b, g)
        # solid wall <-> open coolant film
        for (s_, f_) in ((i, j), (j, i)):
            m = (zf[s_] == SOLID) & (zf[f_] == OPEN)
            a, b = sid[s_[m]], fid[f_[m]]
            g = np.full(len(a), H_OPEN * h * h)
            add(a, b, -g)
            add(b, a, -g)
            np.add.at(diag, a, g)
            np.add.at(diag, b, g)
        # coolant conduction (small) between fluid cells
        m = has_f[i] & has_f[j]
        a, b = fid[i[m]], fid[j[m]]
        g = np.full(len(a), fluid["k"] * 0.5 * h)
        add(a, b, -g)
        add(b, a, -g)
        np.add.at(diag, a, g)
        np.add.at(diag, b, g)
    # lattice exchange
    m = zf >= 3
    a, b = sid[m], fid[m]
    g = ha[m] * h ** 3
    add(a, b, -g)
    add(b, a, -g)
    np.add.at(diag, a, g)
    np.add.at(diag, b, g)
    # advection (upwind) using the conservative face fluxes
    inlet = case["inlet"].ravel()
    outlet = case["outlet"].ravel()
    for (i, j), fl in zip(flow["pairs"], flow["flux"]):
        pos = fl > 0
        for up, dn, ff in ((i[pos], j[pos], fl[pos]), (j[~pos], i[~pos], -fl[~pos])):
            c = rho_cp * ff
            # enthalpy leaves `up`, enters `dn`
            np.add.at(diag, fid[up], c)
            add(fid[dn], fid[up], -c)
    # inlet: coolant enters at t_in through the inlet cells; outlet cells lose their outflow
    Q = flow["q_out"]
    cin = rho_cp * Q / inlet.sum()
    rhs[fid[inlet]] += cin * t_in
    # outlet sink: whatever volume arrives in an outlet cell leaves the domain
    net = np.zeros(N)
    for (i, j), fl in zip(flow["pairs"], flow["flux"]):
        np.add.at(net, j, fl)
        np.add.at(net, i, -fl)
    np.add.at(diag, fid[outlet], rho_cp * np.maximum(net[outlet], 0.0))
    # heaters
    heater = case["heater"].ravel()
    for hid, pw in case["powers"].items():
        m = heater == hid
        if m.sum() == 0:
            raise RuntimeError(f"heater {hid} has no cells")
        rhs[sid[m]] += pw / m.sum()
    Amat = sp.coo_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(nt)]), np.concatenate(cols + [np.arange(nt)]))),
                         shape=(nt, nt)).tocsr()
    import pyamg

    t0 = time.time()
    ml = pyamg.ruge_stuben_solver(Amat, max_coarse=2000, strength=("classical", {"theta": 0.25}))
    res = []
    x, info = spla.gmres(Amat, rhs, M=ml.aspreconditioner(cycle="V"), rtol=1e-9, restart=40, maxiter=40,
                         callback=lambda r: res.append(r), callback_type="pr_norm")
    true_res = float(np.linalg.norm(Amat @ x - rhs) / np.linalg.norm(rhs))
    if verbose:
        print(f"    heat: {nt:,} unknowns, gmres {len(res)} its, residual {true_res:.2e}, {time.time()-t0:.0f} s", flush=True)
    Ts = np.full(N, np.nan)
    Tf = np.full(N, np.nan)
    Ts[has_s] = x[:ns]
    Tf[has_f] = x[ns:]
    t_out = float((Tf[outlet] * np.maximum(net[outlet], 0)).sum() / max(np.maximum(net[outlet], 0).sum(), 1e-30))
    return dict(Ts=Ts.reshape(shape), Tf=Tf.reshape(shape), t_out=t_out, residual=true_res, unknowns=int(nt))


def run(case_name, material="AlSi10Mg", q_lpm=3.0, h=0.5, t_in=0.0, save_fields=True, verbose=True):
    t0 = time.time()
    fluid, lat = closures()
    case = CASES[case_name](h=h)
    zone = case["zone"]
    if verbose:
        cnt = {int(k): int((zone == k).sum()) for k in np.unique(zone)}
        print(f"[{case_name} | {material} | {q_lpm} L/min] grid {zone.shape}, zones {cnt}", flush=True)
    flow = solve_flow(case, fluid, lat, q_lpm, verbose=verbose)
    k = MATERIALS[material]["k"]
    heat = solve_heat(case, fluid, lat, flow, k, t_in=t_in, verbose=verbose)
    Ts, Tf = heat["Ts"], heat["Tf"]
    Q = q_lpm / 60000.0
    p_tot = float(sum(case["powers"].values()))
    q_adv = fluid["rho"] * fluid["cp"] * flow["q_out"] * (heat["t_out"] - t_in)
    heaters = {}
    for hid, pw in case["powers"].items():
        m = case["heater"] == hid
        heaters[str(hid)] = dict(name=case["heater_names"][hid], power_W=pw, cells=int(m.sum()),
                                 T_mean_rise_K=float(Ts[m].mean() - t_in), T_max_rise_K=float(Ts[m].max() - t_in),
                                 R_K_per_W=float((Ts[m].mean() - t_in) / pw),
                                 R_total_K_per_W=float((Ts[m].mean() - t_in) / p_tot))
    lat_used = {name: dict(A=lat[name]["forchheimer"]["A_Pa_s_m2"], B=lat[name]["forchheimer"]["B_Pa_s2_m3"],
                           heat_fit=lat[name]["heat_fit"], k_ratio_z=lat[name]["k_solid_ratio_z"]) for name in case["lattice"].values()}
    Ulat = flow["U"][zone == 3]
    summary = dict(case=case_name, material=material, k_metal=k, flow_lpm=q_lpm, flow_check_lpm=flow["q_out"] * 60000.0,
                   dp_Pa=flow["dp"], picard_iters=flow["iters"], voxel_mm=h,
                   T_solid_max_rise_K=float(np.nanmax(Ts) - t_in), T_out_rise_K=float(heat["t_out"] - t_in),
                   energy_in_W=p_tot, energy_out_W=float(q_adv), energy_closure_pct=float(100 * (q_adv / p_tot - 1)),
                   heaters=heaters, core_superficial_velocity=dict(mean=float(Ulat.mean()), p95=float(np.percentile(Ulat, 95)),
                                                                   max=float(Ulat.max())),
                   closures=lat_used, fluid=fluid, linear_residual=heat["residual"], unknowns=heat["unknowns"],
                   wall_time_s=round(time.time() - t0, 1), h_open_W_m2K=H_OPEN)
    tag = f"{case_name}_{material}_{q_lpm:g}"
    out = OUTD / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if save_fields:
        np.savez_compressed(out / "fields.npz", zone=zone, Ts=Ts.astype(np.float32), Tf=Tf.astype(np.float32),
                            U=flow["U"].astype(np.float32), ux=flow["comp"][0].astype(np.float32),
                            uy=flow["comp"][1].astype(np.float32), uz=flow["comp"][2].astype(np.float32),
                            p=flow["p"].astype(np.float32), origin_mm=np.array(case["grid"].lo), dx_mm=h,
                            inlet=case["inlet"], outlet=case["outlet"], heater=case["heater"])
    if verbose:
        hs = ", ".join(f"{v['name']}: +{v['T_mean_rise_K']:.1f} K (max +{v['T_max_rise_K']:.1f})" for v in heaters.values())
        print(f"  dp {flow['dp']/1000:.2f} kPa | {hs} | coolant +{summary['T_out_rise_K']:.2f} K | "
              f"closure {summary['energy_closure_pct']:+.2f} % | {summary['wall_time_s']} s", flush=True)
    return summary


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else "gpu_block", a[1] if len(a) > 1 else "AlSi10Mg", float(a[2]) if len(a) > 2 else 3.0)
