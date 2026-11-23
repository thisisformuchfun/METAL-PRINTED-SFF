"""Linear-elastic FEA of the assembled 316L skeleton (voxel hex-8 elements, AMG-CG).

The five printed segments are voxelised from their CAD fields, bonded at the bolted
lap joints, and loaded with the real component masses at their mounting pads.

    /opt/venv/bin/python -m sim.fea.frame_fea [voxel_mm]

Load cases
    LC1  standing on its four feet, 1 g
    LC2  standing, 1 g down + 1 g sideways (transport / knock)
    LC3  lifted by the crown, 1 g (everything hangs from the top of the spine)
Outputs: sim/out/fea/summary.json, per-part vertex data for visualisation.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from design import frame as FRM
from design import params as P
from design import sdf

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "sim" / "out" / "fea"

E_MOD = 180e9          # Pa, SLM 316L (JLC3DP datasheet: 180 GPa)
NU = 0.30
RHO = 7980.0           # kg/m3
YIELD = 400e6          # Pa (JLC3DP datasheet)
G0 = 9.81


def hex_ke(h, E, nu):
    """24x24 stiffness of a cube element of side h (m); also B at the centroid and D."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    xi = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)

    def Bmat(p):
        dN = np.zeros((8, 3))
        for a in range(8):
            for d in range(3):
                t = xi[a, d] / 8.0
                for e in range(3):
                    if e != d:
                        t *= 1 + xi[a, e] * p[e]
                dN[a, d] = t * 2.0 / h
        B = np.zeros((6, 24))
        for a in range(8):
            B[0, 3 * a] = dN[a, 0]
            B[1, 3 * a + 1] = dN[a, 1]
            B[2, 3 * a + 2] = dN[a, 2]
            B[3, 3 * a], B[3, 3 * a + 1] = dN[a, 1], dN[a, 0]
            B[4, 3 * a + 1], B[4, 3 * a + 2] = dN[a, 2], dN[a, 1]
            B[5, 3 * a], B[5, 3 * a + 2] = dN[a, 2], dN[a, 0]
        return B

    g = 1 / np.sqrt(3)
    Ke = np.zeros((24, 24))
    for a in (-g, g):
        for b in (-g, g):
            for c in (-g, g):
                B = Bmat((a, b, c))
                Ke += B.T @ D @ B * (h ** 3 / 8.0)
    return Ke, Bmat((0, 0, 0)), D


def voxelise(hv):
    """Global occupancy grid (bool) of the bonded frame + per-voxel part id."""
    lo = np.array([-134.0, 12.0, 0.0])
    hi = np.array([132.0, 310.0, 386.0])
    n = np.ceil((hi - lo) / hv).astype(int)
    occ = np.zeros(n, dtype=np.uint8)
    cx = [lo[i] + hv * (np.arange(n[i]) + 0.5) for i in range(3)]
    for pid, (name, fn) in enumerate(FRM.SEGMENTS.items(), start=1):
        f, g = fn(min(0.75, hv / 2))
        sl = []
        for i in range(3):
            a = int(np.searchsorted(cx[i], g.lo[i] + g.h))
            b = int(np.searchsorted(cx[i], g.hi[i] - g.h))
            sl.append(slice(a, max(b, a)))
        X, Y, Z = np.meshgrid(*[(cx[i][sl[i]] - g.lo[i]) / g.h for i in range(3)], indexing="ij")
        val = ndi.map_coordinates(f, [X, Y, Z], order=1, mode="nearest")
        sub = occ[tuple(sl)]
        sub[(val < 0) & (sub == 0)] = pid
        print(f"  voxelised {name}: {(val < 0).sum():,} voxels", flush=True)
    # bond the bolted joints (the clamped laps act as solid knuckles)
    Xc, Yc, Zc = cx[0][:, None, None], cx[1][None, :, None], cx[2][None, None, :]
    for J in FRM.JOINTS.values():
        a = J["p0"] + J["d"] * (J["t"][0] + 2.0)
        b = J["p0"] + J["d"] * (J["t"][1] - 2.0)
        d = sdf.capsule(Xc, Yc, Zc, a, b, FRM.R_SPINE)
        occ[(d < 0) & (occ == 0)] = 9
    top = FRM.S(0, FRM.SPINE_TOP_S)
    d = sdf.sphere(Xc, Yc, Zc, top, FRM.R_SPINE + 1.0)
    occ[(d < 0) & (occ == 0)] = 9
    lab, nl = ndi.label(occ > 0)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    dropped = int(((lab > 0) & (lab != keep)).sum())
    occ[(lab > 0) & (lab != keep)] = 0
    print(f"  bonded frame: {int((occ > 0).sum()):,} voxels, {nl} components before bonding check, dropped {dropped}")
    return occ, lo, hv


def load_points():
    """(world point, mass kg, label) for every lumped component load."""
    L = []
    gp = FRM.gpu_points()
    m_gpu = P.GPU["mass_pcb_kg"] + 0.507 + 0.06 + 0.10
    for name, (x, s) in gp.items():
        L.append((FRM.S(x, s, FRM.N_GPU_PAD - 2.0), m_gpu / 4, "gpu"))
    m_brd = P.BOARD["mass_kg"] + 0.08 + 0.199 + 0.031 + 0.20
    for (x, s) in FRM.board_points():
        L.append((FRM.S(x, s, FRM.N_BRD_PAD - 2.0), m_brd / 4, "board"))
    m_heart = 0.22 + 0.155 + 0.06
    for (x, s) in FRM.heart_ears():
        L.append((FRM.S(x, s, FRM.N_HEART - 3.0), m_heart / 3, "heart"))
    m_rad = P.MASSES["radiator_wet"]
    for (x, z) in P.rad_holes():
        L.append((np.array([x, P.RAD["y_rear"] + 2.0, z]), m_rad / 8, "radiator"))
    psu_lo, psu_sz = np.array(P.PSU["lo"]), np.array(P.PSU["size"])
    yo = psu_lo[1] + psu_sz[1] / 2
    for (px, py) in ((0, psu_lo[1] + 18), (0, psu_lo[1] + psu_sz[1] - 18), (-68, yo), (68, yo)):
        L.append((np.array([px, py, psu_lo[2] - 1.5]), P.PSU["mass_kg"] / 4, "psu"))
    return L


def solve(hv=1.5):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    occ, lo, hv = voxelise(hv)
    h = hv * 1e-3
    solid = occ > 0
    nx, ny, nz = solid.shape
    el = np.argwhere(solid)
    ne = len(el)
    nid = -np.ones((nx + 1, ny + 1, nz + 1), dtype=np.int64)
    off = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    corner = el[:, None, :] + off[None, :, :]
    flat = np.ravel_multi_index((corner[..., 0], corner[..., 1], corner[..., 2]), nid.shape)
    uniq, inv = np.unique(flat, return_inverse=True)
    conn = inv.reshape(ne, 8)
    nn = len(uniq)
    ncoord = np.stack(np.unravel_index(uniq, nid.shape), axis=1) * hv + lo        # mm
    print(f"  mesh: {ne:,} hex elements, {nn:,} nodes, {3 * nn:,} dof  ({time.time() - t0:.0f}s)", flush=True)

    Ke, Bc, D = hex_ke(h, E_MOD, NU)
    dofs = (3 * conn[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 24)
    rows = np.repeat(dofs, 24, axis=1).ravel()
    cols = np.tile(dofs, (1, 24)).ravel()
    data = np.tile(Ke.ravel(), ne)
    K = sp.coo_matrix((data, (rows, cols)), shape=(3 * nn, 3 * nn)).tocsr()
    del rows, cols, data
    print(f"  K assembled: nnz {K.nnz:,}  ({time.time() - t0:.0f}s)", flush=True)

    ecen = (el + 0.5) * hv + lo                                                    # mm
    tree_n = cKDTree(ncoord)
    # self weight: lumped to nodes
    node_mass = np.bincount(conn.ravel(), minlength=nn) * (RHO * h ** 3 / 8.0)
    loads = load_points()
    lump = np.zeros(nn)
    for (pt, m, _) in loads:
        idx = tree_n.query_ball_point(pt, r=5.5)
        if not idx:
            idx = [tree_n.query(pt)[1]]
        lump[idx] += m / len(idx)
    mass_frame = float(node_mass.sum())
    mass_comp = float(lump.sum())
    mtot = node_mass + lump

    feet = [np.array([fx, fy, 2.0]) for (fx, fy) in P.FRAME["feet"]]
    fix_feet = np.zeros(nn, dtype=bool)
    for ft in feet:
        idx = tree_n.query_ball_point(ft, r=7.5)
        idx = [i for i in idx if ncoord[i, 2] < 3.6]
        fix_feet[idx] = True
    crown = np.array(FRM.crown_path()[4])
    fix_crown = np.zeros(nn, dtype=bool)
    fix_crown[tree_n.query_ball_point(crown, r=20.0)] = True

    import pyamg
    import scipy.sparse.linalg as spla

    AMG = {}

    def run(name, fixed, gvec):
        f = np.zeros(3 * nn)
        for c in range(3):
            f[c::3] = mtot * gvec[c]
        free = np.repeat(~fixed, 3)
        Kff = K[free][:, free]
        Bn = np.zeros((3 * nn, 6))
        Bn[0::3, 0] = 1
        Bn[1::3, 1] = 1
        Bn[2::3, 2] = 1
        x, y, z = (ncoord * 1e-3).T
        Bn[0::3, 3], Bn[1::3, 3] = -y, x
        Bn[1::3, 4], Bn[2::3, 4] = -z, y
        Bn[0::3, 5], Bn[2::3, 5] = z, -x
        key = fixed.tobytes()
        if key not in AMG:
            AMG[key] = pyamg.smoothed_aggregation_solver(
                Kff.tobsr(blocksize=(3, 3)), B=Bn[free], max_coarse=500,
                presmoother=("block_gauss_seidel", {"sweep": "symmetric", "iterations": 2}),
                postsmoother=("block_gauss_seidel", {"sweep": "symmetric", "iterations": 2}))
        ml = AMG[key]
        res = []
        ff = f[free]
        bn = np.linalg.norm(ff)

        def cb(xk):
            res.append(1.0)

        uf, info = spla.cg(Kff, ff, M=ml.aspreconditioner(cycle="V"), rtol=1e-9, maxiter=3000, callback=cb)
        true_res = float(np.linalg.norm(Kff @ uf - ff) / bn)
        res = [1.0] * len(res) + [true_res]
        u = np.zeros(3 * nn)
        u[free] = uf
        ue = u[dofs]                                           # (ne, 24)
        sig = (ue @ Bc.T) @ D.T                                # (ne, 6)
        s = sig
        vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                     + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
        umag = np.linalg.norm(u.reshape(-1, 3), axis=1)
        # smooth element stress over the 26-neighbourhood to tame voxel stair-step peaks
        vol = np.zeros(solid.shape)
        vol[el[:, 0], el[:, 1], el[:, 2]] = vm
        cnt = ndi.uniform_filter(solid.astype(float), 3, mode="constant")
        vms = (ndi.uniform_filter(vol, 3, mode="constant") / np.maximum(cnt, 1e-9))[el[:, 0], el[:, 1], el[:, 2]]
        react = -(K @ u - f).reshape(-1, 3)[fixed].sum(axis=0)
        out = dict(name=name, cg_iters=len(res) - 1, rel_residual=true_res,
                   max_disp_mm=float(umag.max() * 1e3),
                   vm_max_raw_MPa=float(vm.max() / 1e6), vm_max_smoothed_MPa=float(vms.max() / 1e6),
                   vm_p999_MPa=float(np.percentile(vms, 99.9) / 1e6), vm_p99_MPa=float(np.percentile(vms, 99) / 1e6),
                   safety_factor_yield=float(YIELD / max(vms.max(), 1.0)),
                   reaction_N=[float(v) for v in react], applied_N=[float(mtot.sum() * g) for g in gvec])
        print("  ", json.dumps(out), flush=True)
        return out, u.reshape(-1, 3), vms

    cases = []
    r1, u1, vm1 = run("LC1 standing 1g", fix_feet, (0, 0, -G0))
    r2, u2, vm2 = run("LC2 standing 1g + 1g lateral", fix_feet, (G0, 0, -G0))
    r3, u3, vm3 = run("LC3 lifted by the crown 1g", fix_crown, (0, 0, -G0))
    cases = [r1, r2, r3]

    # where is the worst spot?
    for r, vm in ((r1, vm1), (r2, vm2), (r3, vm3)):
        i = int(vm.argmax())
        r["vm_max_location_mm"] = [float(v) for v in ecen[i]]
        r["vm_max_part"] = list(FRM.SEGMENTS)[int(occ[tuple(el[i])]) - 1] if occ[tuple(el[i])] <= 5 else "joint"

    summary = dict(voxel_mm=hv, elements=int(ne), nodes=int(nn), dof=int(3 * nn),
                   material=dict(E_GPa=E_MOD / 1e9, nu=NU, rho=RHO, yield_MPa=YIELD / 1e6,
                                 source="JLC3DP 316L SLM datasheet values"),
                   mass_frame_kg=mass_frame, mass_components_kg=mass_comp,
                   loads=[dict(label=l, mass_kg=round(m, 4), at=[float(v) for v in p]) for (p, m, l) in loads],
                   cases=cases, wall_time_s=time.time() - t0,
                   notes=["joints modelled as bonded (clamped by 2 x M5 each)",
                          "voxel hex-8 mesh: bending of the slimmest ribs is resolved by only ~4-6 elements, "
                          "so deflections are likely under-predicted by 10-25 %; stresses are neighbourhood-averaged"])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(OUT / "fields.npz", ecen=ecen.astype(np.float32), ncoord=ncoord.astype(np.float32),
                        u1=u1.astype(np.float32), u2=u2.astype(np.float32), u3=u3.astype(np.float32),
                        vm1=vm1.astype(np.float32), vm2=vm2.astype(np.float32), vm3=vm3.astype(np.float32))
    print("  wrote", OUT / "summary.json", f"({time.time() - t0:.0f}s)")
    return summary


if __name__ == "__main__":
    solve(float(sys.argv[1]) if len(sys.argv) > 1 else 1.5)
