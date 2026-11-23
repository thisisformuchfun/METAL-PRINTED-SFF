"""Resolved flow + heat transfer in the gyroid lattices used inside the MARROW cold plates.

Scale 1 of the two-scale simulation: a periodic patch of the real lattice (with the real
floor / ceiling walls of the layer it sits in) is resolved on a voxel grid.

  flow    D3Q19 lattice-Boltzmann, TRT collision (magic parameter 3/16) + Smagorinsky
          sub-grid viscosity, halfway bounce-back walls, body-force driven, periodic.
          -> pressure gradient vs superficial velocity  (Darcy-Forchheimer fit)
  heat    finite-volume scalar transport (van Leer TVD advection + central diffusion) on
          the time-mean velocity through a train of 5 cells with isothermal metal.
          -> volumetric heat-transfer coefficient h*a(U)  [W/m3K]
  solid   steady conduction through the metal phase only
          -> effective solid conductivity ratio k_eff / k_metal

    /opt/venv/bin/python -m sim.cell.unit_cell            # runs all three lattices
Outputs sim/out/cell/closures.json and the velocity field of one case for visualisation.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numba as nb
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "sim" / "out" / "cell"

# coolant: premixed water / ~25 % glycol at 40 C  [ASM]
FLUID = dict(rho=1030.0, cp=3850.0, k=0.50, nu=1.0e-6)
FLUID["alpha"] = FLUID["k"] / (FLUID["rho"] * FLUID["cp"])
FLUID["Pr"] = FLUID["nu"] / FLUID["alpha"]

# D3Q19
CX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.int64)
CY = np.array([0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int64)
CZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1], dtype=np.int64)
WQ = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=np.float64)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int64)


@nb.njit(parallel=True, fastmath=True, cache=True)
def lbm_step(f, fn, solid, cx, cy, cz, w, opp, nu, lam, gx, cs2):
    nx, ny, nz = solid.shape
    tau0 = 3.0 * nu + 0.5
    for x in nb.prange(nx):
        fl = np.empty(19)
        fe = np.empty(19)
        for y in range(ny):
            for z in range(nz):
                if solid[x, y, z]:
                    continue
                rho = 0.0
                ux = 0.0
                uy = 0.0
                uz = 0.0
                for i in range(19):
                    xs = x - cx[i]
                    ys = y - cy[i]
                    zs = z - cz[i]
                    if xs < 0:
                        xs += nx
                    elif xs >= nx:
                        xs -= nx
                    if ys < 0:
                        ys += ny
                    elif ys >= ny:
                        ys -= ny
                    if zs < 0:
                        zs += nz
                    elif zs >= nz:
                        zs -= nz
                    if solid[xs, ys, zs]:
                        v = f[opp[i], x, y, z]
                    else:
                        v = f[i, xs, ys, zs]
                    fl[i] = v
                    rho += v
                    ux += v * cx[i]
                    uy += v * cy[i]
                    uz += v * cz[i]
                ux = (ux + 0.5 * gx * rho) / rho
                uy /= rho
                uz /= rho
                usq = 1.5 * (ux * ux + uy * uy + uz * uz)
                pxx = 0.0
                pyy = 0.0
                pzz = 0.0
                pxy = 0.0
                pxz = 0.0
                pyz = 0.0
                for i in range(19):
                    cu = 3.0 * (cx[i] * ux + cy[i] * uy + cz[i] * uz)
                    fe[i] = w[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
                    d = fl[i] - fe[i]
                    pxx += d * cx[i] * cx[i]
                    pyy += d * cy[i] * cy[i]
                    pzz += d * cz[i] * cz[i]
                    pxy += d * cx[i] * cy[i]
                    pxz += d * cx[i] * cz[i]
                    pyz += d * cy[i] * cz[i]
                tau = tau0
                if cs2 > 0.0:
                    q = math.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz))
                    tau = 0.5 * (tau0 + math.sqrt(tau0 * tau0 + 18.0 * 1.41421356 * cs2 * q / rho))
                omp = 1.0 / tau
                omm = 1.0 / (lam / (tau - 0.5) + 0.5)
                fn[0, x, y, z] = fl[0] - omp * (fl[0] - fe[0])
                for i in range(1, 19, 2):
                    j = i + 1
                    fp = 0.5 * (fl[i] + fl[j])
                    fm = 0.5 * (fl[i] - fl[j])
                    ep = 0.5 * (fe[i] + fe[j])
                    em = 0.5 * (fe[i] - fe[j])
                    dp = omp * (fp - ep)
                    dm = omm * (fm - em)
                    # Guo-type forcing reduced to its leading term (uniform body force along x)
                    fo = 3.0 * w[i] * cx[i] * gx * rho * (1.0 - 0.5 * omm)
                    fn[i, x, y, z] = fl[i] - dp - dm + fo
                    fn[j, x, y, z] = fl[j] - dp + dm - fo


@nb.njit(parallel=True, fastmath=True, cache=True)
def moments(f, solid, cx, cy, cz, gx, ux, uy, uz):
    nx, ny, nz = solid.shape
    for x in nb.prange(nx):
        for y in range(ny):
            for z in range(nz):
                if solid[x, y, z]:
                    ux[x, y, z] = 0.0
                    uy[x, y, z] = 0.0
                    uz[x, y, z] = 0.0
                    continue
                rho = 0.0
                a = 0.0
                b = 0.0
                c = 0.0
                for i in range(19):
                    v = f[i, x, y, z]
                    rho += v
                    a += v * cx[i]
                    b += v * cy[i]
                    c += v * cz[i]
                ux[x, y, z] = (a + 0.5 * gx * rho) / rho
                uy[x, y, z] = b / rho
                uz[x, y, z] = c / rho


def run_flow(solid, dx, dpdx, nu_phys=FLUID["nu"], rho_phys=FLUID["rho"], u_peak_guess=None, cs=0.12,
             settle=8.0, avg=4.0, max_steps=120000, verbose=True, perm_guess=None):
    """Body-force driven periodic flow. dpdx in Pa/m. Returns mean velocity field (m/s) + diagnostics."""
    nx, ny, nz = solid.shape
    phi = 1.0 - solid.mean()
    if u_peak_guess is None:
        u_peak_guess = 0.3
    nu_lb = float(np.clip(0.09 * nu_phys / (u_peak_guess * dx), 2.5e-4, 0.1))
    dt = nu_lb * dx * dx / nu_phys
    g_lb = dpdx / rho_phys * dt * dt / dx
    if nu_lb >= 0.002:
        cs = 0.0                 # laminar-resolved runs need no sub-grid model (and V1 shows it biases them)
    f = np.zeros((19,) + solid.shape)
    for i in range(19):
        f[i] = WQ[i]
    fn = f.copy()
    ux = np.zeros(solid.shape)
    uy = np.zeros(solid.shape)
    uz = np.zeros(solid.shape)
    # time scale for the flow to respond: tau ~ K / (nu * phi); refine on the fly
    acc = np.zeros((3,) + solid.shape)
    nacc = 0
    hist = []
    t0 = time.time()
    chunk = 250
    step = 0
    settled_at = None
    umax = 0.0
    while step < max_steps:
        for _ in range(chunk // 2):
            lbm_step(f, fn, solid, CX, CY, CZ, WQ, OPP, nu_lb, 3.0 / 16.0, g_lb, cs * cs)
            lbm_step(fn, f, solid, CX, CY, CZ, WQ, OPP, nu_lb, 3.0 / 16.0, g_lb, cs * cs)
        step += chunk
        moments(f, solid, CX, CY, CZ, g_lb, ux, uy, uz)
        U = float(ux.mean())                       # superficial (averaged over metal too)
        umax = float(np.sqrt(ux * ux + uy * uy + uz * uz).max())
        if not np.isfinite(U) or umax > 0.45:
            raise FloatingPointError(f"LBM diverged (U={U}, umax={umax}) at step {step}")
        hist.append((step, U))
        if settled_at is None:
            # settled when the superficial velocity changed < 0.5 % over the last quarter of the history
            n = len(hist)
            if n >= 12:
                a = np.mean([h[1] for h in hist[n // 2: 3 * n // 4]])
                b = np.mean([h[1] for h in hist[3 * n // 4:]])
                if abs(b - a) < 0.005 * abs(b) and step * dt > 0.02:
                    settled_at = step
        else:
            acc[0] += ux
            acc[1] += uy
            acc[2] += uz
            nacc += 1
            if step - settled_at >= max(0.6 * settled_at, 3000):
                break
    if nacc == 0:
        acc[0], acc[1], acc[2] = ux, uy, uz
        nacc = 1
    um = acc / nacc * dx / dt                        # m/s
    U = float(um[0].mean())
    out = dict(dpdx_Pa_m=float(dpdx), U_superficial=U, U_interstitial=U / phi, porosity=float(phi),
               u_max=float(np.sqrt((um ** 2).sum(0)).max()), nu_lb=nu_lb, dt=dt, steps=step,
               settled_at=settled_at, converged=settled_at is not None, mach_peak=umax * math.sqrt(3.0), smagorinsky=cs,
               seconds=round(time.time() - t0, 1), cells=int(solid.size))
    if verbose:
        print(f"    dp/dx {dpdx/1000:7.2f} kPa/m -> U {U:6.3f} m/s (peak {out['u_max']:.2f}), steps {step}, "
              f"settled {settled_at}, Ma {out['mach_peak']:.2f}, {out['seconds']} s", flush=True)
    return um, out


# ----------------------------------------------------------------------------
# scalar transport (heat) on a frozen mean velocity field
# ----------------------------------------------------------------------------
@nb.njit(fastmath=True, cache=True)
def _vl(r):
    return (r + abs(r)) / (1.0 + abs(r))


@nb.njit(parallel=True, fastmath=True, cache=True)
def scalar_step(T, Tn, solid, fx, fy, fz, alpha, dt):
    """fx/fy/fz: face velocities (lattice units) on the -x/-y/-z faces of each cell."""
    nx, ny, nz = T.shape
    for x in nb.prange(1, nx - 1):
        for y in range(ny):
            ym = y - 1 if y > 0 else ny - 1
            yp = y + 1 if y < ny - 1 else 0
            ymm = ym - 1 if ym > 0 else ny - 1
            ypp = yp + 1 if yp < ny - 1 else 0
            for z in range(nz):
                if solid[x, y, z]:
                    Tn[x, y, z] = 0.0
                    continue
                zm = z - 1 if z > 0 else nz - 1
                zp = z + 1 if z < nz - 1 else 0
                zmm = zm - 1 if zm > 0 else nz - 1
                zpp = zp + 1 if zp < nz - 1 else 0
                t0 = T[x, y, z]
                div = 0.0
                adv = 0.0
                dif = 0.0
                # ---- x faces
                xm2 = x - 2 if x >= 2 else 0
                xp2 = x + 2 if x <= nx - 3 else nx - 1
                for s in range(2):
                    if s == 0:
                        u = fx[x, y, z]
                        a, b, c, d = T[xm2, y, z], T[x - 1, y, z], t0, T[x + 1, y, z]
                        sol = solid[x - 1, y, z]
                    else:
                        u = fx[x + 1, y, z]
                        a, b, c, d = T[x - 1, y, z], t0, T[x + 1, y, z], T[xp2, y, z]
                        sol = solid[x + 1, y, z]
                    if u >= 0:
                        den = c - b
                        r = (b - a) / den if abs(den) > 1e-12 else 0.0
                        tf = b + 0.5 * _vl(r) * den
                    else:
                        den = b - c
                        r = (c - d) / den if abs(den) > 1e-12 else 0.0
                        tf = c + 0.5 * _vl(r) * den
                    if s == 0:
                        adv += u * tf
                        div -= u
                        dif += (2.0 * (0.0 - t0)) if sol else (b - t0)
                    else:
                        adv -= u * tf
                        div += u
                        dif += (2.0 * (0.0 - t0)) if sol else (c - t0)
                # ---- y faces
                for s in range(2):
                    if s == 0:
                        u = fy[x, y, z]
                        a, b, c, d = T[x, ymm, z], T[x, ym, z], t0, T[x, yp, z]
                        sol = solid[x, ym, z]
                    else:
                        u = fy[x, yp, z]
                        a, b, c, d = T[x, ym, z], t0, T[x, yp, z], T[x, ypp, z]
                        sol = solid[x, yp, z]
                    if u >= 0:
                        den = c - b
                        r = (b - a) / den if abs(den) > 1e-12 else 0.0
                        tf = b + 0.5 * _vl(r) * den
                    else:
                        den = b - c
                        r = (c - d) / den if abs(den) > 1e-12 else 0.0
                        tf = c + 0.5 * _vl(r) * den
                    if s == 0:
                        adv += u * tf
                        div -= u
                        dif += (2.0 * (0.0 - t0)) if sol else (b - t0)
                    else:
                        adv -= u * tf
                        div += u
                        dif += (2.0 * (0.0 - t0)) if sol else (c - t0)
                # ---- z faces
                for s in range(2):
                    if s == 0:
                        u = fz[x, y, z]
                        a, b, c, d = T[x, y, zmm], T[x, y, zm], t0, T[x, y, zp]
                        sol = solid[x, y, zm]
                    else:
                        u = fz[x, y, zp]
                        a, b, c, d = T[x, y, zm], t0, T[x, y, zp], T[x, y, zpp]
                        sol = solid[x, y, zp]
                    if u >= 0:
                        den = c - b
                        r = (b - a) / den if abs(den) > 1e-12 else 0.0
                        tf = b + 0.5 * _vl(r) * den
                    else:
                        den = b - c
                        r = (c - d) / den if abs(den) > 1e-12 else 0.0
                        tf = c + 0.5 * _vl(r) * den
                    if s == 0:
                        adv += u * tf
                        div -= u
                        dif += (2.0 * (0.0 - t0)) if sol else (b - t0)
                    else:
                        adv -= u * tf
                        div += u
                        dif += (2.0 * (0.0 - t0)) if sol else (c - t0)
                Tn[x, y, z] = t0 + dt * (adv + t0 * div + alpha * dif)


def face_velocities(u, solid):
    """Cell-centred -> face-normal velocities on the minus faces (zero through metal), periodic in y, z."""
    f = []
    for ax in range(3):
        v = u[ax]
        vm = np.roll(v, 1, axis=ax)
        fv = 0.5 * (v + vm)
        fv[solid | np.roll(solid, 1, axis=ax)] = 0.0
        f.append(np.ascontiguousarray(fv))
    return f


def run_heat(um, solid, dx, n_cells=5, alpha=FLUID["alpha"], verbose=True):
    """Isothermal-wall heat transfer through a train of n_cells copies along x.
    Returns the volumetric coefficient h*a = rho cp U / L_T  [W/m3K]."""
    t0 = time.time()
    nx = solid.shape[0]
    sol = np.concatenate([solid] * n_cells, axis=0)
    u = np.concatenate([um] * n_cells, axis=1)
    sol = np.ascontiguousarray(sol)
    umax = float(np.abs(u).max())
    dt = min(0.25 * dx / umax, 0.12 * dx * dx / alpha)          # s
    u_lb = u * dt / dx
    a_lb = alpha * dt / (dx * dx)
    fx, fy, fz = face_velocities(u_lb, sol)
    fx = np.concatenate([fx, fx[:1]], axis=0)                    # +x face of the last cell
    T = np.zeros(sol.shape)
    T[0] = np.where(sol[0], 0.0, 1.0)
    Tn = T.copy()
    U = float(um[0].mean())
    flow_through = sol.shape[0] * dx / max(U / (1 - solid.mean()), 1e-6)
    nsteps = int(min(max(3.0 * flow_through / dt, 3000), 250000))
    prev = None
    for it in range(nsteps // 500 + 1):
        for _ in range(250):
            scalar_step(T, Tn, sol, fx, fy, fz, a_lb, 1.0)
            Tn[0] = T[0]
            Tn[-1] = Tn[-2]
            scalar_step(Tn, T, sol, fx, fy, fz, a_lb, 1.0)
            T[0] = Tn[0]
            T[-1] = T[-2]
        flux = (fx[:-1] * T).sum(axis=(1, 2))
        q = fx[:-1].sum(axis=(1, 2))
        tb = flux / np.maximum(q, 1e-30)
        cur = tb[nx * (n_cells - 1)]
        if prev is not None and abs(cur - prev) < 2e-4 * max(abs(cur), 1e-9) and it > 6:
            break
        prev = cur
    xs = np.arange(sol.shape[0]) * dx
    idx = [nx * k for k in range(1, n_cells)]
    tbk = np.array([tb[i] for i in idx])
    ok = tbk > 1e-6
    if ok.sum() >= 2:
        slope = np.polyfit(xs[idx][ok], np.log(tbk[ok]), 1)[0]
        LT = -1.0 / slope
    else:
        LT = float("nan")
    ha = FLUID["rho"] * FLUID["cp"] * U / LT
    out = dict(U_superficial=U, L_T_mm=LT * 1e3, ha_W_m3K=float(ha), Tb_cells=[float(v) for v in tbk],
               seconds=round(time.time() - t0, 1), steps=int((it + 1) * 500))
    if verbose:
        print(f"    heat: U {U:6.3f} m/s -> L_T {LT*1e3:6.2f} mm, h*a {ha/1e6:6.2f} MW/m3K, Tb {np.round(tbk, 3)}, {out['seconds']} s", flush=True)
    return out


def solid_conductivity_ratio(solid, axis=2):
    """k_eff/k of the metal phase for conduction along `axis` (Dirichlet ends, periodic sides)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    s = np.moveaxis(solid, axis, 0)
    n0 = s.shape[0]
    idx = -np.ones(s.shape, dtype=np.int64)
    idx[s] = np.arange(int(s.sum()))
    N = int(s.sum())
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    b = np.zeros(N)
    for ax in range(3):
        for sh in (1, -1):
            nb_idx = np.roll(idx, -sh, axis=ax)
            valid = s & (nb_idx >= 0)
            if ax == 0:                                  # no wrap along the conduction axis
                edge = np.zeros(s.shape, dtype=bool)
                if sh == 1:
                    edge[-1] = True
                else:
                    edge[0] = True
                valid &= ~edge
            i = idx[valid]
            j = nb_idx[valid]
            rows.append(i)
            cols.append(j)
            vals.append(-np.ones(len(i)))
            np.add.at(diag, i, 1.0)
    # Dirichlet: T = 1 at face below index 0, T = 0 above the last (half-cell distance -> conductance 2)
    i0 = idx[0][s[0]]
    i1 = idx[-1][s[-1]]
    diag[i0] += 2.0
    b[i0] += 2.0
    diag[i1] += 2.0
    A = sp.coo_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(N)]), np.concatenate(cols + [np.arange(N)]))), shape=(N, N)).tocsr()
    T, info = spla.cg(A, b, rtol=1e-8, maxiter=5000)
    q = 2.0 * (1.0 - T[i0]).sum()                        # total flux (per unit k, dx)
    area = s.shape[1] * s.shape[2]
    return float(q * n0 / area)


# ----------------------------------------------------------------------------
# lattice patches
# ----------------------------------------------------------------------------
def gyroid_patch(cell, wall, dx, lz=None, root=0.0, root_len=1.3, plates=True, phase=(0.0, 0.0, 0.0), kind="sheet", bias=0.0):
    """Sheet-gyroid patch, periodic in x and y over one cell. If lz is given the layer is closed by solid
    plates in z (the real floor and ceiling); otherwise fully periodic."""
    n = int(round(cell / dx))
    nzl = n if lz is None else int(round(lz / dx))
    a = 2 * np.pi / cell
    c = (np.arange(n) + 0.5) * dx
    cz = (np.arange(nzl) + 0.5) * dx
    X, Y, Z = np.meshgrid(c * a + phase[0], c * a + phase[1], cz * a + phase[2], indexing="ij")
    g = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
    gx = np.cos(X) * np.cos(Y) - np.sin(Z) * np.sin(X)
    gy = np.cos(Y) * np.cos(Z) - np.sin(X) * np.sin(Y)
    gz = np.cos(Z) * np.cos(X) - np.sin(Y) * np.sin(Z)
    gm = np.maximum(np.sqrt(gx * gx + gy * gy + gz * gz), 0.35) * a
    w = wall + root * np.exp(-cz / root_len)[None, None, :]
    solid = (np.abs(g) / gm < 0.5 * w) if kind == "sheet" else (g / gm + bias < 0.0)
    if lz is not None and plates:
        pad = np.ones((n, n, 2), dtype=bool)
        solid = np.concatenate([pad, solid, pad], axis=2)
    return np.ascontiguousarray(solid)


LATTICES = dict(
    core=dict(cell=5.0e-3, wall=0.85e-3, lz=6.5e-3, root=0.7e-3, dp=[1.5e3, 6e3, 20e3, 60e3, 180e3], upk=[0.15, 0.45, 0.9, 1.6, 2.8], max_steps=30000,
              note="5 mm sheet gyroid, 0.85 mm wall with root thickening, between floor and divider (6.5 mm)"),
    gallery=dict(cell=8.0e-3, wall=0.9e-3, lz=5.0e-3, root=0.0, dp=[0.4e3, 1.5e3, 5e3, 15e3], upk=[0.15, 0.4, 0.8, 1.4],
                 note="8 mm sheet gyroid, 0.9 mm wall, 5 mm layer between divider and roof"),
    arm=dict(cell=12.0e-3, wall=0.0, lz=14.0e-3, root=0.0, kind="network", bias=1.4e-3, dx=2.5e-4, max_steps=20000,
             dp=[0.5e3, 2e3, 6e3], upk=[0.5, 1.0, 1.8],
             note="12 mm skeletal (strut) gyroid, ~2 mm struts, 14 mm tall VRM-arm channel"),
)


def fit_forchheimer(U, dp):
    """dp/dx = A U + B U^2 (least squares, non-negative)."""
    U, dp = np.asarray(U), np.asarray(dp)
    M = np.stack([U, U * U], axis=1)
    coef, *_ = np.linalg.lstsq(M / dp[:, None], np.ones(len(U)), rcond=None)
    A, B = float(max(coef[0], 0.0)), float(max(coef[1], 0.0))
    return A, B


def run_lattice(name, dx=1.25e-4, heat=True, save_field=False):
    spec = LATTICES[name]
    dx = spec.get("dx", dx)
    solid = gyroid_patch(spec["cell"], spec["wall"], dx, lz=spec["lz"], root=spec["root"], kind=spec.get("kind", "sheet"),
                         bias=spec.get("bias", 0.0))
    phi = 1.0 - solid[:, :, 2:-2].mean()
    print(f"[{name}] {spec['note']}; grid {solid.shape}, dx {dx*1e3:.3f} mm, layer porosity {phi:.3f}", flush=True)
    runs = []
    for dp, upk in zip(spec["dp"], spec["upk"]):
        try:
            um, r = run_flow(solid, dx, dp, u_peak_guess=upk, max_steps=spec.get("max_steps", 120000))
        except FloatingPointError as ex:
            print("    skipped:", ex, flush=True)
            continue
        # report superficial velocity over the fluid layer only (exclude the two plate cells each side)
        layer = um[:, :, 2:-2]
        r["U_superficial"] = float(layer[0].mean())
        Dh = None
        if heat:
            h = run_heat(um, solid, dx)
            h["U_superficial"] = r["U_superficial"]
            lay_frac = (solid.shape[2] - 4) / solid.shape[2]
            h["ha_W_m3K"] = h["ha_W_m3K"] / lay_frac           # per unit volume of the layer itself
            r["heat"] = h
        runs.append(r)
        if save_field and dp == spec["dp"][3]:
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT / f"{name}_field.npz", u=um.astype(np.float32), solid=solid, dx=dx, dpdx=dp)
    U = [r["U_superficial"] for r in runs]
    dps = [r["dpdx_Pa_m"] for r in runs]
    A, B = fit_forchheimer(U, dps)
    res = dict(name=name, spec={k: v for k, v in spec.items() if k not in ("dp", "upk")}, dx=dx, porosity=float(phi),
               runs=runs, forchheimer=dict(A_Pa_s_m2=A, B_Pa_s2_m3=B,
                                           K_m2=FLUID["nu"] * FLUID["rho"] / A if A > 0 else None))
    if heat:
        Uh = np.array([r["heat"]["U_superficial"] for r in runs])
        ha = np.array([r["heat"]["ha_W_m3K"] for r in runs])
        n_, lk = np.polyfit(np.log(Uh), np.log(ha), 1)
        res["heat_fit"] = dict(c=float(math.exp(lk)), n=float(n_), form="h*a = c * U^n  [W/m3K], U superficial in m/s")
    res["k_solid_ratio_z"] = solid_conductivity_ratio(solid[:, :, 2:-2], axis=2)
    res["k_solid_ratio_x"] = solid_conductivity_ratio(solid[:, :, 2:-2], axis=0)
    res["solid_fraction"] = float(solid[:, :, 2:-2].mean())
    print(f"[{name}] Forchheimer A {A:.3e} B {B:.3e}; k_s ratio z {res['k_solid_ratio_z']:.3f} x {res['k_solid_ratio_x']:.3f}", flush=True)
    return res


def main(names=None, dx=1.25e-4):
    OUT.mkdir(parents=True, exist_ok=True)
    names = names or list(LATTICES)
    p = OUT / "closures.json"
    res = json.loads(p.read_text()) if p.exists() else dict(fluid=FLUID, lattices={})
    res["fluid"] = FLUID
    for n in names:
        res["lattices"][n] = run_lattice(n, dx=dx, heat=True, save_field=(n == "core"))
        p.write_text(json.dumps(res, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args or None)
