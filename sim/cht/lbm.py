"""D3Q19 lattice-Boltzmann flow solver (lattice units) on a voxel :class:`Domain`.

Model summary (details and rationale in README.md):
  * incompressible-equilibrium TRT collision, magic parameter Lambda = 3/16,
    separate bulk-mode relaxation, WALE (default) or Smagorinsky eddy viscosity;
  * halfway bounce-back on every solid / void / domain-edge link;
  * inlet: velocity (momentum-injecting) bounce-back on the open faces of the
    tagged inlet cells, scaled so the injected volume flow is exactly ``q_lb``;
  * outlet: anti-bounce-back pressure condition (rho' = 0) on the open faces
    of the tagged outlet cells plus a short viscous sponge in front of it.

Everything here is in lattice units (dx = dt = rho0 = 1); ``units.py`` /
``run.py`` do the conversion.
"""
from __future__ import annotations

import ctypes
import time

import numpy as np

from . import _native
from .geometry import INLET, OUTLET, Domain
from .lattice import C, CS2, NQ, W

ACC_FIELDS = ("rho", "ux", "uy", "uz", "uxux", "uyuy", "uzuz", "uxuy", "uxuz", "uyuz", "nut")


class LBM:
    def __init__(self, dom: Domain, nu: float, q_lb: float, *, collision: str = "rr",
                 lam: float = 3.0 / 16.0, omega_minus_min: float = 0.0, omega_bulk: float = 1.0,
                 sgs: str = "wale", c_wale: float = 0.5, c_smag: float = 0.1,
                 c_limiter: float = 0.0, sgs_every: int = 1, ramp_steps: int = 2000,
                 sponge_cells: float = 10.0, sponge_nu: float = 0.02, threads: int = 2):
        self.dom = dom
        self.N = N = dom.N
        self.nu = float(nu)
        self.q_lb = float(q_lb)
        self.lam = float(lam)
        self.omega_bulk = float(omega_bulk)
        self.om_min = float(omega_minus_min)
        self.sgs = sgs
        if sgs not in ("wale", "smagorinsky", "none"):
            raise ValueError("sgs must be 'wale', 'smagorinsky' or 'none'")
        modes = {"trt": 0.0, "regularized": 1.0, "rr": 2.0}
        if collision not in modes:
            raise ValueError(f"collision must be one of {sorted(modes)}")
        self.collision = collision
        self.cw2 = float(c_wale) ** 2
        # in-kernel strain-based viscosity: classic Smagorinsky, or the
        # cell-Reynolds limiter nu_eff >= c_limiter^2 |S| that backs up WALE
        if sgs == "smagorinsky":
            c2, lim = float(c_smag) ** 2, 0.0
        else:
            c2, lim = float(c_limiter) ** 2, 1.0
        self.par = np.array([self.nu, self.lam, self.om_min, self.omega_bulk, c2, lim,
                             modes[collision], 0.0], dtype=np.float32)
        self.sgs_every = max(int(sgs_every), 1)
        self.ramp_steps = max(int(ramp_steps), 1)
        self.threads = int(threads)
        self.lib = _native.lib()

        self.src = dom.pull_pointers()
        self.fa = np.zeros(NQ * N, dtype=np.float32)
        self.fb = np.zeros(NQ * N, dtype=np.float32)
        self.rho = np.zeros(N, dtype=np.float32)
        self.u = np.zeros(3 * N, dtype=np.float32)
        self.step_count = 0

        # ---- open boundaries
        self.inlet = dom.open_links(INLET)
        self.outlet = dom.open_links(OUTLET)
        self.in_nodes = dom.idmap[dom.mat == INLET]
        self.out_nodes = dom.idmap[dom.mat == OUTLET]
        if self.q_lb != 0.0 and len(self.inlet.flat) == 0:
            raise ValueError("flow requested but no inlet cell has an exterior face")
        if len(self.inlet.flat) and len(self.outlet.flat) == 0:
            raise ValueError("inlet present but no outlet cell has an exterior face")
        w_in = 6.0 * W[self.inlet.q]
        s = w_in.sum()
        #: per-link population increment at full flow; sums to q_lb (mass / step)
        self.in_amp = (w_in * (self.q_lb / s) if s > 0 else w_in).astype(np.float32)
        self.u_in = self.q_lb / max(self.inlet.n_faces, 1)        # mean inlet velocity
        oq = self.outlet.q
        self.out_w2 = (2.0 * W[oq]).astype(np.float32)
        self.out_c = C[oq].astype(np.float32)
        self.q_out_last = 0.0

        # ---- eddy / sponge viscosity
        self.sponge = self._make_sponge(sponge_cells, sponge_nu)
        self.nut = self.sponge.copy()
        if sgs == "wale":
            self.nb, self.gcode = dom.gradient_tables()

        # ---- time averaging
        self.acc = None
        self.n_samples = 0
        self.wall_time = 0.0
        self.node_updates = 0

    # ------------------------------------------------------------------ setup
    def _make_sponge(self, cells: float, nu_max: float) -> np.ndarray:
        """Extra viscosity rising quadratically towards the outlet faces."""
        sp = np.zeros(self.N, dtype=np.float32)
        if cells <= 0 or nu_max <= 0 or len(self.outlet.cells) == 0:
            return sp
        from scipy import ndimage

        seed = np.ones(self.dom.shape, dtype=bool)
        idx = np.flatnonzero(self.dom.fluid.ravel())[self.outlet.cells]
        seed.ravel()[idx] = False
        # distance transform only in a box around the outlet cells
        ijk = np.array(np.unravel_index(idx, self.dom.shape))
        pad = int(np.ceil(cells)) + 1
        lo = np.maximum(ijk.min(1) - pad, 0)
        hi = np.minimum(ijk.max(1) + pad + 1, self.dom.shape)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        dist = ndimage.distance_transform_edt(seed[box])
        prof = np.clip(1.0 - dist / cells, 0.0, 1.0) ** 2 * nu_max
        dense = np.zeros(self.dom.shape, dtype=np.float32)
        dense[box] = prof
        sp[:] = dense[self.dom.fluid]
        return sp

    def ramp(self, step: int | None = None) -> float:
        t = self.step_count if step is None else step
        if t >= self.ramp_steps:
            return 1.0
        return 0.5 * (1.0 - np.cos(np.pi * t / self.ramp_steps))

    # ------------------------------------------------------------- stepping
    def _boundaries(self):
        """Edit the post-collision populations on open links before streaming."""
        fa, N = self.fa, self.N
        if len(self.inlet.flat):
            fa[self.inlet.flat] += self.in_amp * np.float32(self.ramp())
        if len(self.outlet.flat):
            nd = self.outlet.node
            ux, uy, uz = self.u[nd], self.u[N + nd], self.u[2 * N + nd]
            cu = self.out_c[:, 0] * ux + self.out_c[:, 1] * uy + self.out_c[:, 2] * uz
            usq = ux * ux + uy * uy + uz * uz
            old = fa[self.outlet.flat]
            new = -old + self.out_w2 * (4.5 * cu * cu - 1.5 * usq)   # rho'_wall = 0
            fa[self.outlet.flat] = new
            self.q_out_last = float(np.sum(old.astype(np.float64) - new))

    def step(self, n: int = 1):
        """Advance ``n`` time steps."""
        t0 = time.perf_counter()
        L = self.lib
        for _ in range(n):
            self._boundaries()
            L.lbm_step(self.fa, self.fb, self.src, self.rho, self.u, self.nut, self.N,
                       self.par, self.threads)
            self.fa, self.fb = self.fb, self.fa
            self.step_count += 1
            if self.sgs == "wale" and self.step_count % self.sgs_every == 0:
                L.lbm_wale(self.u, self.nb, self.gcode,
                           self.sponge.ctypes.data_as(ctypes.c_void_p), self.nut, self.N,
                           self.cw2, self.threads)
        self.wall_time += time.perf_counter() - t0
        self.node_updates += n * self.N

    # ------------------------------------------------------------- sampling
    def sample(self):
        """Add the current state to the time-average accumulators."""
        if self.acc is None:
            self.acc = np.zeros(len(ACC_FIELDS) * self.N, dtype=np.float64)
        self.lib.lbm_accumulate(self.rho, self.u, self.nut, self.acc, self.N, self.threads)
        self.n_samples += 1

    def reset_average(self):
        self.acc = None
        self.n_samples = 0

    def mean(self, name: str) -> np.ndarray:
        """Time-mean of one accumulated field (per fluid node)."""
        if self.acc is None or self.n_samples == 0:
            raise RuntimeError("no samples accumulated")
        k = ACC_FIELDS.index(name)
        return self.acc[k * self.N:(k + 1) * self.N] / self.n_samples

    # ------------------------------------------------------------- monitors
    def monitor(self) -> dict:
        out = np.zeros(3)
        self.lib.lbm_monitor(self.rho, self.u, self.N, out, self.threads)
        p_in = float(self.rho[self.in_nodes].mean()) * CS2 if len(self.in_nodes) else 0.0
        p_out = float(self.rho[self.out_nodes].mean()) * CS2 if len(self.out_nodes) else 0.0
        return {"step": self.step_count, "q_in": self.q_lb * self.ramp(self.step_count - 1),
                "q_out": self.q_out_last, "dp": p_in - p_out, "p_in": p_in, "p_out": p_out,
                "ke": out[0], "u_max": out[1], "mass": out[2]}

    @property
    def mlups(self) -> float:
        return self.node_updates / self.wall_time / 1e6 if self.wall_time > 0 else 0.0

    def velocity(self):
        N = self.N
        return self.u[:N], self.u[N:2 * N], self.u[2 * N:]

    # ----------------------------------------------------------- checkpoint
    def save_checkpoint(self, path):
        data = {"fa": self.fa, "rho": self.rho, "u": self.u, "nut": self.nut,
                "step_count": self.step_count, "n_samples": self.n_samples,
                "N": self.N, "nu": self.nu, "q_lb": self.q_lb,
                "wall_time": self.wall_time, "node_updates": self.node_updates}
        if self.acc is not None:
            data["acc"] = self.acc
        np.savez(path, **data)

    def load_checkpoint(self, path):
        with np.load(path, allow_pickle=False) as d:
            if int(d["N"]) != self.N:
                raise ValueError("checkpoint belongs to a different domain")
            if abs(float(d["nu"]) - self.nu) > 1e-6 * self.nu or \
                    abs(float(d["q_lb"]) - self.q_lb) > 1e-6 * abs(self.q_lb):
                raise ValueError("checkpoint was written with different lattice parameters")
            self.fa[:] = d["fa"]
            self.rho[:] = d["rho"]
            self.u[:] = d["u"]
            self.nut[:] = d["nut"]
            self.step_count = int(d["step_count"])
            self.n_samples = int(d["n_samples"])
            self.wall_time = float(d["wall_time"])
            self.node_updates = int(d["node_updates"])
            self.acc = d["acc"].copy() if "acc" in d.files else None
