"""Voxel domain: material map, sparse fluid indexing, boundary link tables.

Conventions (see README.md):
  * arrays are indexed [ix, iy, iz]; cell (ix, iy, iz) is the cube
    [ix, ix+1] x [iy, iy+1] x [iz, iz+1] * dx, its centre at (ix+0.5, ...) * dx.
  * ``mat``: 0 fluid, 1..9 solid material ids, 250 void (not simulated,
    adiabatic), 251 inlet fluid cells, 252 outlet fluid cells.
  * a face of a cell is "exterior" when the neighbour across it is void (250)
    or lies outside the array along a non-periodic axis.
  * open boundary faces = exterior faces of inlet / outlet tagged cells.
  * heater faces = exterior faces of solid cells with ``heater`` id k > 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lattice import C, NQ, OPP, W

FLUID, VOID, INLET, OUTLET = 0, 250, 251, 252
# (axis, side) for the six faces in the order -x,+x,-y,+y,-z,+z
FACES = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]


def is_fluid(mat: np.ndarray) -> np.ndarray:
    return (mat == FLUID) | (mat == INLET) | (mat == OUTLET)


def is_solid(mat: np.ndarray) -> np.ndarray:
    return (mat >= 1) & (mat <= 9)


def _padded(a: np.ndarray, fill, periodic) -> np.ndarray:
    """Pad by one cell: ``fill`` on non-periodic axes, wrapped on periodic ones."""
    p = np.pad(a, 1, mode="constant", constant_values=fill)
    for ax in range(3):
        if periodic[ax]:
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[ax], hi[ax] = 0, -2
            p[tuple(lo)] = p[tuple(hi)]
            lo[ax], hi[ax] = -1, 1
            p[tuple(lo)] = p[tuple(hi)]
    return p


def shifted(pad: np.ndarray, c) -> np.ndarray:
    """View v of a padded array with v[x] = a[x + c] (c components in -1..1)."""
    nx, ny, nz = (s - 2 for s in pad.shape)
    return pad[1 + c[0]:1 + c[0] + nx, 1 + c[1]:1 + c[1] + ny, 1 + c[2]:1 + c[2] + nz]


@dataclass
class OpenLinks:
    """Lattice links that cross an open (inlet / outlet) boundary face."""
    node: np.ndarray      # fluid node index of the boundary cell
    q: np.ndarray         # outward link direction
    flat: np.ndarray      # index q*N + node into the population array
    n_faces: int = 0      # number of open cell faces
    cells: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # nodes owning them


@dataclass
class Domain:
    mat: np.ndarray
    heater: np.ndarray
    dx: float
    periodic: tuple = (False, False, False)

    def __post_init__(self):
        mat = np.ascontiguousarray(self.mat, dtype=np.uint8)
        if mat.ndim != 3:
            raise ValueError("mat must be a 3-D array [nx, ny, nz]")
        bad = ~(is_fluid(mat) | is_solid(mat) | (mat == VOID))
        if bad.any():
            raise ValueError(f"mat contains unknown ids: {np.unique(mat[bad])[:10]}")
        self.mat = mat
        if self.heater is None:
            self.heater = np.zeros_like(mat)
        self.heater = np.ascontiguousarray(self.heater, dtype=np.uint8)
        if self.heater.shape != mat.shape:
            raise ValueError("heater must have the same shape as mat")
        self.periodic = tuple(bool(p) for p in self.periodic)
        self.shape = mat.shape
        self.fluid = is_fluid(mat)
        self.solid = is_solid(mat)
        self.N = int(self.fluid.sum())
        if self.N == 0:
            raise ValueError("domain has no fluid cells")
        if NQ * self.N >= 2 ** 31:
            raise ValueError("too many fluid nodes for 32-bit pull pointers")
        self.idmap = np.full(mat.shape, -1, dtype=np.int32)
        self.idmap[self.fluid] = np.arange(self.N, dtype=np.int32)
        self._pmat = _padded(mat, VOID, self.periodic)

    # ------------------------------------------------------------------ faces
    def exterior_face(self, mask: np.ndarray, axis: int, side: int) -> np.ndarray:
        """Cells of ``mask`` whose face (axis, side) looks at void / outside."""
        c = [0, 0, 0]
        c[axis] = side
        return mask & (shifted(self._pmat, c) == VOID)

    def n_open_faces(self, tag: int) -> int:
        m = self.mat == tag
        return int(sum(self.exterior_face(m, a, s).sum() for a, s in FACES))

    # ---------------------------------------------------------- LBM tables
    def pull_pointers(self) -> np.ndarray:
        """src[(q-1)*N + i]: flat index into the old population array."""
        N = self.N
        pid = _padded(self.idmap, -1, self.periodic)
        own = np.arange(N, dtype=np.int64)
        src = np.empty(18 * N, dtype=np.int32)
        for q in range(1, NQ):
            nb = shifted(pid, -C[q])[self.fluid].astype(np.int64)   # node at x - c_q
            src[(q - 1) * N:q * N] = np.where(nb >= 0, q * N + nb, OPP[q] * N + own)
        return src

    def open_links(self, tag: int) -> OpenLinks:
        """Links leaving tagged cells through their exterior (open) faces.

        A diagonal link c = e + t is open when the cell has an open face with
        normal e and the tangential neighbour x + t is a tagged cell with the
        same open face; otherwise it stays a no-slip bounce-back link (rim).
        """
        tagm = self.mat == tag
        ptag = np.pad(tagm, 1, constant_values=False)
        nodes, qs, cells = [], [], []
        n_faces = 0
        for axis, side in FACES:
            openf = self.exterior_face(tagm, axis, side)
            if not openf.any():
                continue
            n_faces += int(openf.sum())
            cells.append(self.idmap[openf])
            popen = np.pad(openf, 1, constant_values=False)
            for q in range(1, NQ):
                if C[q][axis] != side:
                    continue
                t = C[q].copy()
                t[axis] = 0
                ok = openf & shifted(popen, t)       # tangential neighbour open too
                if ok.any():
                    nodes.append(self.idmap[ok].astype(np.int64))
                    qs.append(np.full(int(ok.sum()), q, dtype=np.int64))
        if not nodes:
            return OpenLinks(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64))
        node = np.concatenate(nodes)
        q = np.concatenate(qs)
        # a link reached through two different faces (corner cell) is kept once
        flat, first = np.unique(q * self.N + node, return_index=True)
        return OpenLinks(node[first], q[first], flat, n_faces, np.unique(np.concatenate(cells)))

    def gradient_tables(self):
        """Neighbour indices and stencil codes for velocity gradients (WALE)."""
        N = self.N
        pid = _padded(self.idmap, -1, self.periodic)
        own = np.arange(N, dtype=np.int32)
        opentag = (self.mat == INLET) | (self.mat == OUTLET)
        nb = np.empty(6 * N, dtype=np.int32)
        kinds = []
        for f, (axis, side) in enumerate(FACES):
            c = [0, 0, 0]
            c[axis] = side
            n = shifted(pid, c)[self.fluid]
            isopen = self.exterior_face(opentag, axis, side)[self.fluid]
            nb[f * N:(f + 1) * N] = np.where(n >= 0, n, own)
            kinds.append(np.where(n >= 0, 0, np.where(isopen, 2, 1)).astype(np.uint8))  # F,W,O
        # code[(L kind, R kind)]
        lut = np.array([[0, 2, 5], [1, 3, 6], [4, 7, 8]], dtype=np.uint8)
        gcode = np.empty(3 * N, dtype=np.uint8)
        for a in range(3):
            gcode[a * N:(a + 1) * N] = lut[kinds[2 * a], kinds[2 * a + 1]]
        return nb, gcode

    # -------------------------------------------------------------- heaters
    def heater_faces(self):
        """{id: list of (axis, side, boolean cell mask)} for exterior faces of heater cells."""
        out = {}
        for k in np.unique(self.heater):
            if k == 0:
                continue
            hm = (self.heater == k) & self.solid
            faces = []
            for axis, side in FACES:
                m = self.exterior_face(hm, axis, side)
                if m.any():
                    faces.append((axis, side, m))
            out[int(k)] = faces
        return out

    # ------------------------------------------------------------ geometry
    def wetted_area(self) -> float:
        """Fluid-solid interface area [m^2] from a marching-cubes surface.

        Counting voxel faces overestimates the area of inclined surfaces by up
        to 73 % (staircase), so the hydraulic diameter uses a triangulated
        iso-surface of the lightly smoothed fluid indicator instead.
        """
        from scipy import ndimage
        from skimage import measure

        ind = np.pad(self.fluid.astype(np.float32), 2, mode="edge")
        for ax in range(3):
            if not self.periodic[ax]:
                sl = [slice(None)] * 3
                sl[ax] = slice(0, 2)
                ind[tuple(sl)] = 0.0
                sl[ax] = slice(-2, None)
                ind[tuple(sl)] = 0.0
        ind = ndimage.gaussian_filter(ind, 0.6)
        if ind.max() <= 0.5 or ind.min() >= 0.5:
            return 0.0
        verts, faces, _, _ = measure.marching_cubes(ind, 0.5)
        area = measure.mesh_surface_area(verts, faces) * self.dx ** 2
        # open boundary faces are not walls
        nopen = self.n_open_faces(INLET) + self.n_open_faces(OUTLET)
        return max(area - nopen * self.dx ** 2, 0.0)

    def voxel_wetted_faces(self) -> int:
        """Number of fluid-cell faces that touch a non-fluid cell (staircase area / dx^2)."""
        n = 0
        pf = _padded(self.fluid, False, self.periodic)
        for axis, side in FACES:
            c = [0, 0, 0]
            c[axis] = side
            n += int((self.fluid & ~shifted(pf, c)).sum())
        return n - self.n_open_faces(INLET) - self.n_open_faces(OUTLET)

    def to_dense(self, values: np.ndarray, fill=0.0, dtype=np.float32) -> np.ndarray:
        """Scatter a per-fluid-node array back onto the voxel grid."""
        out = np.full(self.shape, fill, dtype=dtype)
        out[self.fluid] = values
        return out


def load_domain(path, dx=None, periodic=(False, False, False)) -> Domain:
    """Load ``domain.npz`` (arrays ``mat``, ``heater``; optional scalar ``dx_m``)."""
    with np.load(path, allow_pickle=False) as d:
        mat = d["mat"]
        heater = d["heater"] if "heater" in d.files else None
        dx_file = float(d["dx_m"]) if "dx_m" in d.files else None
    if dx is None:
        dx = dx_file
    if dx is None:
        raise ValueError("dx_m must be given in the case file or in domain.npz")
    if dx_file is not None and abs(dx_file - dx) > 1e-9 * dx:
        raise ValueError(f"case dx_m={dx} disagrees with domain.npz dx_m={dx_file}")
    return Domain(mat, heater, float(dx), periodic)
