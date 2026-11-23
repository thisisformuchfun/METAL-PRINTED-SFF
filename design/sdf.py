"""Implicit (signed-distance) modelling toolkit used by every MARROW part.

Convention: field < 0 inside metal, > 0 outside; values are millimetres (true distance
for primitives, a good first-order estimate for lattices).  Everything is evaluated on
a regular voxel grid; features are applied on local sub-boxes so cost scales with the
feature, not the part.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

F = np.float32
BIG = F(1e3)


# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
class Grid:
    def __init__(self, lo, hi, h):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.h = float(h)
        self.n = np.ceil((np.asarray(hi, dtype=np.float64) - self.lo) / self.h).astype(int) + 1
        self.hi = self.lo + (self.n - 1) * self.h
        self.ax = [(self.lo[i] + self.h * np.arange(self.n[i])).astype(F) for i in range(3)]

    # coordinates for a slice triple, broadcastable
    def coords(self, sl=None):
        if sl is None:
            sl = (slice(None),) * 3
        return (self.ax[0][sl[0]][:, None, None], self.ax[1][sl[1]][None, :, None],
                self.ax[2][sl[2]][None, None, :])

    def box(self, lo, hi, pad=0.0):
        """Slices covering the world-space box [lo, hi] grown by pad (clipped)."""
        sl = []
        for i in range(3):
            a = int(math.floor((lo[i] - pad - self.lo[i]) / self.h))
            b = int(math.ceil((hi[i] + pad - self.lo[i]) / self.h)) + 1
            a = max(a, 0)
            b = min(b, int(self.n[i]))
            if b <= a:
                return None
            sl.append(slice(a, b))
        return tuple(sl)

    def new(self, value=BIG):
        return np.full(tuple(self.n), value, dtype=F)

    def origin_of(self, sl):
        return [float(self.ax[i][sl[i].start or 0]) for i in range(3)]

    @property
    def shape(self):
        return tuple(int(v) for v in self.n)

    def memory_mb(self):
        return float(np.prod(self.n)) * 4 / 1e6


# ----------------------------------------------------------------------------
# Boolean operators (array level)
# ----------------------------------------------------------------------------
def smin(a, b, k):
    """Polynomial smooth minimum (C1), blend radius k."""
    if k <= 0:
        return np.minimum(a, b)
    hh = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (b + (a - b) * hh - k * hh * (1.0 - hh)).astype(F, copy=False)


def smax(a, b, k):
    return -smin(-a, -b, k)


def union(a, b):
    return np.minimum(a, b)


def inter(a, b):
    return np.maximum(a, b)


def diff(a, b):
    return np.maximum(a, -b)


def shell(d, t):
    """Hollow shell of thickness t centred on the surface of d."""
    return np.abs(d) - 0.5 * t


# ----------------------------------------------------------------------------
# Primitives (x, y, z are broadcastable arrays)
# ----------------------------------------------------------------------------
def sphere(x, y, z, c, r):
    return np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) - r


def box(x, y, z, lo, hi, r=0.0):
    """Axis-aligned box [lo, hi] with edge rounding r."""
    cx, cy, cz = [(lo[i] + hi[i]) * 0.5 for i in range(3)]
    hx, hy, hz = [(hi[i] - lo[i]) * 0.5 - r for i in range(3)]
    qx = np.abs(x - cx) - hx
    qy = np.abs(y - cy) - hy
    qz = np.abs(z - cz) - hz
    outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
    inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)
    return (outside + inside - r).astype(F, copy=False)


def rect2(u, v, lo, hi, r=0.0):
    """2-D rounded rectangle distance in the (u, v) plane."""
    cu, cv = (lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5
    hu, hv = (hi[0] - lo[0]) * 0.5 - r, (hi[1] - lo[1]) * 0.5 - r
    qu = np.abs(u - cu) - hu
    qv = np.abs(v - cv) - hv
    return (np.sqrt(np.maximum(qu, 0) ** 2 + np.maximum(qv, 0) ** 2)
            + np.minimum(np.maximum(qu, qv), 0) - r).astype(F, copy=False)


def circle2(u, v, c, r):
    return np.sqrt((u - c[0]) ** 2 + (v - c[1]) ** 2) - r


def extrude(d2, w, lo, hi):
    """Turn a 2-D distance d2 into a prism between w = lo..hi (w is the third coord)."""
    dz = np.maximum(lo - w, w - hi)
    return np.maximum(d2, dz).astype(F, copy=False)


def cyl_z(x, y, z, c, r, z0, z1):
    d2 = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2) - r
    return np.maximum(d2, np.maximum(z0 - z, z - z1)).astype(F, copy=False)


def capsule(x, y, z, a, b, ra, rb=None):
    """Round cone / capsule from point a (radius ra) to b (radius rb)."""
    if rb is None:
        rb = ra
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = bx - ax, by - ay, bz - az
    ll = dx * dx + dy * dy + dz * dz
    if ll < 1e-9:
        return sphere(x, y, z, a, max(ra, rb))
    t = ((x - ax) * dx + (y - ay) * dy + (z - az) * dz) / ll
    t = np.clip(t, 0.0, 1.0)
    px = x - (ax + t * dx)
    py = y - (ay + t * dy)
    pz = z - (az + t * dz)
    return (np.sqrt(px * px + py * py + pz * pz) - (ra + (rb - ra) * t)).astype(F, copy=False)


def cylinder(x, y, z, a, b, r):
    """Flat-capped cylinder between points a and b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    L = float(np.linalg.norm(ab))
    u = ab / L
    t = (x - a[0]) * u[0] + (y - a[1]) * u[1] + (z - a[2]) * u[2]
    px = x - (a[0] + t * u[0])
    py = y - (a[1] + t * u[1])
    pz = z - (a[2] + t * u[2])
    rad = np.sqrt(px * px + py * py + pz * pz) - r
    ax = np.maximum(-t, t - L)
    return np.maximum(rad, ax).astype(F, copy=False)


def halfspace(x, y, z, p, n):
    """Negative on the side the normal n points AWAY from (i.e. solid behind the plane)."""
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return ((x - p[0]) * n[0] + (y - p[1]) * n[1] + (z - p[2]) * n[2]).astype(F, copy=False)


def torus_z(x, y, z, c, R, r):
    q = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2) - R
    return np.sqrt(q * q + (z - c[2]) ** 2) - r


# ----------------------------------------------------------------------------
# In-place local application
# ----------------------------------------------------------------------------
def apply(field, grid, lo, hi, fn, op="union", k=0.0, pad=0.0):
    """Evaluate fn(x, y, z) on the sub-box [lo, hi]+pad and combine into field in place.

    op: 'union' | 'cut' (subtract) | 'inter' ; k > 0 makes it a smooth blend.
    """
    sl = grid.box(lo, hi, pad=pad + (k if k > 0 else 0.0) + 2 * grid.h)
    if sl is None:
        return
    x, y, z = grid.coords(sl)
    d = fn(x, y, z)
    cur = field[sl]
    if op == "union":
        field[sl] = smin(cur, d, k) if k > 0 else np.minimum(cur, d)
    elif op == "cut":
        field[sl] = smax(cur, -d, k) if k > 0 else np.maximum(cur, -d)
    elif op == "inter":
        field[sl] = smax(cur, d, k) if k > 0 else np.maximum(cur, d)
    else:
        raise ValueError(op)


def add_capsule(field, grid, a, b, ra, rb=None, k=0.0, op="union"):
    rb = ra if rb is None else rb
    r = max(ra, rb)
    lo = [min(a[i], b[i]) - r for i in range(3)]
    hi = [max(a[i], b[i]) + r for i in range(3)]
    apply(field, grid, lo, hi, lambda x, y, z: capsule(x, y, z, a, b, ra, rb), op=op, k=k)


def add_polyline(field, grid, pts, radii, k=0.0, op="union"):
    """Chain of round cones through pts with per-vertex radii."""
    for i in range(len(pts) - 1):
        add_capsule(field, grid, pts[i], pts[i + 1], radii[i], radii[i + 1], k=k, op=op)


def add_cyl(field, grid, a, b, r, k=0.0, op="union"):
    lo = [min(a[i], b[i]) - r for i in range(3)]
    hi = [max(a[i], b[i]) + r for i in range(3)]
    apply(field, grid, lo, hi, lambda x, y, z: cylinder(x, y, z, a, b, r), op=op, k=k)


def add_box(field, grid, lo, hi, r=0.0, k=0.0, op="union"):
    apply(field, grid, lo, hi, lambda x, y, z: box(x, y, z, lo, hi, r), op=op, k=k)


# ----------------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------------
def catmull(points, n_per=8, closed=False):
    """Centripetal-ish Catmull-Rom through points -> dense polyline (numpy Nx3)."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 3:
        t = np.linspace(0, 1, n_per + 1)[:, None]
        return P[0] * (1 - t) + P[-1] * t
    if closed:
        Q = np.vstack([P[-1], P, P[0], P[1]])
    else:
        Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(Q[-2])
    return np.asarray(out)


def taper(n, r0, r1, belly=0.0):
    """Radius profile along a member: r0 -> r1 with an optional mid-span waist (belly<0)
    or swell (belly>0), like a long bone."""
    t = np.linspace(0, 1, n)
    return r0 + (r1 - r0) * t + belly * np.sin(np.pi * t)


def add_bone(field, grid, points, r0, r1=None, belly=0.0, k=3.0, n_per=6):
    """Organic member through control points. The capsule chain is unioned exactly on a
    local scratch field (no beading), then blended into the part with radius k."""
    r1 = r0 if r1 is None else r1
    pl = catmull(points, n_per=n_per)
    rad = taper(len(pl), r0, r1, belly)
    rmax = float(np.max(rad))
    lo = pl.min(axis=0) - rmax
    hi = pl.max(axis=0) + rmax
    sl = grid.box(lo, hi, pad=k + 2 * grid.h)
    if sl is None:
        return pl
    x, y, z = grid.coords(sl)
    tmp = None
    for i in range(len(pl) - 1):
        a, b = pl[i], pl[i + 1]
        r = max(rad[i], rad[i + 1])
        # evaluate each capsule only on its own little box
        sub = grid.box(np.minimum(a, b) - r, np.maximum(a, b) + r, pad=k + 2 * grid.h)
        if tmp is None:
            tmp = np.full(tuple(s_.stop - s_.start for s_ in sl), BIG, dtype=F)
        loc = tuple(slice(max(sub[j].start, sl[j].start) - sl[j].start, min(sub[j].stop, sl[j].stop) - sl[j].start) for j in range(3))
        xs = x[loc[0], :, :]
        ys = y[:, loc[1], :]
        zs = z[:, :, loc[2]]
        d = capsule(xs, ys, zs, a, b, rad[i], rad[i + 1])
        tmp[loc] = np.minimum(tmp[loc], d)
    field[sl] = smin(field[sl], tmp, k) if k > 0 else np.minimum(field[sl], tmp)
    return pl


# ----------------------------------------------------------------------------
# Lattices
# ----------------------------------------------------------------------------
@nb.njit(parallel=True, fastmath=True, cache=True)
def _gyroid(out, x0, y0, z0, h, cell, px, py, pz, wa, wl):
    """Signed pseudo-distance g/|grad g| of the gyroid (mm). Optional smooth domain warp
    (amplitude wa mm, wavelength wl mm) gives an irregular, bone-like look."""
    a = 2.0 * np.pi / cell
    nx, ny, nz = out.shape
    w = 2.0 * np.pi / wl if wl > 0 else 0.0
    for i in nb.prange(nx):
        x = x0 + i * h
        for j in range(ny):
            y = y0 + j * h
            for k in range(nz):
                z = z0 + k * h
                if wa > 0.0:
                    xx = x + wa * np.sin(w * y + 1.3) * np.cos(w * z * 0.83 + 0.4)
                    yy = y + wa * np.sin(w * z * 1.17 + 2.1) * np.cos(w * x * 0.91 + 1.1)
                    zz = z + wa * np.sin(w * x * 1.07 + 0.7) * np.cos(w * y * 0.79 + 2.9)
                else:
                    xx, yy, zz = x, y, z
                X = xx * a + px
                Y = yy * a + py
                Z = zz * a + pz
                sx, cx = np.sin(X), np.cos(X)
                sy, cy = np.sin(Y), np.cos(Y)
                sz, cz = np.sin(Z), np.cos(Z)
                g = sx * cy + sy * cz + sz * cx
                gx = cx * cy - sz * sx
                gy = cy * cz - sx * sy
                gz = cz * cx - sy * sz
                gm = np.sqrt(gx * gx + gy * gy + gz * gz)
                if gm < 0.35:
                    gm = 0.35
                out[i, j, k] = g / (gm * a)


def gyroid_dist(grid, sl, cell, phase=(0.0, 0.0, 0.0), warp_amp=0.0, warp_len=0.0):
    """Signed pseudo-distance to the gyroid mid-surface on the sub-box sl."""
    shape = tuple(s.stop - s.start for s in sl)
    out = np.empty(shape, dtype=F)
    o = grid.origin_of(sl)
    _gyroid(out, o[0], o[1], o[2], grid.h, float(cell), float(phase[0]), float(phase[1]),
            float(phase[2]), float(warp_amp), float(warp_len))
    return out


def gyroid_sheet(grid, sl, cell, wall, **kw):
    """Sheet gyroid: metal where |d| < wall/2 (wall may be an array broadcastable to sl)."""
    return np.abs(gyroid_dist(grid, sl, cell, **kw)) - 0.5 * wall


def gyroid_network(grid, sl, cell, bias, **kw):
    """Skeletal gyroid: metal on one side of the surface, offset by bias (mm).
    bias > 0 thins the struts."""
    return gyroid_dist(grid, sl, cell, **kw) + bias


def voronoi_edges3(lo, hi, cell, seed=1, jitter=0.42, margin=None):
    """3-D Voronoi diagram of a jittered lattice of seeds -> list of finite edges
    (a, b) whose midpoints lie inside [lo, hi]."""
    from scipy.spatial import Voronoi

    rng = np.random.default_rng(seed)
    margin = 1.6 * cell if margin is None else margin
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    n = np.maximum(np.round((hi - lo + 2 * margin) / cell).astype(int), 2)
    gx, gy, gz = np.meshgrid(*[np.arange(n[i]) for i in range(3)], indexing="ij")
    pts = np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(float)
    pts[:, 0] += 0.5 * (pts[:, 2] % 2)          # stagger layers -> less cubic look
    pts[:, 1] += 0.5 * (pts[:, 2] % 2)
    pts = (pts + 0.5 + rng.uniform(-jitter, jitter, pts.shape)) * cell + (lo - margin)
    vor = Voronoi(pts)
    edges = set()
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        m = len(ridge)
        for i in range(m):
            a, b = ridge[i], ridge[(i + 1) % m]
            if a != b:
                edges.add((min(a, b), max(a, b)))
    V = vor.vertices
    out = []
    for a, b in edges:
        mid = 0.5 * (V[a] + V[b])
        if np.all(mid >= lo - 0.25 * cell) and np.all(mid <= hi + 0.25 * cell):
            if np.linalg.norm(V[a] - V[b]) > 0.12 * cell:
                out.append((V[a], V[b]))
    return out


def add_voronoi3(field, grid, lo, hi, cell, strut_r, seed=1, k=None, clip=None):
    """Union a 3-D Voronoi strut lattice into field (optionally clipped by clip(x,y,z))."""
    k = 0.8 * strut_r if k is None else k
    edges = voronoi_edges3(lo, hi, cell, seed=seed)
    tmp = grid.new(BIG)
    for a, b in edges:
        add_capsule(tmp, grid, a, b, strut_r, k=k)
    sl = grid.box(lo, hi, pad=cell)
    if clip is not None and sl is not None:
        x, y, z = grid.coords(sl)
        tmp[sl] = np.maximum(tmp[sl], clip(x, y, z))
    np.minimum(field, tmp, out=field)
    return len(edges)


def voronoi_edges2(lo, hi, cell, seed=1, jitter=0.40):
    """2-D Voronoi edges (segments) of a jittered hex-ish seed lattice inside [lo, hi]."""
    from scipy.spatial import Voronoi

    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    m = 2.0 * cell
    nx = int((hi[0] - lo[0] + 2 * m) / cell) + 2
    ny = int((hi[1] - lo[1] + 2 * m) / (cell * 0.866)) + 2
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append([(i + 0.5 * (j % 2)) * cell, j * cell * 0.866])
    pts = np.asarray(pts) + rng.uniform(-jitter, jitter, (len(pts), 2)) * cell + (lo - m)
    vor = Voronoi(pts)
    segs = []
    for (a, b) in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue
        pa, pb = vor.vertices[a], vor.vertices[b]
        mid = 0.5 * (pa + pb)
        if np.all(mid >= lo - cell) and np.all(mid <= hi + cell):
            segs.append((pa, pb))
    return segs


def seg2_dist(u, v, segs, shape=None):
    """Distance in the (u, v) plane to the nearest of a list of 2-D segments."""
    d = None
    for pa, pb in segs:
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        ll = dx * dx + dy * dy
        if ll < 1e-9:
            continue
        t = np.clip(((u - pa[0]) * dx + (v - pa[1]) * dy) / ll, 0, 1)
        dd = np.sqrt((u - (pa[0] + t * dx)) ** 2 + (v - (pa[1] + t * dy)) ** 2)
        d = dd if d is None else np.minimum(d, dd)
    return d



def hex_seeds(lo, hi, cell, seed=1, jitter=0.33):
    """Jittered hexagonal seed lattice covering the 2-D box [lo, hi] (+ one cell margin)."""
    rng = np.random.default_rng(seed)
    pts = []
    ny = int((hi[1] - lo[1]) / (cell * 0.866)) + 4
    nx = int((hi[0] - lo[0]) / cell) + 4
    for j in range(ny):
        for i in range(nx):
            pts.append([lo[0] + (i - 1 + 0.5 * (j % 2)) * cell, lo[1] + (j - 1) * cell * 0.866])
    pts = np.asarray(pts)
    return pts + rng.uniform(-jitter, jitter, pts.shape) * cell


def voronoi2_edge(u, v, seeds):
    """Approximate distance (mm) to the nearest Voronoi cell boundary of 2-D seeds."""
    d1 = np.full(np.broadcast(u, v).shape, 1e9, dtype=F)
    d2 = d1.copy()
    for sx, sy in seeds:
        d = np.sqrt((u - sx) ** 2 + (v - sy) ** 2).astype(F)
        d2 = np.minimum(d2, np.maximum(d1, d))
        d1 = np.minimum(d1, d)
    return 0.5 * (d2 - d1)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# ----------------------------------------------------------------------------
# Voxel-level checks and clean-up
# ----------------------------------------------------------------------------
def keep_largest_solid(field, h):
    """Remove floating metal islands (anything not connected to the largest body)."""
    from scipy import ndimage as ndi

    lab, n = ndi.label(field < 0)
    if n <= 1:
        return 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    drop = (lab > 0) & (lab != keep)
    cnt = int(drop.sum())
    field[drop] = F(h)
    return cnt


def fill_closed_voids(field, h, open_mask=None):
    """Fill internal cavities that do not connect to the outside (trapped powder).
    Returns number of voxels filled. The outside is whatever touches the grid boundary
    (or open_mask if given)."""
    from scipy import ndimage as ndi

    lab, n = ndi.label(field >= 0)
    if n <= 1:
        return 0
    edge = np.zeros(n + 1, dtype=bool)
    for face in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
        edge[np.unique(face)] = True
    if open_mask is not None:
        edge[np.unique(lab[open_mask])] = True
    edge[0] = True
    closed = ~edge[lab]
    cnt = int(closed.sum())
    field[closed] = F(-h)
    return cnt


def volume_cm3(field, h):
    """Metal volume with sub-voxel accuracy (linear occupancy ramp across the surface)."""
    occ = np.clip(0.5 - field / h, 0.0, 1.0)
    return float(occ.sum(dtype=np.float64)) * h ** 3 / 1000.0


def min_feature_report(field, h, sample=4):
    """Local metal thickness statistics from the distance transform of the solid:
    thickness ~ 2 x (distance to surface at the medial ridge)."""
    from scipy import ndimage as ndi

    solid = field < 0
    edt = ndi.distance_transform_edt(solid, sampling=h)
    # medial ridge = local maxima of the distance transform
    mx = ndi.maximum_filter(edt, size=3)
    ridge = solid & (edt >= mx - 1e-6) & (edt > 0)
    th = 2.0 * edt[ridge]
    if th.size == 0:
        return {}
    return dict(p01=float(np.percentile(th, 1)), p05=float(np.percentile(th, 5)),
                p50=float(np.percentile(th, 50)), min=float(th.min()), max=float(th.max()))


# ----------------------------------------------------------------------------
# Meshing / export
# ----------------------------------------------------------------------------
def mesh(field, grid, sl=None, step=1):
    """Marching cubes -> trimesh.Trimesh in world millimetres."""
    import trimesh
    from skimage import measure

    if sl is None:
        sl = (slice(None),) * 3
    sub = field[sl]
    # pad with 'outside' so the surface closes at the grid boundary
    sub = np.pad(sub, 1, mode="constant", constant_values=float(grid.h))
    v, f, _, _ = measure.marching_cubes(sub, level=0.0, spacing=(grid.h,) * 3, step_size=step,
                                        gradient_direction="ascent", allow_degenerate=False)
    o = np.array(grid.origin_of(sl)) - grid.h
    v = v + o
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return m


def decimate(m, target_faces=None, ratio=None, agg=5):
    import fast_simplification
    import trimesh

    nf = len(m.faces)
    if target_faces is None:
        target_faces = int(nf * ratio)
    if target_faces >= nf:
        return m
    v, f = fast_simplification.simplify(np.asarray(m.vertices, dtype=np.float32),
                                        np.asarray(m.faces, dtype=np.int32),
                                        target_count=int(target_faces), agg=agg)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def smooth(m, iters=2, lam=0.5):
    """Light Taubin smoothing to remove voxel stair-stepping before decimation."""
    import trimesh

    trimesh.smoothing.filter_taubin(m, lamb=lam, nu=-0.53, iterations=iters)
    return m


def mesh_report(m):
    return dict(verts=int(len(m.vertices)), faces=int(len(m.faces)), watertight=bool(m.is_watertight),
                volume_cm3=float(abs(m.volume)) / 1000.0, area_cm2=float(m.area) / 100.0,
                bbox=[float(v) for v in (m.bounds[1] - m.bounds[0])])
