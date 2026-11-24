"""pcp_geom - small procedural mesh toolkit used by pc_parts (1 Blender unit = 1 mm).

Everything is built as a `Soup` (numpy polygon soup with per-face material names),
turned into a bpy Mesh once, and then instanced through linked object data.
"""
import math

import bpy
import bmesh
import numpy as np

import materials as MAT

TAU = 2.0 * math.pi


# ----------------------------------------------------------------------------
# Soup
# ----------------------------------------------------------------------------
class Soup:
    """Polygon soup: V (n,3) verts, L flat loop->vertex, S face sizes, M face material idx."""

    def __init__(self):
        self.V = np.zeros((0, 3))
        self.L = np.zeros(0, np.int64)
        self.S = np.zeros(0, np.int32)
        self.M = np.zeros(0, np.int32)
        self.UV = None            # optional per-loop (nl,2)
        self.mats = []

    # -- building ------------------------------------------------------------
    def _mi(self, name):
        try:
            return self.mats.index(name)
        except ValueError:
            self.mats.append(name)
            return len(self.mats) - 1

    def add(self, verts, faces, mat="plastic_black", uv=None):
        verts = np.asarray(verts, float).reshape(-1, 3)
        if isinstance(faces, np.ndarray) and faces.ndim == 2:
            sizes = np.full(len(faces), faces.shape[1], np.int32)
            loops = faces.reshape(-1).astype(np.int64)
        else:
            sizes = np.array([len(f) for f in faces], np.int32)
            loops = np.array([i for f in faces for i in f], np.int64)
        nl0 = len(self.L)
        self.L = np.concatenate([self.L, loops + len(self.V)])
        self.V = np.concatenate([self.V, verts])
        self.S = np.concatenate([self.S, sizes])
        self.M = np.concatenate([self.M, np.full(len(sizes), self._mi(mat), np.int32)])
        self._uv_extend(nl0, len(loops), uv)
        return self

    def _uv_extend(self, nl0, n, uv):
        if uv is None and self.UV is None:
            return
        if self.UV is None:
            self.UV = np.zeros((nl0, 2))
        new = np.zeros((n, 2)) if uv is None else np.asarray(uv, float).reshape(-1, 2)
        self.UV = np.concatenate([self.UV, new])

    def __iadd__(self, o):
        if o is None or len(o.S) == 0:
            return self
        remap = np.array([self._mi(m) for m in o.mats], np.int32)
        nl0 = len(self.L)
        self.L = np.concatenate([self.L, o.L + len(self.V)])
        self.V = np.concatenate([self.V, o.V])
        self.S = np.concatenate([self.S, o.S])
        self.M = np.concatenate([self.M, remap[o.M]])
        self._uv_extend(nl0, len(o.L), o.UV)
        return self

    def copy(self):
        s = Soup()
        s.V, s.L, s.S, s.M = self.V.copy(), self.L.copy(), self.S.copy(), self.M.copy()
        s.UV = None if self.UV is None else self.UV.copy()
        s.mats = list(self.mats)
        return s

    # -- transforms (in place, chainable) --------------------------------------
    def move(self, x=0.0, y=0.0, z=0.0):
        self.V = self.V + np.array([x, y, z], float)
        return self

    def scale(self, sx=1.0, sy=None, sz=None):
        sy = sx if sy is None else sy
        sz = sx if sz is None else sz
        self.V = self.V * np.array([sx, sy, sz], float)
        if sx * sy * sz < 0:
            self.flip()
        return self

    def rot(self, axis, deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        if axis in ("z", 2):
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        elif axis in ("x", 0):
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        else:
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        self.V = self.V @ R.T
        return self

    def rotz(self, deg):
        return self.rot("z", deg)

    def xform(self, M):
        M = np.array(M, float)
        self.V = self.V @ M[:3, :3].T + M[:3, 3]
        if np.linalg.det(M[:3, :3]) < 0:
            self.flip()
        return self

    def flip(self):
        nf = len(self.S)
        if nf == 0:
            return self
        starts = np.zeros(nf, np.int64)
        starts[1:] = np.cumsum(self.S)[:-1]
        fo = np.repeat(np.arange(nf), self.S)
        k = np.arange(len(self.L)) - starts[fo]
        src = starts[fo] + self.S[fo] - 1 - k
        self.L = self.L[src]
        if self.UV is not None:
            self.UV = self.UV[src]
        return self

    def mirror(self, axis=0):
        self.V[:, axis] *= -1.0
        return self.flip()

    def setmat(self, name):
        self.mats = [name]
        self.M[:] = 0
        return self

    def swapmat(self, old, new):
        self.mats = [new if m == old else m for m in self.mats]
        return self

    def tile(self, proto, pos, rotz=None, sc=None):
        """Append `proto` once per row of pos (n,3) [rotz degrees (n,), sc (n,) or (n,3)]."""
        pos = np.asarray(pos, float).reshape(-1, 3)
        n = len(pos)
        if n == 0 or len(proto.S) == 0:
            return self
        V = np.repeat(proto.V[None, :, :], n, 0)
        if sc is not None:
            sc = np.asarray(sc, float)
            V = V * (sc.reshape(n, 1, 1) if sc.ndim == 1 else sc.reshape(n, 1, 3))
        if rotz is not None:
            a = np.radians(np.broadcast_to(np.asarray(rotz, float), (n,)))
            c, s = np.cos(a)[:, None], np.sin(a)[:, None]
            V = np.stack([V[:, :, 0] * c - V[:, :, 1] * s, V[:, :, 0] * s + V[:, :, 1] * c, V[:, :, 2]], -1)
        V = V + pos[:, None, :]
        nv = len(proto.V)
        L = (proto.L[None, :] + (np.arange(n) * nv)[:, None]).reshape(-1) + len(self.V)
        remap = np.array([self._mi(m) for m in proto.mats], np.int32)
        nl0 = len(self.L)
        self.L = np.concatenate([self.L, L])
        self.V = np.concatenate([self.V, V.reshape(-1, 3)])
        self.S = np.concatenate([self.S, np.tile(proto.S, n)])
        self.M = np.concatenate([self.M, np.tile(remap[proto.M], n)])
        self._uv_extend(nl0, len(L), None if proto.UV is None else np.tile(proto.UV, (n, 1)))
        return self

    # -- info --------------------------------------------------------------------
    def tris(self):
        return int((self.S - 2).sum())

    def bbox(self):
        return self.V.min(0), self.V.max(0)

    # -- output ------------------------------------------------------------------
    def mesh(self, name, smooth=40.0, weighted=None):
        me = bpy.data.meshes.new(name)
        nv, nl, nf = len(self.V), len(self.L), len(self.S)
        me.vertices.add(nv)
        me.loops.add(nl)
        me.polygons.add(nf)
        me.vertices.foreach_set("co", self.V.astype(np.float32).ravel())
        me.loops.foreach_set("vertex_index", self.L.astype(np.int32))
        starts = np.zeros(nf, np.int32)
        if nf:
            starts[1:] = np.cumsum(self.S)[:-1]
        me.polygons.foreach_set("loop_start", starts)
        try:
            me.polygons.foreach_set("loop_total", self.S.astype(np.int32))
        except Exception:
            pass
        me.polygons.foreach_set("material_index", self.M.astype(np.int32))
        for m in self.mats:
            me.materials.append(MAT.get(m))
        if self.UV is not None:
            uvl = me.uv_layers.new(name="UVMap")
            uvl.data.foreach_set("uv", self.UV.astype(np.float32).ravel())
        me.update(calc_edges=True)
        finish(me, smooth, weighted)
        return me


def soup_from_mesh(me, mat="plastic_black"):
    s = Soup()
    nv, nl, nf = len(me.vertices), len(me.loops), len(me.polygons)
    V = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", V)
    L = np.empty(nl, np.int32)
    me.loops.foreach_get("vertex_index", L)
    S = np.empty(nf, np.int32)
    me.polygons.foreach_get("loop_total", S)
    s.V = V.reshape(-1, 3).astype(float)
    s.L = L.astype(np.int64)
    s.S = S
    s.M = np.zeros(nf, np.int32)
    s.mats = [mat]
    return s


# ----------------------------------------------------------------------------
# Normals / shading
# ----------------------------------------------------------------------------
def finish(me, smooth=40.0, weighted=None):
    nf = len(me.polygons)
    if nf == 0:
        return
    me.polygons.foreach_set("use_smooth", np.ones(nf, bool))
    if weighted is None:
        weighted = len(me.loops) < 1500000
    if smooth is None:
        return
    if weighted:
        try:
            if weighted_normals(me, angle=max(smooth, 50.0)):
                return
        except Exception as e:           # pragma: no cover
            print("weighted_normals failed:", e)
    me.set_sharp_from_angle(angle=math.radians(smooth))


def weighted_normals(me, angle=50.0, power=2.0, kmax=14):
    """Face-area weighted custom normals (bevels blend, big flats stay flat)."""
    nl, nf = len(me.loops), len(me.polygons)
    lv = np.empty(nl, np.int32)
    me.loops.foreach_get("vertex_index", lv)
    fn = np.empty(nf * 3, np.float32)
    me.polygons.foreach_get("normal", fn)
    fn = fn.reshape(-1, 3).astype(np.float64)
    fa = np.empty(nf, np.float32)
    me.polygons.foreach_get("area", fa)
    lt = np.empty(nf, np.int32)
    me.polygons.foreach_get("loop_total", lt)
    lf = np.repeat(np.arange(nf), lt)
    order = np.argsort(lv, kind="stable")
    sv = lv[order]
    first = np.r_[True, sv[1:] != sv[:-1]]
    gstart = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    rank = np.arange(nl) - gstart[gid]
    K = int(rank.max()) + 1
    if K > kmax:
        keep = rank < kmax           # huge fans: only the first kmax faces take part
        K = kmax
    else:
        keep = np.ones(nl, bool)
    G = np.full((len(gstart), K), -1, np.int64)
    G[gid[keep], rank[keep]] = lf[order][keep]
    loop_gid = np.empty(nl, np.int64)
    loop_gid[order] = gid
    own = fn[lf]
    N = np.zeros((nl, 3))
    cosang = math.cos(math.radians(angle))
    fw = fa.astype(np.float64) ** power
    for k in range(K):               # loop over slots keeps memory small
        cand = G[loop_gid, k]
        ok = cand >= 0
        c = np.where(ok, cand, 0)
        cn = fn[c]
        w = fw[c] * ok * ((cn * own).sum(1) > cosang)
        N += cn * w[:, None]
    ln = np.linalg.norm(N, axis=1)
    bad = ln < 1e-20
    N[~bad] /= ln[~bad, None]
    N[bad] = own[bad]
    me.normals_split_custom_set(N.astype(np.float32).tolist())
    return True


# ----------------------------------------------------------------------------
# 2D polygon helpers
# ----------------------------------------------------------------------------
def signed_area(P):
    P = np.asarray(P, float)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ccw(P):
    P = np.asarray(P, float)
    return P if signed_area(P) > 0 else P[::-1].copy()


def offset_poly(P, d):
    """Miter offset of a CCW polygon; d > 0 grows it."""
    P = ccw(P)
    e = np.roll(P, -1, 0) - P
    ln = np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    n = np.stack([e[:, 1], -e[:, 0]], 1) / ln
    n1 = np.roll(n, 1, 0)
    den = np.maximum(1.0 + (n * n1).sum(1, keepdims=True), 0.2)
    return P + d * (n + n1) / den


def circle(r, n=24, c=(0.0, 0.0), phase=0.0):
    a = phase + np.arange(n) * TAU / n
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)


def rrect(w, h, r=0.0, n=5, c=(0.0, 0.0)):
    """Rounded rectangle polygon (CCW), n segments per corner."""
    r = min(r, 0.499 * min(w, h))
    if r <= 1e-9:
        return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) + np.array(c)
    pts = []
    for cx, cy, a0 in ((w / 2 - r, -h / 2 + r, -90), (w / 2 - r, h / 2 - r, 0),
                       (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180)):
        for i in range(n + 1):
            a = math.radians(a0 + 90.0 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts) + np.array(c)


def fillet_poly(P, radii, n=6):
    """Round the corners of polygon P; radii = scalar, list, or {index: r}."""
    P = np.asarray(P, float)
    m = len(P)
    if isinstance(radii, dict):
        rr = [radii.get(i, 0.0) for i in range(m)]
    elif np.isscalar(radii):
        rr = [radii] * m
    else:
        rr = list(radii)
    out = []
    for i in range(m):
        p0, p1, p2 = P[i - 1], P[i], P[(i + 1) % m]
        r = rr[i]
        a, b = p0 - p1, p2 - p1
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if r <= 1e-9 or la < 1e-9 or lb < 1e-9:
            out.append(p1)
            continue
        a, b = a / la, b / lb
        cosang = float(np.clip(a @ b, -1, 1))
        ang = math.acos(cosang)
        if ang < 1e-3 or abs(ang - math.pi) < 1e-3:
            out.append(p1)
            continue
        t = min(r / math.tan(ang / 2), 0.49 * la, 0.49 * lb)
        r_eff = t * math.tan(ang / 2)
        bis = a + b
        bis /= np.linalg.norm(bis)
        cen = p1 + bis * (r_eff / math.sin(ang / 2))
        s, e = p1 + a * t, p1 + b * t
        a0 = math.atan2(s[1] - cen[1], s[0] - cen[0])
        a1 = math.atan2(e[1] - cen[1], e[0] - cen[0])
        da = (a1 - a0 + math.pi) % TAU - math.pi
        for k in range(n + 1):
            aa = a0 + da * k / n
            out.append((cen[0] + r_eff * math.cos(aa), cen[1] + r_eff * math.sin(aa)))
    return np.array(out)


def hex_grid(x0, y0, x1, y1, pitch, r, keep=None):
    """Honeycomb hole polygons (flat-top hexagons of circumradius r) inside a rectangle."""
    out = []
    dy = pitch * math.sqrt(3) / 2
    ny = int((y1 - y0 - 2 * r) / dy) + 1
    nx = int((x1 - x0 - 2 * r) / pitch) + 1
    oy = (y1 - y0 - (ny - 1) * dy) / 2
    for j in range(ny):
        sh = 0.5 * pitch * (j % 2)
        ox = (x1 - x0 - (nx - 1) * pitch) / 2
        for i in range(nx - (j % 2)):
            c = (x0 + ox + sh + i * pitch, y0 + oy + j * dy)
            if keep is None or keep(c[0], c[1]):
                out.append(circle(r, 6, c, math.pi / 6))
    return out


# ----------------------------------------------------------------------------
# 3D primitives (all return a Soup)
# ----------------------------------------------------------------------------
def _orient_out(V, faces, cen):
    out = []
    for f in faces:
        p = V[list(f)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        out.append(tuple(f) if n @ (p.mean(0) - cen) >= 0 else tuple(reversed(f)))
    return out


def box(sx, sy, sz, c=(0.0, 0.0, 0.0), bevel=0.0, mat="plastic_black", base=False):
    """Box centred on c (or standing on c if base=True) with a 1-segment chamfer on all edges."""
    h = np.array([sx, sy, sz], float) / 2
    cen = np.array(c, float) + (np.array([0, 0, h[2]]) if base else 0)
    b = min(bevel, 0.45 * float(min(sx, sy, sz)))
    s = Soup()
    if b <= 1e-9:
        V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h
        F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        return s.add(V + cen, _orient_out(V, F, np.zeros(3)), mat)
    V, idx = [], {}
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                sg = np.array([ix, iy, iz]) * 2 - 1
                for a in range(3):
                    p = sg * (h - b)
                    p[a] = sg[a] * h[a]
                    idx[(ix, iy, iz, a)] = len(V)
                    V.append(p)
    V = np.array(V)
    F = []
    for a in range(3):
        b1, b2 = (a + 1) % 3, (a + 2) % 3
        for sd in (0, 1):
            q = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                k = [0, 0, 0]
                k[a], k[b1], k[b2] = sd, u, v
                q.append(idx[(k[0], k[1], k[2], a)])
            F.append(q)
        for u in (0, 1):
            for v in (0, 1):
                q = []
                for t, ax in ((0, b1), (1, b1), (1, b2), (0, b2)):
                    k = [0, 0, 0]
                    k[a], k[b1], k[b2] = t, u, v
                    q.append(idx[(k[0], k[1], k[2], ax)])
                F.append(q)
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                F.append([idx[(ix, iy, iz, a)] for a in range(3)])
    return s.add(V + cen, _orient_out(V, F, np.zeros(3)), mat)


def box2(x0, y0, z0, x1, y1, z1, bevel=0.0, mat="plastic_black"):
    return box(x1 - x0, y1 - y0, z1 - z0, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), bevel, mat)


def lathe(profile, n=32, mat="plastic_black", axis="z", phase=0.0, c=(0.0, 0.0, 0.0)):
    """Revolve (r, z) profile; travelling +z at r>0 gives outward normals."""
    prof = [(float(r), float(z)) for r, z in profile]
    m = len(prof)
    cap0 = prof[0][0] < 1e-9 and abs(prof[1][1] - prof[0][1]) < 1e-9
    cap1 = prof[-1][0] < 1e-9 and abs(prof[-2][1] - prof[-1][1]) < 1e-9
    a = phase + np.arange(n) * TAU / n
    cs, sn = np.cos(a), np.sin(a)
    V, ring = [], []
    for i, (r, z) in enumerate(prof):
        if (i == 0 and cap0) or (i == m - 1 and cap1):
            ring.append(None)
        elif r < 1e-9:
            ring.append((0, len(V)))
            V.append((0.0, 0.0, z))
        else:
            ring.append((1, len(V)))
            V.extend(zip(r * cs, r * sn, [z] * n))
    F = []
    for i in range(m - 1):
        A, B = ring[i], ring[i + 1]
        if A is None:
            F.append(tuple(B[1] + j for j in range(n - 1, -1, -1)))
            continue
        if B is None:
            F.append(tuple(A[1] + j for j in range(n)))
            continue
        (ta, ia), (tb, ib) = A, B
        for j in range(n):
            j2 = (j + 1) % n
            if ta and tb:
                F.append((ia + j, ia + j2, ib + j2, ib + j))
            elif ta:
                F.append((ia + j, ia + j2, ib))
            elif tb:
                F.append((ia, ib + j2, ib + j))
    V = np.array(V, float)
    if axis == "x":
        V = V[:, [2, 0, 1]]
    elif axis == "y":
        V = V[:, [1, 2, 0]]
    return Soup().add(V + np.array(c, float), F, mat)


def cyl(r, z0, z1, n=24, mat="plastic_black", bevel=0.0, axis="z", c=(0.0, 0.0, 0.0), bevel_bottom=None):
    """Capped cylinder between z0 and z1 with optional chamfer (top, and bottom if asked)."""
    b = min(bevel, 0.45 * r, 0.45 * abs(z1 - z0))
    bb = b if bevel_bottom is None else min(bevel_bottom, b if b > 0 else bevel_bottom)
    prof = [(0, z0)]
    if bb > 1e-9:
        prof += [(r - bb, z0), (r, z0 + bb)]
    else:
        prof += [(r, z0)]
    if b > 1e-9:
        prof += [(r, z1 - b), (r - b, z1)]
    else:
        prof += [(r, z1)]
    prof += [(0, z1)]
    return lathe(prof, n, mat, axis, c=c)


def tube_ring(r_out, r_in, z0, z1, n=24, mat="plastic_black", bevel=0.0, axis="z", c=(0.0, 0.0, 0.0)):
    """Hollow cylinder (washer / sleeve)."""
    b = min(bevel, 0.45 * (r_out - r_in), 0.45 * abs(z1 - z0))
    if b > 1e-9:
        prof = [(r_in, z0), (r_out - b, z0), (r_out, z0 + b), (r_out, z1 - b), (r_out - b, z1),
                (r_in + b, z1), (r_in, z1 - b), (r_in, z0)]
    else:
        prof = [(r_in, z0), (r_out, z0), (r_out, z1), (r_in, z1), (r_in, z0)]
    return lathe(prof, n, mat, axis, c=c)


def prism(poly, z0, z1, mat="plastic_black", bevel=0.0, bevel_bottom=False, cap_bottom=True):
    """Extrude a simple polygon; n-gon caps; optional top (and bottom) chamfer."""
    P = ccw(poly)
    n = len(P)
    if bevel > 1e-9:
        Pi = offset_poly(P, -bevel)
        rings = [(Pi, z0), (P, z0 + bevel), (P, z1 - bevel), (Pi, z1)] if bevel_bottom else \
                [(P, z0), (P, z1 - bevel), (Pi, z1)]
    else:
        rings = [(P, z0), (P, z1)]
    V = np.concatenate([np.column_stack([p, np.full(n, z)]) for p, z in rings])
    F = []
    if cap_bottom:
        F.append(tuple(range(n - 1, -1, -1)))
    for k in range(len(rings) - 1):
        a, b = k * n, (k + 1) * n
        for j in range(n):
            j2 = (j + 1) % n
            F.append((a + j, a + j2, b + j2, b + j))
    top = (len(rings) - 1) * n
    F.append(tuple(top + j for j in range(n)))
    return Soup().add(V, F, mat)


def loft(rings, mat="plastic_black", cap0=True, cap1=True):
    """Skin a list of (polygon (n,2), z) rings with equal vertex counts."""
    n = len(rings[0][0])
    V = np.concatenate([np.column_stack([np.asarray(p, float), np.full(n, z)]) for p, z in rings])
    F = []
    if cap0:
        F.append(tuple(range(n - 1, -1, -1)))
    for k in range(len(rings) - 1):
        a, b = k * n, (k + 1) * n
        F += [(a + j, a + (j + 1) % n, b + (j + 1) % n, b + j) for j in range(n)]
    if cap1:
        F.append(tuple((len(rings) - 1) * n + j for j in range(n)))
    return Soup().add(V, F, mat)


def rect(w, h, z=0.0, mat="silk_white", c=(0.0, 0.0)):
    V = [(c[0] - w / 2, c[1] - h / 2, z), (c[0] + w / 2, c[1] - h / 2, z),
         (c[0] + w / 2, c[1] + h / 2, z), (c[0] - w / 2, c[1] + h / 2, z)]
    return Soup().add(V, [(0, 1, 2, 3)], mat)


def poly_face(P, z=0.0, mat="silk_white"):
    P = ccw(P)
    V = np.column_stack([P, np.full(len(P), z)])
    return Soup().add(V, [tuple(range(len(P)))], mat)


def disc(r, z=0.0, n=16, mat="enig_gold", c=(0.0, 0.0)):
    return poly_face(circle(r, n, c), z, mat)


def annulus(r0, r1, z=0.0, n=20, mat="enig_gold", c=(0.0, 0.0)):
    a, b = circle(r0, n, c), circle(r1, n, c)
    V = np.column_stack([np.concatenate([a, b]), np.full(2 * n, z)])
    F = [(j, n + j, n + (j + 1) % n, (j + 1) % n) for j in range(n)]
    return Soup().add(V, F, mat)


def strip(points, w, z=0.0, mat="silk_white", closed=False):
    """Flat poly-line of width w (square caps), facing +z."""
    P = np.asarray(points, float)
    V, F = [], []
    segs = list(zip(P[:-1], P[1:]))
    if closed:
        segs.append((P[-1], P[0]))
    for A, B in segs:
        d = B - A
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        t = d / L
        nrm = np.array([-t[1], t[0]])
        A2, B2 = A - t * w / 2, B + t * w / 2
        i = len(V)
        for q in (A2 - nrm * w / 2, B2 - nrm * w / 2, B2 + nrm * w / 2, A2 + nrm * w / 2):
            V.append((q[0], q[1], z))
        F.append((i, i + 1, i + 2, i + 3))
    s = Soup()
    if V:
        s.add(V, np.array(F), mat)
    return s


def rect_outline(x0, y0, x1, y1, w=0.12, z=0.0, mat="silk_white"):
    return strip([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], w, z, mat, closed=True)


def plate(outline, holes=(), z0=0.0, z1=1.0, bevel=0.0, mat="plastic_black", bevel_res=0):
    """Extruded 2D shape with holes (through Blender's 2D curve fill); edges rounded by `bevel`."""
    cu = bpy.data.curves.new("pcp_tmp_plate", "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    for poly in [outline] + list(holes):
        poly = np.asarray(poly, float)
        sp = cu.splines.new("POLY")
        sp.points.add(len(poly) - 1)
        co = np.zeros((len(poly), 4))
        co[:, :2] = poly
        co[:, 3] = 1.0
        sp.points.foreach_set("co", co.ravel())
        sp.use_cyclic_u = True
    h = abs(z1 - z0) / 2
    b = min(bevel, 0.9 * h)
    cu.extrude = h - b
    if b > 1e-9:
        cu.bevel_depth = b
        cu.bevel_resolution = bevel_res
        cu.offset = -b
    ob = bpy.data.objects.new("pcp_tmp_plate", cu)
    me = bpy.data.meshes.new_from_object(ob)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    s = soup_from_mesh(me, mat)
    s.move(0, 0, (z0 + z1) / 2)
    bpy.data.meshes.remove(me)
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    return s


# ----------------------------------------------------------------------------
# bmesh bridge (general bevels)
# ----------------------------------------------------------------------------
def to_bm(s):
    bm = bmesh.new()
    vs = [bm.verts.new(v) for v in s.V]
    k = 0
    for sz, mi in zip(s.S, s.M):
        try:
            f = bm.faces.new([vs[i] for i in s.L[k:k + sz]])
            f.material_index = int(mi)
        except ValueError:
            pass
        k += sz
    bm.normal_update()
    return bm


def from_bm(bm, mats):
    s = Soup()
    bm.verts.index_update()
    s.V = np.array([v.co[:] for v in bm.verts], float).reshape(-1, 3)
    s.S = np.array([len(f.verts) for f in bm.faces], np.int32)
    s.L = np.array([v.index for f in bm.faces for v in f.verts], np.int64)
    s.M = np.array([f.material_index for f in bm.faces], np.int32)
    s.mats = list(mats)
    return s


def bevel(s, r, seg=2, angle=30.0, sel=None, profile=0.5):
    """Round every edge sharper than `angle` (optionally filtered by sel(mid, dir))."""
    bm = to_bm(s)
    ca = math.cos(math.radians(angle))
    edges = []
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        if e.link_faces[0].normal.dot(e.link_faces[1].normal) > ca:
            continue
        if sel is not None:
            a, b = e.verts[0].co, e.verts[1].co
            if not sel((a + b) / 2, (b - a).normalized()):
                continue
        edges.append(e)
    if edges:
        bmesh.ops.bevel(bm, geom=edges, offset=r, offset_type="OFFSET", segments=seg,
                        profile=profile, affect="EDGES", clamp_overlap=True)
    out = from_bm(bm, s.mats)
    bm.free()
    return out


def rbox(sx, sy, sz, c=(0.0, 0.0, 0.0), r=1.0, seg=3, mat="plastic_black", base=False, sel=None):
    """Box with properly rounded edges (bmesh bevel)."""
    return bevel(box(sx, sy, sz, c, 0.0, mat, base), r, seg, sel=sel)


# ----------------------------------------------------------------------------
# Paths and sweeps
# ----------------------------------------------------------------------------
def catmull(points, sub=12, closed=False):
    """Centripetal Catmull-Rom through points -> (m,3) array."""
    P = [np.asarray(p, float) for p in points]
    if len(P) < 2:
        return np.array(P)
    if len(P) == 2:
        return np.linspace(P[0], P[1], sub + 1)
    if closed:
        Q = [P[-1]] + P + [P[0], P[1]]
    else:
        Q = [2 * P[0] - P[1]] + P + [2 * P[-1] - P[-2]]
    out = []
    for i in range(len(Q) - 3):
        p0, p1, p2, p3 = Q[i:i + 4]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
        for k in range(sub):
            t = t1 + (t2 - t1) * k / sub
            A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
            out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    if not closed:
        out.append(P[-1])
    return np.array(out)


def resample(P, step):
    """Evenly spaced points along a polyline (keeps both ends)."""
    P = np.asarray(P, float)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(d)]
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, P[:, k]) for k in range(3)])


def smooth_path(points, step=2.0, sub=16):
    return resample(catmull(points, sub), step)


def fillet_path(points, r, n=6):
    """Round the corners of a 3D poly-line with radius ~r (quadratic blends)."""
    P = [np.asarray(p, float) for p in points]
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b = P[i - 1] - P[i], P[i + 1] - P[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        d = min(r, 0.45 * la, 0.45 * lb)
        s, e = P[i] + a / la * d, P[i] + b / lb * d
        for k in range(n + 1):
            t = k / n
            out.append((1 - t) ** 2 * s + 2 * t * (1 - t) * P[i] + t * t * e)
    out.append(P[-1])
    return np.array(out)


def path_frames(P, up=None):
    """Parallel-transport frames. Returns T, N, B (each (n,3)); N starts closest to `up`."""
    P = np.asarray(P, float)
    n = len(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    up = np.array([0.0, 0.0, 1.0] if up is None else up, float)
    n0 = up - (up @ T[0]) * T[0]
    if np.linalg.norm(n0) < 1e-6:
        alt = np.array([1.0, 0.0, 0.0]) if abs(T[0][0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n0 = alt - (alt @ T[0]) * T[0]
    n0 /= np.linalg.norm(n0)
    N = np.zeros_like(P)
    N[0] = n0
    for i in range(1, n):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        c = float(np.clip(T[i - 1] @ T[i], -1, 1))
        nn = N[i - 1]
        if s > 1e-9:
            k = v / s
            ang = math.atan2(s, c)
            nn = nn * math.cos(ang) + np.cross(k, nn) * math.sin(ang) + k * (k @ nn) * (1 - math.cos(ang))
        nn = nn - (nn @ T[i]) * T[i]
        N[i] = nn / np.linalg.norm(nn)
    B = np.cross(T, N)
    return T, N, B


def sweep(P, section, mat="plastic_black", caps=True, up=None, frames=None, scale=None, twist=None,
          uv=False):
    """Sweep a closed 2D section (k,2) [CCW in the (N,B) plane] along path P."""
    P = np.asarray(P, float)
    sec = ccw(section)
    k, n = len(sec), len(P)
    T, N, B = frames if frames is not None else path_frames(P, up)
    sc = np.ones(n) if scale is None else np.broadcast_to(np.asarray(scale, float), (n,))
    if twist is not None:
        tw = np.broadcast_to(np.asarray(twist, float), (n,))
        c, s = np.cos(tw)[:, None], np.sin(tw)[:, None]
        N, B = N * c + B * s, B * c - N * s
    V = (P[:, None, :] + sc[:, None, None] * (sec[None, :, 0:1] * N[:, None, :] + sec[None, :, 1:2] * B[:, None, :]))
    V = V.reshape(-1, 3)
    i = np.arange(n - 1)[:, None] * k
    j = np.arange(k)[None, :]
    j2 = (j + 1) % k
    F = np.stack([i + j, i + j2, i + k + j2, i + k + j], -1).reshape(-1, 4)
    s = Soup()
    UV = None
    if uv:
        per = np.r_[0.0, np.cumsum(np.linalg.norm(np.roll(sec, -1, 0) - sec, axis=1))]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        ii = np.repeat(np.arange(n - 1), k)
        jj = np.tile(np.arange(k), n - 1)
        UV = np.stack([np.stack([per[jj], arc[ii]], -1), np.stack([per[jj + 1], arc[ii]], -1),
                       np.stack([per[jj + 1], arc[ii + 1]], -1), np.stack([per[jj], arc[ii + 1]], -1)], 1)
        UV = UV.reshape(-1, 2)
    s.add(V, F, mat, uv=UV)
    if caps:
        s.add(V[:k], [tuple(range(k - 1, -1, -1))], mat, uv=np.zeros((k, 2)) if uv else None)
        s.add(V[-k:], [tuple(range(k))], mat, uv=np.zeros((k, 2)) if uv else None)
    return s


def tube(P, r, n=16, mat="plastic_black", caps=True, up=None, frames=None, uv=False):
    return sweep(P, circle(r, n), mat, caps, up, frames, uv=uv)


# ----------------------------------------------------------------------------
# Stroke font (silkscreen / laser marks) - generic characters only
# ----------------------------------------------------------------------------
_O = [(1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0)]
_P = [(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)]
GLYPHS = {
    "A": [[(0, 0), (0, 4), (2, 6), (4, 4), (4, 0)], [(0, 2), (4, 2)]],
    "B": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)], [(3, 3), (4, 2), (4, 1), (3, 0), (0, 0)]],
    "C": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1)]],
    "D": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 1), (3, 0), (0, 0)]],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3), (3, 3)]],
    "F": [[(4, 6), (0, 6), (0, 0)], [(0, 3), (3, 3)]],
    "G": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 3), (2, 3)]],
    "H": [[(0, 0), (0, 6)], [(4, 0), (4, 6)], [(0, 3), (4, 3)]],
    "I": [[(2, 0), (2, 6)], [(1, 0), (3, 0)], [(1, 6), (3, 6)]],
    "J": [[(4, 6), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "K": [[(0, 0), (0, 6)], [(4, 6), (0, 2)], [(1, 3), (4, 0)]],
    "L": [[(0, 6), (0, 0), (4, 0)]],
    "M": [[(0, 0), (0, 6), (2, 3), (4, 6), (4, 0)]],
    "N": [[(0, 0), (0, 6), (4, 0), (4, 6)]],
    "O": [_O],
    "P": [_P],
    "Q": [_O, [(2, 2), (4, 0)]],
    "R": [_P, [(2, 3), (4, 0)]],
    "S": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3), (3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "T": [[(0, 6), (4, 6)], [(2, 6), (2, 0)]],
    "U": [[(0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)]],
    "V": [[(0, 6), (2, 0), (4, 6)]],
    "W": [[(0, 6), (1, 0), (2, 4), (3, 0), (4, 6)]],
    "X": [[(0, 0), (4, 6)], [(0, 6), (4, 0)]],
    "Y": [[(0, 6), (2, 3), (4, 6)], [(2, 3), (2, 0)]],
    "Z": [[(0, 6), (4, 6), (0, 0), (4, 0)]],
    "0": [_O, [(1, 1), (3, 5)]],
    "1": [[(1, 5), (2, 6), (2, 0)], [(1, 0), (3, 0)]],
    "2": [[(0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (0, 0), (4, 0)]],
    "3": [[(0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1, 3)], [(3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "4": [[(3, 0), (3, 6), (0, 2), (4, 2)]],
    "5": [[(4, 6), (0, 6), (0, 3), (3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "6": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (3, 3), (0, 3)]],
    "7": [[(0, 6), (4, 6), (1, 0)]],
    "8": [[(1, 3), (0, 4), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1, 3), (0, 2), (0, 1), (1, 0),
           (3, 0), (4, 1), (4, 2), (3, 3)]],
    "9": [[(0, 1), (1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3), (4, 3)]],
    "-": [[(1, 3), (3, 3)]],
    "_": [[(0, 0), (4, 0)]],
    ".": [[(1.8, 0), (2.2, 0)]],
    ":": [[(1.8, 1.5), (2.2, 1.5)], [(1.8, 4.5), (2.2, 4.5)]],
    "/": [[(0, 0), (4, 6)]],
    "+": [[(0, 3), (4, 3)], [(2, 1), (2, 5)]],
    "=": [[(0, 2), (4, 2)], [(0, 4), (4, 4)]],
    "~": [[(0, 2.5), (1, 3.5), (3, 2.5), (4, 3.5)]],
    "(": [[(3, 6), (2, 5), (2, 1), (3, 0)]],
    ")": [[(1, 6), (2, 5), (2, 1), (1, 0)]],
    "#": [[(1, 0), (1, 6)], [(3, 0), (3, 6)], [(0, 2), (4, 2)], [(0, 4), (4, 4)]],
    "%": [[(0, 0), (4, 6)], [(0.5, 5), (1.5, 5)], [(2.5, 1), (3.5, 1)]],
    ">": [[(0, 5), (4, 3), (0, 1)]],
    "<": [[(4, 5), (0, 3), (4, 1)]],
    "^": [[(0, 1), (2, 5), (4, 1), (0, 1)]],
    "*": [[(2, 2.6), (2, 3.4)]],
}


def text(s, h=1.0, w=None, mat="silk_white", z=0.0, align="c", spacing=1.5):
    """Stroke text as flat quads facing +z, centred vertically on y = 0."""
    u = h / 6.0
    w = h * 0.14 if w is None else w
    adv = (4 + spacing) * u
    total = len(s) * adv - spacing * u
    x0 = {"c": -total / 2, "l": 0.0, "r": -total}[align]
    V, F = [], []
    for ch in s.upper():
        for pl in GLYPHS.get(ch, ()):
            for (ax, ay), (bx, by) in zip(pl[:-1], pl[1:]):
                A = np.array([x0 + ax * u, (ay - 3) * u])
                B = np.array([x0 + bx * u, (by - 3) * u])
                d = B - A
                L = np.linalg.norm(d)
                t = d / L if L > 1e-9 else np.array([1.0, 0.0])
                nrm = np.array([-t[1], t[0]]) * (w / 2)
                A2, B2 = A - t * (w / 2), B + t * (w / 2)
                i = len(V)
                for q in (A2 - nrm, B2 - nrm, B2 + nrm, A2 + nrm):
                    V.append((q[0], q[1], z))
                F.append((i, i + 1, i + 2, i + 3))
        x0 += adv
    out = Soup()
    if V:
        out.add(V, np.array(F), mat)
    return out


def text_width(s, h=1.0, spacing=1.5):
    return (len(s) * (4 + spacing) - spacing) * h / 6.0


# ----------------------------------------------------------------------------
# Objects / collections / mesh cache
# ----------------------------------------------------------------------------
_CACHE = {}


def proto(key, fn, smooth=40.0, weighted=None):
    """Mesh datablock cache: the same key always returns the same (linked) mesh."""
    me = _CACHE.get(key)
    if me is not None:
        try:
            if me.name and me.users >= 0:
                return me
        except ReferenceError:
            pass
    res = fn()
    me = res.mesh("pcp." + key, smooth, weighted) if isinstance(res, Soup) else res
    _CACHE[key] = me
    return me


def clear_cache():
    _CACHE.clear()


class Part:
    """A collection + root Empty; every object created through it is parented to the root."""

    def __init__(self, name):
        scene = bpy.context.scene
        self.coll = bpy.data.collections.new(name)
        scene.collection.children.link(self.coll)
        self.root = bpy.data.objects.new(name, None)
        self.root.empty_display_type = "PLAIN_AXES"
        self.root.empty_display_size = 10.0
        self.coll.objects.link(self.root)
        self.name = self.root.name

    def obj(self, name, mesh, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), parent=None, scale=None):
        if isinstance(mesh, Soup):
            mesh = mesh.mesh(self.name + "_" + name)
        ob = bpy.data.objects.new(self.name + "_" + name, mesh)
        ob.parent = parent or self.root
        ob.location = loc
        ob.rotation_euler = tuple(math.radians(a) for a in rot)
        if scale is not None:
            ob.scale = scale if hasattr(scale, "__len__") else (scale, scale, scale)
        self.coll.objects.link(ob)
        return ob

    def empty(self, name, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), parent=None):
        ob = bpy.data.objects.new(self.name + "_" + name, None)
        ob.empty_display_size = 5.0
        ob.parent = parent or self.root
        ob.location = loc
        ob.rotation_euler = tuple(math.radians(a) for a in rot)
        self.coll.objects.link(ob)
        return ob


def tri_count(root_or_objs, unique=False):
    """Triangles of a hierarchy (instances counted unless unique=True)."""
    if hasattr(root_or_objs, "children_recursive"):
        objs = [root_or_objs] + list(root_or_objs.children_recursive)
    else:
        objs = list(root_or_objs)
    seen, tot = set(), 0
    for ob in objs:
        if ob.type != "MESH":
            continue
        me = ob.data
        if unique and me.name in seen:
            continue
        seen.add(me.name)
        lt = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_total", lt)
        tot += int((lt - 2).sum())
    return tot
