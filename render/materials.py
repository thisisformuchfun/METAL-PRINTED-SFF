"""materials.py - Principled-BSDF node material library for the MARROW pc_parts models.

All procedural patterns live in OBJECT space with 1 unit = 1 mm (a few cable/ribbon
materials use a UV map that is also authored in mm: u = around/across, v = along).

Scale-dependent nodes (Bump distance, volume density) are authored in mm and tagged;
call `set_unit_scale(0.001)` once after parenting the parts under a 0.001-scaled root
so bump strength and coolant absorption keep their look (Cycles evaluates both in
world space).

    import materials
    m = materials.get("enig_gold")          # creates on first use -> bpy material "pcp_enig_gold"
    materials.set_unit_scale(0.001)
"""
import math

import bpy

PREFIX = "pcp_"
UNIT = 1.0            # world units per authored millimetre
_REG = {}
TAU = 2.0 * math.pi


def reg(name):
    def deco(fn):
        _REG[name] = fn
        return fn
    return deco


def names():
    return sorted(_REG)


def get(name):
    """Return (creating on first use) the library material `name`."""
    m = bpy.data.materials.get(PREFIX + name)
    if m is not None:
        return m
    fn = _REG.get(name)
    if fn is None:
        print("materials: unknown material '%s' -> plastic_black" % name)
        fn = _REG["plastic_black"]
    m = bpy.data.materials.new(PREFIX + name)
    m.use_nodes = True
    fn(NT(m))
    _apply_unit(m)
    return m


def set_unit_scale(s):
    """World units per mm (1.0 = scene in mm, 0.001 = parts under a 0.001 root)."""
    global UNIT
    UNIT = float(s)
    for m in bpy.data.materials:
        if m.name.startswith(PREFIX):
            _apply_unit(m)


def _apply_unit(m):
    if not m.use_nodes:
        return
    for n in m.node_tree.nodes:
        if n.label.startswith("mm="):
            n.inputs["Distance"].default_value = float(n.label[3:]) * UNIT
        elif n.label.startswith("per_mm="):
            n.inputs["Density"].default_value = float(n.label[7:]) / UNIT


def rgba(c, a=1.0):
    if isinstance(c, (int, float)):
        return (c, c, c, a)
    return (c[0], c[1], c[2], a) if len(c) == 3 else tuple(c)


# ----------------------------------------------------------------------------
# node-tree helper
# ----------------------------------------------------------------------------
class NT:
    PNAMES = {"color": "Base Color", "metal": "Metallic", "rough": "Roughness", "ior": "IOR",
              "spec": "Specular IOR Level", "coat": "Coat Weight", "coat_rough": "Coat Roughness",
              "trans": "Transmission Weight", "sheen": "Sheen Weight", "sheen_rough": "Sheen Roughness",
              "normal": "Normal", "aniso": "Anisotropic", "alpha": "Alpha", "tf": "Thin Film Thickness",
              "tf_ior": "Thin Film IOR", "emit": "Emission Color", "emit_s": "Emission Strength",
              "sss": "Subsurface Weight", "sss_scale": "Subsurface Scale", "coat_normal": "Coat Normal",
              "spec_tint": "Specular Tint", "aniso_rot": "Anisotropic Rotation", "tangent": "Tangent"}

    def __init__(self, m):
        self.m = m
        self.t = m.node_tree
        self.N = self.t.nodes
        self.K = self.t.links
        self.b = self.N.get("Principled BSDF")
        self.out = self.N.get("Material Output")
        self._co = {}

    def node(self, typ, **kw):
        n = self.N.new(typ)
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def set(self, sock, val):
        if isinstance(val, bpy.types.NodeSocket):
            self.K.new(val, sock)
        elif val is not None:
            try:
                sock.default_value = val
            except Exception:
                sock.default_value = rgba(val) if not isinstance(val, (int, float)) else (val, val, val)

    def P(self, **kw):
        for k, v in kw.items():
            self.set(self.b.inputs[self.PNAMES[k]], rgba(v) if k in ("color", "emit") and not isinstance(v, bpy.types.NodeSocket) else v)

    def coord(self, kind="Object"):
        if "tc" not in self._co:
            self._co["tc"] = self.node("ShaderNodeTexCoord")
        return self._co["tc"].outputs[kind]

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op)
        n.use_clamp = clamp
        self.set(n.inputs[0], a)
        if b is not None:
            self.set(n.inputs[1], b)
        if c is not None:
            self.set(n.inputs[2], c)
        return n.outputs[0]

    def add(self, a, b):
        return self.math("ADD", a, b)

    def sub(self, a, b):
        return self.math("SUBTRACT", a, b)

    def mul(self, a, b):
        return self.math("MULTIPLY", a, b)

    def sat(self, a):
        return self.math("ADD", a, 0.0, clamp=True)

    def step(self, x, edge, k=40.0):
        """soft step: clamp((x-edge)*k)."""
        return self.math("MULTIPLY", self.math("SUBTRACT", x, edge), k, clamp=True)

    def sep(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self.K.new(v, n.inputs[0])
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def comb(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self.set(n.inputs[i], v)
        return n.outputs[0]

    def sepcol(self, c):
        n = self.node("ShaderNodeSeparateColor")
        self.K.new(c, n.inputs[0])
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def vadd(self, v, off):
        n = self.node("ShaderNodeVectorMath", operation="ADD")
        self.set(n.inputs[0], v)
        self.set(n.inputs[1], off)
        return n.outputs[0]

    def vscale(self, v, s):
        n = self.node("ShaderNodeVectorMath", operation="MULTIPLY")
        self.set(n.inputs[0], v)
        self.set(n.inputs[1], s if not isinstance(s, (int, float)) else (s, s, s))
        return n.outputs[0]

    def mix(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="RGBA")
        self.set(n.inputs[0], fac)
        self.set(n.inputs[6], rgba(a) if not isinstance(a, bpy.types.NodeSocket) else a)
        self.set(n.inputs[7], rgba(b) if not isinstance(b, bpy.types.NodeSocket) else b)
        return n.outputs[2]

    def mixf(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="FLOAT")
        self.set(n.inputs[0], fac)
        self.set(n.inputs[2], a)
        self.set(n.inputs[3], b)
        return n.outputs[0]

    def ramp(self, fac, stops, interp="LINEAR"):
        n = self.node("ShaderNodeValToRGB")
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for e, (p, c) in zip(cr.elements, stops):
            e.position = p
            e.color = rgba(c)
        self.set(n.inputs[0], fac)
        return n.outputs[0]

    def noise(self, vec=None, scale=1.0, detail=2.0, rough=0.5, dim="3D", out="Fac", distortion=0.0):
        n = self.node("ShaderNodeTexNoise", noise_dimensions=dim)
        if vec is not None:
            self.set(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        return n.outputs[out]

    def voronoi(self, vec, scale=1.0, feature="F1", dist="EUCLIDEAN", rand=1.0, dim="3D"):
        n = self.node("ShaderNodeTexVoronoi", feature=feature, distance=dist, voronoi_dimensions=dim)
        self.set(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Randomness"].default_value = rand
        return n

    def bump(self, height, dist_mm, strength=1.0, normal=None):
        n = self.node("ShaderNodeBump")
        n.label = "mm=%g" % dist_mm
        n.inputs["Strength"].default_value = strength
        n.inputs["Distance"].default_value = dist_mm * UNIT
        self.set(n.inputs["Height"], height)
        if normal is not None:
            self.set(n.inputs["Normal"], normal)
        return n.outputs[0]


def surf(nt, color, rough, metal=0.0, bump=0.0, bscale=10.0, rvar=0.0, cvar=0.0, detail=2.0, **kw):
    """Generic micro-surface: noise-driven roughness / colour variation and fine bump."""
    nt.P(color=color, rough=rough, metal=metal, **kw)
    if not (bump or rvar or cvar):
        return
    co = nt.coord()
    if rvar or cvar:
        lo = nt.noise(co, scale=bscale * 0.12, detail=3.0)
        if rvar:
            nt.P(rough=nt.add(rough - rvar, nt.mul(lo, 2.0 * rvar)))
        if cvar:
            c = rgba(color)
            nt.P(color=nt.mix(lo, tuple(v * (1 - cvar) for v in c[:3]) + (1,),
                              tuple(min(0.95, v * (1 + cvar)) for v in c[:3]) + (1,)))
    if bump:
        hi = nt.noise(co, scale=bscale, detail=detail)
        nt.P(normal=nt.bump(hi, bump))


def simple(name, color, rough, metal=0.0, **kw):
    def fn(nt):
        surf(nt, color, rough, metal, **kw)
    _REG[name] = fn


# ----------------------------------------------------------------------------
# plain / micro-textured materials
# ----------------------------------------------------------------------------
# plastics
simple("plastic_black", (0.020, 0.020, 0.022), 0.42, bump=0.003, bscale=22.0, rvar=0.05)
simple("plastic_gloss_black", (0.016, 0.016, 0.018), 0.22, rvar=0.03, bscale=8.0)
simple("pa66_natural", (0.74, 0.70, 0.58), 0.46, bump=0.003, bscale=18.0, rvar=0.05, cvar=0.04)
simple("slot_grey", (0.19, 0.20, 0.22), 0.40, bump=0.003, bscale=20.0, rvar=0.05)
simple("plastic_white", (0.78, 0.78, 0.76), 0.40, bump=0.002, bscale=20.0, rvar=0.04)
simple("plastic_red", (0.42, 0.025, 0.025), 0.38, rvar=0.04, bscale=20.0)
simple("plastic_blue", (0.02, 0.10, 0.42), 0.36, rvar=0.04, bscale=20.0)
simple("plastic_teal", (0.02, 0.28, 0.30), 0.38, rvar=0.04, bscale=20.0)
simple("plastic_grey", (0.33, 0.33, 0.34), 0.42, rvar=0.04, bscale=20.0)
simple("fan_plastic", (0.020, 0.020, 0.022), 0.38, bump=0.004, bscale=14.0, rvar=0.05)
simple("fan_blade", (0.022, 0.022, 0.025), 0.27, bump=0.002, bscale=6.0, rvar=0.05)
simple("rubber", (0.022, 0.022, 0.022), 0.72, bump=0.006, bscale=16.0, rvar=0.05)
simple("rubber_grey", (0.07, 0.07, 0.072), 0.78, bump=0.006, bscale=16.0, rvar=0.05)
simple("oring", (0.018, 0.018, 0.018), 0.48, rvar=0.05, bscale=10.0)
simple("impeller", (0.045, 0.047, 0.05), 0.30, rvar=0.05, bscale=8.0)
simple("ceramic", (0.82, 0.82, 0.80), 0.18)
# paints
simple("paint_satin_black", (0.016, 0.016, 0.017), 0.36, bump=0.004, bscale=7.0, rvar=0.05)
simple("paint_fin_black", (0.018, 0.018, 0.019), 0.42, rvar=0.06, bscale=5.0)
simple("powdercoat_black", (0.018, 0.018, 0.019), 0.46, bump=0.030, bscale=1.7, rvar=0.05, detail=1.0)
# metals
simple("black_nickel", (0.150, 0.145, 0.140), 0.25, 1.0, bump=0.0015, bscale=30.0, rvar=0.04)
simple("steel_sheet", (0.70, 0.69, 0.67), 0.28, 1.0, bump=0.002, bscale=12.0, rvar=0.06)
simple("steel_bright", (0.78, 0.78, 0.77), 0.16, 1.0, rvar=0.04, bscale=10.0)
simple("stiffener", (0.74, 0.74, 0.73), 0.44, 1.0, bump=0.002, bscale=40.0, rvar=0.05)
simple("tin", (0.80, 0.80, 0.78), 0.32, 1.0, rvar=0.06, bscale=20.0)
simple("solder", (0.74, 0.74, 0.72), 0.20, 1.0, bump=0.004, bscale=25.0, rvar=0.06)
simple("enig_gold", (0.88, 0.68, 0.33), 0.30, 1.0, bump=0.001, bscale=40.0, rvar=0.05)
simple("gold_pin", (0.94, 0.74, 0.33), 0.18, 1.0, rvar=0.03, bscale=20.0)
simple("copper", (0.92, 0.50, 0.35), 0.28, 1.0, rvar=0.05, bscale=10.0)
simple("alu_can", (0.87, 0.88, 0.89), 0.22, 1.0, bump=0.001, bscale=40.0, rvar=0.05)
simple("alu_raw", (0.80, 0.81, 0.82), 0.40, 1.0, bump=0.003, bscale=20.0, rvar=0.06)
simple("nickel_ihs", (0.76, 0.74, 0.70), 0.20, 1.0, bump=0.0012, bscale=45.0, rvar=0.05)
simple("ihs_mark", (0.40, 0.39, 0.38), 0.52, 1.0)
simple("anodized_black", (0.028, 0.028, 0.030), 0.36, 0.85, bump=0.0015, bscale=30.0, rvar=0.05)
simple("screw_black", (0.07, 0.07, 0.075), 0.40, 1.0, rvar=0.05, bscale=20.0)
simple("screw_zinc", (0.76, 0.76, 0.78), 0.30, 1.0, rvar=0.05, bscale=20.0)
simple("brass", (0.86, 0.66, 0.30), 0.26, 1.0, rvar=0.04, bscale=20.0)
# electronics
simple("ic_epoxy", (0.020, 0.020, 0.022), 0.55, bump=0.004, bscale=38.0, rvar=0.06, detail=3.0)
simple("laser_mark", (0.135, 0.135, 0.13), 0.72)
simple("laser_mark_dim", (0.075, 0.075, 0.072), 0.70)
simple("inductor_grey", (0.080, 0.080, 0.085), 0.60, 0.25, bump=0.010, bscale=28.0, rvar=0.06, cvar=0.25, detail=4.0)
simple("ferrite", (0.028, 0.028, 0.030), 0.50, bump=0.006, bscale=28.0, rvar=0.05, detail=3.0)
simple("mlcc_tan", (0.33, 0.22, 0.13), 0.45, rvar=0.05, cvar=0.12, bscale=6.0)
simple("mlcc_grey", (0.27, 0.26, 0.24), 0.45, rvar=0.05, cvar=0.10, bscale=6.0)
simple("res_black", (0.018, 0.018, 0.02), 0.30, rvar=0.04, bscale=20.0)
simple("alumina", (0.72, 0.72, 0.70), 0.5)
simple("silk_white", (0.78, 0.78, 0.76), 0.55)
simple("silk_grey", (0.30, 0.30, 0.30), 0.55)
simple("cap_print", (0.030, 0.030, 0.10), 0.42)
simple("underfill", (0.014, 0.014, 0.016), 0.16)
simple("pcb_edge", (0.050, 0.043, 0.030), 0.70, bump=0.01, bscale=12.0, rvar=0.05)
simple("hole_dark", (0.012, 0.012, 0.012), 0.8)
simple("led_lens", (0.80, 0.78, 0.60), 0.25)
simple("tant_body", (0.030, 0.027, 0.024), 0.46, bump=0.003, bscale=30.0, rvar=0.05)
# labels
simple("label_paper", (0.80, 0.80, 0.78), 0.48, rvar=0.04, bscale=3.0)
simple("label_silver", (0.66, 0.66, 0.67), 0.36, 0.9, rvar=0.05, bscale=4.0)
simple("sticker_black", (0.028, 0.028, 0.03), 0.33, rvar=0.04, bscale=4.0)
simple("ink_black", (0.03, 0.03, 0.03), 0.6)
simple("ink_grey", (0.42, 0.42, 0.42), 0.55)
simple("ink_white", (0.80, 0.80, 0.78), 0.55)
simple("backdrop", (0.42, 0.42, 0.43), 0.6)


# ----------------------------------------------------------------------------
# PCB solder mask with procedural trace / via relief
# ----------------------------------------------------------------------------
def _pcb(nt, base, cu, rough=0.5, cell=14.0, seed=0.0, relief=0.03, weave=0.06, via_gate=0.52,
         pitch=(0.30, 0.38), coat=0.0):
    co = nt.vadd(nt.coord(), (seed * 37.13, seed * 11.71, 0.0))
    x, y, _z = nt.sep(co)
    xy = nt.comb(x, y, 0.0)
    vor = nt.voronoi(xy, scale=1.0 / cell, dim="2D")
    r, g, b = nt.sepcol(vor.outputs["Color"])
    ang = nt.mul(nt.math("FLOOR", nt.mul(r, 4.0)), math.pi / 4)
    ca, sa = nt.math("COSINE", ang), nt.math("SINE", ang)
    u = nt.add(nt.mul(x, ca), nt.mul(y, sa))
    w = nt.sub(nt.mul(y, ca), nt.mul(x, sa))
    p = nt.add(pitch[0], nt.mul(g, pitch[1]))
    wp = nt.math("DIVIDE", w, p)
    lt = nt.math("FRACT", wp)
    lid = nt.math("FLOOR", wp)
    d = nt.math("ABSOLUTE", nt.sub(lt, 0.5))
    line = nt.math("MULTIPLY", nt.sub(0.24, d), 10.0, clamp=True)
    segv = nt.comb(nt.mul(u, 0.055), nt.mul(lid, 1.731), nt.mul(b, 50.0))
    seg = nt.step(nt.noise(segv, scale=1.0, detail=0.0), 0.47, 25.0)
    regn = nt.step(b, 0.30, 500.0)
    traces = nt.mul(nt.mul(line, seg), regn)
    # tented vias
    vv = nt.voronoi(xy, scale=1.0 / 1.45, dim="2D", rand=0.8)
    d2 = vv.outputs["Distance"]
    gate = nt.step(nt.noise(xy, scale=0.045, detail=1.0), via_gate, 20.0)
    ring = nt.sub(nt.math("MULTIPLY", nt.sub(0.30, d2), 22.0, clamp=True),
                  nt.mul(nt.math("MULTIPLY", nt.sub(0.13, d2), 22.0, clamp=True), 1.7))
    vias = nt.mul(ring, gate)
    wv = nt.mul(nt.math("SINE", nt.mul(x, TAU / 0.56)), nt.math("SINE", nt.mul(y, TAU / 0.56)))
    height = nt.add(nt.add(traces, nt.mul(vias, 1.2)), nt.mul(wv, weave))
    cfac = nt.math("ADD", nt.mul(traces, 0.75), nt.mul(nt.sub(1.0, regn), 0.45), clamp=True)
    cfac = nt.math("MAXIMUM", cfac, nt.sat(vias))
    lo = nt.noise(xy, scale=0.02, detail=3.0)
    nt.P(color=nt.mix(cfac, base, cu), rough=nt.add(nt.sub(rough - 0.05, nt.mul(cfac, 0.06)), nt.mul(lo, 0.10)),
         normal=nt.bump(height, relief), coat=coat, coat_rough=0.15)


@reg("pcb_black")
def _pcb_black(nt):
    _pcb(nt, (0.0125, 0.0135, 0.0145), (0.0075, 0.0085, 0.0100), rough=0.50, seed=0.0)


@reg("pcb_black_gpu")
def _pcb_black_gpu(nt):
    _pcb(nt, (0.0120, 0.0125, 0.0135), (0.0070, 0.0078, 0.0092), rough=0.47, seed=3.0, cell=11.0)


@reg("pcb_green")
def _pcb_green(nt):
    _pcb(nt, (0.012, 0.085, 0.070), (0.020, 0.150, 0.110), rough=0.32, seed=5.0, cell=5.0, relief=0.012,
         weave=0.02, via_gate=0.35, pitch=(0.16, 0.14))


@reg("pcb_blue")
def _pcb_blue(nt):
    _pcb(nt, (0.010, 0.030, 0.110), (0.016, 0.055, 0.200), rough=0.34, seed=7.0, cell=6.0, relief=0.02,
         pitch=(0.2, 0.2))


@reg("pcb_darkgreen")
def _pcb_dgreen(nt):
    _pcb(nt, (0.010, 0.035, 0.022), (0.014, 0.060, 0.035), rough=0.36, seed=9.0, cell=6.0, relief=0.02,
         pitch=(0.2, 0.2))


# ----------------------------------------------------------------------------
# special materials
# ----------------------------------------------------------------------------
@reg("silicon_die")
def _silicon(nt):
    lw = nt.node("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.30
    col = nt.ramp(lw.outputs["Facing"], [(0.0, (0.29, 0.31, 0.37)), (0.55, (0.33, 0.30, 0.40)),
                                         (1.0, (0.42, 0.36, 0.31))])
    nt.P(color=col, metal=1.0, rough=0.035, tf=290.0, tf_ior=1.9)


@reg("pvc_clear")
def _pvc(nt):
    nt.P(color=(0.93, 0.95, 0.96), rough=0.045, trans=1.0, ior=1.48)


@reg("glass_clear")
def _glass(nt):
    nt.P(color=(0.95, 0.95, 0.95), rough=0.01, trans=1.0, ior=1.5)


def _coolant(nt, col, dens):
    nt.P(color=(0.95, 0.94, 0.94), rough=0.0, trans=1.0, ior=1.34)
    va = nt.node("ShaderNodeVolumeAbsorption")
    va.label = "per_mm=%g" % dens
    va.inputs["Color"].default_value = rgba(col)
    va.inputs["Density"].default_value = dens / UNIT
    nt.K.new(va.outputs[0], nt.out.inputs["Volume"])


@reg("coolant_ruby")
def _ruby(nt):
    _coolant(nt, (0.94, 0.015, 0.07), 0.46)


@reg("coolant_clear")
def _cclear(nt):
    _coolant(nt, (0.90, 0.93, 0.95), 0.02)


def _sleeve(color):
    def fn(nt):
        x, y, _ = nt.sep(nt.coord("UV"))
        k = TAU / 1.15
        a = nt.math("SINE", nt.mul(nt.add(x, y), k))
        b = nt.math("SINE", nt.mul(nt.sub(x, y), k))
        h = nt.math("MAXIMUM", a, b)
        f = nt.math("ADD", nt.mul(h, 0.5), 0.5, clamp=True)
        c = rgba(color)
        col = nt.mix(f, tuple(v * 0.45 for v in c[:3]) + (1,), c)
        nt.P(color=col, rough=0.58, sheen=0.35, sheen_rough=0.45, normal=nt.bump(h, 0.07))
    return fn


_REG["sleeve_black"] = _sleeve((0.030, 0.030, 0.032))
_REG["sleeve_grey"] = _sleeve((0.16, 0.16, 0.17))
_REG["sleeve_white"] = _sleeve((0.72, 0.72, 0.70))
_REG["sleeve_red"] = _sleeve((0.45, 0.02, 0.03))


@reg("ribbon_black")
def _ribbon(nt):
    x, y, _ = nt.sep(nt.coord("UV"))
    ridge = nt.math("ABSOLUTE", nt.math("SINE", nt.mul(x, math.pi / 1.27)))
    nz = nt.noise(nt.comb(nt.mul(x, 4.0), nt.mul(y, 0.4), 0.0), scale=1.0, detail=2.0)
    h = nt.add(nt.mul(ridge, 0.6), nt.mul(nz, 0.4))
    nt.P(color=(0.020, 0.020, 0.022), rough=nt.add(0.36, nt.mul(nz, 0.14)), sheen=0.2, sheen_rough=0.4,
         normal=nt.bump(h, 0.05))


@reg("wire_black")
def _wire(nt):
    surf(nt, (0.022, 0.022, 0.024), 0.42, rvar=0.05, bscale=6.0)


@reg("brushed_alu")
def _brushed(nt):
    co = nt.coord()
    x, y, z = nt.sep(co)
    nz = nt.noise(nt.comb(nt.mul(x, 0.6), nt.mul(y, 40.0), nt.mul(z, 40.0)), scale=1.0, detail=2.0)
    nt.P(color=(0.80, 0.81, 0.82), metal=1.0, rough=nt.add(0.24, nt.mul(nz, 0.16)), normal=nt.bump(nz, 0.002))
