"""test_parts.py - render every pc_parts builder alone on a neutral studio.

    blender -b --python render/test_parts.py -- [--only board,gpu] [--views a,b] [--res 960x540]
                                                [--spp 48] [--out render/out/parts] [--sheet-only]

Each part is built in an empty scene (1 unit = 1 mm), set on a grey seamless backdrop under a
procedural soft-box world plus two area lights, and rendered from its view list (>= two angles).
A contact sheet of everything in the output folder is written at the end.
"""
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

import pc_parts          # noqa: E402
import materials         # noqa: E402
import pcp_geom          # noqa: E402

THREADS = 2


# ----------------------------------------------------------------------------
# scene / studio
# ----------------------------------------------------------------------------
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    pcp_geom.clear_cache()
    materials.set_unit_scale(1.0)
    sc = bpy.context.scene
    sc.unit_settings.scale_length = 0.001
    return sc


def setup_render(sc, res=(960, 540), spp=48):
    sc.render.engine = "CYCLES"
    c = sc.cycles
    c.device = "CPU"
    c.samples = spp
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.03
    c.use_denoising = True
    try:
        c.denoiser = "OPENIMAGEDENOISE"
        c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except Exception:
        pass
    c.max_bounces = 8
    c.diffuse_bounces = 3
    c.glossy_bounces = 5
    c.transmission_bounces = 10
    c.transparent_max_bounces = 8
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.sample_clamp_indirect = 8.0
    sc.render.threads_mode = "FIXED"
    sc.render.threads = THREADS
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    sc.view_settings.exposure = 0.0


def studio_world(sc, strength=1.0):
    """Procedural 'HDRI-like' world: soft gradient dome with three large soft boxes."""
    w = bpy.data.worlds.new("pcp_studio")
    w.use_nodes = True
    sc.world = w
    nt = w.node_tree
    N, K = nt.nodes, nt.links
    N.clear()
    out = N.new("ShaderNodeOutputWorld")
    bg = N.new("ShaderNodeBackground")
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    K.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = N.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.35, (0.05, 0.05, 0.055, 1)
    cr.elements[1].position, cr.elements[1].color = 1.0, (0.55, 0.56, 0.58, 1)
    m = N.new("ShaderNodeMath")
    m.operation = "MULTIPLY_ADD"
    K.new(sep.outputs[2], m.inputs[0])
    m.inputs[1].default_value, m.inputs[2].default_value = 0.5, 0.5
    K.new(m.outputs[0], ramp.inputs[0])
    col = ramp.outputs[0]

    def softbox(direction, cos_in, cos_out, power, tint=(1, 1, 1)):
        nonlocal col
        d = Vector(direction).normalized()
        dot = N.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        nrm = N.new("ShaderNodeVectorMath")
        nrm.operation = "NORMALIZE"
        K.new(tc.outputs["Generated"], nrm.inputs[0])
        K.new(nrm.outputs[0], dot.inputs[0])
        dot.inputs[1].default_value = d
        mr = N.new("ShaderNodeMapRange")
        mr.interpolation_type = "SMOOTHSTEP"
        K.new(dot.outputs["Value"], mr.inputs[0])
        mr.inputs[1].default_value, mr.inputs[2].default_value = cos_out, cos_in
        mr.inputs[3].default_value, mr.inputs[4].default_value = 0.0, power
        mix = N.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "ADD"
        mix.inputs[0].default_value = 1.0
        K.new(col, mix.inputs[6])
        mul = N.new("ShaderNodeVectorMath")
        mul.operation = "SCALE"
        mul.inputs[0].default_value = tint
        K.new(mr.outputs[0], mul.inputs["Scale"])
        K.new(mul.outputs[0], mix.inputs[7])
        col = mix.outputs[2]

    softbox((-0.55, -0.45, 0.70), 0.90, 0.78, 3.2, (1.0, 0.98, 0.95))     # key side
    softbox((0.75, -0.25, 0.45), 0.93, 0.84, 1.6, (0.92, 0.96, 1.0))      # fill side
    softbox((0.10, 0.80, 0.55), 0.95, 0.86, 2.4, (1.0, 1.0, 1.0))         # back / rim strip
    softbox((0.0, 0.0, 1.0), 0.97, 0.80, 0.8)                             # overhead
    K.new(col, bg.inputs[0])
    bg.inputs[1].default_value = strength
    K.new(bg.outputs[0], out.inputs[0])


def backdrop(sc, z_floor, R, centre):
    """Grey seamless sweep: floor curving up into a back wall."""
    prof = [(-14 * R, 0.0)]
    r = 5 * R
    y0 = 6 * R
    for a in np.linspace(-math.pi / 2, 0, 17):
        prof.append((y0 + r * math.cos(a), r + r * math.sin(a)))
    prof.append((y0 + r, 16 * R))
    V, F = [], []
    for i, (y, z) in enumerate(prof):
        V += [(-22 * R, y, z), (22 * R, y, z)]
        if i:
            F.append((2 * i - 2, 2 * i - 1, 2 * i + 1, 2 * i))
    s = pcp_geom.Soup().add(V, F, "backdrop")
    me = s.mesh("pcp_backdrop", smooth=60, weighted=False)
    ob = bpy.data.objects.new("pcp_backdrop", me)
    ob.location = (centre[0], centre[1], z_floor)
    sc.collection.objects.link(ob)
    return ob


def area_light(sc, name, target, direction, dist, size, radiance, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "SQUARE"
    ld.size = size
    ld.energy = radiance * 4.0 * math.pi * size * size       # W for the wanted surface radiance
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    d = Vector(direction).normalized()
    ob.location = Vector(target) + d * dist
    ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(ob)
    return ob


def camera(sc, target, az, el, dist, lens=70.0, fstop=None):
    cd = bpy.data.cameras.new("pcp_cam")
    cd.lens = lens
    cd.sensor_width = 36.0
    cd.clip_start, cd.clip_end = max(0.5, dist * 0.02), dist * 60
    ob = bpy.data.objects.new("pcp_cam", cd)
    a, e = math.radians(az), math.radians(el)
    d = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
    ob.location = Vector(target) + d * dist
    ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    if fstop:
        cd.dof.use_dof = True
        cd.dof.focus_distance = dist
        cd.dof.aperture_fstop = fstop
    sc.collection.objects.link(ob)
    sc.camera = ob
    return ob


def world_bbox(root):
    lo, hi = Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))
    bpy.context.view_layer.update()
    for ob in [root] + list(root.children_recursive):
        if ob.type != "MESH":
            continue
        for c in ob.bound_box:
            p = ob.matrix_world @ Vector(c)
            lo = Vector((min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z)))
            hi = Vector((max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z)))
    return lo, hi


# ----------------------------------------------------------------------------
# part registry: name -> (builder, views)
# view = (suffix, target or None (bbox centre), azimuth, elevation, distance (None = auto), lens, f-stop or None)
# ----------------------------------------------------------------------------
def _cable_demo(kind):
    def f(L, G):
        return pc_parts.build_cable([(0, 0, 0), (0, 0, 55), (45, 20, 110), (150, 30, 110), (190, 30, 60)], kind, up=(0, -1, 0))
    return f


def _tube_demo(L, G):
    import pcp_geom as g
    root = pc_parts.build_tube([(0, 0, 20.4), (0, 0, 60), (40, 10, 105), (110, 10, 105), (150, 0, 60), (150, 0, 20.4)], name="tube")
    for i, (kind, x) in enumerate((("straight", 0.0), ("straight", 150.0))):
        r, p, d = pc_parts.build_fitting(kind, name="tube_fit%d" % i)
        r.location = (x, 0, 0)
    return root


def _fittings_demo(L, G):
    P = pcp_geom.Part("fittings")
    for i, kind in enumerate(("straight", "elbow90", "plug")):
        r, p, d = pc_parts.build_fitting(kind)
        r.parent = P.root
        r.location = ((i - 1) * 34.0, 0, 5.0)
    blk = pcp_geom.box(120, 36, 10, (0, 0, -5.0 + 5.0), 1.0, "alu_raw")
    P.obj("block", blk.mesh("fit_block"))
    return P.root


def _riser_demo(L, G):
    return pc_parts.build_riser([(0, 0, 0), (0, 0, 30), (0, -45, 62), (0, -95, 40), (0, -95, 14.0)], width_dir=(1, 0, 0),
                                male_dir=(0, 0, 1), female_dir=(0, 0, 1))


PARTS = {
    "board": (lambda L, G: pc_parts.build_itx_board(L),
              [("a", None, -32, 46, None, 70, None), ("b", (62, 108, 4), 35, 28, 175, 85, None),
               ("c", (34, 104, 6), -62, 22, 190, 85, None), ("d", (96, 32, 3), 18, 22, 170, 85, None),
               ("e", (150, 100, 10), 128, 26, 230, 85, None), ("back", None, -30, 50, None, 70, None)]),
    "gpu": (lambda L, G: pc_parts.build_gpu_4090(L, G),
            [("a", None, -28, 48, None, 70, None), ("b", (116, 59, 2), 40, 24, 170, 85, None),
             ("c", (205, 78, 3), -45, 24, 170, 85, None), ("d", (38, 62, 3), -38, 28, 175, 85, None),
             ("e", (92, -4, 0), -12, 22, 150, 85, None), ("back", None, -28, 50, None, 70, None),
             ("backb", (116, 59, -3), 30, 26, 190, 85, None)]),
    "psu": (lambda L, G: pc_parts.build_psu_atx(L),
            [("a", None, 42, 32, None, 70, None), ("b", None, -140, 26, None, 70, None)]),
    "radiator": (lambda L, G: pc_parts.build_radiator_240(L),
                 [("a", None, -28, 22, None, 70, None), ("b", (100, -14, 0), -30, 16, 260, 85, None),
                  ("c", (0, -13, 0), -8, 6, 120, 85, None)]),
    "fan": (lambda L, G: pc_parts.build_fan_120(L),
            [("a", None, -32, 20, None, 70, None), ("b", None, 148, 18, None, 70, None)]),
    "pump": (lambda L, G: pc_parts.build_ddc_pump(L),
             [("a", None, -35, 52, None, 70, None), ("b", None, 140, 12, None, 70, None)]),
    "fittings": (_fittings_demo, [("a", None, -25, 30, None, 85, None), ("b", (0, 0, 14), 20, 12, 150, 100, None)]),
    "tube": (_tube_demo, [("a", None, -22, 16, None, 70, None), ("b", (10, 0, 48), 30, 8, 170, 85, None)]),
    "cable_atx24": (_cable_demo("atx24"), [("a", None, -35, 28, None, 70, None), ("b", (5, 0, 18), 50, 20, 150, 85, None)]),
    "cable_eps8": (_cable_demo("eps8"), [("a", None, -35, 28, None, 70, None)]),
    "cable_12vhpwr": (_cable_demo("12vhpwr"), [("a", None, -35, 28, None, 70, None), ("b", (5, 0, 14), 50, 20, 110, 85, None)]),
    "cable_fan": (_cable_demo("fan"), [("a", None, -35, 28, None, 70, None)]),
    "riser": (_riser_demo, [("a", None, -40, 28, None, 70, None), ("b", None, 140, 18, None, 70, None)]),
}


def render_part(name, out_dir, res, spp, views=None, L=None, G=None):
    sc = reset()
    setup_render(sc, res, spp)
    build, vlist = PARTS[name]
    t0 = time.time()
    root = build(L, G)
    lo, hi = world_bbox(root)
    if name in ("tube", "fittings"):
        for ob in bpy.data.objects:
            if ob.type == "MESH":
                for c in ob.bound_box:
                    p = ob.matrix_world @ Vector(c)
                    lo = Vector((min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z)))
                    hi = Vector((max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z)))
    cen = (lo + hi) / 2
    R = (hi - lo).length / 2
    studio_world(sc, 0.55)
    backdrop(sc, lo.z - 0.02, R, cen)
    area_light(sc, "key", cen, (-0.55, -0.55, 0.75), 4.0 * R, 2.4 * R, 3.2, (1.0, 0.97, 0.93))
    area_light(sc, "rim", cen, (0.65, 0.55, 0.45), 4.5 * R, 1.6 * R, 3.0, (0.93, 0.96, 1.0))
    print("[test_parts] %s built in %.1fs, bbox %s .. %s" % (name, time.time() - t0, tuple(round(v, 1) for v in lo), tuple(round(v, 1) for v in hi)))
    flipped = False
    for (suf, tgt, az, el, dist, lens, fstop) in vlist:
        if views and suf not in views:
            continue
        if suf.startswith("back") and not flipped:          # show the solder side: turn the part over
            flipped = True
            root.rotation_euler = (0.0, math.pi, 0.0)
            root.location = (lo.x + hi.x, 0.0, lo.z + hi.z)
        target = cen if tgt is None else Vector(tgt)
        if flipped and tgt is not None:
            target = Vector((lo.x + hi.x - tgt[0], tgt[1], lo.z + hi.z - tgt[2]))
        if dist is None:
            dist = R / math.tan(math.atan(18.0 * 9 / 16 / lens)) * 0.80
        for ob in [o for o in bpy.data.objects if o.name.startswith("pcp_cam")]:
            bpy.data.objects.remove(ob)
        camera(sc, target, az, el, dist, lens, fstop)
        sc.render.filepath = os.path.join(out_dir, "%s_%s.png" % (name, suf))
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print("[test_parts] rendered %s (%.0fs)" % (sc.render.filepath, time.time() - t1))


def contact_sheet(out_dir, cols=4, thumb=(480, 270)):
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".png") and not f.startswith("contact"))
    if not files:
        return
    rows = (len(files) + cols - 1) // cols
    tw, th = thumb
    sheet = np.zeros((rows * th, cols * tw, 4), np.float32)
    sheet[..., 3] = 1.0
    for i, f in enumerate(files):
        img = bpy.data.images.load(os.path.join(out_dir, f))
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        px = px.reshape(h, w, 4)
        ys = (np.arange(th) * h / th).astype(int)
        xs = (np.arange(tw) * w / tw).astype(int)
        t = px[ys][:, xs]
        r, c = i // cols, i % cols
        y0 = (rows - 1 - r) * th
        sheet[y0:y0 + th, c * tw:(c + 1) * tw] = t
        bpy.data.images.remove(img)
    out = bpy.data.images.new("contact", cols * tw, rows * th, alpha=False)
    out.pixels.foreach_set(sheet.ravel())
    out.filepath_raw = os.path.join(out_dir, "contact_sheet.png")
    out.file_format = "PNG"
    out.save()
    print("[test_parts] contact sheet:", out.filepath_raw, "(%d images)" % len(files))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opt = {"only": None, "views": None, "res": "960x540", "spp": "48", "out": os.path.join(HERE, "out", "parts")}
    i = 0
    sheet_only = False
    while i < len(argv):
        if argv[i] == "--sheet-only":
            sheet_only = True
            i += 1
            continue
        opt[argv[i].lstrip("-")] = argv[i + 1]
        i += 2
    res = tuple(int(v) for v in opt["res"].split("x"))
    os.makedirs(opt["out"], exist_ok=True)
    L, G = pc_parts.load_layout()
    if not sheet_only:
        names = opt["only"].split(",") if opt["only"] else list(PARTS)
        views = opt["views"].split(",") if opt["views"] else None
        for n in names:
            render_part(n, opt["out"], res, int(opt["spp"]), views, L, G)
    contact_sheet(opt["out"])


if __name__ == "__main__":
    main()
