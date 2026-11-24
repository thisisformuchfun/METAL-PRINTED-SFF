"""Quick look-dev render of one or more meshes (run inside Blender).

blender -b --python design/preview.py -- out.png [--view az,el] [--res 1000x700] [--spp 48]
        [--mat alu|steel|clay] [--dist 1.0] mesh1.ply[:mat] mesh2.stl ...
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
opts = dict(view="35,28", res="1000x700", spp="40", mat="alu", dist="1.0", up="z", threads="4")
files = []
i = 1
while i < len(argv):
    a = argv[i]
    if a.startswith("--"):
        opts[a[2:]] = argv[i + 1]
        i += 2
    else:
        files.append(a)
        i += 1

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def metal(name, base, rough, metallic=1.0, bump=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    if bump > 0:
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 900.0
        tex.inputs["Detail"].default_value = 3.0
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = bump
        bp.inputs["Distance"].default_value = 0.05
        nt.links.new(tex.outputs["Fac"], bp.inputs["Height"])
        nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


MATS = dict(
    alu=metal("alu", (0.78, 0.79, 0.80), 0.42, bump=0.25),
    steel=metal("steel", (0.56, 0.57, 0.58), 0.30, bump=0.15),
    clay=metal("clay", (0.75, 0.74, 0.72), 0.6, metallic=0.0),
    dark=metal("dark", (0.03, 0.03, 0.035), 0.5, metallic=0.0),
    red=metal("red", (0.6, 0.03, 0.03), 0.3, metallic=0.0),
)

objs = []
for spec in files:
    path, _, mname = spec.partition(":")
    if path.lower().endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path)
    else:
        bpy.ops.wm.stl_import(filepath=path)
    o = bpy.context.selected_objects[0]
    o.data.materials.append(MATS.get(mname or opts["mat"], MATS["alu"]))
    for p in o.data.polygons:
        p.use_smooth = True
    objs.append(o)

lo = Vector((1e9, 1e9, 1e9))
hi = Vector((-1e9, -1e9, -1e9))
for o in objs:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        lo = Vector(map(min, lo, w))
        hi = Vector(map(max, hi, w))
ctr = (lo + hi) / 2
size = (hi - lo).length

# ground
bpy.ops.mesh.primitive_plane_add(size=size * 30, location=(ctr.x, ctr.y, lo.z - 0.02))
g = bpy.context.object
gm = metal("ground", (0.06, 0.06, 0.065), 0.35, metallic=0.0)
g.data.materials.append(gm)

az, el = [math.radians(float(v)) for v in opts["view"].split(",")]
d = size * 1.25 * float(opts["dist"])
cam_loc = ctr + Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * d
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
cam.location = cam_loc
cam.rotation_euler = (ctr - cam_loc).to_track_quat("-Z", "Y").to_euler()
cam.data.lens = 70
cam.data.clip_start = size * 0.01
cam.data.clip_end = size * 100
scene.camera = cam


def area(loc, energy, sz, col=(1, 1, 1)):
    L = bpy.data.lights.new("a", "AREA")
    L.energy = energy
    L.size = sz
    L.color = col
    o = bpy.data.objects.new("a", L)
    scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (ctr - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


e = size * size * 14.0
right = Vector((math.cos(az), math.sin(az), 0))
fwd = (ctr - cam_loc).normalized()
area(ctr - fwd * size * 0.9 + right * size * 0.9 + Vector((0, 0, size * 0.9)), e, size * 1.2)
area(ctr - fwd * size * 0.5 - right * size * 1.2 + Vector((0, 0, size * 0.4)), e * 0.35, size * 1.5, (0.8, 0.9, 1.0))
area(ctr + fwd * size * 1.0 + Vector((0, 0, size * 1.0)), e * 0.9, size * 0.8, (1.0, 0.93, 0.85))

w = bpy.data.worlds.new("w")
scene.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.055, 0.06, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 1.0

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = int(opts["spp"])
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 6
scene.render.threads_mode = "FIXED"
scene.render.threads = int(opts["threads"])
rx, ry = [int(v) for v in opts["res"].split("x")]
scene.render.resolution_x, scene.render.resolution_y = rx, ry
scene.view_settings.view_transform = "Filmic" if "Filmic" in [v.name for v in type(scene.view_settings).bl_rna.properties["view_transform"].enum_items] else "Standard"
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print("PREVIEW", out, "bbox", tuple(round(v, 1) for v in (hi - lo)))
