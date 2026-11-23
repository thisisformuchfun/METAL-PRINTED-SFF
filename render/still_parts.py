"""Stills of the printed parts on their own (run inside Blender, no .blend needed):

    blender -b --python render/still_parts.py -- organs|cutaway [--res WxH] [--spp N] [--out dir]
"""
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bpy
from mathutils import Matrix, Vector

import marrow_scene as ms
import shots

argv = sys.argv[sys.argv.index("--") + 1:]
mode = argv[0]
opts = dict(res="1600x900", spp="64", out=str(ms.BUILD / "stills"))
for i in range(1, len(argv) - 1, 2):
    opts[argv[i][2:]] = argv[i + 1]
res = tuple(int(v) for v in opts["res"].split("x"))
out = pathlib.Path(opts["out"])
out.mkdir(parents=True, exist_ok=True)

ms.reset()
mats = ms.materials()
coll = ms.collection("parts")


def bounds(o):
    pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def place_flat(name, path, centre, rot_z=0.0, tilt=0.0, lift=0.0):
    o = ms.import_mesh(path, name, Matrix.Identity(4), mats["alu"], coll)
    o.parent = None
    R = Matrix.Rotation(math.radians(rot_z), 4, "Z") @ Matrix.Rotation(math.radians(tilt), 4, "X")
    o.matrix_world = R
    bpy.context.view_layer.update()
    lo, hi = bounds(o)
    c = (lo + hi) / 2
    o.matrix_world = Matrix.Translation(Vector((centre[0] - c.x, centre[1] - c.y, lift - lo.z))) @ R
    ms.adopt(o, coll)
    return o


if mode == "organs":
    B = ms.BUILD
    place_flat("gpu_block", B / "gpu_block_hero.ply", (-62, 20), rot_z=4)
    place_flat("monoblock", B / "monoblock_hero.ply", (118, 42), rot_z=-8)
    place_flat("heart", B / "heart_hero.ply", (176, -78), rot_z=200)
    place_flat("m2_sink", B / "m2_sink_hero.ply", (20, -84), rot_z=-3)
    ms.CENTER = Vector((40.0, 0.0, 12.0))
    ms.add_floor()
    ms.set_world(strength=0.22, rot_deg=40)
    ms.light_rig(key=0.55, rim=0.75, fill=0.6, top=0.6)
    cam = ms.camera(lens=60)
    loc = ms.CENTER + Vector((-330, -640, 400))
    ms.aim(cam, loc, ms.CENTER + Vector((0, -8, 0)), fstop=8.0)
    name = "organs"
elif mode == "cutaway":
    o = ms.import_mesh(ms.BUILD / "gpu_block_cut_hero.ply", "gpu_cut", Matrix.Identity(4), mats["alu"], coll)
    bpy.context.view_layer.update()
    lo, hi = bounds(o)
    c = (lo + hi) / 2
    o.parent = None
    o.matrix_world = Matrix.Translation(Vector((-c.x, -lo.y, -lo.z)))
    ms.adopt(o, coll)
    ms.CENTER = Vector((0.0, 20.0, 12.0))
    ms.add_floor()
    ms.set_world(strength=0.22, rot_deg=10)
    ms.light_rig(key=0.5, rim=0.6, fill=0.7, top=0.5)
    cam = ms.camera(lens=85)
    ms.aim(cam, Vector((-150, -330, 150)), Vector((-4, 0, 11)), fstop=7.0)
    name = "cut_gpu"
else:
    raise SystemExit("unknown mode")

sc = ms.render_settings(res=res, spp=int(opts["spp"]))
d = (cam.location - ms.CENTER * ms.MM).length
shots.setup_depth_output(out / name, max(d * 0.35, 0.02), d * 2.6)
ms.render_still(out / f"{name}.png")
(out / f"{name}.json").write_text(json.dumps(dict(near=max(d * 0.35, 0.02), far=d * 2.6, focus=d, lens=cam.data.lens, sensor=36.0, res=list(res))))
