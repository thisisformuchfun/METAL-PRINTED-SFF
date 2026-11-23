"""Export the assembled scene for the web viewer (run inside Blender on build/marrow_web.blend):

    blender -b build/marrow_web.blend --python render/export_web.py

Writes build/web_parts_raw.glb (printed parts + loop, with simulated-temperature vertex colours on the
two cold plates) and build/web_host_raw.glb (off-the-shelf hardware). gltfpack then compresses them into
site/model/ (see tools/build_site.sh).
"""
import pathlib
import re

import bpy
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
BUILD = REPO / "build"
root = bpy.data.objects["MARROW_ROOT"]
groups = {}
for n in ("steel", "alu", "loop", "host"):
    e = bpy.data.objects.new(n, None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = root
    groups[n] = e

PRINTED = {"gpu_block": "alu", "monoblock": "alu", "m2_sink": "alu", "heart": "alu"}
for o in list(root.children):
    if o.name in groups:
        continue
    if o.name.startswith("frame_"):
        o.parent = groups["steel"]
    elif o.name in PRINTED:
        o.parent = groups["alu"]
    elif o.name.startswith("fit_") or re.match(r"T\d_", o.name):
        o.parent = groups["loop"]
    else:
        o.parent = groups["host"]

# simulated temperature -> vertex colours (linear RGB) on the two cold plates
for name in ("gpu_block", "monoblock"):
    p = BUILD / f"{name}_thermal.npz"
    ob = bpy.data.objects.get(name)
    if ob is None or not p.exists():
        print("no thermal colours for", name)
        continue
    col = np.load(p)["colors"].astype(np.float32) / 255.0
    me = ob.data
    if len(me.vertices) != len(col):
        print("vertex count mismatch", name, len(me.vertices), len(col))
        continue
    lin = np.where(col <= 0.04045, col / 12.92, ((col + 0.055) / 1.055) ** 2.4)
    rgba = np.concatenate([lin, np.ones((len(lin), 1), np.float32)], axis=1)
    att = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    att.data.foreach_set("color", rgba.ravel())
    me.color_attributes.active_color = att
    me.color_attributes.render_color_index = 0
    print("thermal colours ->", name, len(col))


def select_tree(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
        for c in o.children_recursive:
            c.hide_set(False)
            c.select_set(True)


def export(path, objs):
    select_tree(objs)
    kw = dict(filepath=str(path), export_format="GLB", use_selection=True, export_yup=True, export_cameras=False,
              export_lights=False, export_apply=False, export_animations=False)
    for extra in (dict(export_vertex_color="ACTIVE", export_all_vertex_colors=False, export_active_vertex_color_when_no_material=True),
                  dict(export_vertex_color="ACTIVE"), dict()):
        try:
            bpy.ops.export_scene.gltf(**kw, **extra)
            break
        except TypeError as ex:
            print("retrying export without", list(extra), ex)
    print("EXPORTED", path, round(path.stat().st_size / 1e6, 1), "MB")


export(BUILD / "web_parts_raw.glb", [root, groups["steel"], groups["alu"], groups["loop"]])
export(BUILD / "web_host_raw.glb", [root, groups["host"]])
