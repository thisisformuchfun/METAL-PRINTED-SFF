"""Named camera set-ups and render entry points for the MARROW film and stills.

    blender -b build/marrow_hero.blend --python render/shots.py -- still <shot> [--res WxH] [--spp N] [--out dir]
    blender -b build/marrow_hero.blend --python render/shots.py -- orbit <name> --frames 0:59 [...]
    blender -b build/marrow_hero.blend --python render/shots.py -- list

Every still also writes a 16-bit depth map (<shot>_z.png, metres mapped to 0..1 over [near, far]) so
the editor can add true parallax drift to a single rendered frame.
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bpy
from mathutils import Vector

import marrow_scene as ms

C = ms.CENTER


def S(x, s, n=0.0):
    return ms.show(x, s, n)


# name: dict(loc, target (mm), lens, fstop, lights=dict(scale per light), hide=[collections/objects], world)
SHOTS = {
    # --- wide heroes
    "hero_front": dict(loc=ms.orbit(-27, 12, 1260), tgt=C + Vector((0, -10, -4)), lens=50, fstop=10.0),
    "hero_low": dict(loc=ms.orbit(-40, 2, 1150), tgt=C + Vector((-6, 0, 6)), lens=45, fstop=9.0),
    "hero_right": dict(loc=ms.orbit(34, 15, 1260), tgt=C + Vector((8, -10, -4)), lens=50, fstop=10.0),
    "rear_ribs": dict(loc=ms.orbit(150, 11, 1220), tgt=C + Vector((0, 30, 4)), lens=50, fstop=10.0),
    "side_profile": dict(loc=ms.orbit(-90, 3, 1500), tgt=C + Vector((0, 0, 0)), lens=65, fstop=11.0),
    "top_down": dict(loc=ms.orbit(-14, 56, 1250), tgt=C + Vector((0, -20, 0)), lens=50, fstop=10.0),
    # --- medium
    "board": dict(loc=S(-190, 340, 420), tgt=S(-48, 262, 10), lens=70, fstop=6.3),
    "gpu": dict(loc=S(200, 10, 400), tgt=S(-10, 88, 10), lens=70, fstop=6.3),
    "heart": dict(loc=S(300, 320, 260), tgt=S(90, 250, 22), lens=85, fstop=5.0),
    # --- macro
    "macro_skin": dict(loc=S(-205, 6, 96), tgt=S(-84, 70, 14), lens=90, fstop=5.6),
    "macro_ports": dict(loc=S(-190, 170, 190), tgt=S(-52, 262, 32), lens=90, fstop=6.3),
    "macro_knuckle": dict(loc=Vector((-170, 420, 300)), tgt=Vector((0, 292, 272)), lens=100, fstop=3.5),
    "macro_tube": dict(loc=S(120, 30, 190), tgt=S(-34, 86, 46), lens=90, fstop=5.6),
}


def apply_shot(name, cam=None):
    sh = SHOTS[name]
    cam = cam or bpy.context.scene.camera
    cam.data.lens = sh["lens"]
    ms.aim(cam, sh["loc"], sh["tgt"], fstop=sh.get("fstop"))
    return sh


def setup_depth_output(path_stem, near, far):
    """Compositor: write normalised depth alongside the beauty frame."""
    sc = bpy.context.scene
    sc.view_layers[0].use_pass_z = True
    sc.use_nodes = True
    nt = sc.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
    mr = nt.nodes.new("CompositorNodeMapRange")
    mr.inputs["From Min"].default_value = near
    mr.inputs["From Max"].default_value = far
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 1.0
    mr.use_clamp = True
    nt.links.new(rl.outputs["Depth"], mr.inputs["Value"])
    fo = nt.nodes.new("CompositorNodeOutputFile")
    fo.base_path = str(pathlib.Path(path_stem).parent)
    fo.format.file_format = "PNG"
    fo.format.color_mode = "BW"
    fo.format.color_depth = "16"
    fo.format.color_management = "OVERRIDE"
    fo.format.view_settings.view_transform = "Raw"
    fo.file_slots[0].path = pathlib.Path(path_stem).name + "_z_"
    nt.links.new(mr.outputs["Value"], fo.inputs[0])
    return fo


def show_only(prefixes):
    """Hide everything under the root except objects whose name starts with one of prefixes (None = show all)."""
    root = bpy.data.objects["MARROW_ROOT"]
    for o in root.children_recursive:
        top = o
        while top.parent is not None and top.parent != root:
            top = top.parent
        vis = prefixes is None or any(top.name.startswith(p) for p in prefixes)
        o.hide_render = not vis


VARIANTS = {"frame_only": ("hero_front", ("frame_",))}


def render_still(name, out_dir, res=(1920, 1080), spp=96, depth=True, threads=4, exposure=0.0):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = name
    if name in VARIANTS:
        name, only = VARIANTS[name]
        show_only(only)
    else:
        show_only(None)
    sc = ms.render_settings(res=res, spp=spp, threads=threads, exposure=exposure)
    sh = apply_shot(name)
    cam = sc.camera
    d = (Vector(sh["tgt"]) - Vector(sh["loc"])).length * ms.MM
    near, far = max(d * 0.35, 0.02), d * 2.6
    if depth:
        setup_depth_output(out_dir / out_name, near, far)
    sc.frame_set(1)
    ms.render_still(out_dir / f"{out_name}.png")
    (out_dir / f"{out_name}.json").write_text(__import__("json").dumps(dict(
        near=near, far=far, focus=d, lens=sh["lens"], sensor=36.0, res=list(res))))


def render_orbit(name, out_dir, frames, total, res=(960, 540), spp=40, az0=-62.0, az1=38.0, el=11.0, dist=960.0,
                 lens=55.0, threads=4, tgt=None):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = ms.render_settings(res=res, spp=spp, threads=threads)
    sc.use_nodes = False
    cam = sc.camera
    cam.data.lens = lens
    tgt = C + Vector((0, -6, -4)) if tgt is None else tgt
    for f in frames:
        t = f / max(total - 1, 1)
        e = t * t * (3 - 2 * t)                       # ease in/out
        az = az0 + (az1 - az0) * e
        ms.aim(cam, ms.orbit(az, el + 3.0 * math.sin(math.pi * t), dist - 60.0 * math.sin(math.pi * t), tgt), tgt, fstop=8.0)
        ms.render_still(out_dir / f"{name}_{f:04d}.png")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    mode = argv[0]
    opts = dict(res="1920x1080", spp="96", out=str(ms.BUILD / "stills"), frames="0:0", total="60", threads="4", exposure="0")
    names = []
    i = 1
    while i < len(argv):
        if argv[i].startswith("--"):
            opts[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            names.append(argv[i])
            i += 1
    res = tuple(int(v) for v in opts["res"].split("x"))
    if mode == "list":
        print("SHOTS", " ".join(SHOTS))
    elif mode == "still":
        for n in names:
            render_still(n, opts["out"], res=res, spp=int(opts["spp"]), threads=int(opts["threads"]), exposure=float(opts["exposure"]))
    elif mode == "orbit":
        a, b = [int(v) for v in opts["frames"].split(":")]
        render_orbit(names[0] if names else "orbit", opts["out"], range(a, b + 1), int(opts["total"]), res=res, spp=int(opts["spp"]),
                     threads=int(opts["threads"]))


if __name__ == "__main__":
    main()
