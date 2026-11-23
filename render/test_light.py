import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import marrow_scene as ms
import bpy
argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
ms.build(lod="web", components=False)
ms.set_world(strength=0.25, rot_deg=40)
lights = ms.light_rig()
base = [l.data.energy for l in lights]
cam = ms.camera(lens=60)
ms.render_settings(res=(640, 720), spp=24, threads=4)
ms.aim(cam, ms.orbit(-30, 12, 900), ms.CENTER)
for sc in [float(v) for v in argv[1].split(",")]:
    for l, b in zip(lights, base):
        l.data.energy = b * sc
    ms.render_still(out.replace(".png", f"_{sc}.png"))
