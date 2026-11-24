import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import marrow_scene as ms
from mathutils import Vector
argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
lod = argv[1] if len(argv) > 1 else "web"
views = argv[2].split(";") if len(argv) > 2 else ["-28,14,900"]
comps = (argv[3] == "1") if len(argv) > 3 else False
ms.build(lod=lod, components=comps)
ms.set_world(strength=0.5, rot_deg=40)
ms.light_rig()
cam = ms.camera(lens=55)
ms.render_settings(res=(1100, 1100), spp=40, threads=4)
for i, v in enumerate(views):
    az, el, dist = [float(t) for t in v.split(",")]
    ms.aim(cam, ms.orbit(az, el, dist), ms.CENTER)
    ms.render_still(out.replace(".png", f"_{i}.png"))
