"""Assemble the master scene once and save it:  blender -b --python render/build_blend.py -- [hero|web]"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import marrow_scene as ms
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
lod = argv[0] if argv else "hero"
t0 = time.time()
ms.build(lod=lod, components=True)
ms.set_world(strength=0.25, rot_deg=40)
ms.light_rig()
ms.camera(lens=60)
ms.render_settings()
out = ms.BUILD / f"marrow_{lod}.blend"
ms.save_blend(out)
print("BUILD_TIME", round(time.time() - t0, 1))
