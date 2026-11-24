"""Section the 4090 block through the die centre for cutaway renders:  python -m design.cutaway"""
import numpy as np

from . import gpu_block as GB
from . import part, sdf


def main(h=0.25):
    f, grid, info = GB.build(h=h, verbose=False)
    x, y, z = grid.coords()
    fc = np.maximum(f, (GB.CY - y).astype(np.float32))        # keep the half with y > die centre
    m = sdf.mesh(fc, grid)
    sdf.smooth(m, iters=1)
    m = sdf.decimate(m, 2_600_000)
    part.WORK.mkdir(exist_ok=True)
    m.export(part.WORK / "gpu_block_cut_hero.ply")
    print("cutaway mesh", len(m.faces), "faces")


if __name__ == "__main__":
    main()
