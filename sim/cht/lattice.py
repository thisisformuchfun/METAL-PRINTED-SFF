"""D3Q19 lattice constants (must match csrc/lbm.c)."""
import numpy as np

#: discrete velocities c_q, q = 0..18 (rest, 6 axial, 12 diagonal)
C = np.array(
    [[0, 0, 0],
     [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
     [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
     [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
     [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]], dtype=np.int64)
#: weights
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
#: index of the opposite velocity
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
CS2 = 1.0 / 3.0
NQ = 19

assert all((C[q] == -C[OPP[q]]).all() for q in range(NQ))
