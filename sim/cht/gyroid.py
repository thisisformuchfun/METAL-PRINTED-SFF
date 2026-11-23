"""Sheet-gyroid (TPMS) helpers used by the validation samples and the example case."""
from __future__ import annotations

import numpy as np

#: area of the gyroid minimal surface per unit cell, in units of cell^2
GYROID_AREA = 3.0915


def gyroid_field(shape, dx, cell, origin=(0.0, 0.0, 0.0)):
    """g = sin X cos Y + sin Y cos Z + sin Z cos X at the cell centres, X = 2 pi x / cell."""
    ax = [2.0 * np.pi * ((np.arange(n) + 0.5) * dx + o) / cell for n, o in zip(shape, origin)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij", sparse=True)
    return (np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)).astype(np.float32)


def sheet_threshold(wall, cell, n=96):
    """Level t such that the sheet |g| < t has mean thickness ``wall``.

    Mean thickness is defined as solid volume / mid-surface area, i.e. the
    solid fraction equals wall * 3.0915 / cell.
    """
    target = wall * GYROID_AREA / cell
    if not 0.0 < target < 1.0:
        raise ValueError("wall / cell out of range for a sheet gyroid")
    g = np.abs(gyroid_field((n, n, n), 1.0 / n, 1.0)).ravel()
    return float(np.quantile(g, target))


def sheet_gyroid(shape, dx, cell, wall, origin=(0.0, 0.0, 0.0)):
    """Boolean solid mask of the sheet gyroid |g| < t(wall)."""
    return np.abs(gyroid_field(shape, dx, cell, origin)) < sheet_threshold(wall, cell)
