"""Validation of the two kernels used for the unit-cell simulations.

  V1  body-force Poiseuille flow between plates  -> mean velocity vs g H^2 / (12 nu)
  V2  heat transfer between isothermal plates    -> Nusselt number vs 7.541 (fully developed)
                                                    and vs the Hausen-type mean-Nu correlation (developing)
    /opt/venv/bin/python -m sim.validation.validate_cell
"""
import json
import pathlib

import numpy as np

from sim.cell import unit_cell as uc

OUT = pathlib.Path(__file__).resolve().parents[2] / "sim" / "out" / "validation"


def plates(n, nxy=4):
    s = np.zeros((nxy, nxy, n + 4), dtype=bool)
    s[:, :, :2] = True
    s[:, :, -2:] = True
    return s


def v1():
    rows = []
    for n in (8, 13, 16, 24):
        for cs in (0.0, 0.10):
            dx = 1.625e-3 / n
            H = n * dx
            dpdx = 400.0
            U_an = dpdx / uc.FLUID["rho"] * H * H / (12 * uc.FLUID["nu"])
            um, r = uc.run_flow(plates(n), dx, dpdx, u_peak_guess=1.5 * U_an, cs=cs, verbose=False)
            U = float(um[0][:, :, 2:-2].mean())
            rows.append(dict(cells_across=n, smagorinsky=cs, U_lbm=U, U_analytic=U_an, error_pct=100 * (U / U_an - 1),
                             Re_Dh=U_an * 2 * H / uc.FLUID["nu"]))
            print("V1", rows[-1], flush=True)
    return rows


def v2():
    rows = []
    for n in (8, 13, 16):
        for U in (1.0e-3, 0.02, 0.10):
            dx = 1.625e-3 / n
            H = n * dx
            nx = 40 if U < 0.01 else 160
            z = (np.arange(n) + 0.5) * dx
            prof = 6 * U * (z / H) * (1 - z / H)
            solid = plates(n)
            solid = np.repeat(solid[:1], nx, axis=0)
            um = np.zeros((3,) + solid.shape)
            um[0][:, :, 2:-2] = prof[None, None, :]
            h = uc.run_heat(um, solid, dx, n_cells=5, verbose=False)
            # h*a here is per volume of the padded box; convert to the channel: a = 2/H
            ha = h["ha_W_m3K"] / (n / (n + 4))
            Nu = ha * H * H / uc.FLUID["k"]
            Dh = 2 * H
            Pe = U * Dh / uc.FLUID["alpha"]
            Lfit = 4 * nx * dx
            gz = Dh / Lfit * Pe
            Nu_ref = 7.541 + 0.03 * gz / (1 + 0.016 * gz ** (2.0 / 3.0))
            rows.append(dict(cells_across=n, U=U, Pe=Pe, graetz=gz, Nu=Nu, Nu_ref=Nu_ref, error_pct=100 * (Nu / Nu_ref - 1)))
            print("V2", rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(poiseuille=v1(), plates_heat=v2())
    (OUT / "validate_cell.json").write_text(json.dumps(res, indent=1))
