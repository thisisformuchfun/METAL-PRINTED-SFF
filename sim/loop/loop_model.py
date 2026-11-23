"""MARROW loop model: one pump, one loop - hydraulic operating point and steady
temperatures of every heat source, for a sweep of fan speeds and two cold-plate metals.

    /opt/venv/bin/python -m sim.loop.loop_model

Inputs that come from our own simulations (read from sim/out/<case>/summary.json):
    cold-plate pressure drop and thermal resistance (conjugate LBM/FV runs)
Inputs that are engineering assumptions (flagged [ASM]; every one is in ASSUME below):
    pump curve, radiator air-side performance, TIM and package resistances, loads
Output: sim/out/loop/loop.json (+ a table on stdout)
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from design import params as P

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTD = ROOT / "sim" / "out"

ASSUME = dict(
    coolant="30 % ethylene-glycol / water with aluminium-safe inhibitors",
    rho=1035.0, cp=3750.0,                         # kg/m3, J/kgK at ~40 C
    # DDC-class 12 V pump at full speed: shut-off head 5.2 m, free flow 15 L/min  [ASM]
    pump_p0_kpa=5.2 * 9.81 * 1.035, pump_qmax_lpm=15.0, pump_exp=1.6, pump_heat_w=14.0,
    # restrictions other than our cold plates, as k in dp = k Q^2 (kPa, L/min)       [ASM]
    k_radiator=1.1 / 3.785 ** 2,                   # ~1.1 kPa at 1 gpm, per slim 240
    k_fittings=0.16 / 3.785 ** 2,                  # per G1/4 fitting pair + tube run
    n_fittings=6, k_heart=0.6 / 3.785 ** 2,
    # radiator air side: slim 27 mm aluminium 240 with two 120 mm fans               [ASM]
    rad_wk_at_1500=17.0, rad_rpm_exp=0.72,         # W/K per radiator at 1500 rpm, scaling exponent
    rad_uncertainty=0.20,
    # interface / package resistances                                                  [ASM]
    gpu_tim_k_per_w=0.08e-4 / (23.4e-3 * 26.4e-3),  # 0.08 cm2K/W paste on the bare die
    gpu_hotspot_k=4.0,                             # die-surface mean -> reported hot-spot offset
    cpu_tim_k_per_w=0.020, cpu_package_k_per_w=0.30,   # IHS paste; junction->IHS of a stacked X3D die
    ambient_c=25.0,
)

LOADS = {
    "gaming": dict(label="4K gaming", gpu=380.0, cpu=75.0, vrm=12.0),
    "full": dict(label="Synthetic full load", gpu=450.0, cpu=120.0, vrm=22.0),
}


def read_case(name):
    p = OUTD / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def block_curves():
    """Fit dp = k Q^n and R(Q) for each cold plate from whatever CHT runs exist."""
    out = {}
    for blk in ("gpu_block", "monoblock"):
        runs = []
        for d in sorted(OUTD.glob(f"{blk}_*")):
            s = read_case(d.name)
            if s:
                runs.append((d.name, s))
        out[blk] = runs
    return out


def fit_power(qs, ys, default_n):
    qs, ys = np.asarray(qs, float), np.asarray(ys, float)
    if len(qs) >= 2 and len(set(np.round(qs, 3))) >= 2:
        n, lk = np.polyfit(np.log(qs), np.log(ys), 1)
        return float(math.exp(lk)), float(n)
    return float(ys[0] / qs[0] ** default_n), float(default_n)


def pump_dp(q, speed=1.0):
    a = ASSUME
    qm = a["pump_qmax_lpm"] * speed
    return a["pump_p0_kpa"] * speed ** 2 * max(0.0, 1.0 - (q / qm) ** a["pump_exp"])


def operating_point(sys_dp, speed=1.0):
    lo, hi = 0.01, ASSUME["pump_qmax_lpm"] * speed
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pump_dp(mid, speed) > sys_dp(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rad_wk(rpm):
    return ASSUME["rad_wk_at_1500"] * (rpm / 1500.0) ** ASSUME["rad_rpm_exp"]


def solve(material, blocks, load, rpm, pump_speed=1.0, rad_scale=1.0):
    a = ASSUME
    g, m = blocks["gpu_block"][material], blocks["monoblock"][material]

    def sys_dp(q):
        return (g["k"] * q ** g["n"] + m["k"] * q ** m["n"] + 2 * a["k_radiator"] * q ** 2
                + a["n_fittings"] * a["k_fittings"] * q ** 2 + a["k_heart"] * q ** 2)

    q = operating_point(sys_dp, pump_speed)
    mdot = q / 60000.0 * a["rho"]
    C = mdot * a["cp"]
    p_tot = load["gpu"] + load["cpu"] + load["vrm"] + a["pump_heat_w"] * pump_speed ** 3
    # two radiators in series, each an exchanger against ambient air with capacity UA_eff = W/K
    wk = rad_wk(rpm) * rad_scale
    eff = 1.0 - math.exp(-wk / C)                # water-side effectiveness of one radiator
    # steady state: T after pump (= GPU inlet) is T0; walk the loop
    # unknown T0; heat rejected must equal p_tot -> solve linear relation
    def loop(T0):
        T = T0
        Tg_in = T
        T += load["gpu"] / C
        Tm_in = T
        T += (load["cpu"] + load["vrm"]) / C
        Tr_in = T
        for _ in range(2):
            T -= eff * (T - a["ambient_c"])
        T += a["pump_heat_w"] * pump_speed ** 3 / C
        return T, Tg_in, Tm_in, Tr_in
    lo, hi = a["ambient_c"], a["ambient_c"] + 120.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if loop(mid)[0] > mid:
            lo = mid
        else:
            hi = mid
    T0 = 0.5 * (lo + hi)
    _, Tg_in, Tm_in, Tr_in = loop(T0)
    Rg = g["R_k"] * q ** g["R_n"]
    Rc = m["R_k"] * q ** m["R_n"]
    p_die = load["gpu"] * g["die_share"]
    gpu_die = Tg_in + Rg * p_die + a["gpu_tim_k_per_w"] * p_die
    gpu_hot = gpu_die + a["gpu_hotspot_k"]
    cpu_ihs = Tm_in + Rc * load["cpu"]
    cpu_tj = cpu_ihs + (a["cpu_tim_k_per_w"] + a["cpu_package_k_per_w"]) * load["cpu"]
    return dict(material=material, rpm=rpm, flow_lpm=round(q, 2), loop_dp_kpa=round(sys_dp(q), 1),
                dp_gpu_kpa=round(g["k"] * q ** g["n"], 2), dp_mono_kpa=round(m["k"] * q ** m["n"], 2),
                heat_w=round(p_tot, 0), coolant_gpu_in_c=round(Tg_in, 1), coolant_max_c=round(Tr_in, 1),
                coolant_rise_k=round(Tr_in - Tg_in, 1), delta_coolant_ambient_k=round(Tr_in - a["ambient_c"], 1),
                R_gpu_k_per_w=round(Rg, 4), R_cpu_k_per_w=round(Rc, 4),
                gpu_die_c=round(gpu_die, 1), gpu_hotspot_c=round(gpu_hot, 1),
                cpu_ihs_c=round(cpu_ihs, 1), cpu_tj_c=round(cpu_tj, 1))


def gather_blocks():
    """Reduce CHT summaries to {block: {material: {k, n, R_k, R_n, ...}}}."""
    raw = block_curves()
    out = {}
    for blk, runs in raw.items():
        per = {}
        for name, s in runs:
            mat = "316L" if "316" in name else "AlSi10Mg"
            per.setdefault(mat, []).append(s)
        res = {}
        for mat, lst in per.items():
            qs = [s["flow_lpm"] for s in lst]
            dps = [s["dp_Pa"] / 1000.0 for s in lst]
            h1 = [s["heaters"]["1"] for s in lst]
            Rs = [h["R_K_per_W"] for h in h1]
            k, n = fit_power(qs, dps, 1.8)
            Rk, Rn = fit_power(qs, Rs, -0.25)
            share = h1[0]["power_W"] / lst[0]["energy_in_W"] if blk == "gpu_block" else 1.0
            res[mat] = dict(k=k, n=n, R_k=Rk, R_n=Rn, heater_w=h1[0]["power_W"], die_share=share,
                            runs=[dict(flow_lpm=q_, dp_kpa=d_, R=r_) for q_, d_, r_ in zip(qs, dps, Rs)])
        # hydraulics do not depend on the metal: a metal simulated at one flow rate borrows the exponents
        ref = res.get("AlSi10Mg")
        for mat, v in res.items():
            if ref and mat != "AlSi10Mg" and len(v["runs"]) < 2:
                v["k"], v["n"], v["R_n"] = ref["k"], ref["n"], ref["R_n"]
                v["R_k"] = v["runs"][0]["R"] / v["runs"][0]["flow_lpm"] ** ref["R_n"]
        out[blk] = res
    return out


def main():
    blocks = gather_blocks()
    missing = [b for b in ("gpu_block", "monoblock") if not blocks.get(b)]
    if missing:
        print("No CHT results yet for:", missing, "- run the cold-plate simulations first.")
        return None
    res = dict(assumptions=ASSUME, loads=LOADS, blocks=blocks, sweep=[])
    mats = [m for m in ("AlSi10Mg", "316L") if all(m in blocks[b] for b in blocks)]
    for mat in mats:
        for lk, load in LOADS.items():
            for rpm in (800, 1000, 1200, 1500, 1800, 2000):
                r = solve(mat, blocks, load, rpm)
                r["load"] = lk
                r_lo = solve(mat, blocks, load, rpm, rad_scale=1 - ASSUME["rad_uncertainty"])
                r_hi = solve(mat, blocks, load, rpm, rad_scale=1 + ASSUME["rad_uncertainty"])
                r["gpu_hotspot_band_c"] = [r_hi["gpu_hotspot_c"], r_lo["gpu_hotspot_c"]]
                r["cpu_tj_band_c"] = [r_hi["cpu_tj_c"], r_lo["cpu_tj_c"]]
                res["sweep"].append(r)
    (OUTD / "loop").mkdir(parents=True, exist_ok=True)
    (OUTD / "loop" / "loop.json").write_text(json.dumps(res, indent=1))
    hdr = "material   load    rpm  L/min  dp_kPa  coolant_C  GPU_hot  CPU_Tj"
    print(hdr)
    for r in res["sweep"]:
        print(f"{r['material']:9s} {r['load']:7s} {r['rpm']:4d}  {r['flow_lpm']:5.2f}  {r['loop_dp_kpa']:6.1f}  "
              f"{r['coolant_max_c']:9.1f}  {r['gpu_hotspot_c']:7.1f}  {r['cpu_tj_c']:6.1f}")
    return res


if __name__ == "__main__":
    main()
