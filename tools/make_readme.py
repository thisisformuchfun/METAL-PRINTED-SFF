"""Regenerate README.md from the current design and simulation data:
    /opt/venv/bin/python tools/make_readme.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "site" / "data" / "site.json").read_text())
P, S = d["parts"], d["sims"]
ORDER = ["gpu_block", "monoblock", "heart", "m2_sink", "frame_pelvis", "frame_lumbar", "frame_thoracic", "frame_dorsal", "frame_cervical"]
NAMES = {
    "gpu_block": "Organ I - RTX 4090 full-cover block", "monoblock": "Organ II - CPU + VRM monoblock (AM5)",
    "heart": "Organ III - pump top + 41 mL reservoir", "m2_sink": "Organ IV - M.2 / chipset sink (clamped to the monoblock tail)",
    "frame_pelvis": "Pelvis - keel, feet, PSU cradle, riser shelf", "frame_lumbar": "Lumbar - lower front spine, GPU ribs",
    "frame_thoracic": "Thoracic - upper front spine, board ribs", "frame_dorsal": "Dorsal - lower rear spine, radiator ribs",
    "frame_cervical": "Cervical - upper rear spine, crown",
}


def money(v):
    return f"${v:,.0f}"


rows = []
tot = [0.0, 0.0, 0.0]
steel_g = 0.0
for k in ORDER:
    p = P[k]
    bb = " x ".join(str(int(v + 0.5)) for v in p["bbox_mm"])
    ok = "yes" if p["fits_envelope"] else "NO"
    wt = "yes" if p["print_stl"]["watertight"] else "NO"
    rows.append(f"| `{k}.3mf` | {NAMES[k]} | {p['material']} | {p['mass_g']:.0f} g | {bb} | {ok} | {wt} | {money(p['cost_usd_jlc_pcbway'][0])} | {money(p['cost_usd_jlc_pcbway'][1])} |")
    tot[0] += p["mass_g"]
    tot[1] += p["cost_usd_jlc_pcbway"][0]
    tot[2] += p["cost_usd_jlc_pcbway"][1]
    if p["material"] == "316L":
        steel_g += p["mass_g"]
rows.append(f"| | **All nine** | | **{tot[0]/1000:.2f} kg** | | | | **{money(tot[1])}** | **{money(tot[2])}** |")
parts_table = "\n".join(rows)

m = S.get("macro", {})
g, gs, c = m.get("gpu_block_AlSi10Mg_3"), m.get("gpu_block_316L_3"), m.get("monoblock_AlSi10Mg_3")
loop = S.get("loop", {}).get("sweep", [])


def lp(mat, load, rpm):
    r = [x for x in loop if x["material"] == mat and x["load"] == load and x["rpm"] == rpm]
    return r[0] if r else None


fea = S.get("fea")
fea_rows = ""
if fea:
    nm = ["Standing, 1 g", "Standing, 1 g down + 1 g sideways", "Lifted by the crown, 1 g"]
    fea_rows = "\n".join(f"| {nm[i]} | {cse['max_disp_mm']:.2f} mm | {cse['vm_max_smoothed_MPa']:.0f} MPa | {cse['safety_factor_yield']:.0f}x |" for i, cse in enumerate(fea["cases"]))

sim_rows = []
if g:
    h = g["heaters"]
    sim_rows.append(f"| 4090 block, AlSi10Mg | {g['flow_lpm']:.0f} L/min | {g['dp_Pa']/1000:.1f} kPa | die +{h['1']['T_mean_rise_K']:.1f} K (peak +{h['1']['T_max_rise_K']:.1f}), memory +{h['2']['T_mean_rise_K']:.1f} K, VRM +{h['3']['T_mean_rise_K']:.1f} K | {abs(g['energy_closure_pct']):.2f} % |")
if gs:
    h = gs["heaters"]
    sim_rows.append(f"| 4090 block, 316L (for comparison) | {gs['flow_lpm']:.0f} L/min | {gs['dp_Pa']/1000:.1f} kPa | die +{h['1']['T_mean_rise_K']:.1f} K, memory +{h['2']['T_mean_rise_K']:.1f} K, VRM +{h['3']['T_mean_rise_K']:.1f} K | {abs(gs['energy_closure_pct']):.2f} % |")
if c:
    h = c["heaters"]
    sim_rows.append(f"| CPU + VRM monoblock, AlSi10Mg | {c['flow_lpm']:.0f} L/min | {c['dp_Pa']/1000:.1f} kPa | CPU pad +{h['1']['T_mean_rise_K']:.1f} K, VRM +{h['2']['T_mean_rise_K']:.1f} K | {abs(c['energy_closure_pct']):.2f} % |")
sim_table = "\n".join(sim_rows)

loop_rows = []
for mat in ("AlSi10Mg", "316L"):
    for load in ("gaming", "full"):
        r = lp(mat, load, 1500)
        if r:
            loop_rows.append(f"| {mat} | {'4K gaming' if load == 'gaming' else 'Synthetic full load'} | {r['heat_w']:.0f} W | {r['flow_lpm']:.1f} L/min | {r['coolant_max_c']:.0f} C | {r['gpu_hotspot_c']:.0f} C ({r['gpu_hotspot_band_c'][0]:.0f}-{r['gpu_hotspot_band_c'][1]:.0f}) | {r['cpu_tj_c']:.0f} C |")
loop_table = "\n".join(loop_rows)

m2 = S.get("m2_sink")
m2_line = ""
if m2:
    mid = m2["cases"][1]
    m2_line = (f"**M.2 / chipset sink.** The first version of this part was passive, and the model failed it: with "
               f"{sum(m2['power_w'].values()):.0f} W on it in still air the SSD pad ran +{m2['cases'][2]['rise_ssd_K']:.0f} to "
               f"+{m2['cases'][0]['rise_ssd_K']:.0f} K over the room (film coefficient 10 to 5 W/m2K), so the lattice alone is good "
               f"for about {m2['power_for_45K_W']:.0f} W.")
    cp = m2.get("coupled")
    if cp:
        share = 100 * cp["q_to_loop_W"] / (cp["q_to_loop_W"] + cp["q_to_air_W"])
        m2_line += (f" The monoblock therefore has a solid conduction tail that clamps onto the sink through a {m2['assumptions']['pad_t_mm']} mm gap pad. "
                    f"With that, at full load the model puts the SSD controller at {cp['ssd_c']:.0f} C and the chipset at {cp['chipset_c']:.0f} C "
                    f"(7 W and 6 W continuous, coolant {cp['coolant_mono_in_c']:.0f} C), and {share:.0f} % of the heat leaves through the loop.")

cpu_line = ""
_r = lp("AlSi10Mg", "full", 1500)
_g = lp("AlSi10Mg", "gaming", 1500)
if _r and _g:
    _cool = [x for x in sorted(loop, key=lambda x: x["rpm"]) if x["material"] == "AlSi10Mg" and x["load"] == "full" and x["cpu_tj_c"] <= 89.0]
    _fan = f" until the fans reach {_cool[0]['rpm']} rpm ({_cool[0]['cpu_tj_c']:.1f} C)" if _cool else ""
    cpu_line = (f"**CPU margin.** At the synthetic full load (120 W into an X3D part, which no game does) the model puts the junction at "
                f"{_r['cpu_tj_c']:.1f} C with the fans at 1500 rpm. The limit of the Ryzen 7 X3D parts is 89 C, so the chip would hold temperature by "
                f"throttling{_fan}. Most of that rise is the chip's own junction-to-lid resistance (0.30 K/W assumed), not the block. "
                f"At a 75 W gaming load the junction is {_g['cpu_tj_c']:.0f} C.")

val = S.get("validation")
val_lines = ""
if val:
    p13 = [r for r in val["poiseuille"] if r["cells_across"] == 13 and r["smagorinsky"] == 0][0]
    h13 = [r for r in val["plates_heat"] if r["cells_across"] == 13]
    val_lines = (f"- Flow kernel vs analytic Poiseuille flow at the lattice's channel resolution (13 cells): {p13['error_pct']:+.2f} %.\n"
                 f"- Heat kernel vs plate Nusselt number (7.54 developed, Hausen-type while developing): {h13[0]['error_pct']:.1f} to {h13[-1]['error_pct']:.1f} % "
                 f"(under-predicts, the safe direction).\n"
                 f"- Block models conserve energy to within {max(abs(v['energy_closure_pct']) for v in m.values()):.2f} %.\n"
                 f"- Frame FEA reactions equal the applied weight ({-fea['cases'][0]['reaction_N'][2]:.1f} N).")

env = d["layout"]["envelope"]
r_full = lp("AlSi10Mg", "full", 1500)
headline = f"{r_full['gpu_hotspot_c']:.0f} C" if r_full else "n/a"

README = f"""# MARROW

**Printed bone. Liquid marrow.** An open small-form-factor chassis for a mini-ITX AM5 board and a liquid-cooled
RTX 4090: five 316L steel "bones", four AlSi10Mg "organs", one coolant loop. Every printed part fits a 300 mm
metal printer at JLC3DP or PCBWay.

![MARROW](site/media/hero.jpg)

> **Status: design study.** Nothing in this repository has been printed, assembled or tested. Every temperature,
> pressure and stress quoted here comes from the simulations in `sim/`. Read "Before you order" first.

- **Site (source):** `site/index.html` - serve the `site/` folder, or open the published link from the pull request.
- **Film:** `site/media/film.mp4` (80 s, original synthesised score).
- **Print files:** `exports/print/*.3mf` (nine parts).

## The build this was designed around

| Item | What was assumed |
|---|---|
| Graphics card | MSI GeForce RTX 4090 SUPRIM LIQUID 24G (PCB MS-V510), stock AIO removed |
| Motherboard | mini-ITX, AM5 (exact model not known - a representative layout is modelled) |
| CPU | Ryzen 7 X3D class, 120 W design load |
| Power supply | EVGA SuperNOVA NEX650G, ATX, bay sized 150 x 86 x 180 mm |
| Radiators | 2 x 240 mm slim (27 mm) **aluminium**, four 120 mm fans |
| Pump | DDC-class, on the printed top |

Envelope {env['width']:.0f} x {env['depth']:.0f} x {env['height']:.0f} mm (W x D x H). The board and the card share one show
plane leaning back 25 degrees; all external cables (board I/O, display outputs, mains) leave on the left.

## Printed parts

Prices are automated, pre-review quotes scaled by volume (JLC3DP / PCBWay, November 2026).

| File | Part | Metal | Mass | Bounding box (mm) | Fits 300 x 300 x 290 | Watertight | JLC3DP | PCBWay |
|---|---|---|---|---|---|---|---|---|
{parts_table}

Why two metals: neither service prints copper, and steel cold plates do not work on a 4090 (see below). So the
structure is 316L and everything that carries heat is aluminium.

## What the simulations say

**Cold plates** (two-scale conjugate model, `sim/cell` + `sim/macro`; temperatures are rises over the coolant
entering the block):

| Case | Flow | Pressure drop | Pad temperatures | Energy balance error |
|---|---|---|---|---|
{sim_table}

**Whole loop** at 1500 rpm fans in a 25 C room (`sim/loop`; band = +-20 % radiator performance):

| Cold-plate metal | Load | Heat | Flow | Coolant | 4090 hot spot | CPU junction |
|---|---|---|---|---|---|---|
{loop_table}

Headline: **{headline} simulated 4090 hot spot at full load** with aluminium blocks.

{cpu_line}

{m2_line}

**Frame** (`sim/fea`, {fea['elements']:,} hex elements, {fea['mass_components_kg']:.1f} kg of hardware on a {fea['mass_frame_kg']:.2f} kg frame):

| Load case | Peak deflection | Peak stress | Margin on 400 MPa yield |
|---|---|---|---|
{fea_rows}

**Checks**

{val_lines}

How it works: a patch of each lattice (with its real floor and ceiling) is resolved with a D3Q19 TRT lattice-Boltzmann
solver and a TVD finite-volume heat solver. That gives each lattice a Darcy-Forchheimer resistance, a volumetric
heat-transfer law and an effective metal conductivity. A 0.5 mm block-level model built from the same CAD fields as
the print files then solves porous flow plus a two-temperature energy equation with those laws. Expect about +-25 %
on the cold-plate temperature rises.

## Before you order

1. **Set the board dimensions.** The VRM layout under the monoblock is representative, not yours. Measure your board
   and edit `BOARD['vrm_left']`, `BOARD['vrm_top']`, `BOARD['socket']` and `MONOBLOCK['arm_*']` in `design/params.py`,
   then rerun the generators. The AM5 hole pattern (54 x 90 mm, UNC 6-32) and the mini-ITX holes are standard.
2. **Measure the 4090's component heights.** Hole positions, memory, VRM rows and outline were measured from PCB
   photographs (about +-0.3 mm) - `design/data/gpu_pcb_layout.json`. Heights of the die, inductors and capacitors are
   not published; the `z_*` values in `params.GPU` are assumptions. Also confirm your card says "MS-V510" on the PCB.
3. **Power supply.** MSI lists 850 W as the minimum for this card and recommends 1000 W. A 650 W unit is below that,
   and the NEX650G predates the 16-pin connector, so the card would run from the adapter. Treat the PSU as the first
   thing to upgrade.
4. **Keep the loop all-aluminium.** Aluminium blocks with copper or brass radiators corrode. Use aluminium radiators
   (the SUPRIM's own 240 mm radiator is aluminium), stainless or anodised fittings, and coolant with aluminium inhibitors.
5. **Post-process the prints.** Tap the G1/4 ports (11.8 mm pilots) and spot-face their seats, tap the M3/M4 pilots,
   lap the die / IHS / memory lands flat, then leak-test at 1 bar before filling. Thin printed aluminium often needs sealing.
6. **Tail joint.** The monoblock's tail lands on the M.2 sink with a 0.5 mm gap pad (6 W/mK assumed) and two M3 x 6
   screws into blind holes. Fit the sink first, then the monoblock; check the gap on your board before choosing the pad.
7. **Check the riser.** The card sits below the board on a 200 mm PCIe 4.0 riser that runs behind the PCB.

## Repository map

| Path | What it is |
|---|---|
| `design/params.py` | Single source of truth for every dimension; writes `design/layout.json` |
| `design/sdf.py`, `part.py`, `meshfix.py` | Implicit-modelling toolkit (gyroid, Voronoi, bones), meshing, watertight export |
| `design/gpu_block.py`, `monoblock.py`, `heart.py`, `m2_sink.py`, `frame.py` | Generators for the nine printed parts |
| `sim/cell/` | Resolved lattice flow and heat transfer (LBM + finite volume) |
| `sim/macro/` | Block-level conjugate model, M.2 sink model |
| `sim/loop/`, `sim/fea/` | Loop operating point and temperatures; frame FEA |
| `sim/validation/`, `sim/post/` | Kernel validation; figures and site data |
| `render/` | Procedural PC hardware, scene assembly, shots (Blender 4.5, Cycles) |
| `media/score/`, `media/film/` | Score synthesiser and its QA; film editor and simulation plates |
| `site/` | The website (static) |
| `exports/print/` | 3MF print files; `exports/meta/` per-part mass, size, cost, checks |

Rebuild everything: `tools/build_all.sh` (CAD, simulations, site data), then the render and film steps listed in it.

## Credits

Studio lighting: "Studio Small 09" from Poly Haven (CC0). Typefaces in the film and site: Bodoni Moda and IBM Plex
(SIL Open Font License). The score is original and synthesised by `media/score/score.py`. Product and company names
are used only to say what the parts fit.
"""
(ROOT / "README.md").write_text(README)
print("README.md written,", len(README), "chars")
