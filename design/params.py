"""MARROW - single source of truth for every dimension.

Units: millimetres, degrees, watts. Run `python -m design.params` to (re)write
design/layout.json, which the CAD generators, simulations, the Blender scene and
the website all read.

Frames
------
WORLD      X = viewer's right, Y = back (away from viewer), Z = up. Desk at z = 0.
SHOW       the tilted "show plane" carrying motherboard + GPU:
             e_x = (1,0,0), e_s = up the slope, e_n = out of the plane toward the viewer.
             world = SHOW_O + x*e_x + s*e_s + n*e_n
BOARD      mini-ITX PCB frame: origin at the corner shared by the rear-I/O edge and the
           PCIe-slot edge, +x along the PCIe edge away from the I/O, +y toward the top
           edge, +z out of the component side. PCB top surface is z = 0.
GPU        RTX 4090 PCB frame exactly as in design/data/gpu_pcb_layout.json: origin at
           the bracket-end / PCIe-edge corner, +x along the card, +y toward the top
           edge, +z out of the component side. PCB top surface is z = 0.
Both PCB frames map straight onto SHOW: x -> e_x, y -> e_s, z -> e_n.

Status tags used below
----------------------
[STD]  dimension from a published standard / drawing
[MEAS] measured from PCB photos (see gpu_pcb_layout.json, +-0.3 mm)
[ASM]  engineering assumption - verify on the real part before ordering metal
[DES]  free design choice
"""
from __future__ import annotations

import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
GPU_JSON = HERE / "data" / "gpu_pcb_layout.json"

# ----------------------------------------------------------------------------
# Print-service envelope (binding intersection of JLC3DP and PCBWay, Nov 2026)
# ----------------------------------------------------------------------------
PRINT = dict(
    env_316l=(300.0, 300.0, 290.0),      # PCBWay 300 cube cut by JLC3DP 290 axis
    env_alu=(300.0, 300.0, 300.0),       # PCBWay AlSi10Mg generic 300 cube
    soft_limit=250.0,                    # JLC3DP surcharge threshold for long parts
    min_wall=0.8,                        # "thinnest part >= 0.8 mm" (both vendors)
    min_wall_structural=1.5,             # walls up to 50 x 50 mm
    min_hole=1.5,
    min_escape_hole=2.5,
    tol_small=0.3, tol_pct=0.4,
    rate_316l_usd_cm3=(1.76, 2.68),      # JLC3DP, PCBWay (automated quotes)
    rate_alu_usd_cm3=(1.06, 1.50),
    rho_316l=7.98, rho_alu=2.67,
)

# ----------------------------------------------------------------------------
# Show plane
# ----------------------------------------------------------------------------
TILT = 25.0                              # [DES] lean back from vertical
SHOW_O = (0.0, 42.0, 34.0)               # [DES] world position of show origin
_t = math.radians(TILT)
E_X = (1.0, 0.0, 0.0)
E_S = (0.0, math.sin(_t), math.cos(_t))
E_N = (0.0, -math.cos(_t), math.sin(_t))


def show(x: float, s: float, n: float = 0.0):
    """SHOW coordinates -> WORLD."""
    return (
        SHOW_O[0] + x,
        SHOW_O[1] + s * E_S[1] + n * E_N[1],
        SHOW_O[2] + s * E_S[2] + n * E_N[2],
    )


# ----------------------------------------------------------------------------
# GPU: MSI RTX 4090 SUPRIM LIQUID 24G, PCB MS-V510
# ----------------------------------------------------------------------------
_g = json.loads(GPU_JSON.read_text()) if GPU_JSON.exists() else None

GPU = dict(
    pcb_l=266.6, pcb_h=127.25, pcb_t=1.57,           # [STD] Alphacool drawing
    tongue_y=-12.75, tongue_x=(45.0, 129.3),         # [STD] PCIe CEM
    centre=(116.55, 59.20),                          # [MEAS]
    die=(23.4, 26.4), package=(51.0, 51.0),          # [MEAS]
    stiffener_open=38.0,                             # [MEAS]
    core_holes=[(87.09, 93.07), (145.94, 93.04), (87.11, 25.44), (145.92, 25.49)],  # [MEAS]
    mec={                                            # [MEAS] silkscreen names
        "MEC1": (170.61, 106.63), "MEC2": (139.64, 117.55), "MEC3": (98.71, 117.56),
        "MEC4": (22.31, 109.49), "MEC5": (4.85, 93.44), "MEC6": (222.90, 65.47),
        "MEC7": (222.81, 5.05), "MEC8": (202.42, 119.25), "MEC9": (170.83, 10.76),
        "MEC11": (261.09, 95.38), "MEC13": (22.29, 12.03), "MEC14": (261.08, 5.01),
    },
    mem={                                            # [MEAS] centre x, y, w, h
        "M1": (81.25, 78.42, 14, 12), "M2": (81.25, 65.57, 14, 12),
        "M3": (81.27, 52.78, 14, 12), "M4": (81.27, 39.97, 14, 12),
        "M5": (104.14, 96.07, 12, 14), "M6": (116.94, 96.08, 12, 14),
        "M7": (127.09, 25.74, 14, 12), "M8": (129.81, 96.07, 12, 14),
        "M9": (151.82, 78.36, 14, 12), "M10": (151.81, 65.59, 14, 12),
        "M11": (151.82, 52.74, 14, 12), "M12": (151.83, 39.96, 14, 12),
    },
    vrm_row_y0=7.80, vrm_row_pitch=8.552,            # [MEAS]
    vrm_near=dict(rows=range(0, 14), drmos_x=(30.05, 36.30), ind_x=(37.82, 51.70)),
    vrm_far=dict(rows=range(1, 13), ind_x=(183.95, 197.80), drmos_x=(199.35, 205.60),
                 cap_x=212.95, cap_d=6.3,
                 cap_y=[13.05, 20.15, 27.49, 34.70, 41.96, 49.21, 56.54,
                        74.59, 81.83, 89.04, 96.27, 103.35, 110.47]),
    mem_vrm_ind=[(65.7, 96.7), (65.8, 25.5), (165.4, 94.7), (165.6, 25.8)],   # [MEAS] 7x7 bodies
    mem_vrm_stage=[(77.4, 99.9), (77.2, 22.2), (155.5, 99.4), (155.9, 22.5)],
    small_ind=[(84.0, 110.3, 91.0, 117.3), (93.1, 11.2, 99.8, 17.6), (101.3, 11.2, 108.0, 17.6),
               (186.7, 114.6, 196.3, 123.5), (242.3, 78.0, 256.2, 95.2), (248.5, 54.9, 254.3, 60.4),
               (251.0, 10.8, 256.8, 16.1)],
    hpwr=dict(x=(237.8, 257.0), y=(103.8, 117.2), recess_x=237.1, recess_y=117.2),
    headers_top=[(106.5, 114.4, 118.2, 120.4), (120.1, 114.4, 131.9, 120.4)],
    headers_bot=[(229.9, 2.4, 241.9, 8.5), (243.7, 2.9, 255.6, 8.3)],
    display_cage=dict(x=(-4.0, 15.2), y=(4.0, 85.5), dp_y=[76.7, 55.7, 34.7], hdmi_y=12.9),
    # heights above the PCB top surface  [ASM] - no public source; measure the card
    z_die=3.00, z_stiffener=2.55, z_mem=1.20, z_drmos=0.95, z_ind=4.50, z_cap=6.50,
    z_mem_ind=3.50, z_hpwr=9.0,
    power_w=450.0,                                   # [STD] MSI datasheet (silent BIOS)
    power_split=dict(die=390.0, mem=30.0, vrm=30.0), # [ASM] typical 4090 breakdown
    mass_pcb_kg=0.36,                                # [ASM]
)
# placement on the show plane (connector edge DOWN, bracket on the viewer's LEFT)
GPU_AT = dict(x0=-136.0, s0=24.0, n0=0.0)            # [DES]


def gpu_to_show(xg, yg, zg=0.0):
    return (GPU_AT["x0"] + xg, GPU_AT["s0"] + yg, GPU_AT["n0"] + zg)


def gpu_to_world(xg, yg, zg=0.0):
    return show(*gpu_to_show(xg, yg, zg))


# ----------------------------------------------------------------------------
# Motherboard: generic mini-ITX AM5 (exact model unknown -> representative layout)
# ----------------------------------------------------------------------------
BOARD = dict(
    size=(170.0, 170.0), pcb_t=1.6,                                  # [STD]
    holes=[(6.35, 159.84), (163.83, 136.98), (6.35, 4.90), (163.83, 4.90)],  # [STD] mini-ITX
    socket=(80.0, 96.0),                                             # [ASM] socket centre
    am5_holes=(54.0, 90.0),                                          # [STD] 54 (x) by 90 (y)
    am5_thread="UNC 6-32",
    ihs=(40.0, 40.0), z_ihs=7.60,                                    # [ASM] IHS top above PCB
    sam=(56.0, 76.0),                                                # socket frame footprint (visual)
    hotspot_offset=(0.0, -6.0),                                      # [ASM] CCD sits south of centre
    # VRM, representative 6 + 4 phase "L" around the socket  [ASM]
    vrm_left=dict(n=6, y0=76.0, pitch=12.0, drmos_x=32.0, ind_x=43.0),
    vrm_top=dict(n=4, x0=62.0, pitch=16.0, drmos_y=161.0, ind_y=150.0),
    drmos=(5.0, 6.0), inductor=(10.0, 10.0),
    z_drmos=0.95, z_ind=5.0,
    eps8=dict(x=(8.0, 28.0), y=(153.0, 168.0)),
    io=dict(x=(-2.5, 27.0), y=(18.0, 150.0), z=30.0),
    dimm=dict(x=[143.0, 152.5], y=(22.0, 164.0), w=7.0, z_slot=6.0, z_module=34.0),
    atx24=dict(x=(159.0, 169.0), y=(72.0, 126.0)),
    m2=dict(x=(30.0, 110.0), y=(20.0, 42.0), z_top=3.6),             # 2280 SSD
    chipset=dict(c=(124.0, 31.0), size=(19.0, 19.0), z_top=1.9),
    pcie=dict(y=7.5, x=(40.0, 131.0), z=11.25),
    cpu_power_w=120.0, vrm_power_w=22.0,                             # [ASM] Ryzen 7 X3D class, PBO peak
    m2_power_w=7.0, chipset_power_w=6.0,
    mass_kg=0.55,
)
BOARD_AT = dict(x0=-131.0, s0=GPU_AT["s0"] + GPU["pcb_h"] + 22.75, n0=0.0)   # [DES] 174.0


def board_to_show(xb, yb, zb=0.0):
    return (BOARD_AT["x0"] + xb, BOARD_AT["s0"] + yb, BOARD_AT["n0"] + zb)


def board_to_world(xb, yb, zb=0.0):
    return show(*board_to_show(xb, yb, zb))


# ----------------------------------------------------------------------------
# Power supply: EVGA SuperNOVA NEX650G (ATX). Length not confirmed -> bay takes 180.
# ----------------------------------------------------------------------------
PSU = dict(
    size=(180.0, 150.0, 86.0),            # along X (length), Y (width), Z (height)  [ASM] length
    lo=(-90.0, 116.0, 26.0),              # world min corner  [DES]
    ac_face="-x", modular_face="+x", fan="+z",   # AC cord exits on the I/O side (viewer's left)
    mass_kg=2.0, watts=650,
    # ATX rear-face hole pattern, measured on the 150 x 86 face from its lower-left corner [STD]
    rear_holes=[(6.0, 6.0), (144.0, 6.0), (6.0, 80.0 - 10.0), (120.0, 80.0)],
)

# ----------------------------------------------------------------------------
# Radiators: 2 x 240 mm slim (27 mm) aluminium, 2 x 120 x 25 fans each
# ----------------------------------------------------------------------------
RAD = dict(
    core=(275.0, 27.0, 120.0),            # X length, Y thickness, Z height  [STD-ish]
    fan=(120.0, 25.0), fan_hole=105.0, fan_thread="M3",
    y_fan0=232.0,                         # inner face of fans  [DES]
    centres_z=[298.0, 178.0],             # rad A (upper), rad B (lower)
    tank_end="+x", tank_len=17.5, port_dz=16.0,
    mass_kg=0.42, fan_mass_kg=0.16, coolant_ml=110,
)
RAD["y_core0"] = RAD["y_fan0"] + RAD["fan"][1]            # 257
RAD["y_rear"] = RAD["y_core0"] + RAD["core"][1]           # 284 (outer face, threaded holes)


def rad_holes():
    """Threaded fan holes on the rear (outer) face of both radiators -> list of (x, z)."""
    out = []
    for zc in RAD["centres_z"]:
        for fx in (-60.0, 60.0):
            for dx in (-52.5, 52.5):
                for dz in (-52.5, 52.5):
                    out.append((fx + dx, zc + dz))
    return out


# ----------------------------------------------------------------------------
# Loop
# ----------------------------------------------------------------------------
LOOP = dict(
    tube_od=13.0, tube_id=10.0, port="G1/4", port_tap=11.8, port_major=13.157,
    port_depth=7.0, port_seat_d=21.0, min_bend_r=38.0,
    coolant="ruby", pump="DDC-class 12 V PWM",
    order=["heart", "gpu_block", "monoblock", "rad_A", "rad_B", "heart"],
    design_flow_lpm=3.0, ambient_c=25.0,
)

# ----------------------------------------------------------------------------
# Lattice language
# ----------------------------------------------------------------------------
LATTICE = dict(
    wet_cell=5.0, wet_wall=0.85, wet_root=0.7, wet_root_len=1.3,   # sheet gyroid, coolant side
    gallery_cell=8.0, gallery_wall=0.9,
    skin_cell=11.0, skin_wall=1.35,                                # exterior "bone" gyroid
    voronoi_cell=11.0, voronoi_strut=1.9,
)

# ----------------------------------------------------------------------------
# Cold plate stack-ups (z measured from the PCB top surface each part sits on)
# ----------------------------------------------------------------------------
GPU_BLOCK = dict(
    material="AlSi10Mg",
    z_under=3.40,                 # general underside over the core zone
    floor=2.00,                   # die pedestal face -> wet chamber floor
    core_h=6.5, divider=1.0, gallery_h=5.0, roof=1.6, skin_h=7.5,
    chamber_x=(71.0, 162.0), chamber_y=(20.0, 105.5), wall=2.0,
    riser_gap=3.5,                # perimeter slot where coolant rises from core to gallery
    jet=(4.0, 24.0),              # slot nozzle (x width, y length) over the die
    trench=(5.0, 2.0, 3.2),       # entry trench under the nozzle: top width, bottom width, lattice left on the floor
    inlet=(116.55, 59.20), outlet=(116.55 - 34.0, 59.20),
    pedestal_die=(30.0, 33.0), pad_mem=1.0, pad_vrm=1.5, pad_ind=1.0,
    wing_top=9.7,
    post_thread="M3", post_pilot=2.5,
    posts=["MEC13", "MEC4", "MEC1", "MEC9", "MEC8", "MEC7", "MEC6"],
    frame_holes=["MEC13", "MEC4", "MEC8", "MEC7"],
)
MONOBLOCK = dict(
    material="AlSi10Mg",
    floor=2.0, core_h=6.5, divider=1.0, gallery_h=5.0, roof=1.6, skin_h=7.0,
    contact=(42.0, 42.0), chamber=(46.0, 46.0), wall=2.0, riser_gap=3.0,
    jet=(24.0, 4.0),              # slot nozzle (x length, y width) over the CCD hot spot
    trench=(6.0, 2.5, 2.0),
    inlet_offset=(0.0, -4.0),     # port position relative to socket centre
    arm_left=dict(x=(28.5, 48.6), y=(68.0, 147.0)),
    arm_top=dict(x=(28.5, 120.0), y=(145.5, 167.5)),
    arm_z=(6.0, 23.7), arm_channel_z=(8.0, 22.1), arm_skin_h=4.6,
    arm_cell=12.0, arm_bias=1.4,   # skeletal gyroid in the arms: low pressure drop, still self-supporting
    bridge_y=(72.0, 104.0),
    outlet=(111.0, 156.5),
    lug_d=11.0, lug_z=(9.6, 13.6), lug_hole=4.2, lug_well=8.6,
    heater_cpu=(26.0, 20.0),      # effective hot patch on the IHS  [ASM]
)
M2_SINK = dict(material="AlSi10Mg", x=(28.0, 135.0), y=(17.5, 44.5), base=2.2, lattice_h=13.0)
HEART = dict(
    material="AlSi10Mg", centre_show=(93.0, 236.0), pump_body=(62.0, 62.0, 38.0),
    volute_d=42.0, volute_h=11.0, res_ml=45.0,
)

# ----------------------------------------------------------------------------
# Frame (316L)
# ----------------------------------------------------------------------------
FRAME = dict(
    material="316L",
    spine_n=-24.0, spine_r=7.0, rib_r=4.2, foot_r=5.5,
    gpu_standoff=6.0, board_standoff=10.0,
    rear_spine_y=RAD["y_rear"] + 8.5,      # 292.5
    keel_z=12.0,
    feet=[(-120.0, 26.0), (120.0, 26.0), (-120.0, 286.0), (120.0, 286.0)],
    joint_bolt="M5", joint_len=40.0,
    split_front_s=160.0, split_rear_z=262.0,
    crown_z=368.0,
)

MASSES = dict(  # kg, for FEA load cases  [ASM]
    gpu_pcb=GPU["mass_pcb_kg"], board=BOARD["mass_kg"], ram=0.08, psu=PSU["mass_kg"],
    radiator_wet=RAD["mass_kg"] + 2 * RAD["fan_mass_kg"] + RAD["coolant_ml"] / 1000.0,
    pump=0.22, tubing_coolant=0.25,
)


def layout() -> dict:
    """Everything a downstream consumer needs, JSON-serialisable."""
    def tup(v):
        if isinstance(v, range):
            return list(v)
        if isinstance(v, dict):
            return {k: tup(w) for k, w in v.items()}
        if isinstance(v, (list, tuple)):
            return [tup(w) for w in v]
        return v

    psu_lo = PSU["lo"]
    psu_hi = tuple(psu_lo[i] + PSU["size"][i] for i in range(3))
    L = dict(
        units="mm",
        frames=dict(
            world="X right, Y back, Z up, desk z=0",
            show=dict(origin=SHOW_O, e_x=E_X, e_s=E_S, e_n=E_N, tilt_deg=TILT),
            board=dict(origin_show=(BOARD_AT["x0"], BOARD_AT["s0"], BOARD_AT["n0"]),
                       origin_world=board_to_world(0, 0, 0)),
            gpu=dict(origin_show=(GPU_AT["x0"], GPU_AT["s0"], GPU_AT["n0"]),
                     origin_world=gpu_to_world(0, 0, 0)),
        ),
        print=PRINT, gpu=GPU, board=BOARD, psu=dict(PSU, hi=psu_hi), rad=RAD,
        rad_holes_xz=rad_holes(), loop=LOOP, lattice=LATTICE,
        gpu_block=GPU_BLOCK, monoblock=MONOBLOCK, m2_sink=M2_SINK, heart=HEART,
        frame=FRAME, masses=MASSES,
        envelope=dict(width=300.0, depth=305.0, height=FRAME["crown_z"] + 8.0),
    )
    return tup(L)


if __name__ == "__main__":
    out = HERE / "layout.json"
    out.write_text(json.dumps(layout(), indent=1))
    L = layout()
    print("wrote", out)
    print("board origin (world):", [round(v, 2) for v in L["frames"]["board"]["origin_world"]])
    print("board top edge (world):", [round(v, 2) for v in board_to_world(0, 170, 0)])
    print("gpu origin (world):", [round(v, 2) for v in L["frames"]["gpu"]["origin_world"]])
    print("gpu die centre (world):", [round(v, 2) for v in gpu_to_world(*GPU["centre"], GPU["z_die"])])
