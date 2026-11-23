"""pc_parts - procedural off-the-shelf PC hardware for the MARROW product film.

    import sys; sys.path.append("<repo>/render"); import pc_parts
    L, G = pc_parts.load_layout()                       # design/layout.json + gpu_pcb_layout.json
    board = pc_parts.build_itx_board(L)                 # every builder returns its root Empty

Units: 1 Blender unit = 1 mm (parent the roots under an Empty scaled 0.001 and call
materials.set_unit_scale(0.001) afterwards). See PARTS.md for frames and naming.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

import materials                                             # noqa: E402
from materials import set_unit_scale                         # noqa: E402
from pcp_geom import tri_count, clear_cache, Part            # noqa: E402
from pcp_board import build_itx_board, keepouts as board_keepouts, PCIE_CAV0, PCIE_TONGUE   # noqa: E402
from pcp_gpu import build_gpu_4090                           # noqa: E402
from pcp_cooling import (build_radiator_240, build_fan_120, build_ddc_pump, build_fitting,   # noqa: E402
                         build_tube)
from pcp_power import build_psu_atx, build_cable, build_riser, CABLES   # noqa: E402

__all__ = ["load_layout", "build_itx_board", "build_gpu_4090", "build_psu_atx", "build_radiator_240",
           "build_fan_120", "build_ddc_pump", "build_fitting", "build_tube", "build_cable", "build_riser",
           "board_keepouts", "board_slot_datum", "gpu_tongue_datum", "tri_count", "clear_cache",
           "set_unit_scale", "materials", "parts_stats"]


def load_layout(repo=None):
    """Return (layout.json dict, gpu_pcb_layout.json dict)."""
    repo = repo or os.path.dirname(HERE)
    with open(os.path.join(repo, "design", "layout.json")) as f:
        L = json.load(f)
    with open(os.path.join(repo, "design", "data", "gpu_pcb_layout.json")) as f:
        G = json.load(f)
    return L, G


def board_slot_datum(L):
    """Riser start in the BOARD frame: (centre of the x16 slot opening under the tongue centre, +z)."""
    pc = L["board"]["pcie"]
    return (pc["x"][0] + PCIE_CAV0 + 0.15 + PCIE_TONGUE / 2, pc["y"], pc["z"]), (0.0, 0.0, 1.0)


def gpu_tongue_datum(L, depth=8.4):
    """Riser end in the GPU frame: (female-slot opening centre, direction the slot opens = +y).
    depth = how far the gold-finger tongue enters the slot."""
    g = L["gpu"]
    xc = (g["tongue_x"][0] + g["tongue_x"][1]) / 2
    return (xc, g["tongue_y"] + depth, -g["pcb_t"] / 2), (0.0, 1.0, 0.0)


def parts_stats(roots):
    """{root name: (instanced triangles, unique-mesh triangles, object count)}."""
    out = {}
    for r in roots:
        objs = [r] + list(r.children_recursive)
        out[r.name] = (tri_count(objs), tri_count(objs, unique=True), sum(o.type == "MESH" for o in objs))
    return out
