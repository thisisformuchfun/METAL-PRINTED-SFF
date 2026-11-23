"""MARROW master scene (Blender 4.5, Cycles). Import as a module from shot scripts:

    import marrow_scene as ms
    sc = ms.build(lod="hero", components=True)
    ms.render_still("out.png", cam=..., ...)

Units: all geometry is authored in millimetres under a root Empty scaled 0.001, so the
world is in metres (physically sensible lights and depth of field).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import bpy
from mathutils import Matrix, Vector

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "render"))
L = json.loads((REPO / "design" / "layout.json").read_text())
BUILD = REPO / "build"
MM = 0.001

SH = L["frames"]["show"]
O = Vector(SH["origin"])
EX, ES, EN = Vector(SH["e_x"]), Vector(SH["e_s"]), Vector(SH["e_n"])


def show(x, s, n=0.0):
    return O + EX * x + ES * s + EN * n


def frame_matrix(origin):
    m = Matrix.Identity(4)
    for i, v in enumerate((EX, ES, EN)):
        m[0][i], m[1][i], m[2][i] = v.x, v.y, v.z
    m[0][3], m[1][3], m[2][3] = origin[0], origin[1], origin[2]
    return m


M_GPU = frame_matrix(Vector(L["frames"]["gpu"]["origin_world"]))
M_BOARD = frame_matrix(Vector(L["frames"]["board"]["origin_world"]))
HC = L["heart"]["centre_show"]
M_HEART = frame_matrix(show(HC[0], HC[1], L["heart"].get("n0", 0.0)))


def gpu_w(x, y, z=0.0):
    return M_GPU @ Vector((x, y, z))


def board_w(x, y, z=0.0):
    return M_BOARD @ Vector((x, y, z))


def heart_w(x, y, z=0.0):
    return M_HEART @ Vector((x, y, z))


# ----------------------------------------------------------------------------
# materials for the printed parts
# ----------------------------------------------------------------------------
def _nodes(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    return m, nt, nt.nodes["Principled BSDF"]


def mat_printed(name, base, rough, grain=0.16, sparkle=0.35, tint=(1, 1, 1)):
    """Bead-blasted SLM metal: fine granular micro-relief, slightly uneven roughness."""
    m, nt, b = _nodes(name)
    N, Lk = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    n1 = N.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 1.0 / grain
    n1.inputs["Detail"].default_value = 2.0
    n1.inputs["Roughness"].default_value = 0.6
    n2 = N.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 0.09
    n2.inputs["Detail"].default_value = 4.0
    Lk.new(tc.outputs["Object"], n1.inputs["Vector"])
    Lk.new(tc.outputs["Object"], n2.inputs["Vector"])
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = sparkle
    bump.inputs["Distance"].default_value = 0.05 * MM     # 0.05 mm relief, world is in metres
    Lk.new(n1.outputs["Fac"], bump.inputs["Height"])
    Lk.new(bump.outputs["Normal"], b.inputs["Normal"])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.3
    mr.inputs["From Max"].default_value = 0.7
    mr.inputs["To Min"].default_value = rough - 0.05
    mr.inputs["To Max"].default_value = rough + 0.07
    Lk.new(n2.outputs["Fac"], mr.inputs["Value"])
    Lk.new(mr.outputs["Result"], b.inputs["Roughness"])
    mix = N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (base[0] * 0.93, base[1] * 0.93, base[2] * 0.94, 1)
    mix.inputs["B"].default_value = (base[0] * tint[0], base[1] * tint[1], base[2] * tint[2], 1)
    Lk.new(n2.outputs["Fac"], mix.inputs["Factor"])
    Lk.new(mix.outputs["Result"], b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = 1.0
    return m


def mat_simple(name, color, rough=0.5, metallic=0.0, transmission=0.0, ior=1.45, alpha=1.0, emission=None, strength=0.0):
    m, nt, b = _nodes(name)
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Transmission Weight"].default_value = transmission
    b.inputs["IOR"].default_value = ior
    if emission:
        b.inputs["Emission Color"].default_value = (*emission, 1)
        b.inputs["Emission Strength"].default_value = strength
    return m


def mat_coolant(name="MAT_coolant_ruby", color=(0.78, 0.015, 0.03), density=0.065):
    """Clear ruby liquid: glass surface + volume absorption (depth-dependent colour)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, Lk = nt.nodes, nt.links
    N.remove(N["Principled BSDF"])
    out = N["Material Output"]
    g = N.new("ShaderNodeBsdfGlass")
    g.inputs["Color"].default_value = (1.0, 0.55, 0.55, 1)
    g.inputs["Roughness"].default_value = 0.0
    g.inputs["IOR"].default_value = 1.34
    v = N.new("ShaderNodeVolumeAbsorption")
    v.inputs["Color"].default_value = (*color, 1)
    v.inputs["Density"].default_value = density
    Lk.new(g.outputs["BSDF"], out.inputs["Surface"])
    Lk.new(v.outputs["Volume"], out.inputs["Volume"])
    return m


def mat_tube(name="MAT_tube_clear"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, Lk = nt.nodes, nt.links
    N.remove(N["Principled BSDF"])
    g = N.new("ShaderNodeBsdfGlass")
    g.inputs["Color"].default_value = (0.97, 0.97, 0.98, 1)
    g.inputs["Roughness"].default_value = 0.03
    g.inputs["IOR"].default_value = 1.47
    Lk.new(g.outputs["BSDF"], N["Material Output"].inputs["Surface"])
    return m


MATS = {}


def materials():
    if MATS:
        return MATS
    MATS["alu"] = mat_printed("MAT_printed_AlSi10Mg", (0.70, 0.705, 0.715), 0.36, grain=0.17, sparkle=0.30)
    MATS["steel"] = mat_printed("MAT_printed_316L", (0.33, 0.335, 0.35), 0.22, grain=0.14, sparkle=0.18, tint=(1.04, 1.0, 0.96))
    MATS["coolant"] = mat_coolant()
    MATS["tube"] = mat_tube()
    MATS["fitting"] = mat_simple("MAT_fit_blacknickel", (0.09, 0.09, 0.10), rough=0.22, metallic=1.0)
    MATS["rubber"] = mat_simple("MAT_rubber", (0.02, 0.02, 0.022), rough=0.7)
    MATS["floor"] = mat_simple("MAT_floor", (0.010, 0.010, 0.012), rough=0.28)
    return MATS


# ----------------------------------------------------------------------------
# scene scaffolding
# ----------------------------------------------------------------------------
STATE = {}


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    MATS.clear()
    STATE.clear()
    root = bpy.data.objects.new("MARROW_ROOT", None)
    bpy.context.scene.collection.objects.link(root)
    root.scale = (MM, MM, MM)
    STATE["root"] = root
    return root


def collection(name):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(c)
    return c


def adopt(obj, coll=None):
    """Parent obj under the mm root (keeping its mm-space matrix) and move to a collection."""
    obj.parent = STATE["root"]
    if coll is not None:
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        coll.objects.link(obj)
    return obj


def import_mesh(path, name, matrix=None, mat=None, coll=None, smooth=True):
    path = str(path)
    if path.lower().endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path)
    else:
        bpy.ops.wm.stl_import(filepath=path)
    o = bpy.context.selected_objects[0]
    o.name = name
    if matrix is not None:
        o.matrix_world = matrix
    if mat is not None:
        o.data.materials.clear()
        o.data.materials.append(mat)
    if smooth:
        o.data.polygons.foreach_set("use_smooth", [True] * len(o.data.polygons))
    adopt(o, coll)
    return o


PRINTED = [
    ("gpu_block", M_GPU, "alu"), ("monoblock", M_BOARD, "alu"), ("m2_sink", M_BOARD, "alu"), ("heart", M_HEART, "alu"),
    ("frame_pelvis", None, "steel"), ("frame_lumbar", None, "steel"), ("frame_thoracic", None, "steel"),
    ("frame_dorsal", None, "steel"), ("frame_cervical", None, "steel"),
]


def add_printed(lod="hero", only=None):
    mats = materials()
    coll = collection("printed")
    objs = {}
    for name, M, mk in PRINTED:
        if only and name not in only:
            continue
        p = BUILD / f"{name}_{lod}.ply"
        if not p.exists():
            p = BUILD / f"{name}_web.ply"
        if not p.exists():
            print("MISSING", name)
            continue
        objs[name] = import_mesh(p, name, M.copy() if M is not None else Matrix.Identity(4), mats[mk], coll)
    STATE["printed"] = objs
    return objs


# ----------------------------------------------------------------------------
# tubes
# ----------------------------------------------------------------------------
def port_table():
    """World position + outward direction of every G1/4 port."""
    gb, mb = L["gpu_block"], L["monoblock"]
    zg = L["gpu"]["z_die"] + gb["floor"] + gb["core_h"] + gb["divider"] + gb["gallery_h"] + gb["roof"] + gb["skin_h"] + 0.3
    sx, sy = L["board"]["socket"]
    zc = L["board"]["z_ihs"] + mb["floor"] + mb["core_h"] + mb["divider"] + mb["gallery_h"] + mb["roof"] + mb["skin_h"] + 0.3
    za = mb["arm_z"][1] + mb["arm_skin_h"] + 0.3
    hp = json.loads((REPO / "exports" / "meta" / "heart.json").read_text())["checks"]["ports"]
    R = L["rad"]
    xr = R["core"][0] / 2 - R["tank_len"] / 2
    P = {
        "gpu_in": (gpu_w(*gb["inlet"], zg), EN), "gpu_out": (gpu_w(*gb["outlet"], zg), EN),
        "mono_in": (board_w(sx + mb["inlet_offset"][0], sy + mb["inlet_offset"][1], zc), EN),
        "mono_out": (board_w(*mb["outlet"], za), EN),
        "heart_out": (heart_w(*hp["out"]), EN), "heart_in": (heart_w(*hp["inlet_back"]), -EN),
        "heart_fill": (heart_w(*hp["fill"]), ES),
    }
    for key, zc_ in (("radA", R["centres_z"][0]), ("radB", R["centres_z"][1])):
        P[key + "_hi"] = (Vector((xr, R["y_core0"], zc_ + R["port_dz"])), Vector((0, -1, 0)))
        P[key + "_lo"] = (Vector((xr, R["y_core0"], zc_ - R["port_dz"])), Vector((0, -1, 0)))
    return P


FIT_SEAT = 5.6        # straight fitting: port seat -> where the tube bottoms out
FIT_EXIT = 20.4       # straight fitting: port seat -> end of the compression collar


def tube_routes():
    P = port_table()
    seat = lambda k: P[k][0] + P[k][1] * FIT_SEAT
    out = lambda k, d=FIT_EXIT + 4.0: P[k][0] + P[k][1] * d

    def arc(a, b, lift, via=None):
        da, db = P[a][1], P[b][1]
        pts = [seat(a), out(a), out(a) + da * lift * 0.5]
        if via:
            pts += [Vector(v) for v in via]
        else:
            pts.append((out(a) + out(b)) * 0.5 + (da + db).normalized() * lift)
        pts += [out(b) + db * lift * 0.5, out(b), seat(b)]
        return pts

    routes = {
        "T1_heart_gpu": arc("heart_out", "gpu_in", 56.0),
        "T2_gpu_mono": arc("gpu_out", "mono_in", 52.0),
        "T3_mono_radA": arc("mono_out", "radA_hi", 26.0, via=[(6, 180, 387), (54, 216, 383), (104, 222, 356), (124, 212, 332)]),
        "T4_radA_radB": arc("radA_lo", "radB_hi", 22.0, via=[(127, 196, 238)]),
        "T5_radB_heart": arc("radB_lo", "heart_in", 20.0, via=[(122, 206, 200), (106, 200, 246)]),
    }
    return routes, P


def dir_matrix(origin, zdir, xhint=None):
    """Matrix placing a part's local +Z along zdir at origin."""
    z = Vector(zdir).normalized()
    xh = Vector(xhint) if xhint is not None else (Vector((1, 0, 0)) if abs(z.x) < 0.9 else Vector((0, 1, 0)))
    x = (xh - z * xh.dot(z)).normalized()
    y = z.cross(x)
    m = Matrix.Identity(4)
    for i, v in enumerate((x, y, z)):
        m[0][i], m[1][i], m[2][i] = v.x, v.y, v.z
    m[0][3], m[1][3], m[2][3] = origin[0], origin[1], origin[2]
    return m


def add_loop():
    """Fittings + soft tubing + ruby coolant, using the pc_parts builders."""
    import pc_parts
    routes, P = tube_routes()
    od, idd = L["loop"]["tube_od"], L["loop"]["tube_id"]
    objs = {}
    for key, (p, d) in P.items():
        if key == "heart_fill":
            r = pc_parts.build_fitting("plug", name="fit_" + key)[0]
        else:
            r = pc_parts.build_fitting("straight", name="fit_" + key, od=od, idd=idd)[0]
        place(r, dir_matrix(p, d))
        objs["fit_" + key] = r
    for name, pts in routes.items():
        r = pc_parts.build_tube([tuple(v) for v in pts], od=od, id=idd, coolant=True, name=name)
        r.parent = STATE["root"]
        objs[name] = r
    STATE["ports"] = P
    STATE["loop"] = objs
    return objs


def add_riser_and_cables():
    import pc_parts
    out = {}
    # ---- PCIe riser: board slot -> down behind the graphics card -> female slot on the shelf
    (bx, by, bz), _ = pc_parts.board_slot_datum(L)
    (gx, gy, gz), _ = pc_parts.gpu_tongue_datum(L)
    p0 = board_w(bx, by, bz)
    p1 = gpu_w(gx, gy, gz)
    xb = (p0 - O).dot(EX)
    xg = (p1 - O).dot(EX)
    sb = L["frames"]["board"]["origin_show"][1]
    sg1 = L["frames"]["gpu"]["origin_show"][1] + L["gpu"]["pcb_h"]
    mid = 0.5 * (sb + sg1)
    lerp = lambda t: xb + (xg - xb) * t
    pts = [p0, show(lerp(0.0), sb + 7.5, 30.0), show(lerp(0.05), sb + 0.5, 31.0), show(lerp(0.1), mid + 3.0, 17.0),
           show(lerp(0.2), mid - 2.0, -3.0), show(lerp(0.35), sg1 - 8.0, -8.6), show(lerp(0.7), 90.0, -8.6),
           show(lerp(1.0), 34.0, -8.6), p1]
    try:
        r = pc_parts.build_riser([tuple(v) for v in pts], width_dir=tuple(EX), male_dir=tuple(EN), female_dir=tuple(ES))
        r.parent = STATE["root"]
        out["riser"] = r
    except Exception as ex:  # noqa: BLE001
        print("riser failed:", ex)
    # ---- power cables (PSU modular face is world +x after the 180 deg turn)
    psu_lo = Vector(L["psu"]["lo"])
    sock = lambda u, v, depth=8.5: Vector((psu_lo.x + L["psu"]["size"][0] - depth, psu_lo.y + u, psu_lo.z + v))
    px = Vector((1, 0, 0))
    hp = L["gpu"]["hpwr"]
    g_end = gpu_w(0.5 * (hp["x"][0] + hp["x"][1]), hp["y"][1] - 7.0, 5.4)
    a24 = L["board"]["atx24"]
    b24 = board_w(0.5 * (a24["x"][0] + a24["x"][1]) - 0.4, 0.5 * (a24["y"][0] + a24["y"][1]), 2.0)
    e8 = L["board"]["eps8"]
    b8 = board_w(0.5 * (e8["x"][0] + e8["x"][1]), 0.5 * (e8["y"][0] + e8["y"][1]) - 1.0, 2.0)
    xs24 = (b24 - O).dot(EX)
    s24 = (b24 - O).dot(ES)
    xs8 = (b8 - O).dot(EX)
    s8 = (b8 - O).dot(ES)
    cables = {
        "12vhpwr": dict(pts=[sock(129.9, 60.0), sock(129.9, 60.0, -22.0), Vector((118, 232, 104)), Vector((118, 196, 150)),
                             show(113.0, 186.0, -34.0), show(111.6, 184.0, -8.0), g_end + ES * 46.0 + EN * 3.0, g_end + ES * 22.0, g_end],
                        up=(0, 0, 1), up_end=tuple(EN)),
        "atx24": dict(pts=[sock(26.4, 60.0), sock(26.4, 60.0, -24.0), Vector((116, 150, 124)), Vector((96, 150, 176)),
                           show(52.0, s24 - 16.0, -36.0), show(47.5, s24 - 2.0, -6.0), b24 + EN * 46.0 + EX * 9.0, b24 + EN * 22.0, b24],
                      up=(0, 0, 1), up_end=tuple(EX)),
        "eps8": dict(pts=[sock(42.06, 26.0), sock(42.06, 26.0, -22.0), Vector((112, 170, 70)), Vector((60, 196, 128)),
                          Vector((-40, 204, 250)), show(xs8, s8 + 24.0, -34.0), show(xs8, s8 + 27.0, 4.0),
                          b8 + EN * 40.0 + ES * 12.0, b8 + EN * 22.0, b8],
                     up=(0, 0, 1), up_end=tuple(ES)),
    }
    for kind, c in cables.items():
        try:
            r = pc_parts.build_cable([tuple(v) for v in c["pts"]], kind=kind, up=c["up"], up_end=c["up_end"])
            r.parent = STATE["root"]
            out["cable_" + kind] = r
        except Exception as ex:  # noqa: BLE001
            print("cable failed:", kind, ex)
    STATE["cables"] = out
    return out


def add_cylinder(name, a, b, r, mat, coll, verts=48):
    a, b = Vector(a), Vector(b)
    d = b - a
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=d.length, location=(a + b) / 2)
    o = bpy.context.object
    o.name = name
    o.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    o.data.materials.append(mat)
    for p in o.data.polygons:
        p.use_smooth = True
    bv = o.modifiers.new("bevel", "BEVEL")
    bv.width = 0.6
    bv.segments = 3
    bv.limit_method = "ANGLE"
    adopt(o, coll)
    return o


# ----------------------------------------------------------------------------
# off-the-shelf components (procedural models from pc_parts)
# ----------------------------------------------------------------------------
def place(root_obj, matrix, coll=None):
    root_obj.matrix_world = matrix
    root_obj.parent = STATE["root"]
    return root_obj


def add_components():
    try:
        import pc_parts
    except Exception as ex:  # noqa: BLE001
        print("pc_parts not available:", ex)
        return {}
    G = json.loads((REPO / "design" / "data" / "gpu_pcb_layout.json").read_text())
    out = {}
    out["board"] = place(pc_parts.build_itx_board(L), M_BOARD)
    out["gpu"] = place(pc_parts.build_gpu_4090(L, G), M_GPU)
    # PSU: local box (180,150,86) with AC on local +x; we want AC on world -x -> rotate 180 deg about Z
    lo, hi = Vector(L["psu"]["lo"]), Vector(L["psu"]["hi"])
    psu = pc_parts.build_psu_atx(L)
    place(psu, Matrix.Translation(Vector((hi.x, hi.y, lo.z))) @ Matrix.Rotation(math.pi, 4, "Z"))
    out["psu"] = psu
    R = L["rad"]
    for i, zc in enumerate(R["centres_z"]):
        rad = pc_parts.build_radiator_240(L)
        place(rad, Matrix.Translation(Vector((0, R["y_core0"] + R["core"][1] / 2, zc))))
        out[f"rad{i}"] = rad
        for j, fx in enumerate((-60.0, 60.0)):
            fan = pc_parts.build_fan_120(L)
            place(fan, Matrix.Translation(Vector((fx, R["y_fan0"] + R["fan"][1] / 2, zc))))
            out[f"fan{i}{j}"] = fan
    pump = pc_parts.build_ddc_pump(L)
    place(pump, M_HEART.copy())
    out["pump"] = pump
    STATE["components"] = out
    return out


# ----------------------------------------------------------------------------
# lighting / world / camera / render
# ----------------------------------------------------------------------------
def set_world(hdri="studio_small_09_2k.hdr", strength=0.35, rot_deg=0.0, bg=(0.0045, 0.005, 0.006), bg_strength=1.0):
    w = bpy.data.worlds.new("world")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    N, Lk = nt.nodes, nt.links
    N.clear()
    out = N.new("ShaderNodeOutputWorld")
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0, 0, math.radians(rot_deg))
    env = N.new("ShaderNodeTexEnvironment")
    p = REPO / "render" / "hdri" / hdri
    if p.exists():
        env.image = bpy.data.images.load(str(p))
    bg_l = N.new("ShaderNodeBackground")
    bg_l.inputs["Strength"].default_value = strength
    bg_c = N.new("ShaderNodeBackground")
    bg_c.inputs["Color"].default_value = (*bg, 1)
    bg_c.inputs["Strength"].default_value = bg_strength
    lp = N.new("ShaderNodeLightPath")
    mix = N.new("ShaderNodeMixShader")
    Lk.new(tc.outputs["Generated"], mp.inputs["Vector"])
    Lk.new(mp.outputs["Vector"], env.inputs["Vector"])
    Lk.new(env.outputs["Color"], bg_l.inputs["Color"])
    Lk.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    Lk.new(bg_l.outputs["Background"], mix.inputs[1])
    Lk.new(bg_c.outputs["Background"], mix.inputs[2])
    Lk.new(mix.outputs["Shader"], out.inputs["Surface"])
    return w


def add_floor(z=0.0, size=6.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0.15, z))
    o = bpy.context.object
    o.name = "floor"
    o.data.materials.append(materials()["floor"])
    return o


def area_light(name, loc_mm, target_mm, power, size_mm, color=(1, 1, 1), size_y_mm=None, spread=None):
    Ld = bpy.data.lights.new(name, "AREA")
    Ld.energy = power
    Ld.color = color
    if size_y_mm:
        Ld.shape = "RECTANGLE"
        Ld.size = size_mm * MM
        Ld.size_y = size_y_mm * MM
    else:
        Ld.size = size_mm * MM
    if spread is not None:
        Ld.spread = math.radians(spread)
    o = bpy.data.objects.new(name, Ld)
    bpy.context.scene.collection.objects.link(o)
    o.location = Vector(loc_mm) * MM
    o.rotation_euler = (Vector(target_mm) * MM - o.location).to_track_quat("-Z", "Y").to_euler()
    return o


CENTER = Vector((0.0, 160.0, 190.0))      # visual centre of the machine (mm)


def light_rig(key=1.0, rim=1.0, fill=1.0, top=1.0, warm=(1.0, 0.92, 0.82), cool=(0.72, 0.84, 1.0)):
    c = CENTER
    lights = [
        area_light("key", c + Vector((-620, -760, 520)), c + Vector((0, -40, 20)), 9.0 * key, 620, warm, size_y_mm=900),
        area_light("fill", c + Vector((760, -620, 120)), c, 1.2 * fill, 900, cool, size_y_mm=1200),
        area_light("rim_l", c + Vector((-760, 520, 420)), c + Vector((0, 60, 40)), 14.0 * rim, 300, cool, size_y_mm=1100, spread=120),
        area_light("rim_r", c + Vector((700, 620, 300)), c + Vector((0, 60, 0)), 11.0 * rim, 260, (1.0, 0.86, 0.74), size_y_mm=1000, spread=120),
        area_light("top", c + Vector((0, 60, 900)), c, 5.0 * top, 1100, (1, 1, 1), size_y_mm=700),
    ]
    STATE["lights"] = lights
    return lights


def camera(name="cam", lens=50.0):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36.0
    cd.clip_start = 0.01
    cd.clip_end = 50.0
    o = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(o)
    bpy.context.scene.camera = o
    return o


def aim(cam, loc_mm, target_mm, fstop=None, roll_deg=0.0):
    cam.location = Vector(loc_mm) * MM
    d = Vector(target_mm) * MM - cam.location
    q = d.to_track_quat("-Z", "Y")
    cam.rotation_euler = q.to_euler()
    if roll_deg:
        cam.rotation_euler.rotate_axis("Z", math.radians(roll_deg))
    if fstop:
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = d.length
        cam.data.dof.aperture_fstop = fstop
    else:
        cam.data.dof.use_dof = False
    return cam


def orbit(az_deg, el_deg, dist_mm, target=None):
    t = CENTER if target is None else Vector(target)
    az, el = math.radians(az_deg), math.radians(el_deg)
    return t + Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * dist_mm


def render_settings(res=(1280, 720), spp=64, threads=4, denoise=True, bounces=8, look="AgX - Medium High Contrast",
                    exposure=0.0, film_transparent=False, noise_threshold=0.03):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.device = "CPU"
    cy.samples = spp
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = noise_threshold
    cy.use_denoising = denoise
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
        cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except Exception:  # noqa: BLE001
        pass
    cy.max_bounces = bounces
    cy.glossy_bounces = min(bounces, 6)
    cy.transmission_bounces = 10
    cy.transparent_max_bounces = 10
    cy.volume_bounces = 0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_indirect = 6.0
    cy.blur_glossy = 0.6
    sc.render.threads_mode = "FIXED"
    sc.render.threads = threads
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = film_transparent
    sc.render.use_persistent_data = True
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = look
    except Exception:  # noqa: BLE001
        pass
    sc.view_settings.exposure = exposure
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    return sc


def render_still(path, **kw):
    sc = bpy.context.scene
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print("RENDERED", path, flush=True)


def build(lod="hero", components=True, tubes=True, floor=True, only=None, cables=True):
    reset()
    add_printed(lod=lod, only=only)
    if components:
        add_components()
        if tubes:
            add_loop()
        if cables:
            add_riser_and_cables()
        import pc_parts
        pc_parts.set_unit_scale(MM)
    if floor:
        add_floor()
    return STATE


def save_blend(path):
    bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True)
    print("SAVED", path, flush=True)
