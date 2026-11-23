"""Manifold-preserving decimation and a watertightness report for print files (pymeshlab)."""
import numpy as np


def decimate_manifold(mesh, target_faces):
    """Quadric edge collapse that keeps the surface a closed 2-manifold. Returns (trimesh, report)."""
    import pymeshlab
    import trimesh

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    if ms.current_mesh().face_number() > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_faces), preservetopology=True, preserveboundary=True,
                                                    preservenormal=True, optimalplacement=True, qualitythr=0.3, autoclean=True)
    for _ in range(3):
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        try:
            ms.meshing_remove_connected_component_by_face_number(mincomponentsize=2000)
            ms.meshing_close_holes(maxholesize=200)
        except Exception as ex:  # noqa: BLE001
            print("    meshfix:", ex)
        rep = ms.get_topological_measures()
        if rep.get("boundary_edges", 1) == 0 and rep.get("non_two_manifold_edges", 1) == 0:
            break
    rep = ms.get_topological_measures()
    cm = ms.current_mesh()
    out = trimesh.Trimesh(vertices=cm.vertex_matrix(), faces=cm.face_matrix(), process=False)
    report = dict(faces=int(cm.face_number()), boundary_edges=int(rep.get("boundary_edges", -1)),
                  non_manifold_edges=int(rep.get("non_two_manifold_edges", -1)),
                  non_manifold_vertices=int(rep.get("non_two_manifold_vertices", -1)),
                  components=int(rep.get("connected_components_number", -1)), watertight=bool(out.is_watertight))
    return out, report
