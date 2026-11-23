"""Convert the print STLs (exports/print/*.stl, git-ignored because of their size) into compact,
indexed 3MF files that both print services accept:  /opt/venv/bin/python tools/export_3mf.py"""
import json
import pathlib

import trimesh

ROOT = pathlib.Path(__file__).resolve().parent.parent
out = {}
for p in sorted((ROOT / "exports" / "print").glob("*.stl")):
    m = trimesh.load(p, process=True)
    q = p.with_suffix(".3mf")
    m.export(q)
    out[p.stem] = dict(faces=int(len(m.faces)), watertight=bool(m.is_watertight), stl_mb=round(p.stat().st_size / 1e6, 1),
                       mf3_mb=round(q.stat().st_size / 1e6, 1), volume_cm3=round(abs(m.volume) / 1000.0, 2))
    print(p.stem, out[p.stem], flush=True)
(ROOT / "exports" / "print" / "index.json").write_text(json.dumps(out, indent=1))
