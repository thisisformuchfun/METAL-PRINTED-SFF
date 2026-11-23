#!/usr/bin/env bash
# Regenerate MARROW from source. Needs /opt/venv (see requirements in the README) and Blender 4.5.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/opt/venv/bin/python
BLENDER=${BLENDER:-/opt/tools/blender/blender}

# 1. CAD: parameters -> nine printed parts (print STL, render meshes, metadata)
$PY -m design.params
for p in gpu_block monoblock heart m2_sink; do $PY -m design.$p 0.25; done
$PY -m design.frame 0.5
PYTHONPATH=. $PY tools/export_3mf.py
PYTHONPATH=. $PY tools/check_interference.py
$PY -m design.cutaway

# 2. Simulations
$PY -m sim.validation.validate_cell
$PY -m sim.cell.unit_cell
for m in AlSi10Mg 316L; do $PY -m sim.macro.macro_cht gpu_block $m 3.0; done
for q in 1.5 4.5; do $PY -m sim.macro.macro_cht gpu_block AlSi10Mg $q; $PY -m sim.macro.macro_cht monoblock AlSi10Mg $q; done
for m in AlSi10Mg 316L; do $PY -m sim.macro.macro_cht monoblock $m 3.0; done
$PY -m sim.fea.frame_fea 1.5
$PY -m sim.loop.loop_model
$PY -m sim.macro.passive_sink        # after the monoblock runs and the loop model: it reads both
$PY -m sim.post.site_assets

# 3. Scene, stills, film (long: path tracing on CPU)
$BLENDER -b --python render/build_blend.py -- hero
$BLENDER -b --python render/build_blend.py -- web
$BLENDER -b build/marrow_hero.blend --python render/shots.py -- still hero_front hero_low hero_right rear_ribs board gpu heart \
    macro_skin macro_knuckle macro_tube frame_only --res 1600x900 --spp 64 --exposure 0.25
$BLENDER -b build/marrow_hero.blend --python render/shots.py -- still hero_wide --res 1920x1080 --spp 64 --exposure 0.25
$BLENDER -b --python render/still_parts.py -- organs
$BLENDER -b --python render/still_parts.py -- cutaway
$BLENDER -b build/marrow_hero.blend --python render/shots.py -- orbit orbit --frames 0:59 --total 60 --res 1280x720 --spp 48 \
    --exposure 0.25 --az0 -38 --az1 -17 --side 190 --out build/orbit
(cd media/score && $PY score.py)
PYTHONPATH=. $PY media/film/simviz.py
$PY media/film/make_film.py

# 4. Web model + site
$BLENDER -b build/marrow_web.blend --python render/export_web.py
tools/build_site.sh
$PY tools/make_readme.py
