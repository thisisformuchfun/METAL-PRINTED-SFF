#!/usr/bin/env bash
# Assemble site/ from rendered stills, the film, simulation figures and the web models.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/opt/venv/bin/python}
GLTFPACK=${GLTFPACK:-/opt/tools/gltf/node_modules/.bin/gltfpack}
mkdir -p site/media site/model site/data site/sim

# stills -> JPEG
jpg() { # src dst width quality
  [ -f "$1" ] && ffmpeg -y -loglevel error -i "$1" -vf "scale=$3:-2:flags=lanczos" -q:v "${4:-3}" "$2" && echo "  $2" || echo "  (missing $1)"
}
S=build/stills
if [ -f $S/hero_wide.png ]; then jpg $S/hero_wide.png site/media/hero.jpg 1920 2; else jpg $S/hero_front.png site/media/hero.jpg 1920 2; fi
jpg $S/hero_front.png   site/media/hero_front.jpg    1600
jpg $S/frame_only.png   site/media/frame.jpg         1600
jpg $S/rear_ribs.png    site/media/rear.jpg          1600
jpg $S/cut_gpu.png      site/media/cut_gpu.jpg       1600
jpg $S/board.png        site/media/board.jpg         1600
jpg $S/heart.png        site/media/heart.jpg         1600
jpg $S/macro_skin.png   site/media/macro_skin.jpg    1600
jpg $S/hero_right.png   site/media/viewer_poster.jpg 1400
jpg $S/organs.png       site/media/organs.jpg        1600
jpg $S/gpu.png          site/media/gpu.jpg           1600
jpg $S/hero_low.png     site/media/hero_low.jpg      1600

# film
[ -f build/film/film.mp4 ] && cp build/film/film.mp4 site/media/film.mp4 && cp build/film/film_poster.jpg site/media/film_poster.jpg && echo "  film copied"

# web models (meshopt-compressed)
if [ -f build/web_parts_raw.glb ]; then
  $GLTFPACK -i build/web_parts_raw.glb -o site/model/marrow.glb -cc -kn -km -vp 14 >/dev/null
  $GLTFPACK -i build/web_host_raw.glb -o site/model/marrow_host.glb -cc -kn -km -si 0.22 -vp 13 >/dev/null
  ls -la site/model
fi

# artifact page: the part of index.html between the PAGE markers, without the head/body wrapper
$PY - <<'PYEOF'
import pathlib, re
s = pathlib.Path('site/index.html').read_text()
body = s[s.index('<!--PAGE-->') + len('<!--PAGE-->'):s.index('<!--/PAGE-->')]
body = re.sub(r'<!--HEADEND-->.*?<!--/HEADEND-->', '', body, flags=re.S)
# that host serves web data types only: wrap each GLB as base64 in JSON and tell the viewer
import base64, json
out = pathlib.Path('build/artifact')
(out / 'model').mkdir(parents=True, exist_ok=True)
for name in ('marrow', 'marrow_host'):
    src = pathlib.Path('site/model') / (name + '.glb')
    if src.exists():
        (out / 'model' / (name + '.b64.json')).write_text(json.dumps({'b64': base64.b64encode(src.read_bytes()).decode()}))
flag = '<script>window.MARROW_MODEL_B64 = true;</script>\n'
i = body.index('<script src="js/app.js">')
body = body[:i] + flag + body[i:]
(out / 'marrow.html').write_text(body.strip() + '\n')
print('  artifact page', len(body), 'chars')
PYEOF
du -sh site
