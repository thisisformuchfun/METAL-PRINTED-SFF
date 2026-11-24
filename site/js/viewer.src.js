// MARROW 3-D viewer. Bundled with esbuild into viewer.bundle.js (three.js r160 included).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const stage = document.getElementById('stage');
const btn = document.getElementById('load3d');
const poster = document.getElementById('poster');
const hint = document.getElementById('hint');
const note = document.getElementById('viewer-note');
const LAYERS = ['steel', 'alu', 'loop', 'host'];
const S = { mode: 'material', explode: 0, groups: {}, base: {}, flow: null, mats: {}, ready: false, t: 0 };
// show-plane normal (towards the viewer) in the exported Y-up frame: world (0,-cos25,sin25) -> (x, z, -y)
const EN = new THREE.Vector3(0, Math.sin(25 * Math.PI / 180), Math.cos(25 * Math.PI / 180));
const OFFS = { steel: -0.07, alu: 0.07, loop: 0.16, host: 0 };

// The hardware's render materials are procedural, so their colours do not survive the glTF export.
// Restore the ones that matter by name; give the rest a sensible roughness.
const FIX = [
  [/pcb_black/, { color: 0x0b0e12, roughness: 0.5, metalness: 0 }],
  [/pcb_(dark)?green/, { color: 0x0c2a1d, roughness: 0.5, metalness: 0 }],
  [/sleeve_black|ribbon_black|wire_black/, { color: 0x0b0b0d, roughness: 0.85, metalness: 0 }],
  [/mlcc_tan/, { color: 0x8f6f4c, roughness: 0.6, metalness: 0 }],
  [/mlcc_grey/, { color: 0x5d5f63, roughness: 0.6, metalness: 0 }],
  [/inductor_grey/, { color: 0x3a3d42, roughness: 0.7, metalness: 0.2 }],
  [/silicon_die/, { color: 0x1d2128, roughness: 0.12, metalness: 1 }],
  [/coolant_ruby/, { color: 0xc10b24, roughness: 0.18, metalness: 0, transmission: 0, transparent: true, opacity: 0.92 }],
  [/pvc_clear/, { color: 0xdfe8ee, roughness: 0.06, metalness: 0, transmission: 0, transparent: true, opacity: 0.16, depthWrite: false }],
];
function fixMaterial(m) {
  if (!m || m.userData.fixed) { return; }
  m.userData.fixed = true;
  let hit = false;
  FIX.forEach(([re, props]) => {
    if (!re.test(m.name)) { return; }
    hit = true;
    Object.keys(props).forEach((k) => { if (k === 'color') { m.color.setHex(props.color); } else { m[k] = props[k]; } });
  });
  if (!hit && m.roughness >= 0.99) { m.roughness = m.metalness > 0.7 ? 0.34 : 0.62; }
  if ('sheen' in m) { m.sheen = 0; }
  if ('clearcoat' in m) { m.clearcoat = Math.min(m.clearcoat, 0.3); }
  m.vertexColors = false;
  m.needsUpdate = true;
}

function inferno(t) {
  const c = [[0, 0, 4], [66, 10, 104], [147, 38, 103], [221, 81, 58], [252, 165, 10], [252, 255, 164]];
  const x = Math.min(0.9999, Math.max(0, t)) * 5, i = Math.floor(x), f = x - i;
  return [0, 1, 2].map(k => (c[i][k] + (c[i + 1][k] - c[i][k]) * f) / 255);
}

function start() {
  btn.disabled = true;
  btn.textContent = 'Loading…';
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  } catch (e) {
    btn.textContent = '3D needs WebGL';
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  stage.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  const camera = new THREE.PerspectiveCamera(32, 4 / 3, 0.02, 20);
  camera.position.set(-0.62, 0.42, 0.92);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.185, -0.16);
  controls.enableDamping = true;
  controls.minDistance = 0.25;
  controls.maxDistance = 2.2;
  const key = new THREE.DirectionalLight(0xfff0e0, 1.6);
  key.position.set(-1.2, 1.6, 1.4);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xbfd8ff, 1.2);
  rim.position.set(1.4, 0.8, -1.6);
  scene.add(rim);

  let dirty = true;
  const invalidate = () => { dirty = true; };
  controls.addEventListener('change', invalidate);
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    invalidate();
  }
  new ResizeObserver(resize).observe(stage);

  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  function adopt(gltf) {
    scene.add(gltf.scene);
    LAYERS.forEach((n) => {
      const g = gltf.scene.getObjectByName(n);
      if (g && !S.groups[n]) { S.groups[n] = g; S.base[n] = g.position.clone(); }
    });
    gltf.scene.traverse((o) => {
      if (!o.isMesh) { return; }
      let layer = null, p = o;
      while (p) { if (LAYERS.includes(p.name)) { layer = p.name; break; } p = p.parent; }
      o.userData.layer = layer;
      o.userData.hasHeat = layer === 'alu' && !!o.geometry.getAttribute('color');
      if (layer === 'alu') { o.material = S.mats.alu; }
      if (layer === 'steel') { o.material = S.mats.steel; }
      if (layer === 'host' || layer === 'loop') {
        (Array.isArray(o.material) ? o.material : [o.material]).forEach(fixMaterial);
      }
      o.userData.orig = o.material;
    });
  }
  // the hardware the chassis carries is a second, heavier file: show the printed parts first
  // Hosts that only serve web data types get the same GLB wrapped as base64 in JSON (window.MARROW_MODEL_B64).
  function loadModel(name, onLoad, onErr) {
    if (!window.MARROW_MODEL_B64) { loader.load('model/' + name + '.glb', onLoad, undefined, onErr); return; }
    fetch('model/' + name + '.b64.json').then((r) => r.json()).then((j) => {
      const bin = atob(j.b64), buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) { buf[i] = bin.charCodeAt(i); }
      loader.parse(buf.buffer, '', onLoad, onErr);
    }).catch(onErr);
  }
  const loadHost = () => loadModel('marrow_host', (g2) => { adopt(g2); apply(); }, () => {});
  loadModel('marrow', (gltf) => {
    const alu = new THREE.MeshStandardMaterial({ color: 0xcdd4da, metalness: 1, roughness: 0.42 });
    const steel = new THREE.MeshStandardMaterial({ color: 0x77828d, metalness: 1, roughness: 0.3 });
    const heat = new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, metalness: 0, roughness: 0.75 });
    const ghost = new THREE.MeshStandardMaterial({ color: 0xcfd8df, metalness: 0.2, roughness: 0.5, transparent: true, opacity: 0.13, depthWrite: false });
    const dim = new THREE.MeshStandardMaterial({ color: 0x1a232c, metalness: 0.1, roughness: 0.8, transparent: true, opacity: 0.22, depthWrite: false });
    S.mats = { alu, steel, heat, ghost, dim };
    adopt(gltf);
    loadHost();
    poster.hidden = true;
    btn.hidden = true;
    hint.hidden = false;
    S.ready = true;
    resize();
    apply();
  }, () => { btn.textContent = 'Model failed to load'; });

  function buildFlow(data) {
    const grp = new THREE.Group();
    const pos = [], col = [], uu = [];
    let tmax = 1e-6;
    const packs = [];
    Object.keys(data).forEach((k) => {
      const d = data[k], bin = atob(d.b64), buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) { buf[i] = bin.charCodeAt(i); }
      const a = new Float32Array(buf.buffer);
      packs.push({ a, counts: d.counts, stride: d.stride });
      for (let i = 3; i < a.length; i += d.stride) { tmax = Math.max(tmax, a[i]); }
    });
    packs.forEach(({ a, counts, stride }) => {
      let o = 0;
      counts.forEach((n) => {
        let len = 0;
        for (let i = 0; i < n - 1; i++) {
          const p = (o + i) * stride, q = (o + i + 1) * stride;
          // world mm (X right, Y back, Z up) -> viewer metres (x, z, -y)
          const ax = a[p] * 1e-3, ay = a[p + 2] * 1e-3, az = -a[p + 1] * 1e-3;
          const bx = a[q] * 1e-3, by = a[q + 2] * 1e-3, bz = -a[q + 1] * 1e-3;
          const seg = Math.hypot(bx - ax, by - ay, bz - az);
          const ca = inferno(0.25 + 0.75 * a[p + 3] / tmax), cb = inferno(0.25 + 0.75 * a[q + 3] / tmax);
          pos.push(ax, ay, az, bx, by, bz);
          col.push(ca[0], ca[1], ca[2], cb[0], cb[1], cb[2]);
          uu.push(len, len + seg);
          len += seg;
        }
        o += n;
      });
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('u', new THREE.Float32BufferAttribute(uu, 1));
    const m = new THREE.ShaderMaterial({
      uniforms: { time: { value: 0 } }, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      vertexShader: 'attribute float u; attribute vec3 color; varying float vu; varying vec3 vc; void main(){ vu=u; vc=color; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
      fragmentShader: 'uniform float time; varying float vu; varying vec3 vc; void main(){ float p=fract(vu*14.0-time*0.9); float k=0.28+0.72*smoothstep(0.55,1.0,p); gl_FragColor=vec4(vc*k*1.5, 0.9*k); }'
    });
    grp.add(new THREE.LineSegments(g, m));
    grp.userData.mat = m;
    return grp;
  }

  function apply() {
    if (!S.ready) { return; }
    const m = S.mats;
    scene.traverse((o) => {
      if (!o.isMesh || !o.userData.layer) { return; }
      const L = o.userData.layer;
      if (S.mode === 'material') { o.material = o.userData.orig; }
      else if (S.mode === 'thermal') { o.material = (L === 'alu' && o.userData.hasHeat) ? m.heat : (L === 'alu' ? m.ghost : m.dim); }
      else { o.material = (L === 'alu') ? m.ghost : m.dim; }
    });
    LAYERS.forEach((n) => {
      const g = S.groups[n];
      if (!g) { return; }
      const cb = document.querySelector('[data-layer="' + n + '"]');
      g.visible = !cb || cb.checked;
      g.position.copy(S.base[n]).addScaledVector(EN, OFFS[n] * S.explode);
    });
    if (S.flow) {
      S.flow.visible = S.mode === 'flow';
      S.flow.position.copy(EN).multiplyScalar(OFFS.alu * S.explode);
    }
    document.getElementById('thermal-legend').hidden = S.mode !== 'thermal';
    note.textContent = S.mode === 'thermal'
      ? 'Simulated metal temperature of the two cold plates at full load, as a rise over the coolant entering each block.'
      : S.mode === 'flow' ? 'Mean coolant paths from the block-level model, coloured by how much the coolant has warmed.'
        : 'The web model is simplified; the print files carry the full lattice.';
    invalidate();
  }
  S.apply = apply;

  document.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => {
    S.mode = b.getAttribute('data-mode');
    document.querySelectorAll('[data-mode]').forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
    if (S.mode === 'flow' && !S.flow && !S.flowLoading) {
      S.flowLoading = true;
      fetch('data/flow.json').then((r) => r.json()).then((d) => { S.flow = buildFlow(d); scene.add(S.flow); apply(); }).catch(() => {});
    }
    apply();
  }));
  document.querySelectorAll('[data-layer]').forEach((c) => c.addEventListener('change', apply));
  document.getElementById('explode').addEventListener('input', (e) => { S.explode = Number(e.target.value) / 100; apply(); });

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const clock = new THREE.Clock();
  (function loop() {
    requestAnimationFrame(loop);
    const dt = clock.getDelta();
    controls.update();
    const anim = S.flow && S.flow.visible && !reduce;
    if (anim) { S.t += dt; S.flow.userData.mat.uniforms.time.value = S.t; }
    if (dirty || anim) { renderer.render(scene, camera); dirty = false; }
  })();
}

if (btn) { btn.addEventListener('click', start, { once: true }); }
