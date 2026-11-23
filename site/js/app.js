/* MARROW site: binds simulation and part data into the page and draws the charts. No dependencies. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  function bind(key, text) { $$('[data-b="' + key + '"]').forEach(function (e) { e.textContent = text; }); }
  function f(v, d) { return (v === undefined || v === null || isNaN(v)) ? '–' : Number(v).toFixed(d === undefined ? 1 : d); }
  function el(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) { e.setAttribute(k, attrs[k]); }
    if (text !== undefined) { e.textContent = text; }
    if (parent) { parent.appendChild(e); }
    return e;
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  var PART_NAMES = {
    gpu_block: 'Organ I · 4090 block', monoblock: 'Organ II · CPU + VRM monoblock', heart: 'Organ III · pump top',
    m2_sink: 'Organ IV · M.2 / chipset sink', frame_pelvis: 'Pelvis · keel, feet, PSU cradle',
    frame_lumbar: 'Lumbar · lower front spine', frame_thoracic: 'Thoracic · upper front spine',
    frame_dorsal: 'Dorsal · lower rear spine', frame_cervical: 'Cervical · upper rear spine, crown'
  };
  var ORDER = ['gpu_block', 'monoblock', 'heart', 'm2_sink', 'frame_pelvis', 'frame_lumbar', 'frame_thoracic', 'frame_dorsal', 'frame_cervical'];
  var state = { load: 'full', rpm: 1500, data: null };

  function money(v) { return '$' + Math.round(v).toLocaleString('en-US'); }

  function parts(d) {
    var tb = $('#parts-table tbody');
    if (!tb || !d.parts) { return; }
    tb.innerHTML = '';
    var tot = { m: 0, a: 0, b: 0 }, steel = 0, alu = { a: 0, b: 0 }, longest = 0;
    ORDER.forEach(function (k) {
      var p = d.parts[k];
      if (!p) { return; }
      var tr = document.createElement('tr');
      var isSteel = p.material === '316L';
      var bb = p.bbox_mm.map(function (v) { return Math.round(v); }).join(' × ');
      tr.innerHTML = '<td>' + PART_NAMES[k] + '</td><td><span class="pill ' + (isSteel ? 'steel' : 'alu') + '">' + p.material + '</span></td>' +
        '<td class="n">' + Math.round(p.mass_g) + ' g</td><td class="n">' + bb + '</td>' +
        '<td>' + (p.fits_envelope ? '<span class="pill ok">fits</span>' : '<span class="pill warn">too long</span>') + '</td>' +
        '<td class="n">' + money(p.cost_usd_jlc_pcbway[0]) + '</td><td class="n">' + money(p.cost_usd_jlc_pcbway[1]) + '</td>';
      tb.appendChild(tr);
      tot.m += p.mass_g; tot.a += p.cost_usd_jlc_pcbway[0]; tot.b += p.cost_usd_jlc_pcbway[1];
      if (isSteel) { steel += p.mass_g; longest = Math.max(longest, p.longest_mm); } else { alu.a += p.cost_usd_jlc_pcbway[0]; alu.b += p.cost_usd_jlc_pcbway[1]; }
      bind('p.' + k + '.mass', Math.round(p.mass_g) + ' g ' + p.material);
      bind('p.' + k + '.bbox', bb + ' mm');
    });
    var tr = document.createElement('tr');
    tr.innerHTML = '<td><strong>All nine</strong></td><td></td><td class="n"><strong>' + f(tot.m / 1000, 2) + ' kg</strong></td><td></td><td></td><td class="n"><strong>' +
      money(tot.a) + '</strong></td><td class="n"><strong>' + money(tot.b) + '</strong></td>';
    tb.appendChild(tr);
    bind('steel_kg', f(steel / 1000, 1));
    bind('longest', String(Math.round(longest)));
    bind('alu_cost', money(alu.a) + '–' + money(alu.b));
    bind('all_cost', money(tot.a) + '–' + money(tot.b));
    if (d.layout && d.layout.envelope) {
      var e = d.layout.envelope;
      bind('env', Math.round(e.width) + ' × ' + Math.round(e.depth) + ' × ' + Math.round(e.height));
    }
  }

  function fea(d) {
    var s = d.sims && d.sims.fea, tb = $('#fea-table tbody');
    if (!s || !tb) { return; }
    tb.innerHTML = '';
    var names = ['Standing on its feet, 1 g', 'Standing, 1 g down plus 1 g sideways', 'Lifted by the crown, 1 g'];
    s.cases.forEach(function (c, i) {
      var tr = document.createElement('tr');
      tr.innerHTML = '<td>' + names[i] + '</td><td class="n">' + f(c.max_disp_mm, 2) + ' mm</td><td class="n">' + f(c.vm_max_smoothed_MPa, 0) +
        ' MPa</td><td class="n">' + f(c.safety_factor_yield, 0) + ' ×</td>';
      tb.appendChild(tr);
    });
    bind('fea_el', s.elements.toLocaleString('en-US'));
    bind('fea_mass', f(s.mass_components_kg, 1));
  }

  function macro(d) {
    var m = d.sims && d.sims.macro;
    if (!m) { return; }
    var g = m['gpu_block_AlSi10Mg_3'], gs = m['gpu_block_316L_3'], c = m['monoblock_AlSi10Mg_3'];
    if (g) {
      bind('gpu_rise', '+' + f(g.heaters['1'].T_mean_rise_K, 1) + ' K over coolant');
      bind('gpu_dp', f(g.dp_Pa / 1000, 1) + ' kPa at ' + f(g.flow_lpm, 0) + ' L/min');
      bind('tscale', '+' + f(d.sims.gpu_thermal_scale_K, 0) + ' K');
    }
    if (c) {
      bind('mono_rise', '+' + f(c.heaters['1'].T_mean_rise_K, 1) + ' K over coolant');
      bind('mono_dp', f(c.dp_Pa / 1000, 1) + ' kPa at ' + f(c.flow_lpm, 0) + ' L/min');
    }
    if (g && gs) {
      var a = g.heaters['1'].T_mean_rise_K, b = gs.heaters['1'].T_mean_rise_K;
      bind('steel_finding', 'The same block geometry was simulated in both metals at 3 L/min. In aluminium the die pad sits ' + f(a, 0) +
        ' K above the coolant. In 316L it sits ' + f(b, 0) + ' K above, which puts a 4090 past its limit before the radiators are even counted. ' +
        'Steel conducts about a ninth as well, and the card puts 390 W through six square centimetres. So steel carries the load and aluminium carries the heat.');
    }
    var s = d.sims.m2_sink;
    if (s) {
      var mid = s.cases[1];
      bind('m2_w', f(s.power_for_45K_W, 0) + ' W');
      bind('m2_rise', '+' + f(mid.rise_ssd_K, 0) + ' K at 13 W, which fails');
      var cp = s.coupled;
      if (cp) {
        bind('m2_ssd', f(cp.ssd_c, 0) + ' °C');
        bind('m2_chip', f(cp.chipset_c, 0) + ' °C');
        bind('m2_loop_share', f(100 * cp.q_to_loop_W / (cp.q_to_loop_W + cp.q_to_air_W), 0) + ' % of the heat');
      }
    }
  }

  /* ---------------- loop chart ---------------- */
  function loopRows(d, material, load) {
    var sw = (d.sims && d.sims.loop && d.sims.loop.sweep) || [];
    return sw.filter(function (r) { return r.material === material && r.load === load; }).sort(function (a, b) { return a.rpm - b.rpm; });
  }
  function interp(rows, rpm, key) {
    if (!rows.length) { return NaN; }
    for (var i = 0; i < rows.length - 1; i++) {
      if (rpm >= rows[i].rpm && rpm <= rows[i + 1].rpm) {
        var t = (rpm - rows[i].rpm) / (rows[i + 1].rpm - rows[i].rpm);
        return rows[i][key] + t * (rows[i + 1][key] - rows[i][key]);
      }
    }
    return rpm < rows[0].rpm ? rows[0][key] : rows[rows.length - 1][key];
  }
  function drawLoop() {
    var d = state.data, svg = $('#loop-chart');
    if (!d || !svg) { return; }
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    var al = loopRows(d, 'AlSi10Mg', state.load), st = loopRows(d, '316L', state.load);
    if (!al.length) { el('text', { x: 20, y: 40, fill: css('--ink-3'), 'font-size': 14 }, svg, 'Loop model results are not available.'); return; }
    var W = 900, H = 340, L = 54, R = 20, T = 14, B = 40;
    var ymin = 20, ymax = 130;
    var vals = al.concat(st);
    ymax = Math.min(160, Math.max(100, Math.ceil(Math.max.apply(null, vals.map(function (r) { return r.gpu_hotspot_c; })) / 10) * 10 + 10));
    var X = function (rpm) { return L + (rpm - 800) / 1200 * (W - L - R); };
    var Y = function (t) { return T + (ymax - t) / (ymax - ymin) * (H - T - B); };
    var ink3 = css('--ink-3'), rule = css('--rule'), ruby = css('--ruby'), alu = css('--alu'), steel = css('--steel'), warn = css('--warn');
    for (var t = ymin; t <= ymax; t += 20) {
      el('line', { x1: L, x2: W - R, y1: Y(t), y2: Y(t), stroke: rule, 'stroke-width': 1 }, svg);
      el('text', { x: L - 8, y: Y(t) + 4, 'text-anchor': 'end', fill: ink3, 'font-size': 12, 'font-family': 'IBM Plex Mono, monospace' }, svg, t + '°');
    }
    for (var r = 800; r <= 2000; r += 200) {
      el('text', { x: X(r), y: H - 14, 'text-anchor': 'middle', fill: ink3, 'font-size': 12, 'font-family': 'IBM Plex Mono, monospace' }, svg, String(r));
    }
    el('text', { x: W - R, y: H - 0, 'text-anchor': 'end', fill: ink3, 'font-size': 11, 'font-family': 'IBM Plex Mono, monospace' }, svg, 'fan speed, rpm');
    // 4090 thermal limit
    el('line', { x1: L, x2: W - R, y1: Y(88), y2: Y(88), stroke: warn, 'stroke-width': 1, 'stroke-dasharray': '2 5' }, svg);
    el('text', { x: W - R, y: Y(88) - 6, 'text-anchor': 'end', fill: warn, 'font-size': 11, 'font-family': 'IBM Plex Mono, monospace' }, svg, '4090 begins to throttle near 88 °C');
    function band(rows, key, color) {
      var up = rows.map(function (r) { return X(r.rpm) + ',' + Y(Math.min(ymax, r[key][1])); });
      var dn = rows.map(function (r) { return X(r.rpm) + ',' + Y(Math.min(ymax, r[key][0])); }).reverse();
      el('polygon', { points: up.concat(dn).join(' '), fill: color, opacity: 0.13 }, svg);
    }
    function line(rows, key, color, dash, w) {
      var pts = rows.map(function (r) { return X(r.rpm) + ',' + Y(Math.min(ymax, r[key])); }).join(' ');
      var a = { points: pts, fill: 'none', stroke: color, 'stroke-width': w || 2.2, 'stroke-linejoin': 'round' };
      if (dash) { a['stroke-dasharray'] = dash; }
      el('polyline', a, svg);
    }
    band(al, 'gpu_hotspot_band_c', ruby);
    line(al, 'coolant_max_c', steel, null, 1.6);
    line(al, 'cpu_tj_c', alu, null, 1.8);
    if (st.length) { line(st, 'gpu_hotspot_c', ruby, '7 5', 1.8); }
    line(al, 'gpu_hotspot_c', ruby, null, 2.6);
    var x = X(state.rpm);
    el('line', { x1: x, x2: x, y1: T, y2: H - B, stroke: css('--ink-2'), 'stroke-width': 1 }, svg);
    [['gpu_hotspot_c', ruby], ['cpu_tj_c', alu], ['coolant_max_c', steel]].forEach(function (p) {
      el('circle', { cx: x, cy: Y(Math.min(ymax, interp(al, state.rpm, p[0]))), r: 4.5, fill: p[1], stroke: css('--plate'), 'stroke-width': 2 }, svg);
    });
    $$('[data-r]').forEach(function (e) {
      var k = e.getAttribute('data-r');
      if (k === 'gpu') { e.textContent = f(interp(al, state.rpm, 'gpu_hotspot_c'), 0); }
      if (k === 'cpu') { e.textContent = f(interp(al, state.rpm, 'cpu_tj_c'), 0); }
      if (k === 'cool') { e.textContent = f(interp(al, state.rpm, 'coolant_max_c'), 0); }
      if (k === 'flow') { e.textContent = f(al[0].flow_lpm, 1) + ' L/min · ' + f(al[0].loop_dp_kpa, 0) + ' kPa'; }
    });
    var full1500 = loopRows(d, 'AlSi10Mg', 'full').filter(function (r) { return r.rpm === 1500; })[0];
    if (full1500) { bind('gpu_hot', f(full1500.gpu_hotspot_c, 0)); }
  }

  /* ---------------- closure charts (log-log) ---------------- */
  function drawFit(id, kind) {
    var d = state.data, svg = $(id);
    var lat = d && d.sims && d.sims.cell && d.sims.cell.lattices;
    if (!svg || !lat) { return; }
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    var W = 440, H = 280, L = 52, R = 14, T = 10, B = 40;
    var colors = { core: css('--ruby'), gallery: css('--steel'), arm: css('--alu') };
    var names = { core: '5 mm core', gallery: '8 mm gallery', arm: 'VRM arm' };
    var xs = [], ys = [], series = {};
    Object.keys(lat).forEach(function (k) {
      var pts = lat[k].runs.map(function (r) {
        return kind === 'dp' ? [r.U_superficial, r.dpdx_Pa_m / 1000] : (r.heat ? [r.heat.U_superficial, r.heat.ha_W_m3K / 1e6] : null);
      }).filter(Boolean);
      series[k] = pts;
      pts.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); });
    });
    if (!xs.length) { return; }
    var lx0 = Math.floor(Math.log10(Math.min.apply(null, xs)) * 2) / 2, lx1 = Math.log10(0.5);
    var ly0 = Math.floor(Math.log10(Math.min.apply(null, ys))), ly1 = Math.ceil(Math.log10(Math.max.apply(null, ys)) + 0.6);
    var X = function (v) { return L + (Math.log10(v) - lx0) / (lx1 - lx0) * (W - L - R); };
    var Y = function (v) { return T + (ly1 - Math.log10(v)) / (ly1 - ly0) * (H - T - B); };
    var ink3 = css('--ink-3'), rule = css('--rule');
    for (var e = ly0; e <= ly1; e++) {
      el('line', { x1: L, x2: W - R, y1: Y(Math.pow(10, e)), y2: Y(Math.pow(10, e)), stroke: rule }, svg);
      el('text', { x: L - 6, y: Y(Math.pow(10, e)) + 4, 'text-anchor': 'end', fill: ink3, 'font-size': 11, 'font-family': 'IBM Plex Mono, monospace' }, svg, String(Math.pow(10, e)));
    }
    [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5].forEach(function (v) {
      if (Math.log10(v) < lx0) { return; }
      el('text', { x: X(v), y: H - 20, 'text-anchor': 'middle', fill: ink3, 'font-size': 11, 'font-family': 'IBM Plex Mono, monospace' }, svg, String(v));
    });
    el('text', { x: W - R, y: H - 4, 'text-anchor': 'end', fill: ink3, 'font-size': 10.5, 'font-family': 'IBM Plex Mono, monospace' }, svg,
      'superficial speed, m/s   ·   ' + (kind === 'dp' ? 'kPa per metre' : 'MW/m³K'));
    var row = 0;
    Object.keys(series).forEach(function (k) {
      var c = colors[k] || ink3, L_ = lat[k];
      var path = [];
      for (var i = 0; i <= 40; i++) {
        var u = Math.pow(10, lx0 + (lx1 - lx0) * i / 40), v;
        if (kind === 'dp') { v = (L_.forchheimer.A_Pa_s_m2 * u + L_.forchheimer.B_Pa_s2_m3 * u * u) / 1000; }
        else if (L_.heat_fit) { v = L_.heat_fit.c * Math.pow(u, L_.heat_fit.n) / 1e6; }
        if (v > 0 && Math.log10(v) <= ly1 && Math.log10(v) >= ly0) { path.push(X(u) + ',' + Y(v)); }
      }
      el('polyline', { points: path.join(' '), fill: 'none', stroke: c, 'stroke-width': 1.6 }, svg);
      series[k].forEach(function (p) { el('circle', { cx: X(p[0]), cy: Y(p[1]), r: 3.6, fill: css('--plate'), stroke: c, 'stroke-width': 1.8 }, svg); });
      el('text', { x: L + 10, y: T + 16 + row * 16, fill: c, 'font-size': 11.5, 'font-family': 'IBM Plex Mono, monospace' }, svg, '○ ' + (names[k] || k));
      row++;
    });
  }

  function validation(d) {
    var v = d.sims && d.sims.validation, tb = $('#val-table tbody');
    if (!tb) { return; }
    tb.innerHTML = '';
    function row(a, b, c, e) {
      var tr = document.createElement('tr');
      tr.innerHTML = '<td class="wraps">' + a + '</td><td class="wraps">' + b + '</td><td class="n">' + c + '</td><td class="wraps">' + e + '</td>';
      tb.appendChild(tr);
    }
    if (v) {
      var p13 = v.poiseuille.filter(function (r) { return r.cells_across === 13 && r.smagorinsky === 0; })[0];
      var p13s = v.poiseuille.filter(function (r) { return r.cells_across === 13 && r.smagorinsky > 0; })[0];
      var h13 = v.plates_heat.filter(function (r) { return r.cells_across === 13; });
      if (p13) { row('Flow between plates, 13 cells across (the lattice channel width)', 'Analytic Poiseuille flow', (p13.error_pct > 0 ? '+' : '') + f(p13.error_pct, 2) + ' %', 'Flow solver is accurate at the resolution used.'); }
      if (p13s) { row('Same, with the sub-grid model switched on', 'Analytic Poiseuille flow', f(p13s.error_pct, 1) + ' %', 'The sub-grid model costs a few percent in laminar flow, so it is only used for the fastest runs.'); }
      if (h13.length) { row('Heat transfer between isothermal plates', 'Nu = 7.54 fully developed; Hausen-type correlation while developing', f(h13[0].error_pct, 1) + ' to ' + f(h13[h13.length - 1].error_pct, 1) + ' %', 'Heat transfer is under-predicted, which is the safe direction.'); }
    }
    var m = d.sims && d.sims.macro;
    if (m) {
      Object.keys(m).forEach(function (k) {
        if (k.indexOf('AlSi10Mg_3') < 0) { return; }
        row('Energy balance, ' + k.split('_AlSi')[0].replace('_', ' '), 'Heat in = coolant enthalpy out', (m[k].energy_closure_pct > 0 ? '+' : '') + f(m[k].energy_closure_pct, 2) + ' %',
          'Block model conserves energy to solver tolerance.');
      });
    }
    var fe = d.sims && d.sims.fea;
    if (fe) { row('Frame FEA, reaction at the feet', 'Weight of frame plus hardware, ' + f(fe.cases[0].applied_N[2] * -1, 1) + ' N', f(-fe.cases[0].reaction_N[2], 1) + ' N', 'Static equilibrium holds.'); }
  }

  function wire() {
    $$('[data-load]').forEach(function (b) {
      b.addEventListener('click', function () {
        state.load = b.getAttribute('data-load');
        $$('[data-load]').forEach(function (o) { o.setAttribute('aria-pressed', String(o === b)); });
        drawLoop();
      });
    });
    var rpm = $('#rpm');
    if (rpm) { rpm.addEventListener('input', function () { state.rpm = Number(rpm.value); $('#rpm-v').textContent = rpm.value; drawLoop(); }); }
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var redraw = function () { drawLoop(); drawFit('#dp-chart', 'dp'); drawFit('#ha-chart', 'ha'); };
    if (mq.addEventListener) { mq.addEventListener('change', redraw); }
    new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  wire();
  fetch('data/site.json').then(function (r) { return r.json(); }).then(function (d) {
    state.data = d;
    window.MARROW_DATA = d;
    parts(d); fea(d); macro(d); validation(d);
    $$('[data-load]').forEach(function (b) {
      var lk = b.getAttribute('data-load');
      var row = loopRows(d, 'AlSi10Mg', lk)[0];
      if (row) { b.textContent = (lk === 'full' ? 'Full load ' : 'Gaming ') + f(row.heat_w, 0) + ' W'; }
    });
    drawLoop(); drawFit('#dp-chart', 'dp'); drawFit('#ha-chart', 'ha');
  }).catch(function (e) { console.warn('site data unavailable', e); });
})();
