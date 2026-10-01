/* Isotope fingerprints — the client.
 *
 * Two rules run through all of this.
 *
 * 1. The server holds the state. Every control change posts to /api/state and
 *    the response *is* the authoritative document, which is then re-applied to
 *    every control. So a value the server clamped or coerced shows up
 *    immediately instead of leaving a control disagreeing with the analysis.
 *
 * 2. Molecule-derived text — keys, SMILES, formulas, and synonyms that can come
 *    from PubChem — is inserted with textContent, never by building an HTML
 *    string. The only innerHTML in this file takes RDKit's SVG, which the
 *    server generated from our own numeric arrays.
 */
'use strict';

const $ = (id) => document.getElementById(id);

/* Only one request is in flight at a time. The server answers 409 when it is
 * busy, but suppressing at source means that is a safety net rather than the
 * normal path. */
let inFlight = null;
let state = null;

/* Counted, not a flag. setState() posts and then renders, so a flag would drop
 * to '0' between the two and anything waiting for the app to settle could pass
 * in that gap. */
let pending = 0;
function markBusy() { document.body.dataset.busy = pending ? '1' : '0'; }

async function api(method, path, body) {
  if (inFlight) { await inFlight.catch(() => {}); }
  pending += 1;
  markBusy();
  const controller = new AbortController();
  // fetch has no default timeout, so a hung request would disable the UI forever.
  const timer = setTimeout(() => controller.abort(), 60000);
  const run = (async () => {
    try {
      const response = await fetch(path, {
        method,
        signal: controller.signal,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      const payload = await response.json();
      if (!response.ok) { throw new Error(payload?.error?.message || response.statusText); }
      return payload;
    } finally {
      clearTimeout(timer);
      pending -= 1;
      markBusy();
      inFlight = null;
    }
  })();
  inFlight = run;
  return run;
}

let toastTimer = null;
function toast(message) {
  const node = $('toast');
  node.textContent = message;      // untrusted: server text, possibly data-derived
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 6000);
}

/* Crash reporting. In a normal install POST /api/_client_error is a 404 and
 * this does nothing; started with --dev the server prints the report to the
 * terminal, which is the only place a browser-side error is otherwise
 * invisible. Bounded, so one bad loop cannot flood the log. */
const MAX_REPORTS = 20;
let reported = 0;
function report(kind, message, extra) {
  if (reported >= MAX_REPORTS) { return; }
  reported += 1;
  fetch('/api/_client_error', {
    method: 'POST',
    keepalive: true,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(Object.assign({ kind, message: String(message) }, extra || {})),
  }).catch(() => {});
}

window.addEventListener('error', (event) => report('error', event.message, {
  source: event.filename,
  line: event.lineno,
  column: event.colno,
  stack: event.error && event.error.stack ? String(event.error.stack).slice(0, 2000) : '',
}));

window.addEventListener('unhandledrejection', (event) => {
  const reason = event.reason;
  report('rejection', (reason && reason.message) || reason, {
    stack: reason && reason.stack ? String(reason.stack).slice(0, 2000) : '',
  });
});

const fail = (error) => {
  // Toasts are legitimate UX, not faults — but a transcript of what the user
  // was told is the narrative that makes a bug report readable.
  report('toast', error.message || String(error), {
    stack: error && error.stack ? String(error.stack).slice(0, 2000) : '',
  });
  toast(error.message || String(error));
};

function clear(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) { node.textContent = String(text); }
  if (className) { node.className = className; }
  return node;
}

/* ------------------------------------------------------------------- state */

// Each write and the render it triggers finish before the next write starts.
// api() alone orders single requests, but several callers waiting on the same
// in-flight request resume together, so their writes could interleave.
let stateWrites = Promise.resolve();

function setState(patch) {
  const run = stateWrites.then(async () => {
    try { applyState(await api('POST', '/api/state', patch)); await render(); }
    catch (error) { fail(error); }
  });
  stateWrites = run;
  return run;
}

function fillSelect(select, options, current) {
  clear(select);
  for (const option of options) {
    const node = el('option', option.label);
    node.value = option.value;
    if (String(option.value) === String(current)) { node.selected = true; }
    select.appendChild(node);
  }
}

//: The option value that means "not one of the stored temperatures".
const CUSTOM_TEMPERATURE = 'custom';

// 273.15 and 298.15 are the ones a chemist reaches for; say why.
const TEMPERATURE_NOTES = { 273.15: '0 °C', 298.15: '25 °C' };

function temperatureLabel(t) {
  const note = TEMPERATURE_NOTES[t];
  return note ? `${t} K (${note})` : `${t} K`;
}

function applyState(next) {
  state = next;
  if (!next.ready) { return; }
  const settings = next.settings;

  fillSelect($('dataset'), next.datasets.map((d) => ({ value: d, label: d })), next.dataset);
  $('dataset-subsample').textContent = next.subsample || '';
  $('dataset-subsample').hidden = !next.subsample;
  fillSelect($('element'),
    [{ value: '', label: 'all elements' }].concat(next.elements.map((e) => ({ value: e, label: e }))),
    settings.element ?? '');
  fillSelect($('kind'), next.quantities.map((q) => ({ value: q.value, label: q.label })), settings.kind);
  fillSelect($('reference'),
    next.reference_modes.map((r) => ({ value: r.value, label: r.label })), settings.reference_mode);

  // Absent, not disabled: a disabled control asks "why can't I click this?",
  // an absent one asks nothing. Beta needs no reference at all.
  $('reference-field').hidden = !next.needs_reference;
  $('anchor-field').hidden = !(next.needs_reference && settings.reference_mode === 'self');
  fillSelect($('anchor'),
    (next.anchor_options.a || []).map((o) => ({ value: o.value, label: o.label })),
    settings.anchor_atom_index);

  // The stored temperatures are a closed list, so they are a list. Anything
  // else is a different kind of answer -- fitted rather than read -- and asks
  // for a choice of fit, so it gets its own field rather than hiding inside
  // the same box.
  fillSelect($('temperature'),
    next.temperatures.map((t) => ({ value: String(t), label: temperatureLabel(t) }))
      .concat([{ value: CUSTOM_TEMPERATURE, label: 'another temperature…' }]),
    next.on_grid ? String(settings.temperature) : CUSTOM_TEMPERATURE);

  $('temperature-custom-field').hidden = next.on_grid;
  if (!next.on_grid) { $('temperature-custom').value = String(settings.temperature); }

  $('show_hydrogens').checked = !!settings.show_hydrogens;
  $('show_labels').checked = !!settings.show_labels;

  // The interpolant decides what happens *between* stored temperatures and
  // nothing else, so it appears only when the temperature is between them.
  // Absent rather than disabled: a disabled control asks "why can't I click
  // this?", an absent one asks nothing.
  fillSelect($('beta_method'),
    next.beta_methods.map((m) => ({ value: m.value, label: m.label })), settings.beta_method);
  $('beta-method-field').hidden = next.on_grid;
  $('top_n').value = String(settings.top_n);
  $('dataset-count').textContent = next.status || '';

  for (const button of document.querySelectorAll('#scope button')) {
    button.classList.toggle('on', (button.dataset.scope === 'results') === !!settings.use_match_pool);
  }
}

/* ------------------------------------------------------------------ render */

async function render() {
  if (!state?.ready) { return; }
  if (!state.slots.a) { showEmptyStage(); return; }
  try {
    draw(await api('GET', '/view/render'));
  } catch (error) {
    showEmptyStage(error.message);
  }
  if (currentPanel() === 'ranked') { await loadRanks(); }
  if (currentPanel() === 'quality') { await loadQc(); }
  if (currentPanel() === 'derived') { await loadDerived(); }
}

function showEmptyStage(message) {
  clear($('panes'));
  $('legend').textContent = '';
  $('site-table').textContent = '';
  const note = el('p', message || 'Search for a molecule, or browse by functional group.', 'muted');
  $('panes').appendChild(note);
}

function draw(fragment) {
  const panes = $('panes');
  clear(panes);

  for (const pane of fragment.panes) {
    const box = el('div', null, 'pane');
    box.appendChild(el('h4', pane.label));           // data-derived: textContent

    const figures = el('div', null, 'figures');
    if (pane.viewer_url) {
      const plate = el('div', null, 'plate');
      const frame = document.createElement('iframe');
      frame.src = pane.viewer_url;
      frame.width = 420; frame.height = 340;
      frame.title = '3D structure';
      // The viewer runs 3Dmol.js, so it needs scripts — and nothing else.
      frame.setAttribute('sandbox', 'allow-scripts allow-same-origin');
      plate.appendChild(frame);
      figures.appendChild(plate);
    }
    if (pane.svg) {
      const plate = el('div', null, 'plate');
      plate.innerHTML = pane.svg;   // trusted: RDKit output, built from our arrays
      figures.appendChild(plate);
    }
    box.appendChild(figures);

    if (pane.depiction_note) {
      box.appendChild(notice(pane.depiction_note, 'warn'));
    }
    panes.appendChild(box);
    if (pane.slot === 'a') { $('site-table').textContent = pane.table; }
  }

  $('legend').textContent = [fragment.caption, fragment.caption_b].filter(Boolean).join('   ·   ');

  const provenance = $('provenance');
  provenance.textContent = fragment.provenance.text;
  provenance.dataset.level = fragment.provenance.level;

  const notices = $('notices');
  clear(notices);
  for (const warning of fragment.warnings) { notices.appendChild(notice(warning, 'warn')); }
  if (fragment.note_b) { notices.appendChild(notice(fragment.note_b, 'warn')); }
}

function notice(text, level) {
  const node = el('p', text, 'notice');
  node.dataset.level = level || 'info';
  return node;
}

/* ------------------------------------------------------------------ search */

// The QC checkbox is sugar for a query token, not a second filtering path:
// appending it here means the grammar in io/search stays the only thing that
// decides what a filtered search means.
function withFilters(query) {
  return $('qc_pass').checked ? query + ', qc:pass' : query;
}

async function search() {
  const query = $('q').value.trim();
  if (!query) { return; }
  try { showResults(await api('POST', '/api/search', { query: withFilters(query) })); }
  catch (error) { fail(error); }
}

function showResults(payload) {
  const list = $('results');
  clear(list);

  $('results-head').textContent = payload.shown === 1
    ? '1 result'
    : `${payload.shown} results  ·  ${payload.elapsed_ms} ms`;

  for (const hit of payload.hits) {
    const item = el('li');
    item.appendChild(el('div', hit.key, 'key'));
    const detail = [hit.formula, hit.smiles].filter(Boolean).join('  ');
    if (detail) { item.appendChild(el('div', detail, 'mono')); }
    if (hit.names.length) { item.appendChild(el('div', hit.names.join(', '))); }
    item.appendChild(el('div', hit.why, 'why'));
    item.onclick = () => load(hit.key, item);
    list.appendChild(item);
  }

  // No cursor in the index and no relevance ranking, so say what was left out
  // rather than offering a "more" button that pages through arbitrary order.
  $('results-note').textContent = payload.capped
    ? `First ${payload.shown} in file order. Narrow the search to see the rest.`
    : (payload.shown === 0 ? 'Nothing matched. Try a functional group or a molecule ID.' : '');

  if (payload.hits.length) { load(payload.hits[0].key, list.firstChild); }
}

/* A subsample holds whichever molecules it holds, and small famous ones are
 * usually not among them -- on the published 1% tier "ethanol" matches nothing.
 * The server already computes a few keys that will actually render, so offer
 * those rather than leaving a first-time user to guess and conclude the search
 * is broken. */
async function loadExamples() {
  const target = $('examples');
  let payload;
  try { payload = await api('GET', '/api/examples'); }
  catch { return; }               // a missing example list is not worth a toast
  if (!payload.keys.length) { return; }

  clear(target);
  target.appendChild(document.createTextNode('Try: '));
  payload.keys.slice(0, 4).forEach((key, i) => {
    if (i) { target.appendChild(document.createTextNode(' \u00b7 ')); }
    const button = el('button', key, 'linky');
    button.type = 'button';
    button.onclick = () => { $('q').value = key; search(); };
    target.appendChild(button);
  });
  target.hidden = false;
}

async function load(key, item) {
  for (const node of document.querySelectorAll('#results li')) { node.classList.remove('on'); }
  if (item) { item.classList.add('on'); }
  try { applyState(await api('POST', '/api/load', { key })); await render(); }
  catch (error) { fail(error); }
}

/* ------------------------------------------------------------------ charts */

/* Charts are built here as inline SVG rather than pulled from a library: this
 * app has no build step, and an in-document <svg> inherits the CSS variables,
 * so light and dark are two selected palettes rather than one baked-in image
 * flipped. The server sends numbers; nothing here is ever inserted as markup. */

const SVG_NS = 'http://www.w3.org/2000/svg';
const SERIES_SLOTS = 6;

function svgEl(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    node.setAttribute(key, String(value));
  }
  return node;
}

const seriesColor = (i) => `var(--series-${(i % SERIES_SLOTS) + 1})`;

function niceTicks(lo, hi, count) {
  if (!(hi > lo)) { return [lo]; }
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) { out.push(v); }
  return out;
}

const fmtTick = (v) => (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3)
  ? v.toExponential(0) : String(Number(v.toPrecision(4))));

/* One line chart. `series` is [{label, temperatures, values, curve_*}]; a series
 * with curve points draws them as the line and the stored points as markers,
 * and one without draws markers only — which is how mass law says "these are
 * the only temperatures where this exists" rather than implying a fit. */
function lineChart(series, options) {
  const opts = Object.assign({ log: false, yLabel: '', xLabel: 'temperature (K)' }, options);
  const W = 640, H = 290;
  const m = { top: 12, right: 54, bottom: 34, left: 58 };
  const plotW = W - m.left - m.right, plotH = H - m.top - m.bottom;

  const allX = [], allY = [];
  for (const s of series) {
    allX.push(...s.temperatures, ...(s.curve_temperatures || []));
    allY.push(...s.values, ...(s.curve_values || []));
  }
  const finite = (xs) => xs.filter((v) => Number.isFinite(v));
  const xs = finite(allX);
  let ys = finite(allY);
  if (opts.log) { ys = ys.filter((v) => v > 0); }
  if (!xs.length || !ys.length) { return null; }

  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const tf = opts.log ? Math.log10 : (v) => v;
  let y0 = Math.min(...ys.map(tf)), y1 = Math.max(...ys.map(tf));
  if (y1 - y0 < 1e-9) { y0 -= 0.5; y1 += 0.5; }    // a flat series still needs a band
  const pad = (y1 - y0) * 0.06;
  y0 -= pad; y1 += pad;

  const sx = (v) => m.left + ((v - x0) / (x1 - x0 || 1)) * plotW;
  const sy = (v) => m.top + plotH - ((tf(v) - y0) / (y1 - y0 || 1)) * plotH;

  const svg = svgEl('svg', {
    class: 'chart', viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': opts.yLabel + ' against temperature',
  });

  // Recessive solid hairlines — never dashed, which would read as a threshold.
  // On a log axis step in whole decades: half-decades print as 3.162, which is
  // a number nobody is looking for.
  const yTicks = opts.log && (y1 - y0) >= 1.5
    ? (() => {
      const out = [];
      for (let d = Math.ceil(y0); d <= Math.floor(y1); d += 1) { out.push(d); }
      return out.length >= 2 ? out : niceTicks(y0, y1, 5);
    })()
    : niceTicks(y0, y1, 5);
  for (const v of yTicks) {
    const y = m.top + plotH - ((v - y0) / (y1 - y0 || 1)) * plotH;
    if (y < m.top || y > m.top + plotH) { continue; }
    svg.appendChild(svgEl('line', { class: 'grid', x1: m.left, x2: m.left + plotW, y1: y, y2: y }));
    const label = opts.log ? fmtTick(10 ** v) : fmtTick(v);
    const text = svgEl('text', { class: 'tick', x: m.left - 6, y: y + 3, 'text-anchor': 'end' });
    text.textContent = label;
    svg.appendChild(text);
  }
  for (const v of niceTicks(x0, x1, 6)) {
    if (v < x0 || v > x1) { continue; }
    const text = svgEl('text', {
      class: 'tick', x: sx(v), y: m.top + plotH + 15, 'text-anchor': 'middle',
    });
    text.textContent = fmtTick(v);
    svg.appendChild(text);
  }
  svg.appendChild(svgEl('line', {
    class: 'axis', x1: m.left, x2: m.left + plotW, y1: m.top + plotH, y2: m.top + plotH,
  }));

  const xTitle = svgEl('text', {
    class: 'axis-label', x: m.left + plotW / 2, y: H - 4, 'text-anchor': 'middle',
  });
  xTitle.textContent = opts.xLabel;
  svg.appendChild(xTitle);
  const yTitle = svgEl('text', {
    class: 'axis-label', x: 12, y: m.top + plotH / 2,
    'text-anchor': 'middle', transform: `rotate(-90 12 ${m.top + plotH / 2})`,
  });
  yTitle.textContent = opts.yLabel;
  svg.appendChild(yTitle);

  series.forEach((s, i) => {
    const color = seriesColor(i);
    const curveX = s.curve_temperatures || [], curveY = s.curve_values || [];
    if (curveX.length > 1) {
      const d = curveX.map((t, k) => {
        const v = curveY[k];
        if (!Number.isFinite(v) || (opts.log && v <= 0)) { return null; }
        return `${sx(t).toFixed(2)},${sy(v).toFixed(2)}`;
      }).filter(Boolean);
      if (d.length > 1) {
        svg.appendChild(svgEl('polyline', { class: 'line', stroke: color, points: d.join(' ') }));
      }
    } else if (s.temperatures.length > 1) {
      // No fit exists, so the connector is a plain hairline between stored
      // points and the markers carry the meaning.
      const pts = s.temperatures.map((t, k) => `${sx(t).toFixed(2)},${sy(s.values[k]).toFixed(2)}`);
      svg.appendChild(svgEl('polyline', {
        class: 'line', stroke: color, 'stroke-dasharray': '', points: pts.join(' '),
      }));
    }
    s.temperatures.forEach((t, k) => {
      const v = s.values[k];
      if (!Number.isFinite(v) || (opts.log && v <= 0)) { return; }
      svg.appendChild(svgEl('circle', {
        class: 'marker', cx: sx(t), cy: sy(v), r: 4, fill: color,
      }));
    });
  });

  // Direct-label the endpoints, but only where they do not sit on top of each
  // other. Sites of one element converge at high temperature, so on those
  // charts the labels would overprint into an unreadable smudge -- the legend
  // already carries identity, so dropping the collided ones loses nothing.
  if (series.length <= 4) {
    const ends = series.map((s, i) => {
      const last = s.temperatures.length - 1;
      const v = last >= 0 ? s.values[last] : NaN;
      if (!Number.isFinite(v) || (opts.log && v <= 0)) { return null; }
      return { i, label: s.label, x: sx(s.temperatures[last]), y: sy(v) };
    }).filter(Boolean).sort((a, b) => a.y - b.y);

    const MIN_GAP = 11;
    const collides = ends.some((e, k) => k > 0 && Math.abs(e.y - ends[k - 1].y) < MIN_GAP);
    if (!collides) {
      for (const end of ends) {
        const tag = svgEl('text', { class: 'endpoint', x: end.x + 7, y: end.y + 3 });
        tag.textContent = end.label;
        svg.appendChild(tag);
      }
    }
  }

  return { svg, sx, sy, bounds: { x0, x1, m, plotW, plotH } };
}

/* A legend is always present for two or more series: three of the light-mode
 * steps sit below 3:1 contrast, so identity must never be colour alone. */
function chartLegend(series) {
  const box = el('div', null, 'chart-legend');
  if (series.length < 2) { return box; }
  series.forEach((s, i) => {
    const item = el('span');
    const swatch = el('i');
    swatch.style.background = seriesColor(i);
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(s.label));
    box.appendChild(item);
  });
  return box;
}

/* The values behind a chart, always reachable without hovering. */
function chartTable(series, valueHeading) {
  const wrap = el('details', null, 'more');
  wrap.appendChild(el('summary', 'Show the numbers'));
  const table = el('table', null, 'pair-table');
  const head = el('tr');
  head.appendChild(el('th', 'T (K)'));
  for (const s of series) { head.appendChild(el('th', `${s.label} ${valueHeading}`)); }
  table.appendChild(head);
  const temps = series[0] ? series[0].temperatures : [];
  temps.forEach((t, k) => {
    const row = el('tr');
    row.appendChild(el('td', fmtTick(t)));
    for (const s of series) {
      const v = s.values[k];
      row.appendChild(el('td', Number.isFinite(v) ? Number(v.toPrecision(6)) : '—'));
    }
    table.appendChild(row);
  });
  wrap.appendChild(table);
  return wrap;
}

/* Crosshair + tooltip. Enhances reading; never the only way to a value. */
function attachHover(wrap, built, series, format) {
  const { svg, sx, bounds } = built;
  const tip = el('div', null, 'chart-tip');
  tip.hidden = true;
  wrap.appendChild(tip);
  const cross = svgEl('line', { class: 'crosshair', y1: bounds.m.top, y2: bounds.m.top + bounds.plotH });
  cross.setAttribute('visibility', 'hidden');
  svg.appendChild(cross);

  svg.addEventListener('mousemove', (event) => {
    const box = svg.getBoundingClientRect();
    const scale = bounds.plotW ? box.width / svg.viewBox.baseVal.width : 1;
    const xUser = (event.clientX - box.left) / scale;
    if (xUser < bounds.m.left || xUser > bounds.m.left + bounds.plotW) { return; }
    const temps = series[0] ? series[0].temperatures : [];
    if (!temps.length) { return; }
    let best = 0;
    temps.forEach((t, k) => {
      if (Math.abs(sx(t) - xUser) < Math.abs(sx(temps[best]) - xUser)) { best = k; }
    });
    cross.setAttribute('x1', sx(temps[best]));
    cross.setAttribute('x2', sx(temps[best]));
    cross.setAttribute('visibility', 'visible');
    tip.textContent = [`${fmtTick(temps[best])} K`].concat(
      series.map((s) => {
        const v = s.values[best];
        return `${s.label}  ${Number.isFinite(v) ? format(v) : '—'}`;
      })).join('\n');
    tip.hidden = false;
    tip.style.left = `${Math.min(box.width - 130, event.clientX - box.left + 12)}px`;
    tip.style.top = '8px';
  });
  svg.addEventListener('mouseleave', () => {
    tip.hidden = true;
    cross.setAttribute('visibility', 'hidden');
  });
}

/* The clumping matrix: a diverging scale centred on zero, because Delta is
 * signed and "no anomaly" must read as nothing rather than as a hue. */
function clumpingMatrix(labels, matrix, onPick, include) {
  const keep = include || (() => true);
  const n = labels.length;
  const cell = Math.max(9, Math.min(22, Math.floor(360 / Math.max(1, n))));
  const pad = 34;
  const size = pad + cell * n + 4;
  const svg = svgEl('svg', {
    class: 'chart', viewBox: `0 0 ${size} ${size}`, role: 'img',
    'aria-label': 'clumping anomaly between every pair of atoms',
  });

  // Scaled to the pairs actually on show. H-H clumping runs an order of
  // magnitude above heavy-atom clumping, so a scale fixed to the whole matrix
  // would flatten every C-C pair to nothing the moment hydrogen is excluded.
  let peak = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const v = matrix[i][j];
      if (v !== null && Number.isFinite(v) && keep(i, j)) { peak = Math.max(peak, Math.abs(v)); }
    }
  }
  if (peak <= 0) { peak = 1; }

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const v = keep(i, j) ? matrix[i][j] : null;
      const rect = svgEl('rect', {
        x: pad + j * cell, y: pad + i * cell,
        width: cell - 1, height: cell - 1, rx: 1,
      });
      if (v === null) {
        // The diagonal is an atom paired with itself: no information, so it stays
        // off the scale rather than anchoring one end of it.
        rect.setAttribute('fill', 'var(--sunk)');
      } else {
        const t = Math.max(-1, Math.min(1, v / peak));
        const pole = t >= 0 ? 'var(--diverge-high)' : 'var(--diverge-low)';
        rect.setAttribute('fill', pole);
        rect.setAttribute('fill-opacity', String(0.12 + 0.88 * Math.abs(t)));
        rect.style.cursor = 'pointer';
        rect.addEventListener('click', () => onPick(i, j));
        const title = svgEl('title', {});
        title.textContent = `${labels[i]} · ${labels[j]}   ${v.toPrecision(4)} ‰`;
        rect.appendChild(title);
      }
      svg.appendChild(rect);
    }
  }
  labels.forEach((label, i) => {
    const row = svgEl('text', {
      class: 'tick', x: pad - 4, y: pad + i * cell + cell * 0.72, 'text-anchor': 'end',
    });
    row.textContent = label;
    svg.appendChild(row);
    const col = svgEl('text', {
      class: 'tick', x: pad + i * cell + cell * 0.5, y: pad - 6, 'text-anchor': 'start',
      transform: `rotate(-90 ${pad + i * cell + cell * 0.5} ${pad - 6})`,
    });
    col.textContent = label;
    svg.appendChild(col);
  });
  return { svg, peak };
}


/* --------------------------------------------------------------- derived */

function derivedSection(title, note) {
  const box = el('div', null, 'derived-section');
  box.appendChild(el('h3', title));
  if (note) { box.appendChild(el('p', note, 'muted small')); }
  return box;
}

function unavailable(box, message, nearest) {
  box.appendChild(el('p', message, 'muted small'));
  // An off-grid temperature is normal use, not an error, so offer the way out
  // rather than only naming the problem.
  if (nearest && nearest.length) {
    const row = el('p', null, 'row');
    row.appendChild(el('span', 'Jump to a stored temperature: ', 'muted small'));
    for (const t of nearest) {
      const button = el('button', `${t} K`, 'linky');
      button.type = 'button';
      button.onclick = () => setState({ temperature: t });
      row.appendChild(button);
    }
    box.appendChild(row);
  }
}

function renderBeta(panel, payload) {
  const sites = payload.beta.sites;
  // The count comes from the data, not the text: it was "thirteen" until
  // schema 2.2 added 460 K and 500 K, and a caption that states the grid size
  // goes stale the next time the grid changes.
  const stored = (state?.temperatures || []).length;
  const box = derivedSection('β against temperature',
    `Markers: the ${stored || ''} stored values. Line: `
    + `${methodLabel(payload.beta.method)} fit of ln β against 1/T. Log scale.`);
  if (!sites.length) {
    box.appendChild(el('p', 'No sites to draw at this element lock.', 'muted small'));
    panel.appendChild(box);
    return;
  }
  const built = lineChart(sites, { log: true, yLabel: 'β (log)' });
  if (built) {
    const wrap = el('div', null, 'chart-wrap');
    wrap.appendChild(built.svg);
    attachHover(wrap, built, sites, (v) => v.toPrecision(5));
    box.appendChild(wrap);
    box.appendChild(chartLegend(sites));
    box.appendChild(chartTable(sites, 'β'));
  }
  panel.appendChild(box);
}

function renderMassLaw(panel, payload) {
  const section = payload.mass_law;
  const box = derivedSection('Mass-law exponent λ', section.note || '');
  if (!section.available) {
    unavailable(box, section.message, payload.nearest_grid);
    panel.appendChild(box);
    return;
  }
  const sites = section.sites;
  const built = lineChart(sites, { log: false, yLabel: 'λ' });
  if (built) {
    const wrap = el('div', null, 'chart-wrap');
    wrap.appendChild(built.svg);
    attachHover(wrap, built, sites, (v) => v.toPrecision(6));
    box.appendChild(wrap);
    box.appendChild(chartLegend(sites));
    box.appendChild(chartTable(sites, 'λ'));
  }
  panel.appendChild(box);
}

/* Which pairs the clumping matrix shows. Client-side: the payload already
 * carries every pair, so filtering is a redraw rather than a round trip. */
let clumpingFilter = 'all';

const CLUMPING_FILTERS = [
  ['all', 'all pairs'],
  ['heavy', 'no hydrogen'],
  ['h', 'with hydrogen'],
];

function clumpingKeeps(isH, i, j) {
  const involvesH = isH[i] || isH[j];
  if (clumpingFilter === 'heavy') { return !involvesH; }
  if (clumpingFilter === 'h') { return involvesH; }
  return true;
}

function renderClumping(panel, payload) {
  const section = payload.clumping;
  const box = derivedSection('Clumping Δ',
    'Δᵢⱼ = 1000 ln(βᵢⱼ / βᵢβⱼ), in per mil. Click a cell or row to highlight the pair.');
  if (!section.available) {
    unavailable(box, section.message, payload.nearest_grid);
    panel.appendChild(box);
    return;
  }

  const isH = section.is_hydrogen || [];
  const pick = (i, j) => setState({ highlight_pair: [i, j] });

  const controls = el('div', null, 'row');
  controls.appendChild(el('span', 'Show ', 'muted small'));
  const seg = el('span', null, 'segmented');
  for (const [value, text] of CLUMPING_FILTERS) {
    const button = el('button', text, value === clumpingFilter ? 'on' : '');
    button.type = 'button';
    button.onclick = () => {
      clumpingFilter = value;
      loadDerived();            // one render path, so the scale and table agree
    };
    seg.appendChild(button);
  }
  controls.appendChild(seg);
  box.appendChild(controls);

  const wrap = el('div', null, 'matrix-wrap');
  const built = clumpingMatrix(
    section.labels, section.matrix, pick, (i, j) => clumpingKeeps(isH, i, j),
  );
  wrap.appendChild(built.svg);

  const pairs = section.pairs.filter(
    (p) => clumpingKeeps(isH, p.i, p.j),
  );

  const table = el('table', null, 'pair-table');
  const head = el('tr');
  head.appendChild(el('th', 'pair'));
  head.appendChild(el('th', section.unit || 'Δ (‰)'));
  table.appendChild(head);
  const body = el('tbody');
  for (const pair of pairs.slice(0, 25)) {
    const row = el('tr');
    row.appendChild(el('td', `${pair.label_i} · ${pair.label_j}`));
    row.appendChild(el('td', pair.delta.toPrecision(4)));
    row.onclick = () => pick(pair.i, pair.j);
    body.appendChild(row);
  }
  table.appendChild(body);

  const side = el('div');
  side.appendChild(el('p', pairs.length
    ? `Strongest pairs · scale ±${built.peak.toPrecision(3)} ‰`
    : 'No pairs of this kind in this molecule.', 'muted small'));
  side.appendChild(table);
  wrap.appendChild(side);
  box.appendChild(wrap);
  panel.appendChild(box);
}

async function loadDerived() {
  const panel = $('panel-derived');
  let payload;
  try { payload = await api('GET', '/view/derived'); }
  catch (error) { return fail(error); }

  clear(panel);
  if (!payload.available) {
    panel.appendChild(el('p', payload.message, 'muted small'));
    return;
  }
  const header = el('p',
    `${payload.molecule} at ${payload.temperature} K, `
    + (payload.on_grid ? 'stored' : 'fitted'),
    'muted small');
  panel.appendChild(header);

  renderBeta(panel, payload);
  renderMassLaw(panel, payload);
  renderClumping(panel, payload);
}

// The chart is the one place the choice of interpolant is visible, so name it
// there rather than making the reader remember what the sidebar says.
function methodLabel(value) {
  const found = (state?.beta_methods || []).find((m) => m.value === value);
  return found ? found.label.toLowerCase() : 'the default fit';
}

/* ------------------------------------------------------------------ panels */

const currentPanel = () => document.querySelector('#panels button.on').dataset.panel;

async function loadRanks() {
  try {
    const payload = await api('GET', '/view/ranks');
    const scope = payload.scope === 'results'
      ? `among ${payload.pool_size} molecules in the results`
      : 'in this molecule';
    $('rank-caption').textContent =
      `Largest ${payload.quantity_symbol || payload.quantity} for ${payload.element || 'all elements'}, `
      + `at ${payload.temperature} K, ${scope}.`;

    const table = el('table');
    const head = el('tr');
    for (const [label, cls] of [['#', ''], ['molecule', ''], ['site', 'num'],
                                ['element', ''], ['value', 'num'], ['neighbourhood', '']]) {
      head.appendChild(Object.assign(el('th', label), { className: cls }));
    }
    table.appendChild(head);

    payload.rows.forEach((row, i) => {
      const tr = el('tr');
      tr.appendChild(el('td', i + 1));
      tr.appendChild(Object.assign(el('td', row.key), { className: 'mono' }));
      tr.appendChild(Object.assign(el('td', row.atom_index), { className: 'num' }));
      tr.appendChild(el('td', row.element));
      tr.appendChild(Object.assign(
        el('td', row.value === null ? '—' : row.value.toFixed(5)), { className: 'num' }));
      tr.appendChild(Object.assign(el('td', row.r1), { className: 'mono' }));
      tr.onclick = () => load(row.key);
      table.appendChild(tr);
    });

    const box = $('ranks');
    clear(box);
    box.appendChild(payload.rows.length ? table : el('p', 'Nothing to rank yet.', 'muted'));
  } catch (error) { fail(error); }
}

async function loadQc() {
  try {
    const payload = await api('GET', '/view/qc');
    const verdict = $('qc-verdict');
    clear(verdict);
    const dot = el('span', '', 'dot');
    dot.dataset.verdict = payload.verdict || '';
    verdict.appendChild(dot);
    // The words carry the meaning; the dot only echoes it.
    verdict.appendChild(el('span', payload.verdict === 'pass'
      ? 'Data quality: all checks pass'
      : payload.verdict === 'check' ? 'Data quality: some checks need attention'
      : 'Data quality: not known'));
    $('qc-text').textContent = payload.text;
  } catch (error) { fail(error); }
}

/* -------------------------------------------------------------- statistics */

async function loadStatistics() {
  try {
    const stats = await api('GET', '/api/statistics');

    const table = el('table');
    const head = el('tr');
    for (const label of ['dataset', 'molecules', 'searchable by structure', 'quality checks']) {
      head.appendChild(el('th', label));
    }
    table.appendChild(head);
    const notes = [];
    for (const row of stats.datasets) {
      const tr = el('tr');
      if (row.active) { tr.className = 'on'; }
      tr.appendChild(el('td', row.dataset + (row.active ? '  ← open' : '')));
      tr.appendChild(Object.assign(el('td', row.n.toLocaleString()), { className: 'num' }));
      tr.appendChild(Object.assign(
        el('td', `${row.n_with_structure.toLocaleString()} (${row.structure_pct}%)`),
        { className: 'num' }));
      tr.appendChild(el('td', `${row.n_qc_pass.toLocaleString()} pass`
        + (row.n_qc_fail ? `, ${row.n_qc_fail.toLocaleString()} check` : '')));
      table.appendChild(tr);
      if (row.note) { notes.push(row.note); }
    }
    const box = $('stats-datasets');
    clear(box);
    box.appendChild(table);
    for (const note of notes) { box.appendChild(notice(note, 'warn')); }

    $('stats-temperatures').textContent = stats.temperatures.join('   ') + '  K';
    $('stats-sample-note').textContent =
      `The opacity scale: β over ${stats.sampled_molecules} sampled molecules `
      + `from ${stats.dataset}, at ${stats.temperature} K.`;

    const elements = el('table');
    const ehead = el('tr');
    for (const label of ['element', '5th pct', 'median', '95th pct']) {
      ehead.appendChild(Object.assign(el('th', label), { className: label === 'element' ? '' : 'num' }));
    }
    elements.appendChild(ehead);
    for (const row of stats.elements) {
      const tr = el('tr');
      tr.appendChild(el('td', row.element));
      for (const key of ['p5', 'p50', 'p95']) {
        tr.appendChild(Object.assign(
          el('td', row[key] === null ? '—' : row[key].toFixed(4)), { className: 'num' }));
      }
      elements.appendChild(tr);
    }
    clear($('stats-elements'));
    $('stats-elements').appendChild(elements);

    const groups = el('table');
    const ghead = el('tr');
    for (const label of ['group', 'molecules', 'what it is']) { ghead.appendChild(el('th', label)); }
    groups.appendChild(ghead);
    for (const row of stats.groups) {
      const tr = el('tr');
      tr.appendChild(el('td', row.name));
      tr.appendChild(Object.assign(
        el('td', row.count === null ? '—' : row.count.toLocaleString()), { className: 'num' }));
      // Groups with no members stay listed: absence is information.
      tr.appendChild(el('td', row.count === 0 ? 'none in this dataset' : row.description));
      groups.appendChild(tr);
    }
    clear($('stats-groups'));
    $('stats-groups').appendChild(groups);

    const p = stats.provenance;
    $('stats-provenance').textContent = p.index
      ? `Index: ${p.molecules.toLocaleString()} molecules, version ${p.index_version}, `
        + `built with RDKit ${p.rdkit_built_with}`
        + (p.rdkit_drift
            ? `. You are running RDKit ${p.rdkit_running}, so some structure lookups may `
              + 'miss molecules. Rebuild the index to fix this.'
            : '.')
      : 'No search index is open.';
  } catch (error) { fail(error); }
}

/* ----------------------------------------------------------------- download */

function currentDownloadScope() {
  const on = document.querySelector('#dl-scope button.on');
  return on ? on.dataset.scope : 'molecule';
}

async function loadDownloadPreview() {
  try {
    const preview = await api('GET', '/api/download?scope=' + currentDownloadScope());
    const sites = preview.n_sites_estimate == null
      ? ''
      : ` · about ${preview.n_sites_estimate.toLocaleString()} site rows`;
    const element = preview.element
      ? ` · ${preview.element} sites only`
      : ' · every element';
    const quantity = preview.quantity_symbol || preview.quantity || 'β';
    // One temperature is named; a list is counted. Naming only the first of a
    // list would describe a file that holds more than it says.
    const spread = preview.temperatures || [];
    const when = spread.length > 1
      ? `${spread.length} temperatures`
      : `${spread[0] ?? preview.temperature} K`;
    $('download-summary').textContent = preview.can_download
      ? `${preview.n_molecules.toLocaleString()} molecule`
        + (preview.n_molecules === 1 ? '' : 's')
        + sites
        + ` · ${quantity} at ${when}`
        + element
      : preview.reason;
    // Refill the box from the saved list, but never while it is being typed
    // into: a preview refresh would yank a half-typed list from under the cursor.
    if (document.activeElement !== $('download-temperatures')) {
      $('download-temperatures').value = (state?.settings?.download_temperatures || []).join(', ');
    }
    $('download-note').textContent = preview.shown < preview.n_molecules
      ? `Explore lists ${preview.shown.toLocaleString()}; this file has all `
        + `${preview.n_molecules.toLocaleString()}.`
      : (preview.temperature_note || '');
    $('download-filename').textContent = preview.filename || '';
    $('download-go').disabled = !preview.can_download;
    $('download-columns').textContent = (preview.columns || []).join(', ');
  } catch (error) { fail(error); }
}

async function downloadCsv() {
  const scope = currentDownloadScope();
  const button = $('download-go');
  if (button.disabled) { return; }
  button.disabled = true;
  try {
    if (inFlight) { await inFlight.catch(() => {}); }
    pending += 1;
    markBusy();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60000);
    const response = await fetch('/api/download.csv?scope=' + scope, { signal: controller.signal });
    clearTimeout(timer);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload?.error?.message || response.statusText);
    }
    const blob = await response.blob();
    const header = response.headers.get('Content-Disposition') || '';
    const match = /filename="([^"]+)"/.exec(header);
    const name = match ? match[1] : $('download-filename').textContent || 'rpfr-visualizer.csv';
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) { fail(error); }
  finally {
    pending -= 1;
    markBusy();
    loadDownloadPreview();
  }
}

/* ----------------------------------------------------------------- presets */

/* The functional-group catalog on the Guide tab.
 *
 * Pictures arrive as <img src="/api/fg/depict?name=...">, not as markup: the
 * server draws them with RDKit, the browser caches them, and nothing has to be
 * inserted as HTML -- so the single-innerHTML rule is untouched. A missing
 * RDKit shows a line of text instead of a broken image icon. */
let guideLoaded = false;

async function loadGuide() {
  if (guideLoaded) { return; }
  let payload;
  try { payload = await api('GET', '/api/fg/reference'); }
  catch (error) { return fail(error); }

  const grid = $('guide-groups');
  clear(grid);
  $('guide-groups-note').textContent = payload.groups.length
    + ' curated groups. The count is how many molecules in the open dataset'
    + ' carry each one; click a card to search for it.';

  for (const group of payload.groups) {
    const card = el('div', null, 'fg-card');

    if (payload.depictions_available) {
      const img = el('img', null, 'fg-image');
      img.src = '/api/fg/depict?name=' + encodeURIComponent(group.name);
      img.alt = group.name + ' pattern';
      img.loading = 'lazy';
      img.onerror = () => img.replaceWith(el('p', 'no preview', 'muted small'));
      card.appendChild(img);
    } else {
      card.appendChild(el('p', 'RDKit is not installed, so there is no picture.',
        'muted small'));
    }

    card.appendChild(el('h4', group.name));
    card.appendChild(el('p', group.description, 'muted small'));
    card.appendChild(el('p', group.smarts || '', 'mono small'));
    card.appendChild(el('p',
      group.count === null ? 'no index open'
        : `${group.count.toLocaleString()} molecule${group.count === 1 ? '' : 's'} here`,
      'muted small'));

    const button = el('button', 'Search this');
    button.type = 'button';
    button.disabled = group.count === 0;
    button.onclick = () => {
      document.querySelector('#views button[data-view=\"explore\"]').click();
      $('q').value = 'fg:' + group.name;
      search();
    };
    card.appendChild(button);
    grid.appendChild(card);
  }
  guideLoaded = true;
}

async function loadPresets() {
  try {
    const payload = await api('GET', '/api/fg/presets');
    const list = $('preset-groups');
    clear(list);
    for (const preset of payload.presets) {
      const row = el('label', null, 'check');
      const box = el('input');
      box.type = 'checkbox';
      box.dataset.fg = preset.name;
      // Shown but not selectable rather than hidden: "there are none here" is
      // an answer, and a group that vanished would read as a missing feature.
      box.disabled = preset.count === 0;
      if (preset.count === 0) { row.classList.add('empty'); }
      row.title = preset.description || '';
      row.appendChild(box);
      row.appendChild(document.createTextNode(' ' + preset.name));
      if (preset.count !== null) {
        row.appendChild(el('span', `  ${preset.count.toLocaleString()}`, 'count'));
      }
      list.appendChild(row);
    }
  } catch (error) { fail(error); }
}

/* The group picker composes query text rather than calling a second endpoint.
 * One grammar decides what a search means, and the user watches the syntax they
 * could have typed appear in the box. */
function selectedGroups() {
  return [...document.querySelectorAll('#preset-groups input:checked')]
    .map((box) => box.dataset.fg);
}

function groupModeIsAll() {
  return document.querySelector('#fg-mode button.on').dataset.mode === 'all';
}

function groupQuery(names) {
  // "&" is the conjunction because a comma already means "either"; it appears
  // in neither SMILES nor SMARTS, so it cannot collide with a pattern.
  return groupModeIsAll()
    ? 'fg:' + names.join('&')
    : names.map((name) => 'fg:' + name).join(', ');
}

function syncGroupsToQuery() {
  const names = selectedGroups();
  const note = $('fg-note');
  if (!names.length) {
    note.textContent = '';
    return;
  }
  note.textContent = names.length === 1
    ? ''
    : (groupModeIsAll()
      ? `Molecules carrying all ${names.length} groups.`
      : `Molecules carrying any of these ${names.length}.`);
  $('q').value = groupQuery(names);
  search();
}

/* ------------------------------------------------------------------- setup */

function showSetup(payload) {
  document.body.dataset.view = 'setup';
  $('setup').hidden = false;
  $('explore').hidden = true;
  $('dataset-view').hidden = true;
  $('download-view').hidden = true;
  $('guide-view').hidden = true;
  document.querySelector('.bar .chip').hidden = true;

  const report = $('setup-report');
  clear(report);
  const found = payload.discovered;
  const table = el('table');
  const rows = [
    ['Data files', found.data_dirs[0] || 'not found'],
    ['External drive', found.drive_root || 'not connected'],
    ['$RPFR_VISUALIZER_INDEX', found.index_env || 'not set'],
  ];
  for (const [label, value] of rows) {
    const tr = el('tr');
    tr.appendChild(el('th', label));
    tr.appendChild(Object.assign(el('td', value), { className: 'mono' }));
    table.appendChild(tr);
  }
  report.appendChild(table);

  const problems = $('setup-problems');
  clear(problems);
  for (const message of payload.problems.concat(payload.warnings)) {
    problems.appendChild(notice(message, 'warn'));
  }

  const indexes = $('setup-indexes');
  clear(indexes);
  if (!found.index_candidates.length) {
    indexes.appendChild(el('p', 'No index found. Build one with:', 'muted'));
    indexes.appendChild(Object.assign(
      el('p', 'rpfr-visualizer index build --raw data/raw'), { className: 'mono' }));
  }
  for (const candidate of found.index_candidates) {
    const row = el('div', null, 'row');
    const button = el('button', 'Use this');
    button.onclick = () => openWith({ index_path: candidate.path });
    row.appendChild(button);
    row.appendChild(Object.assign(
      el('span', `${candidate.path}  ·  ${candidate.size_mb} MB  ·  `
        + `${candidate.molecules ? candidate.molecules.toLocaleString() + ' molecules' : candidate.error}`),
      { className: 'mono small' }));
    indexes.appendChild(row);
  }

  $('setup-remember-note').textContent = 'Saved for next time.';
}

async function openWith(body) {
  try {
    const payload = await api('POST', '/api/setup', Object.assign({ remember: true }, body));
    if (payload.ready) { await start(); }
    else { showSetup(payload); }
  } catch (error) { fail(error); }
}

/* -------------------------------------------------------------------- boot */

function wire() {
  $('go').onclick = search;
  $('q').addEventListener('keydown', (event) => { if (event.key === 'Enter') { search(); } });
  $('preset-groups').addEventListener('change', syncGroupsToQuery);
  for (const button of document.querySelectorAll('#fg-mode button')) {
    button.onclick = () => {
      for (const other of document.querySelectorAll('#fg-mode button')) {
        other.classList.toggle('on', other === button);
      }
      // Only re-run if the toggle can change anything: with fewer than two
      // groups picked, any and all mean the same query.
      if (selectedGroups().length > 1) { syncGroupsToQuery(); }
    };
  }
  // Re-run rather than silently leaving stale results under a changed filter.
  $('qc_pass').onchange = () => { if ($('q').value.trim()) { search(); } };

  // Cheap changes are automatic — the picture always matches the controls.
  $('element').onchange = (e) => setState({ element: e.target.value });
  $('kind').onchange = (e) => setState({ kind: e.target.value });
  $('reference').onchange = (e) => setState({ reference_mode: e.target.value });
  $('anchor').onchange = (e) => setState({ anchor_atom_index: Number(e.target.value) });
  $('temperature').onchange = (e) => {
    if (e.target.value !== CUSTOM_TEMPERATURE) {
      // Take focus off the custom box before anything else. Leaving it focused
      // with a stale value meant that blurring it fired a second change event,
      // which set the temperature straight back to whatever had been typed
      // there — so choosing a stored temperature silently did nothing.
      $('temperature-custom').blur();
      setState({ temperature: e.target.value });
      return;
    }
    // Reveal the box before the round trip, so the click that asked for a
    // custom temperature is answered immediately rather than after a render.
    $('temperature-custom-field').hidden = false;
    $('temperature-custom').focus();
    $('temperature-custom').select();
  };
  $('temperature-custom').onchange = (e) => {
    // Belt and braces for the same race: a change arriving from this box after
    // the dropdown has moved on is stale, whatever fired it.
    if ($('temperature').value !== CUSTOM_TEMPERATURE) { return; }
    // A blur re-fires change with the value already applied; sending it again
    // is at best a wasted render and at worst overwrites a newer choice.
    if (Number(e.target.value) === Number(state?.settings?.temperature)) { return; }
    setState({ temperature: e.target.value });
  };
  $('show_hydrogens').onchange = (e) => setState({ show_hydrogens: e.target.checked });
  $('show_labels').onchange = (e) => setState({ show_labels: e.target.checked });
  $('beta_method').onchange = (e) => setState({ beta_method: e.target.value });
  $('top_n').onchange = (e) => setState({ top_n: Number(e.target.value) });

  $('dataset').onchange = async (event) => {
    try {
      applyState(await api('POST', '/api/dataset', { name: event.target.value }));
      await loadPresets();
      await render();
      if (!$('download-view').hidden) { await loadDownloadPreview(); }
    } catch (error) { fail(error); }
  };

  for (const button of document.querySelectorAll('#scope button')) {
    button.onclick = async () => {
      await setState({ use_match_pool: button.dataset.scope === 'results' });
      await loadRanks();
    };
  }

  for (const button of document.querySelectorAll('#panels button')) {
    button.onclick = () => {
      for (const other of document.querySelectorAll('#panels button')) {
        other.classList.toggle('on', other === button);
      }
      for (const name of ['molecule', 'ranked', 'quality', 'derived']) {
        $('panel-' + name).hidden = name !== button.dataset.panel;
      }
      if (button.dataset.panel === 'ranked') { loadRanks(); }
      if (button.dataset.panel === 'quality') { loadQc(); }
      if (button.dataset.panel === 'derived') { loadDerived(); }
    };
  }

  for (const button of document.querySelectorAll('#views button')) {
    button.onclick = () => {
      for (const other of document.querySelectorAll('#views button')) {
        other.classList.toggle('on', other === button);
      }
      const view = button.dataset.view;
      $('explore').hidden = view !== 'explore';
      $('dataset-view').hidden = view !== 'dataset';
      $('download-view').hidden = view !== 'download';
      $('guide-view').hidden = view !== 'guide';
      if (view === 'dataset') { loadStatistics(); }
      if (view === 'download') { loadDownloadPreview(); }
      if (view === 'guide') { loadGuide(); }
    };
  }

  for (const button of document.querySelectorAll('#dl-scope button')) {
    button.onclick = () => {
      for (const other of document.querySelectorAll('#dl-scope button')) {
        other.classList.toggle('on', other === button);
      }
      loadDownloadPreview();
    };
  }
  $('download-go').onclick = downloadCsv;
  $('download-temperatures').onchange = async (e) => {
    try {
      applyState(await api('POST', '/api/state', { download_temperatures: e.target.value }));
      await loadDownloadPreview();
    } catch (error) { fail(error); }
  };

  $('setup-use').onclick = () => openWith({ data_dir: $('setup-path').value.trim() });
}

async function start() {
  const setup = await api('GET', '/api/setup');
  if (!setup.ready) { showSetup(setup); document.body.dataset.ready = 'setup'; return; }

  document.body.dataset.view = 'explore';
  $('setup').hidden = true;
  $('explore').hidden = false;
  document.querySelector('.bar .chip').hidden = false;

  applyState(await api('GET', '/api/state'));
  for (const message of setup.warnings) { toast(message); }
  await loadPresets();
  await loadExamples();
  await render();
  /* start() is async and dataset.view is set before its awaits, so view is not
   * a completion signal. This is one. */
  document.body.dataset.ready = 'explore';
}

wire();
start().catch((error) => {
  fail(error);
  document.body.dataset.ready = 'error';
});
