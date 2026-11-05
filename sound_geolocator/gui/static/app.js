const COLORS = ['#4cc9f0', '#80ed99', '#f8961e', '#f72585', '#b5179e', '#ffd166', '#90be6d', '#43aa8b'];
const $ = (id) => document.getElementById(id);
let spec = null, img = null, view = null, curves = [], tracks = [], viewMode = 'enhanced';

// ---------------------------------------------------------------- helpers
function status(msg, err = false, busy = false) {
  const s = $('status');
  if (!msg) { s.style.display = 'none'; return; }
  s.innerHTML = (busy ? '<span class="spin"></span>' : '') + msg;
  s.className = err ? 'err' : '';
  s.style.display = 'block';
  if (!busy && !err) setTimeout(() => { if (s.innerHTML.endsWith(msg)) s.style.display = 'none'; }, 3000);
}
async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
const post = (url, body) => api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
async function upload(url, file) { const fd = new FormData(); fd.append('file', file); return api(url, { method: 'POST', body: fd }); }

// ---------------------------------------------------------------- spectrogram canvas
const cv = $('spec'), ctx = cv.getContext('2d');
const PAD = { l: 58, r: 12, t: 10, b: 32 };
function plotRect() { return { x: PAD.l, y: PAD.t, w: cv.width - PAD.l - PAD.r, h: cv.height - PAD.t - PAD.b }; }
function toTF(px, py) { const p = plotRect(); return { t: view.t0 + (px - p.x) / p.w * (view.t1 - view.t0), f: view.f1 - (py - p.y) / p.h * (view.f1 - view.f0) }; }
function toXY(t, f) { const p = plotRect(); return { x: p.x + (t - view.t0) / (view.t1 - view.t0) * p.w, y: p.y + (view.f1 - f) / (view.f1 - view.f0) * p.h }; }
function niceStep(range, n) { const r = range / n, m = Math.pow(10, Math.floor(Math.log10(r))); for (const k of [1, 2, 5, 10]) if (k * m >= r) return k * m; }

function resize() { const r = cv.getBoundingClientRect(); cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio; ctx.setTransform(1, 0, 0, 1, 0, 0); draw(); }
function draw(box) {
  const W = cv.width, H = cv.height, dpr = devicePixelRatio;
  ctx.fillStyle = '#0f1115'; ctx.fillRect(0, 0, W, H);
  if (!spec || !img) return;
  const p = plotRect();
  // source rect inside the image (image row 0 = highest frequency)
  const iw = img.naturalWidth, ih = img.naturalHeight;
  const sx = (view.t0 - spec.t0) / (spec.t1 - spec.t0) * iw, sw = (view.t1 - view.t0) / (spec.t1 - spec.t0) * iw;
  const sy = (spec.f1 - view.f1) / (spec.f1 - spec.f0) * ih, sh = (view.f1 - view.f0) / (spec.f1 - spec.f0) * ih;
  ctx.imageSmoothingEnabled = sw < p.w;
  ctx.drawImage(img, sx, sy, sw, sh, p.x, p.y, p.w, p.h);
  // axes
  ctx.strokeStyle = '#ffffff22'; ctx.fillStyle = '#9aa3b2'; ctx.font = `${11 * dpr}px system-ui`; ctx.lineWidth = 1;
  const ts = niceStep(view.t1 - view.t0, 10), fs = niceStep(view.f1 - view.f0, 6);
  ctx.textAlign = 'center';
  for (let t = Math.ceil(view.t0 / ts) * ts; t <= view.t1; t += ts) { const { x } = toXY(t, 0); ctx.beginPath(); ctx.moveTo(x, p.y); ctx.lineTo(x, p.y + p.h); ctx.stroke(); ctx.fillText(+t.toFixed(2) + ' s', x, p.y + p.h + 18 * dpr); }
  ctx.textAlign = 'right';
  for (let f = Math.ceil(view.f0 / fs) * fs; f <= view.f1; f += fs) { const { y } = toXY(0, f); ctx.beginPath(); ctx.moveTo(p.x, y); ctx.lineTo(p.x + p.w, y); ctx.stroke(); ctx.fillText(+f.toFixed(1) + ' Hz', p.x - 6, y + 4); }
  // curves
  ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
  curves.forEach((c, i) => {
    ctx.strokeStyle = COLORS[i % COLORS.length]; ctx.lineWidth = 2 * dpr; ctx.beginPath();
    let prev = null;
    c.t.forEach((t, k) => { const q = toXY(t, c.f[k]); if (prev !== null && t - prev > 3) ctx.moveTo(q.x, q.y); else if (k === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y); prev = t; });
    ctx.stroke();
    const q = toXY(c.t[0], c.f[0]); ctx.fillStyle = COLORS[i % COLORS.length]; ctx.textAlign = 'left'; ctx.fillText(c.name, q.x + 4, q.y - 6);
  });
  if (box) { ctx.strokeStyle = '#fff'; ctx.setLineDash([5, 4]); ctx.strokeRect(box.x, box.y, box.w, box.h); ctx.setLineDash([]); }
  ctx.restore();
}
function loadImage() {
  return new Promise((res) => { img = new Image(); img.onload = () => { draw(); res(); }; img.src = `/api/spectrogram.png?view=${viewMode}&v=${Date.now()}`; });
}
async function setSpec(s) {
  spec = s; view = { t0: s.t0, t1: s.t1, f0: s.f0, f1: s.f1 };
  $('audioName').textContent = `${s.name} · ${s.duration.toFixed(1)} s`;
  $('fmin').value = s.params.fmin; $('fmax').value = s.params.fmax; $('win').value = s.params.window;
  $('hint').textContent = 'Click a falling line to trace it · Shift-drag for a box · scroll to zoom';
  await loadImage();
}

let drag = null;
cv.addEventListener('mousedown', (e) => { if (!spec) return; const r = cv.getBoundingClientRect(); drag = { x0: (e.clientX - r.left) * devicePixelRatio, y0: (e.clientY - r.top) * devicePixelRatio, shift: e.shiftKey, moved: false, v0: { ...view } }; });
window.addEventListener('mousemove', (e) => {
  if (!spec) return;
  const r = cv.getBoundingClientRect(); const x = (e.clientX - r.left) * devicePixelRatio, y = (e.clientY - r.top) * devicePixelRatio;
  if (e.target === cv) { const { t, f } = toTF(x, y); $('readout').textContent = `${t.toFixed(2)} s · ${f.toFixed(1)} Hz`; }
  if (!drag) return;
  if (Math.abs(x - drag.x0) + Math.abs(y - drag.y0) > 6) drag.moved = true;
  if (!drag.moved) return;
  if (drag.shift) { draw({ x: Math.min(x, drag.x0), y: Math.min(y, drag.y0), w: Math.abs(x - drag.x0), h: Math.abs(y - drag.y0) }); }
  else { // pan
    const p = plotRect(); const dt = (x - drag.x0) / p.w * (drag.v0.t1 - drag.v0.t0), df = (y - drag.y0) / p.h * (drag.v0.f1 - drag.v0.f0);
    view = { t0: drag.v0.t0 - dt, t1: drag.v0.t1 - dt, f0: drag.v0.f0 + df, f1: drag.v0.f1 + df }; clampView(); draw();
  }
});
window.addEventListener('mouseup', async (e) => {
  if (!drag) return; const d = drag; drag = null;
  const r = cv.getBoundingClientRect(); const x = (e.clientX - r.left) * devicePixelRatio, y = (e.clientY - r.top) * devicePixelRatio;
  const allow_rise = $('allowRise').checked;
  try {
    if (!d.moved && e.target === cv) {
      const { t, f } = toTF(x, y); status(`Tracing line through ${t.toFixed(1)} s, ${f.toFixed(1)} Hz…`, false, true);
      const j = await post('/api/curves', { mode: 'seed', t, f, allow_rise }); setCurves(j.curves); status('Curve added');
    } else if (d.moved && d.shift) {
      const a = toTF(Math.min(x, d.x0), Math.min(y, d.y0)), b = toTF(Math.max(x, d.x0), Math.max(y, d.y0));
      status('Tracing line in box…', false, true);
      const j = await post('/api/curves', { mode: 'box', t0: a.t, t1: b.t, f0: b.f, f1: a.f, allow_rise }); setCurves(j.curves); status('Curve added');
    }
  } catch (err) { status(err.message, true); draw(); }
});
function clampView() {
  const tw = Math.min(view.t1 - view.t0, spec.t1 - spec.t0), fw = Math.min(view.f1 - view.f0, spec.f1 - spec.f0);
  view.t0 = Math.max(spec.t0, Math.min(view.t0, spec.t1 - tw)); view.t1 = view.t0 + tw;
  view.f0 = Math.max(spec.f0, Math.min(view.f0, spec.f1 - fw)); view.f1 = view.f0 + fw;
}
cv.addEventListener('wheel', (e) => {
  if (!spec) return; e.preventDefault();
  const r = cv.getBoundingClientRect(); const { t, f } = toTF((e.clientX - r.left) * devicePixelRatio, (e.clientY - r.top) * devicePixelRatio);
  const k = e.deltaY > 0 ? 1.25 : 0.8;
  const onlyF = e.altKey, onlyT = e.ctrlKey || e.metaKey;
  if (!onlyF) { view.t0 = t - (t - view.t0) * k; view.t1 = t + (view.t1 - t) * k; }
  if (!onlyT) { view.f0 = f - (f - view.f0) * k; view.f1 = f + (view.f1 - f) * k; }
  clampView(); draw();
}, { passive: false });
cv.addEventListener('dblclick', () => { if (spec) { view = { t0: spec.t0, t1: spec.t1, f0: spec.f0, f1: spec.f1 }; draw(); } });
new ResizeObserver(resize).observe($('specwrap'));

// ---------------------------------------------------------------- curves list
function setCurves(cs) {
  curves = cs; const ul = $('curves'); ul.innerHTML = '';
  cs.forEach((c, i) => {
    const li = document.createElement('li');
    const fit = c.fit ? `≈${c.fit.f0} Hz tone · closest at ${c.fit.t_cpa} s · ${Math.round(c.fit.speed * 1.944)} kt · ${(c.fit.d_cpa / 1000).toFixed(1)} km` : '';
    li.innerHTML = `<span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span><div><b>${c.name}</b> · ${c.t[0].toFixed(0)}–${c.t[c.t.length - 1].toFixed(0)} s<div class="meta">${fit}</div></div><button class="x" title="remove">×</button>`;
    li.querySelector('.x').onclick = async () => { const j = await api(`/api/curves/${i}`, { method: 'DELETE' }); setCurves(j.curves); };
    ul.appendChild(li);
  });
  draw();
}

// ---------------------------------------------------------------- map
const map = L.map('map', { worldCopyJump: true }).setView([45.46, 9.19], 5);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { attribution: '© OpenStreetMap, © CARTO', maxZoom: 19 }).addTo(map);
const trackLayer = L.layerGroup().addTo(map), resultLayer = L.layerGroup().addTo(map);
let centerMarker = null, searchBox = null;
function setCenter(lat, lon) {
  $('lat').value = (+lat).toFixed(5); $('lon').value = (+lon).toFixed(5);
  if (centerMarker) centerMarker.setLatLng([lat, lon]); else centerMarker = L.circleMarker([lat, lon], { radius: 6, color: '#fff' }).addTo(map).bindTooltip('search centre');
  drawSearchBox();
}
function drawSearchBox() {
  const lat = +$('lat').value, lon = +$('lon').value, r = +$('radius').value; if (!lat && !lon) return;
  const dlat = r / 111.2, dlon = r / (111.2 * Math.cos(lat * Math.PI / 180));
  if (searchBox) map.removeLayer(searchBox);
  searchBox = L.rectangle([[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]], { color: '#4cc9f0', weight: 1, fill: false, dashArray: '4 4' }).addTo(map);
}
map.on('click', (e) => setCenter(e.latlng.lat, e.latlng.lng));
['lat', 'lon', 'radius'].forEach((id) => $(id).addEventListener('change', drawSearchBox));
function setTracks(ts, fit = true) {
  tracks = ts; trackLayer.clearLayers();
  ts.forEach((t) => L.polyline(t.path, { color: '#9aa3b2', weight: 1.5, opacity: 0.7 }).bindTooltip(t.label).addTo(trackLayer));
  $('tracksInfo').textContent = ts.length ? `${ts.length} flights loaded` : '';
  if (fit && ts.length) map.fitBounds(L.latLngBounds(ts.flatMap((t) => t.path)));
}

// ---------------------------------------------------------------- actions
$('audioFile').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  status(`Loading ${f.name}…`, false, true);
  try { const j = await upload('/api/audio', f); setCurves([]); await setSpec(j.spec); status('Recording loaded'); } catch (err) { status(err.message, true); }
  e.target.value = '';
};
$('reSpec').onclick = async () => {
  status('Recomputing spectrogram…', false, true);
  try { const j = await post('/api/params', { fmin: +$('fmin').value, fmax: +$('fmax').value, window: +$('win').value, hop: Math.min(0.25, +$('win').value / 4) }); await setSpec(j.spec); status('Updated'); } catch (err) { status(err.message, true); }
};
$('vEnh').onclick = () => { viewMode = 'enhanced'; $('vEnh').classList.add('on'); $('vRaw').classList.remove('on'); if (spec) loadImage(); };
$('vRaw').onclick = () => { viewMode = 'raw'; $('vRaw').classList.add('on'); $('vEnh').classList.remove('on'); if (spec) loadImage(); };
$('curvesFile').onchange = async (e) => { try { const j = await upload('/api/curves/import', e.target.files[0]); setCurves(j.curves); } catch (err) { status(err.message, true); } e.target.value = ''; };
$('tracksFile').onchange = async (e) => {
  status('Loading flights…', false, true);
  try { const j = await upload('/api/tracks', e.target.files[0]); setTracks(j.tracks); status(`${j.tracks.length} flights loaded`); } catch (err) { status(err.message, true); }
  e.target.value = '';
};
$('openskyBtn').onclick = async () => {
  if (!$('lat').value || !$('start').value) { status('Set the search centre (click the map) and the recording start first', true); return; }
  status('Fetching from OpenSky…', false, true);
  try { const j = await post('/api/tracks/opensky', { lat: +$('lat').value, lon: +$('lon').value, radius: Math.max(30, +$('radius').value + 15), start: $('start').value }); setTracks(j.tracks); status(`${j.tracks.length} flights fetched`); } catch (err) { status(err.message, true); }
};
$('locateBtn').onclick = async () => {
  const btn = $('locateBtn'); btn.disabled = true; status('Searching positions and start times… (can take a minute)', false, true);
  try {
    const r = await post('/api/locate', { lat: +$('lat').value, lon: +$('lon').value, radius: +$('radius').value, step: +$('step').value, start: $('start').value, uncertainty: +$('unc').value, alt: +$('alt').value, c: +$('c').value });
    showResult(r); status('Done');
  } catch (err) { status(err.message, true); }
  btn.disabled = false;
};
function showResult(r) {
  resultLayer.clearLayers();
  L.imageOverlay(`/api/costmap.png?v=${Date.now()}`, r.bounds, { opacity: 0.85 }).addTo(resultLayer);
  L.marker([r.lat, r.lon]).bindPopup(`<b>Estimated location</b><br>${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}`).addTo(resultLayer).openPopup();
  let truth = '';
  if (r.truth) {
    L.circleMarker([r.truth.lat, r.truth.lon], { radius: 9, color: '#80ed99', fill: false, weight: 2 }).bindTooltip('true position (demo)').addTo(resultLayer);
    const d = map.distance([r.lat, r.lon], [r.truth.lat, r.truth.lon]);
    truth = `<div>Demo truth is <b>${Math.round(d)} m</b> away, start time off by ${(r.t_start - r.truth.t_start).toFixed(1)} s.</div>`;
  }
  const rows = r.assignments.map((a) => `<div>${a.curve} → <b>${a.flight}</b>, tone ${a.f_src.toFixed(1)} Hz, misfit ${a.rms_hz.toFixed(2)} Hz</div>`).join('');
  $('result').innerHTML = `<div class="result"><div><b>${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}</b> · <a href="https://www.openstreetmap.org/?mlat=${r.lat}&mlon=${r.lon}#map=15/${r.lat}/${r.lon}" target="_blank">open map</a></div><div>Recording started ${r.t_start_iso}</div>${rows}${truth}</div>`;
  map.fitBounds(r.bounds);
}
$('demoBtn').onclick = async () => {
  status('Rendering the synthetic demo recording…', false, true);
  try {
    const j = await post('/api/demo', {});
    setCurves([]); await setSpec(j.spec); setTracks(j.tracks, false);
    const g = j.guess; $('start').value = g.start; $('unc').value = g.uncertainty; $('radius').value = g.radius; $('alt').value = g.alt;
    setCenter(g.lat, g.lon); map.setView([g.lat, g.lon], 9);
    status('Demo loaded: click the two falling lines near 100 Hz at ~100 s and ~200 s, then Find location');
  } catch (err) { status(err.message, true); }
};

// restore state on reload
(async () => {
  try {
    const s = await api('/api/state');
    if (s.spec.loaded) await setSpec(s.spec);
    setCurves(s.curves); if (s.tracks.length) setTracks(s.tracks);
  } catch (e) { /* fresh start */ }
})();
