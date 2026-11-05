const COLORS = ['#4cc9f0', '#80ed99', '#f8961e', '#f72585', '#b5179e', '#ffd166', '#90be6d', '#43aa8b'];
const $ = (id) => document.getElementById(id);
let spec = null, img = null, imgExt = null, view = null, curves = [], tracks = [], viewMode = 'enhanced';
let tool = 'zoom', history = [];

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
const patch = (url, body) => api(url, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
async function upload(url, file) { const fd = new FormData(); fd.append('file', file); return api(url, { method: 'POST', body: fd }); }
const gmaps = (lat, lon) => `https://www.google.com/maps/search/?api=1&query=${lat.toFixed(6)},${lon.toFixed(6)}`;

// ---------------------------------------------------------------- spectrogram canvas
const cv = $('spec'), ctx = cv.getContext('2d');
const PAD = { l: 58, r: 12, t: 10, b: 32 };
function plotRect() { return { x: PAD.l, y: PAD.t, w: cv.width - PAD.l - PAD.r, h: cv.height - PAD.t - PAD.b }; }
function toTF(px, py) { const p = plotRect(); return { t: view.t0 + (px - p.x) / p.w * (view.t1 - view.t0), f: view.f1 - (py - p.y) / p.h * (view.f1 - view.f0) }; }
function toXY(t, f) { const p = plotRect(); return { x: p.x + (t - view.t0) / (view.t1 - view.t0) * p.w, y: p.y + (view.f1 - f) / (view.f1 - view.f0) * p.h }; }
function fmtT(t, step) {
  if (spec && spec.t1 < 600) return +t.toFixed(2) + ' s';
  const tt = step < 1 ? Math.round(t * 10) / 10 : Math.round(t);
  const m = Math.floor(tt / 60), sec = tt - 60 * m;
  return `${m}:${(step < 1 ? sec.toFixed(1) : sec.toString()).padStart(step < 1 ? 4 : 2, '0')}`;
}
function niceTime(range, n) { const r = range / n; for (const k of [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]) if (k >= r) return k; return 3600; }
function niceStep(range, n) { const r = range / n, m = Math.pow(10, Math.floor(Math.log10(r))); for (const k of [1, 2, 5, 10]) if (k * m >= r) return k * m; }
function evXY(e) { const r = cv.getBoundingClientRect(); return { x: (e.clientX - r.left) * devicePixelRatio, y: (e.clientY - r.top) * devicePixelRatio }; }

function resize() { const r = cv.getBoundingClientRect(); cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio; draw(); if (spec) scheduleTile(); }
function draw(box) {
  const W = cv.width, H = cv.height, dpr = devicePixelRatio;
  ctx.fillStyle = '#0f1115'; ctx.fillRect(0, 0, W, H);
  if (!spec) return;
  const p = plotRect();
  ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
  if (img) { const a = toXY(imgExt[0], imgExt[3]), b = toXY(imgExt[1], imgExt[2]); ctx.imageSmoothingEnabled = false; ctx.drawImage(img, a.x, a.y, b.x - a.x, b.y - a.y); }
  ctx.restore();
  // axes
  ctx.strokeStyle = '#ffffff22'; ctx.fillStyle = '#9aa3b2'; ctx.font = `${11 * dpr}px system-ui`; ctx.lineWidth = 1;
  const ts = niceTime(view.t1 - view.t0, 10), fs = niceStep(view.f1 - view.f0, 6);
  ctx.textAlign = 'center';
  for (let t = Math.ceil(view.t0 / ts) * ts; t <= view.t1; t += ts) { const { x } = toXY(t, 0); ctx.beginPath(); ctx.moveTo(x, p.y); ctx.lineTo(x, p.y + p.h); ctx.stroke(); ctx.fillText(fmtT(t, ts), x, p.y + p.h + 18 * dpr); }
  ctx.textAlign = 'right';
  for (let f = Math.ceil(view.f0 / fs) * fs; f <= view.f1; f += fs) { const { y } = toXY(0, f); ctx.beginPath(); ctx.moveTo(p.x, y); ctx.lineTo(p.x + p.w, y); ctx.stroke(); ctx.fillText(+f.toFixed(1) + ' Hz', p.x - 6, y + 4); }
  // fits
  ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
  if ($('showFits').checked) curves.forEach((c, i) => {
    if (!c.show) return;
    const col = COLORS[i % COLORS.length];
    if ($('showModel').checked && c.model_t) {
      ctx.strokeStyle = col; ctx.globalAlpha = 0.9; ctx.lineWidth = 1.2 * dpr; ctx.setLineDash([6 * dpr, 4 * dpr]); ctx.beginPath();
      c.model_t.forEach((t, k) => { const q = toXY(t, c.model_f[k]); k ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); });
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    ctx.fillStyle = col; const r = 1.4 * dpr;
    c.t.forEach((t, k) => { const q = toXY(t, c.f[k]); ctx.fillRect(q.x - r, q.y - r, 2 * r, 2 * r); });
    const q = toXY(c.t[0], c.f[0]); ctx.textAlign = 'left'; ctx.font = `bold ${12 * dpr}px system-ui`; ctx.fillText(c.name, q.x + 4, q.y - 8);
  });
  if (box) { ctx.strokeStyle = box.color || '#fff'; ctx.lineWidth = 1.5 * dpr; ctx.setLineDash([5, 4]); ctx.strokeRect(box.x, box.y, box.w, box.h); ctx.setLineDash([]); }
  ctx.restore();
}
let tileReq = 0, tileTimer = null;
async function loadImage() {
  const p = plotRect(), id = ++tileReq;
  const q = new URLSearchParams({ view: viewMode, t0: view.t0, t1: view.t1, f0: view.f0, f1: view.f1, w: Math.round(p.w), h: Math.round(p.h), v: Date.now() });
  const r = await fetch(`/api/spectrogram.png?${q}`);
  if (!r.ok) return;
  const ext = r.headers.get('X-Extent').split(',').map(Number);
  const bmp = await createImageBitmap(await r.blob());
  if (id !== tileReq) return;
  img = bmp; imgExt = ext; draw();
}
function scheduleTile() { clearTimeout(tileTimer); tileTimer = setTimeout(loadImage, 120); }
function clampView() {
  const tw = Math.min(view.t1 - view.t0, spec.t1 - spec.t0), fw = Math.min(view.f1 - view.f0, spec.f1 - spec.f0);
  view.t0 = Math.max(spec.t0, Math.min(view.t0, spec.t1 - tw)); view.t1 = view.t0 + tw;
  view.f0 = Math.max(spec.f0, Math.min(view.f0, spec.f1 - fw)); view.f1 = view.f0 + fw;
}
function setView(v, remember = true) { if (remember && view) { history.push({ ...view }); if (history.length > 50) history.shift(); } view = v; clampView(); draw(); scheduleTile(); }
async function setSpec(s, keepView = false) {
  const old = view; spec = s; history = [];
  view = keepView && old ? old : { t0: s.t0, t1: s.t1, f0: s.f0, f1: s.f1 }; clampView();
  $('audioName').textContent = `${s.name} · ${(s.duration / 60).toFixed(1)} min`;
  $('fmin').value = s.params.fmin; $('fmax').value = s.params.fmax; $('win').value = s.params.window;
  $('hint').textContent = '';
  img = null; draw(); await loadImage();
}

// tools
function setTool(t) { tool = t; for (const [id, n] of [['tZoom', 'zoom'], ['tPan', 'pan'], ['tFit', 'fit']]) $(id).classList.toggle('on', n === t); cv.style.cursor = t === 'pan' ? 'grab' : 'crosshair'; }
$('tZoom').onclick = () => setTool('zoom'); $('tPan').onclick = () => setTool('pan'); $('tFit').onclick = () => setTool('fit');
$('zBack').onclick = () => { if (history.length) setView(history.pop(), false); };
$('zReset').onclick = () => spec && setView({ t0: spec.t0, t1: spec.t1, f0: spec.f0, f1: spec.f1 });
$('showFits').onchange = () => draw(); $('showModel').onchange = () => draw();
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'z') setTool('zoom'); else if (e.key === 'h') setTool('pan'); else if (e.key === 'f') setTool('fit');
  else if (e.key === 'Backspace') { e.preventDefault(); $('zBack').click(); } else if (e.key === '0') $('zReset').click();
});

let drag = null;
cv.addEventListener('mousedown', (e) => { if (!spec || e.button !== 0) return; const q = evXY(e); drag = { x0: q.x, y0: q.y, moved: false, v0: { ...view }, mode: (e.shiftKey && tool !== 'pan') ? 'fit' : tool }; if (drag.mode === 'pan') cv.style.cursor = 'grabbing'; });
window.addEventListener('mousemove', (e) => {
  if (!spec) return;
  const { x, y } = evXY(e);
  if (e.target === cv) { const { t, f } = toTF(x, y); $('readout').textContent = `${fmtT(t, 0.1)} (${t.toFixed(1)} s) · ${f.toFixed(1)} Hz`; }
  if (!drag) return;
  if (Math.abs(x - drag.x0) + Math.abs(y - drag.y0) > 6) drag.moved = true;
  if (!drag.moved) return;
  if (drag.mode === 'pan') {
    const p = plotRect(); const dt = (x - drag.x0) / p.w * (drag.v0.t1 - drag.v0.t0), df = (y - drag.y0) / p.h * (drag.v0.f1 - drag.v0.f0);
    view = { t0: drag.v0.t0 - dt, t1: drag.v0.t1 - dt, f0: drag.v0.f0 + df, f1: drag.v0.f1 + df }; clampView(); draw(); scheduleTile();
  } else draw({ x: Math.min(x, drag.x0), y: Math.min(y, drag.y0), w: Math.abs(x - drag.x0), h: Math.abs(y - drag.y0), color: drag.mode === 'fit' ? '#80ed99' : '#fff' });
});
window.addEventListener('mouseup', async (e) => {
  if (!drag) return; const d = drag; drag = null; setTool(tool);
  const { x, y } = evXY(e);
  if (d.mode === 'pan') { if (d.moved) { history.push(d.v0); } return; }
  const a = toTF(Math.min(x, d.x0), Math.min(y, d.y0)), b = toTF(Math.max(x, d.x0), Math.max(y, d.y0));
  if (d.mode === 'zoom') {
    if (d.moved) setView({ t0: a.t, t1: b.t, f0: b.f, f1: a.f });
    else { draw(); }
    return;
  }
  // fit
  const allow_rise = $('allowRise').checked;
  try {
    if (!d.moved && e.target === cv) {
      const { t, f } = toTF(x, y); status(`Tracing and fitting the line through ${fmtT(t, 0.1)}, ${f.toFixed(1)} Hz…`, false, true);
      const j = await post('/api/curves', { mode: 'seed', t, f, allow_rise, tmin: view.t0, tmax: view.t1 }); setCurves(j.curves);
    } else if (d.moved) {
      status('Tracing and fitting the line in the box…', false, true);
      const j = await post('/api/curves', { mode: 'box', t0: a.t, t1: b.t, f0: b.f, f1: a.f, allow_rise }); setCurves(j.curves);
    }
    const c = curves[curves.length - 1];
    status(c && c.fit ? `${c.name}: closest at ${fmtT(c.fit.t_cpa, 0.1)}, ${Math.round(c.fit.speed)} m/s, ${(c.fit.d_cpa / 1000).toFixed(2)} km, tone ${c.fit.f0} Hz (fit rms ${c.fit.rms_hz} Hz). Saved.` : 'Curve added');
  } catch (err) { status(err.message, true); draw(); }
});
cv.addEventListener('wheel', (e) => {
  if (!spec) return; e.preventDefault();
  const q = evXY(e); const { t, f } = toTF(q.x, q.y);
  if (e.ctrlKey || e.metaKey) { // pinch on a trackpad, or Ctrl+scroll: zoom around the cursor
    const k = Math.exp(e.deltaY * 0.01);
    const onlyT = e.altKey && false;
    view.t0 = t - (t - view.t0) * k; view.t1 = t + (view.t1 - t) * k;
    if (!e.shiftKey) { view.f0 = f - (f - view.f0) * k; view.f1 = f + (view.f1 - f) * k; }
  } else { // two-finger scroll: move in both directions (Shift+wheel = sideways for a mouse)
    const p = plotRect();
    let dx = e.deltaX, dy = e.deltaY; if (e.shiftKey && !dx) { dx = dy; dy = 0; }
    const st = dx * devicePixelRatio / p.w * (view.t1 - view.t0), sf = -dy * devicePixelRatio / p.h * (view.f1 - view.f0);
    view.t0 += st; view.t1 += st; view.f0 += sf; view.f1 += sf;
  }
  clampView(); draw(); scheduleTile();
}, { passive: false });
cv.addEventListener('dblclick', () => $('zReset').click());
new ResizeObserver(resize).observe(cv);

// ---------------------------------------------------------------- fits list
function setCurves(cs) {
  curves = cs; const ul = $('curves'); ul.innerHTML = '';
  cs.forEach((c, i) => {
    const li = document.createElement('li');
    const fit = c.fit ? `closest ${fmtT(c.fit.t_cpa, 0.1)} · ${Math.round(c.fit.speed)} m/s · ${(c.fit.d_cpa / 1000).toFixed(1)} km · ${c.fit.f0} Hz · rms ${c.fit.rms_hz}` : 'no fit';
    li.innerHTML = `<span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span>
      <div><input class="nm" value="${c.name}" style="width:9em;padding:2px 4px"> <span class="meta">${fmtT(c.t[0], 1)}–${fmtT(c.t[c.t.length - 1], 1)}</span>
      <div class="meta">${fit}</div>
      <label class="chk" style="display:inline-flex;margin:2px 8px 0 0"><input type="checkbox" class="sh" ${c.show ? 'checked' : ''}> show</label>
      <label class="chk" style="display:inline-flex;margin:2px 8px 0 0"><input type="checkbox" class="us" ${c.use ? 'checked' : ''}> use</label>
      <a href="#" class="go meta">go to</a></div>
      <button class="x" title="delete">×</button>`;
    li.querySelector('.x').onclick = async () => { if (!confirm(`Delete ${c.name}?`)) return; const j = await api(`/api/curves/${i}`, { method: 'DELETE' }); setCurves(j.curves); };
    li.querySelector('.nm').onchange = async (e) => setCurves((await patch(`/api/curves/${i}`, { name: e.target.value })).curves);
    li.querySelector('.sh').onchange = async (e) => setCurves((await patch(`/api/curves/${i}`, { show: e.target.checked })).curves);
    li.querySelector('.us').onchange = async (e) => setCurves((await patch(`/api/curves/${i}`, { use: e.target.checked })).curves);
    li.querySelector('.go').onclick = (e) => { e.preventDefault(); const pad = 30, fmn = Math.min(...c.f), fmx = Math.max(...c.f); setView({ t0: c.t[0] - pad, t1: c.t[c.t.length - 1] + pad, f0: fmn - (fmx - fmn) * 0.3, f1: fmx + (fmx - fmn) * 0.3 }); };
    ul.appendChild(li);
  });
  draw();
}

// ---------------------------------------------------------------- map
const map = L.map('map', { worldCopyJump: true }).setView([45.46, 9.19], 5);
const layers = {
  'Map (OpenStreetMap)': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap contributors' }),
  'Satellite (Esri)': L.layerGroup([
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 19, attribution: 'Imagery © Esri, Maxar, Earthstar Geographics' }),
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', { maxZoom: 19 }),
  ]),
  'Terrain (OpenTopoMap)': L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', { maxZoom: 17, attribution: '© OpenStreetMap, SRTM | © OpenTopoMap' }),
  'Dark (CARTO)': L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 19, attribution: '© OpenStreetMap, © CARTO' }),
};
layers['Satellite (Esri)'].addTo(map);
L.control.layers(layers, null, { position: 'topright' }).addTo(map);
L.control.scale({ imperial: false }).addTo(map);
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
  searchBox = L.rectangle([[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]], { color: '#4cc9f0', weight: 1, fill: false, dashArray: '4 4', interactive: false }).addTo(map);
}
map.on('click', (e) => setCenter(e.latlng.lat, e.latlng.lng));
map.on('contextmenu', (e) => L.popup().setLatLng(e.latlng).setContent(`${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}<br><a href="${gmaps(e.latlng.lat, e.latlng.lng)}" target="_blank">Open in Google Maps</a>`).openOn(map));
['lat', 'lon', 'radius'].forEach((id) => $(id).addEventListener('change', drawSearchBox));
function setTracks(ts, fit = true) {
  tracks = ts; trackLayer.clearLayers();
  const col = { prop: '#80ed99', heli: '#ffd166', jet: '#c0c6d0', unknown: '#8a93a3' };
  ts.forEach((t) => L.polyline(t.path, { color: col[t.category] || '#9aa3b2', weight: 1.5, opacity: 0.85 }).bindTooltip(`${t.label}${t.type ? ' · ' + t.type : ''} (${t.category})`).addTo(trackLayer));
  const n = (c) => ts.filter((t) => t.category === c).length;
  $('tracksInfo').textContent = ts.length ? `${ts.length} flights: ${n('prop')} propeller, ${n('heli')} helicopter, ${n('jet')} jet, ${n('unknown')} unknown type` : '';
  if (fit && ts.length) map.fitBounds(L.latLngBounds(ts.flatMap((t) => t.path)));
}

// ---------------------------------------------------------------- actions
function pollProgress(label) {
  const t = setInterval(async () => { try { const p = await api('/api/progress'); if (p.stage !== 'done' && p.stage !== 'idle') status(`${label}: ${p.stage} ${Math.round(p.fraction * 100)}%`, false, true); } catch (e) { } }, 700);
  return () => clearInterval(t);
}
async function afterLoad(j) { setCurves(j.curves || []); await setSpec(j.spec); status(j.curves && j.curves.length ? `Recording loaded with ${j.curves.length} saved fits` : 'Recording loaded'); }
$('audioFile').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  status(`Loading ${f.name}…`, false, true); const stop = pollProgress(f.name);
  try { const j = await upload('/api/audio', f); stop(); await afterLoad(j); } catch (err) { stop(); status(err.message, true); }
  e.target.value = '';
};
$('pathBtn').onclick = async () => {
  const path = $('path').value.trim().replace(/\\ /g, ' '); if (!path) return;
  status(`Loading ${path}…`, false, true); const stop = pollProgress('Loading');
  try { const j = await post('/api/audio/path', { path }); stop(); await afterLoad(j); } catch (err) { stop(); status(err.message, true); }
};
$('reSpec').onclick = async () => {
  status('Recomputing spectrogram…', false, true); const stop = pollProgress('Recomputing');
  try { const j = await post('/api/params', { fmin: +$('fmin').value, fmax: +$('fmax').value, window: +$('win').value, hop: Math.min(0.25, +$('win').value / 4) }); stop(); await setSpec(j.spec, true); status('Updated'); } catch (err) { stop(); status(err.message, true); }
};
$('vEnh').onclick = () => { viewMode = 'enhanced'; $('vEnh').classList.add('on'); $('vRaw').classList.remove('on'); if (spec) loadImage(); };
$('vRaw').onclick = () => { viewMode = 'raw'; $('vRaw').classList.add('on'); $('vEnh').classList.remove('on'); if (spec) loadImage(); };
$('curvesFile').onchange = async (e) => { try { const j = await upload('/api/curves/import', e.target.files[0]); setCurves(j.curves); } catch (err) { status(err.message, true); } e.target.value = ''; };
$('tracksFile').onchange = async (e) => {
  status('Loading flights…', false, true);
  try { const j = await upload('/api/tracks', e.target.files[0]); setTracks(j.tracks); status(`${j.tracks.length} flights loaded`); } catch (err) { status(err.message, true); }
  e.target.value = '';
};
$('histBtn').onclick = async () => {
  if (!$('lat').value || !$('start').value) { status('Set the search centre (click the map) and the recording start in step 4 first', true); return; }
  status('Reading flight history…', false, true); const stop = pollProgress('Flights');
  try {
    const j = await post('/api/tracks/history', { path: $('histPath').value.trim(), lat: +$('lat').value, lon: +$('lon').value, radius: +$('radius').value + 20, start: $('start').value, uncertainty: +$('unc').value });
    stop(); setTracks(j.tracks); status(`${j.tracks.length} flights loaded (also saved to ${j.saved})`);
  } catch (err) { stop(); status(err.message, true); }
};
$('openskyBtn').onclick = async () => {
  if (!$('lat').value || !$('start').value) { status('Set the search centre (click the map) and the recording start first', true); return; }
  status('Fetching from OpenSky…', false, true);
  try { const j = await post('/api/tracks/opensky', { lat: +$('lat').value, lon: +$('lon').value, radius: Math.max(30, +$('radius').value + 15), start: $('start').value }); setTracks(j.tracks); status(`${j.tracks.length} flights fetched`); } catch (err) { status(err.message, true); }
};
$('demFile').onchange = async (e) => {
  status('Loading DEM…', false, true);
  try { const j = await upload('/api/dem', e.target.files[0]); $('dem').value = 'file'; $('demInfo').textContent = `${j.name}: lat ${j.lat[0].toFixed(2)}–${j.lat[1].toFixed(2)}, lon ${j.lon[0].toFixed(2)}–${j.lon[1].toFixed(2)}`; status('DEM loaded'); } catch (err) { status(err.message, true); }
  e.target.value = '';
};
$('locateBtn').onclick = async () => {
  const btn = $('locateBtn'); btn.disabled = true; status('Searching positions and start times… (can take a minute)', false, true);
  try {
    const types = [...document.querySelectorAll('.ty:checked')].map((x) => x.value);
    const r = await post('/api/locate', {
      lat: +$('lat').value, lon: +$('lon').value, radius: +$('radius').value, step: +$('step').value, start: $('start').value,
      uncertainty: +$('unc').value, alt: +$('alt').value, c: +$('c').value, types,
      cpa_slack: +$('slack').value, silence_km: $('silence').value ? +$('silence').value : null, fit_gate: $('fitGate').checked,
      dem: $('dem').value, geoid: $('geoid').value ? +$('geoid').value : 0,
    });
    showResult(r); status('Done');
  } catch (err) { status(err.message, true); }
  btn.disabled = false;
};
function showResult(r) {
  resultLayer.clearLayers();
  L.imageOverlay(`/api/costmap.png?v=${Date.now()}`, r.bounds, { opacity: 0.85 }).addTo(resultLayer);
  L.marker([r.lat, r.lon]).bindPopup(`<b>Estimated location</b><br>${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}<br><a href="${gmaps(r.lat, r.lon)}" target="_blank">Open in Google Maps</a>`).addTo(resultLayer).openPopup();
  let truth = '';
  if (r.truth) {
    L.circleMarker([r.truth.lat, r.truth.lon], { radius: 9, color: '#80ed99', fill: false, weight: 2 }).bindTooltip('true position (demo)').addTo(resultLayer);
    const d = map.distance([r.lat, r.lon], [r.truth.lat, r.truth.lon]);
    truth = `<div>Demo truth is <b>${Math.round(d)} m</b> away, start time off by ${(r.t_start - r.truth.t_start).toFixed(1)} s.</div>`;
  }
  const rows = r.assignments.map((a) => `<div>${a.curve} → <b>${a.flight}</b>, tone ${a.f_src.toFixed(1)} Hz, misfit ${a.rms_hz.toFixed(2)} Hz${a.time_shift_s ? `, shift ${a.time_shift_s > 0 ? '+' : ''}${a.time_shift_s.toFixed(1)} s` : ''}</div>`).join('');
  (r.candidates || []).slice(1).forEach(([la, lo, co], k) => L.circleMarker([la, lo], { radius: 5, color: '#f8961e' }).bindTooltip(`candidate ${k + 2}: cost ${(co / r.cost).toFixed(1)}× the best`).addTo(resultLayer));
  const others = (r.candidates || []).slice(1).map(([la, lo, co]) => `${la.toFixed(4)}, ${lo.toFixed(4)} (${(co / r.cost).toFixed(1)}×)`).join(' · ');
  $('result').innerHTML = `<div class="result"><div><b>${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}</b> · <a href="${gmaps(r.lat, r.lon)}" target="_blank">Google Maps</a></div><div>Recording started ${r.t_start_iso} · listener at ${Math.round(r.observer_alt)} m · ${r.n_tracks} flights considered</div>${rows}${others ? `<div class="sub" style="margin-top:4px">Other candidates (cost vs best): ${others}</div>` : ''}${truth}</div>`;
  map.fitBounds(r.bounds);
}
$('demoBtn').onclick = async () => {
  status('Rendering the synthetic demo recording…', false, true);
  try {
    const j = await post('/api/demo', {});
    setCurves(j.curves || []); await setSpec(j.spec); setTracks(j.tracks, false);
    const g = j.guess; $('start').value = g.start; $('unc').value = g.uncertainty; $('radius').value = g.radius; $('alt').value = g.alt;
    setCenter(g.lat, g.lon); map.setView([g.lat, g.lon], 9);
    status('Demo loaded: use Fit on the two falling lines near 100 Hz at ~100 s and ~200 s, then Find location');
  } catch (err) { status(err.message, true); }
};

// restore state on reload
(async () => {
  try { const s = await api('/api/state'); setCurves(s.curves); if (s.spec.loaded) await setSpec(s.spec); if (s.tracks.length) setTracks(s.tracks); } catch (e) { }
})();

// draggable split between the sound map and the map
(() => {
  const h = $('split'), main = document.querySelector('main'); let on = false;
  h.addEventListener('mousedown', (e) => { on = true; e.preventDefault(); });
  window.addEventListener('mousemove', (e) => {
    if (!on) return; const r = main.getBoundingClientRect();
    const frac = Math.min(0.85, Math.max(0.15, (e.clientY - r.top) / r.height));
    main.style.gridTemplateRows = `${frac * 100}% 6px 1fr`; map.invalidateSize();
  });
  window.addEventListener('mouseup', () => { if (on) { on = false; map.invalidateSize(); } });
})();
