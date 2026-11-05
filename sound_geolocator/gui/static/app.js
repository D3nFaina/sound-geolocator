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
const PADCSS = { l: 68, r: 14, t: 12, b: 38 };  // CSS pixels, room for the axis labels
const qs = new URLSearchParams(location.search);
const SOLO = qs.get('solo') === '1';
if (SOLO) document.body.classList.add('solo');
let sel = -1, guide = [], knownVersion = -1, overlays = {};
const OCOLORS = ['#ffd166', '#f72585', '#90be6d', '#4895ef', '#f3722c', '#b5179e'];
function plotRect() { const d = devicePixelRatio; return { x: PADCSS.l * d, y: PADCSS.t * d, w: cv.width - (PADCSS.l + PADCSS.r) * d, h: cv.height - (PADCSS.t + PADCSS.b) * d }; }
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
const bandHalf = (f) => { const w = +$('bandW').value; return w > 0 ? w : Math.max(0.04 * f, 2); };

function resize() { const r = cv.getBoundingClientRect(); cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio; draw(); if (spec) scheduleTile(); }
function draw(box) {
  const W = cv.width, H = cv.height, dpr = devicePixelRatio;
  ctx.fillStyle = '#0f1115'; ctx.fillRect(0, 0, W, H);
  if (!spec) return;
  const p = plotRect();
  ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
  if (img) { const a = toXY(imgExt[0], imgExt[3]), b = toXY(imgExt[1], imgExt[2]); ctx.imageSmoothingEnabled = false; ctx.drawImage(img, a.x, a.y, b.x - a.x, b.y - a.y); }
  ctx.restore();
  // axes: labels sit in the margins, outside the plot
  ctx.strokeStyle = '#ffffff22'; ctx.fillStyle = '#c3c9d4'; ctx.font = `${11 * dpr}px system-ui`; ctx.lineWidth = 1;
  const ts = niceTime(view.t1 - view.t0, Math.max(3, p.w / dpr / 90)), fs = niceStep(view.f1 - view.f0, Math.max(3, p.h / dpr / 45));
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let t = Math.ceil(view.t0 / ts) * ts; t <= view.t1; t += ts) { const { x } = toXY(t, 0); ctx.beginPath(); ctx.moveTo(x, p.y); ctx.lineTo(x, p.y + p.h); ctx.stroke(); ctx.fillText(fmtT(t, ts), x, p.y + p.h + 6 * dpr); }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let f = Math.ceil(view.f0 / fs) * fs; f <= view.f1; f += fs) { const { y } = toXY(0, f); ctx.beginPath(); ctx.moveTo(p.x, y); ctx.lineTo(p.x + p.w, y); ctx.stroke(); ctx.fillText(+f.toFixed(1) + ' Hz', p.x - 6 * dpr, y); }
  ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = '#9aa3b2';
  ctx.fillText(spec.t1 < 600 ? 'time' : 'time (min:s)', p.x, H - 2 * dpr);
  ctx.textBaseline = 'alphabetic';
  // curves
  ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
  if ($('showFits').checked) curves.forEach((c, i) => {
    if (!c.show && i !== sel) return;
    const col = COLORS[i % COLORS.length], isSel = i === sel;
    if ($('showModel').checked && c.model_t) {
      ctx.strokeStyle = col; ctx.globalAlpha = 0.95; ctx.lineWidth = (isSel ? 1.8 : 1.2) * dpr; ctx.setLineDash([6 * dpr, 4 * dpr]); ctx.beginPath();
      c.model_t.forEach((t, k) => { const q = toXY(t, c.model_f[k]); k ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); });
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    const r = (isSel ? 2.2 : 1.4) * dpr;
    if (isSel) { ctx.fillStyle = '#000'; c.t.forEach((t, k) => { const q = toXY(t, c.f[k]); ctx.fillRect(q.x - r - dpr, q.y - r - dpr, 2 * r + 2 * dpr, 2 * r + 2 * dpr); }); }
    ctx.fillStyle = col;
    c.t.forEach((t, k) => { const q = toXY(t, c.f[k]); ctx.fillRect(q.x - r, q.y - r, 2 * r, 2 * r); });
    if (c.t.length) { const q = toXY(c.t[0], c.f[0]); ctx.textAlign = 'left'; ctx.font = `bold ${12 * dpr}px system-ui`; ctx.fillText(c.name + (c.fit ? '' : ' (not fitted)'), q.x + 4, q.y - 8); }
  });
  // predicted lines of candidate flights
  const refF = (() => { const f = curves.filter((c) => c.fit).map((c) => c.fit.f0).sort((a, b) => a - b); return f.length ? f[f.length >> 1] : 150; })();
  Object.values(overlays).forEach((ov, oi) => {
    if (!ov.show) return;
    const col = OCOLORS[oi % OCOLORS.length];
    ov.flights.forEach((fl) => {
      const matched = fl.f0 != null;
      if (!matched && !$('showUnmatched').checked) return;
      const f0 = matched ? fl.f0 : refF;
      ctx.strokeStyle = matched ? col : '#c0c6d0'; ctx.globalAlpha = matched ? 0.95 : 0.55; ctx.lineWidth = 1.3 * dpr;
      ctx.setLineDash(matched ? [] : [2 * dpr, 3 * dpr]); ctx.beginPath();
      let prev = null;
      fl.t.forEach((t, k) => { const q = toXY(t, f0 * fl.r[k]); if (prev != null && t - prev > 3) ctx.moveTo(q.x, q.y); else k ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); prev = t; });
      ctx.stroke(); ctx.setLineDash([]);
      const kc = fl.t.reduce((b, t, k) => (Math.abs(t - fl.t_cpa) < Math.abs(fl.t[b] - fl.t_cpa) ? k : b), 0);
      const q = toXY(fl.t[kc], f0 * fl.r[kc]);
      ctx.fillStyle = matched ? col : '#c0c6d0'; ctx.font = `${11 * dpr}px system-ui`; ctx.textAlign = 'left';
      ctx.fillText(`${fl.label}${matched ? ` → ${fl.match} (${fl.rms_hz.toFixed(1)} Hz)` : ` ${(fl.d_min / 1000).toFixed(1)} km, no pass`}`, q.x + 5, q.y + 14 * dpr);
      ctx.globalAlpha = 1;
    });
  });
  // guide being drawn, with the band that will be searched
  if (guide.length) {
    const g = [...guide].sort((a, b) => a.t - b.t);
    ctx.fillStyle = '#ffffff18'; ctx.beginPath();
    g.forEach((q, k) => { const a = toXY(q.t, q.f + bandHalf(q.f)); k ? ctx.lineTo(a.x, a.y) : ctx.moveTo(a.x, a.y); });
    [...g].reverse().forEach((q) => { const a = toXY(q.t, q.f - bandHalf(q.f)); ctx.lineTo(a.x, a.y); });
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2 * dpr; ctx.beginPath();
    g.forEach((q, k) => { const a = toXY(q.t, q.f); k ? ctx.lineTo(a.x, a.y) : ctx.moveTo(a.x, a.y); }); ctx.stroke();
    ctx.fillStyle = '#fff'; g.forEach((q) => { const a = toXY(q.t, q.f); ctx.beginPath(); ctx.arc(a.x, a.y, 3.5 * dpr, 0, 7); ctx.fill(); });
  }
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
  view = keepView && old ? old : { t0: s.t0, t1: s.t1, f0: s.f0, f1: s.f1 };
  const v = qs.get('v'); if (v && !keepView) { const [a, b, c, d] = v.split(',').map(Number); view = { t0: a, t1: b, f0: c, f1: d }; }
  clampView();
  $('audioName').textContent = `${s.name} · ${(s.duration / 60).toFixed(1)} min`;
  $('fmin').value = s.params.fmin; $('fmax').value = s.params.fmax; $('win').value = s.params.window;
  $('hint').textContent = '';
  if (SOLO) document.title = `Sound Geolocator · ${fmtT(view.t0, 1)}–${fmtT(view.t1, 1)}`;
  img = null; draw(); await loadImage();
}

// tools
const TOOLS = [['tZoom', 'zoom'], ['tPan', 'pan'], ['tTrace', 'trace'], ['tErase', 'erase']];
function setTool(t) {
  if (t !== 'trace' && guide.length) { guide = []; draw(); }
  tool = t; for (const [id, n] of TOOLS) $(id).classList.toggle('on', n === t);
  cv.style.cursor = t === 'pan' ? 'grab' : t === 'erase' ? 'cell' : 'crosshair';
  const hints = { trace: 'Click points along one line, then Enter or double-click. Esc cancels, Backspace removes the last point.', erase: sel >= 0 ? `Drag a box over stray points of ${curves[sel].name}` : 'Select a pass in the list first' };
  $('hint').textContent = spec ? (hints[t] || '') : $('hint').textContent;
}
for (const [id, n] of TOOLS) $(id).onclick = () => setTool(n);
$('zBack').onclick = () => { if (history.length) setView(history.pop(), false); };
$('zReset').onclick = () => spec && setView({ t0: spec.t0, t1: spec.t1, f0: spec.f0, f1: spec.f1 });
$('showFits').onchange = () => draw(); $('showModel').onchange = () => draw(); $('bandW').oninput = () => draw();
$('fitSel').onclick = () => fitCurve(sel);
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'Escape') { guide = []; draw(); $('modelBox').style.display = 'none'; return; }
  if (e.key === 'Enter') { e.preventDefault(); if (guide.length) finishGuide(); else if (sel >= 0) fitCurve(sel); return; }
  if (e.key === 'Backspace') { e.preventDefault(); if (guide.length) { guide.pop(); draw(); } else $('zBack').click(); return; }
  const k = e.key.toLowerCase();
  if (k === 'z') setTool('zoom'); else if (k === 'h') setTool('pan'); else if (k === 't' || k === 'f') setTool('trace'); else if (k === 'e') setTool('erase');
  else if (e.key === '0') $('zReset').click();
});

async function finishGuide() {
  const g = guide.filter((q, k) => k === 0 || Math.abs(q.t - guide[k - 1].t) > 1e-3 || Math.abs(q.f - guide[k - 1].f) > 1e-3);
  guide = []; draw();
  if (g.length < 2) { status('Click at least two points along the line', true); return; }
  status('Digitising the line along your points…', false, true);
  try {
    const w = +$('bandW').value;
    const j = await post('/api/curves', { mode: 'guide', t: g.map((q) => q.t), f: g.map((q) => q.f), width: w > 0 ? w : null });
    sel = j.curves.length - 1; setCurves(j.curves);
    const c = curves[sel];
    status(`${c.name}: ${c.n} points digitised. Check they sit on the line (Erase removes strays), then press Fit or Enter.`);
  } catch (err) { status(err.message, true); }
}
async function fitCurve(i) {
  if (i < 0 || !curves[i]) { status('Select a pass in the list first', true); return; }
  status(`Fitting ${curves[i].name}…`, false, true);
  try {
    const j = await post(`/api/curves/${i}/fit`, {}); setCurves(j.curves);
    const c = curves[i];
    status(`${c.name}: closest at ${fmtT(c.fit.t_cpa, 0.1)}, ${Math.round(c.fit.speed)} m/s, ${(c.fit.d_cpa / 1000).toFixed(2)} km, tone ${c.fit.f0} Hz (fit rms ${c.fit.rms_hz} Hz)`);
  } catch (err) { status(err.message, true); }
}

let drag = null;
cv.addEventListener('mousedown', (e) => { if (!spec || e.button !== 0) return; const q = evXY(e); drag = { x0: q.x, y0: q.y, moved: false, v0: { ...view }, mode: (e.shiftKey && tool !== 'pan') ? 'trace' : tool }; if (drag.mode === 'pan') cv.style.cursor = 'grabbing'; });
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
  } else draw({ x: Math.min(x, drag.x0), y: Math.min(y, drag.y0), w: Math.abs(x - drag.x0), h: Math.abs(y - drag.y0), color: { trace: '#80ed99', erase: '#f72585' }[drag.mode] || '#fff' });
});
window.addEventListener('mouseup', async (e) => {
  if (!drag) return; const d = drag; drag = null; setTool(tool);
  const { x, y } = evXY(e);
  if (d.mode === 'pan') { if (d.moved) { history.push(d.v0); } return; }
  const a = toTF(Math.min(x, d.x0), Math.min(y, d.y0)), b = toTF(Math.max(x, d.x0), Math.max(y, d.y0));
  if (d.mode === 'zoom') { if (d.moved) setView({ t0: a.t, t1: b.t, f0: b.f, f1: a.f }); else draw(); return; }
  if (d.mode === 'erase') {
    if (!d.moved) { draw(); return; }
    if (sel < 0) { status('Select a pass in the list first', true); draw(); return; }
    try { const j = await post(`/api/curves/${sel}/erase`, { t0: a.t, t1: b.t, f0: b.f, f1: a.f }); setCurves(j.curves); status(`${j.removed} points removed from ${curves[sel].name}: press Fit to refit`); } catch (err) { status(err.message, true); draw(); }
    return;
  }
  // trace
  if (!d.moved) { if (e.target === cv) { guide.push(toTF(x, y)); draw(); } return; }
  status('Tracing the line in the box…', false, true);
  try {
    const j = await post('/api/curves', { mode: 'box', t0: a.t, t1: b.t, f0: b.f, f1: a.f, allow_rise: $('allowRise').checked });
    sel = j.curves.length - 1; setCurves(j.curves);
    status(`${curves[sel].name}: ${curves[sel].n} points traced. Check them, then Fit.`);
  } catch (err) { status(err.message, true); draw(); }
});
cv.addEventListener('wheel', (e) => {
  if (!spec) return; e.preventDefault();
  const q = evXY(e); const { t, f } = toTF(q.x, q.y);
  if (e.ctrlKey || e.metaKey) { // pinch on a trackpad, or Ctrl+scroll: zoom around the cursor
    const k = Math.exp(e.deltaY * 0.01);
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
cv.addEventListener('dblclick', (e) => {
  if (tool === 'trace') { guide.pop(); finishGuide(); return; }  // the second click of the double-click added a duplicate point
  if (tool === 'zoom' || tool === 'pan') $('zReset').click();
});
new ResizeObserver(resize).observe(cv);

// ---------------------------------------------------------------- side-by-side windows
function openWindow(v) {
  v = v || view; if (!v) return;
  const url = `/?solo=1&v=${[v.t0, v.t1, v.f0, v.f1].map((x) => x.toFixed(2)).join(',')}`;
  window.open(url, '_blank', 'popup,width=1100,height=650');
}
$('popOut').onclick = () => openWindow();
// every window follows changes made in the others
setInterval(async () => {
  try {
    const r = await api('/api/version');
    if (knownVersion >= 0 && r.version !== knownVersion) { const j = await api('/api/curves'); knownVersion = j.version; setCurves(j.curves, false); setOverlays(j.overlays); }
    else knownVersion = r.version;
    if (spec && r.name && r.name !== spec.name) location.reload();
  } catch (e) { }
}, 1500);

// ---------------------------------------------------------------- fits list
function setCurves(cs, bump = true) {
  curves = cs; if (sel >= cs.length) sel = cs.length - 1;
  const ul = $('curves'); ul.innerHTML = '';
  cs.forEach((c, i) => {
    const li = document.createElement('li'); if (i === sel) li.className = 'sel';
    const fit = c.fit ? `closest ${fmtT(c.fit.t_cpa, 0.1)} · ${Math.round(c.fit.speed)} m/s · ${(c.fit.d_cpa / 1000).toFixed(2)} km · ${c.fit.f0} Hz · rms ${c.fit.rms_hz} Hz` : `${c.n} points, not fitted yet`;
    li.innerHTML = `<span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span>
      <div><input class="nm" value="${c.name}" style="width:8em;padding:2px 4px"> <span class="meta">${c.t.length ? fmtT(c.t[0], 1) + '–' + fmtT(c.t[c.t.length - 1], 1) : ''}</span>
      <span class="badge ${c.fit ? 'ok' : 'todo'}">${c.fit ? 'fitted' : 'digitised'}</span>
      <div class="meta">${fit}</div>
      <div class="acts"><button class="ft ${c.fit ? '' : 'primary'}">${c.fit ? 'Refit' : 'Fit'}</button><button class="go">Go to</button><button class="win" title="Open this pass in its own window">⧉ Window</button>
      <label class="chk" style="display:inline-flex;margin:0"><input type="checkbox" class="sh" ${c.show ? 'checked' : ''}> show</label>
      <label class="chk" style="display:inline-flex;margin:0"><input type="checkbox" class="us" ${c.use ? 'checked' : ''}> use</label></div></div>
      <button class="x" title="delete">×</button>`;
    const passView = () => { const pad = Math.max(15, (c.t[c.t.length - 1] - c.t[0]) * 0.25), fmn = Math.min(...c.f), fmx = Math.max(...c.f), fp = Math.max(10, (fmx - fmn) * 0.35); return { t0: c.t[0] - pad, t1: c.t[c.t.length - 1] + pad, f0: fmn - fp, f1: fmx + fp }; };
    li.onclick = (e) => { if (e.target.closest('input,button')) return; sel = i; setCurves(curves); setTool(tool); };
    li.querySelector('.x').onclick = async () => { if (!confirm(`Delete ${c.name}?`)) return; const j = await api(`/api/curves/${i}`, { method: 'DELETE' }); if (sel === i) sel = -1; setCurves(j.curves); };
    li.querySelector('.nm').onchange = async (e) => setCurves((await patch(`/api/curves/${i}`, { name: e.target.value })).curves);
    li.querySelector('.sh').onchange = async (e) => setCurves((await patch(`/api/curves/${i}`, { show: e.target.checked })).curves);
    li.querySelector('.us').onchange = async (e) => setCurves((await patch(`/api/curves/${i}`, { use: e.target.checked })).curves);
    li.querySelector('.ft').onclick = () => { sel = i; fitCurve(i); };
    li.querySelector('.go').onclick = () => { sel = i; if (c.t.length) setView(passView()); setCurves(curves); };
    li.querySelector('.win').onclick = () => c.t.length && openWindow(passView());
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
// flights per aircraft type, each a map overlay that can be switched on and off
const CAT = { prop: ['Flights: propeller', '#80ed99'], jet: ['Flights: jet', '#c0c6d0'], heli: ['Flights: helicopter', '#ffd166'], unknown: ['Flights: unknown type', '#8a93a3'] };
const trackLayers = Object.fromEntries(Object.keys(CAT).map((k) => [k, L.layerGroup().addTo(map)]));
const resultLayer = L.layerGroup().addTo(map), worldLayer = L.layerGroup().addTo(map);
const mapOverlays = { ...Object.fromEntries(Object.entries(CAT).map(([k, [name, col]]) => [`<span style="color:${col}">━</span> ${name}`, trackLayers[k]])), 'Location result': resultLayer, 'Worldwide candidates': worldLayer };
L.control.layers(layers, mapOverlays, { position: 'topright', collapsed: false }).addTo(map);
L.control.scale({ imperial: false }).addTo(map);
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
  tracks = ts; Object.values(trackLayers).forEach((l) => l.clearLayers());
  ts.forEach((t) => { const c = CAT[t.category] ? t.category : 'unknown'; L.polyline(t.path, { color: CAT[c][1], weight: 1.5, opacity: 0.85 }).bindTooltip(`${t.label}${t.type ? ' · ' + t.type : ''} (${t.category})`).addTo(trackLayers[c]); });
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
  $('result').innerHTML = `<div class="result"><div><b>${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}</b> · <a href="${gmaps(r.lat, r.lon)}" target="_blank">Google Maps</a> <button class="mini" id="ovBest">on spectrogram</button></div><div>Recording started ${r.t_start_iso} · listener at ${Math.round(r.observer_alt)} m · ${r.n_tracks} flights considered</div>${rows}${others ? `<div class="sub" style="margin-top:4px">Other candidates (cost vs best): ${others}</div>` : ''}${truth}</div>`;
  map.fitBounds(r.bounds);
  $('ovBest').onclick = () => predictAt(r.lat, r.lon, r.t_start_iso, `best ${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}`);
  (r.candidates || []).slice(1).forEach(([la, lo], k) => { const b = document.createElement('button'); b.className = 'mini'; b.textContent = `candidate ${k + 2} on spectrogram`; b.onclick = () => predictAt(la, lo, r.t_start_iso, `candidate ${k + 2}`); $('result').firstChild.appendChild(b); });
}
// ---------------------------------------------------------------- candidate flights on the sound map
function setOverlays(ovs) {
  overlays = ovs || {}; const box = $('ovList'); box.innerHTML = '';
  Object.values(overlays).forEach((ov, oi) => {
    const m = ov.flights.filter((f) => f.f0 != null);
    const byPass = {}; m.forEach((f) => { if (!byPass[f.match] || f.rms_hz < byPass[f.match].rms_hz) byPass[f.match] = f; });
    const summary = curves.filter((c) => c.t.length).map((c) => byPass[c.name] ? `${c.name} → <b>${byPass[c.name].label}</b> ${byPass[c.name].type || ''} (${byPass[c.name].rms_hz.toFixed(1)} Hz)` : `${c.name} → <span style="color:#f8961e">nothing</span>`).join('<br>');
    const div = document.createElement('div'); div.className = 'ov';
    div.innerHTML = `<input type="checkbox" ${ov.show ? 'checked' : ''}><span class="swatch" style="background:${OCOLORS[oi % OCOLORS.length]};margin-top:3px"></span>
      <div><b>${ov.title}</b> · start ${ov.start.slice(11, 19)} UTC<div class="m">${ov.flights.length} flights within ${$('ovKm').value} km</div>${summary}</div><button class="x" title="remove">×</button>`;
    div.querySelector('input').onchange = async (e) => setOverlays((await patch(`/api/overlays/${encodeURIComponent(ov.key)}`, { show: e.target.checked })).overlays);
    div.querySelector('.x').onclick = async () => setOverlays((await api(`/api/overlays/${encodeURIComponent(ov.key)}`, { method: 'DELETE' })).overlays);
    box.appendChild(div);
  });
  draw();
}
async function predictAt(lat, lon, start, title) {
  status('Predicting the flights\' Doppler lines…', false, true);
  try {
    const j = await post('/api/predict', { lat, lon, start, alt: +$('alt').value, max_km: +$('ovKm').value, title });
    setOverlays(j.overlays); const ov = j.overlays[j.key];
    status(`${ov.flights.length} flights drawn on the spectrogram (${ov.flights.filter((f) => f.f0 != null).length} match a pass)`);
  } catch (err) { status(err.message, true); }
}
$('predictHere').onclick = () => {
  if (!$('lat').value || !$('start').value) { status('Set the search centre (click the map) and the recording start first', true); return; }
  predictAt(+$('lat').value, +$('lon').value, $('start').value, `centre ${(+$('lat').value).toFixed(4)}, ${(+$('lon').value).toFixed(4)}`);
};
$('showUnmatched').onchange = () => draw();

// ---------------------------------------------------------------- whole-day worldwide archive
function showDay(d) {
  if (!d) { $('dayInfo').textContent = ''; return; }
  const c = d.categories;
  $('dayInfo').innerHTML = `<b>${d.day}</b>: ${d.aircraft.toLocaleString()} aircraft below ${Math.round(d.max_alt_m / 1000)} km, worldwide (${c.prop.toLocaleString()} propeller, ${c.jet.toLocaleString()} jet, ${c.heli.toLocaleString()} helicopter, ${c.unknown.toLocaleString()} unknown)`;
}
$('dayBtn').onclick = async () => {
  const path = $('dayPath').value.trim(); if (!path) return; localStorage.setItem('dayPath', path);
  status('Opening the day…', false, true);
  try {
    let j = await post('/api/day/open', { path });
    while (j.building) {
      status(`Converting the day (one time only): ${Math.round(j.progress.fraction * 100)}%, ${(j.progress.aircraft || 0).toLocaleString()} aircraft so far`, false, true);
      await new Promise((r) => setTimeout(r, 2000)); j = await api('/api/day/status');
    }
    showDay(j.day); status('Day opened');
  } catch (err) { status(err.message, true); }
};
$('dayTracksBtn').onclick = async () => {
  if (!$('lat').value || !$('start').value) { status('Set the search centre (click the map) and the recording start in step 4 first', true); return; }
  status('Loading flights from the day…', false, true);
  try {
    const j = await post('/api/day/tracks', { lat: +$('lat').value, lon: +$('lon').value, radius: +$('radius').value + 20, start: $('start').value, uncertainty: +$('unc').value });
    setTracks(j.tracks); status(`${j.tracks.length} flights loaded`);
  } catch (err) { status(err.message, true); }
};
$('worldBtn').onclick = async () => {
  const btn = $('worldBtn'); btn.disabled = true; status('Searching the whole day worldwide…', false, true); const stop = pollProgress('Worldwide');
  try {
    const types = [...document.querySelectorAll('.ty:checked')].map((x) => x.value);
    const j = await post('/api/day/search', { types: types.length === 4 ? null : types, c: +$('c').value });
    stop(); showWorld(j.candidates); status(`${j.candidates.length} candidate places`);
  } catch (err) { stop(); status(err.message, true); }
  btn.disabled = false;
};
function showWorld(cs) {
  worldLayer.clearLayers(); const box = $('worldRes'); box.innerHTML = '';
  if (!cs.length) { box.innerHTML = '<div class="sub">No place matched every pass. Try fitting the passes again, or untick a pass.</div>'; return; }
  const best = cs.find((c) => c.cost != null);
  cs.forEach((c, k) => {
    const lat = c.lat ?? c.vote.lat, lon = c.lon ?? c.vote.lon, iso = c.t_start_iso || c.vote_iso;
    const rel = c.cost != null && best ? (c.cost / best.cost).toFixed(1) + '×' : 'no match';
    const div = document.createElement('div'); div.className = 'wc';
    div.innerHTML = `<b>#${k + 1}</b> ${lat.toFixed(4)}, ${lon.toFixed(4)} · start ${iso.slice(11, 19)} UTC<br><span class="meta">misfit ${rel} of the best · ${c.vote.busy} low-traffic counts nearby · <a href="${gmaps(lat, lon)}" target="_blank">Google Maps</a></span> <button class="mini ovw">on spectrogram</button>`;
    const use = () => { setCenter(lat, lon); $('start').value = iso; $('unc').value = 60; $('radius').value = 8; map.setView([lat, lon], 11); $('dayTracksBtn').click(); };
    div.querySelector('.ovw').onclick = () => predictAt(lat, lon, iso, `world #${k + 1}`);
    div.onclick = (e) => { if (e.target.tagName !== 'A' && e.target.tagName !== 'BUTTON') use(); };
    box.appendChild(div);
    L.circleMarker([lat, lon], { radius: k ? 6 : 9, color: k ? '#f8961e' : '#80ed99', weight: 2 }).bindTooltip(`#${k + 1}: ${iso.slice(11, 19)} UTC, misfit ${rel}`).on('click', use).addTo(worldLayer);
  });
  const pts = cs.map((c) => [c.lat ?? c.vote.lat, c.lon ?? c.vote.lon]);
  map.fitBounds(L.latLngBounds(pts).pad(0.3));
}
// remember the paths typed last time
for (const id of ['path', 'histPath', 'dayPath']) { const v = localStorage.getItem(id); if (v && !$(id).value) $(id).value = v; $(id).addEventListener('change', () => localStorage.setItem(id, $(id).value)); }

// ---------------------------------------------------------------- the model, explained
const MODEL = [
  ['h3', '1. What we measure'],
  ['p', 'A propeller or engine emits a steady tone $f_0$ (blade-passing frequency and its harmonics). A listener at $\\mathbf{O}$ hears it shifted by the aircraft\'s motion: higher while it approaches, lower once it has passed. The spectrogram shows this as a falling S-shaped line, one per pass.'],
  ['h3', '2. Sound takes time to arrive'],
  ['p', 'The sound heard at time $t$ left the aircraft earlier, at the emission time $\\tau$, when the aircraft was at $\\mathbf{P}(\\tau)$:'],
  ['d', 't = \\tau + \\frac{\\lVert \\mathbf{P}(\\tau) - \\mathbf{O} \\rVert}{c}'],
  ['p', 'with $c \\approx 340\\ \\mathrm{m/s}$ the speed of sound. It is solved for $\\tau$ by fixed-point iteration, $\\tau_{k+1} = t - \\lVert \\mathbf{P}(\\tau_k) - \\mathbf{O}\\rVert / c$, which converges in a few steps because the aircraft is much slower than sound.'],
  ['h3', '3. Doppler shift for a moving source and a still listener'],
  ['d', 'f(t) = f_0 \\, \\frac{c}{c - v_r(\\tau)}, \\qquad v_r(\\tau) = \\mathbf{v}(\\tau) \\cdot \\frac{\\mathbf{O} - \\mathbf{P}(\\tau)}{\\lVert \\mathbf{O} - \\mathbf{P}(\\tau) \\rVert}'],
  ['p', '$v_r$ is the speed of the aircraft towards the listener at the moment of emission ($v_r>0$ approaching). With ADS-B, $\\mathbf{P}(\\tau)$ and $\\mathbf{v}(\\tau)$ come from a smoothing spline through the reported positions (in Earth-centred coordinates), so each flight predicts a whole curve $f(t)$ for any listener position.'],
  ['h3', '4. Straight-line flyover: the curve fitted to each pass'],
  ['p', 'For one pass we assume a straight, level flight at constant speed $v$. Let $t_\\mathrm{cpa}$ be the (emission) time of closest approach and $d$ the 3D distance then. The aircraft is $x(\\tau) = v\\,(\\tau - t_\\mathrm{cpa})$ along its track, and'],
  ['d', 'v_r(\\tau) = -\\,\\frac{v\\,x(\\tau)}{\\sqrt{x(\\tau)^2 + d^2}}, \\qquad t = \\tau + \\frac{\\sqrt{x(\\tau)^2 + d^2}}{c}'],
  ['d', 'f(t) = f_0\\,\\frac{c}{c - v_r(\\tau(t))}'],
  ['p', 'Four unknowns $(f_0,\\ t_\\mathrm{cpa},\\ v,\\ d)$ shape the curve: $f_0$ sets its middle, $v$ sets how far apart the start and end tones are, $\\frac{f_\\text{approach}}{f_\\text{recede}} \\to \\frac{c+v}{c-v}$, and $d$ sets how sharp the drop is (the slope at closest approach is about $-f_0 v^2 / (c\\,d)$).'],
  ['h3', '5. Digitising and fitting'],
  ['p', 'Your guide clicks $(t_i, g_i)$ are joined by a monotone spline $g(t)$. In every spectrogram frame only the band $|f - g(t)| \\le w$ is searched ($w$ = the ± Hz box, or 4 % of the frequency), the strongest bin is taken and refined by a parabola through its neighbours:'],
  ['d', '\\hat f_j = f_{b} + \\frac{\\Delta f}{2}\\,\\frac{S_{b-1} - S_{b+1}}{S_{b-1} - 2S_b + S_{b+1}}'],
  ['p', 'Weak peaks, peaks on the band edge, and points more than 2 Hz off a running median are dropped. Then the four parameters are found by robust non-linear least squares (soft-L1 loss, so a few stray points barely matter), started from several distances to avoid a wrong local minimum:'],
  ['d', '\\min_{f_0,\\,t_\\mathrm{cpa},\\,v,\\,d}\\ \\sum_j \\big(\\hat f_j - f(t_j;\\,f_0, t_\\mathrm{cpa}, v, d)\\big)^2'],
  ['p', 'The rms residual (Hz) says how well a straight flyover explains the points. A turning or climbing aircraft fits worse.'],
  ['h3', '6. Finding the place'],
  ['p', 'For a candidate listener position $\\mathbf{O}$ (a grid over the search area, with the ground height from a DEM) and recording start $T$, every flight $k$ predicts a curve shape $r_k(t) = c / (c - v_r)$. Each pass $j$ is compared with each flight, solving for its unknown tone in closed form:'],
  ['d', '\\hat f_{0} = \\frac{\\sum_i \\hat f_i\\, r_k(t_i)}{\\sum_i r_k(t_i)^2}, \\qquad \\mathrm{cost}_{jk} = \\frac{\\sqrt{\\tfrac1n \\sum_i (\\hat f_i - \\hat f_0\\, r_k(t_i))^2}}{\\hat f_0}'],
  ['p', 'Flights are first filtered by type (propeller / jet), and by the speed and closest distance fitted in step 5. The passes are locked together: they share one start time $T$, so the fixed gaps between them in the recording (e.g. B and C, 920 s apart) must match the flights. Each pass may shift by a couple of seconds (timing slack) and its tone is free. The total cost over all passes, plus a penalty for loud flights that should have been heard but are not in the recording (silence), is minimised over $(\\mathbf{O}, T)$, first on a coarse grid then refined.'],
  ['h3', '7. Checking a candidate by eye'],
  ['p', 'For a chosen place and start time, every flight within the set distance predicts its line $r_k(t)$. For each picked pass it overlaps, its tone is set to $\\hat f_0$ above and the rms misfit is shown; the line is drawn on the spectrogram. A good candidate puts a flight on top of every pass, and puts no loud line where the recording is silent.'],
  ['h3', '7. Worldwide search'],
  ['p', 'Without a known area, every low aircraft sample of the whole day with the right ground speed "votes" for the places abeam of it at up to the fitted distance, and for the start time $T = \\tau_\\text{sample} - t_\\mathrm{cpa}$. Places and times voted by every pass (by different aircraft) are candidates; busy places such as airports, where low aircraft pass all the time, are ranked down because the recording is mostly quiet. The best candidates are then checked with the full curve match above.'],
];
function renderModel() {
  const box = $('modelText'); if (box.dataset.done) return;
  const tex = (src, display) => { try { return katex.renderToString(src, { displayMode: display, throwOnError: false }); } catch (e) { return src; } };
  box.innerHTML = '<h2 style="margin:0 0 4px;color:var(--text);text-transform:none;letter-spacing:0;font-size:18px">How Doppler curves are modelled and fitted</h2>' + MODEL.map(([k, t]) => {
    if (k === 'd') return `<div>${tex(t, true)}</div>`;
    const html = t.replace(/\$([^$]+)\$/g, (_, m) => tex(m, false));
    return k === 'h3' ? `<h3>${html}</h3>` : `<p>${html}</p>`;
  }).join('');
  box.dataset.done = '1';
}
document.querySelectorAll('.openModel').forEach((b) => b.addEventListener('click', (e) => { e.preventDefault(); renderModel(); $('modelBox').style.display = 'block'; }));
$('modelClose').onclick = () => { $('modelBox').style.display = 'none'; };
$('modelBox').onclick = (e) => { if (e.target.id === 'modelBox') $('modelBox').style.display = 'none'; };

$('demoBtn').onclick = async () => {
  status('Rendering the synthetic demo recording…', false, true);
  try {
    const j = await post('/api/demo', {});
    setCurves(j.curves || []); await setSpec(j.spec); setTracks(j.tracks, false);
    const g = j.guess; $('start').value = g.start; $('unc').value = g.uncertainty; $('radius').value = g.radius; $('alt').value = g.alt;
    setCenter(g.lat, g.lon); map.setView([g.lat, g.lon], 9);
    status('Demo loaded: with ✎ Trace, click along each falling line near 100 Hz (~100 s and ~200 s), Enter, Fit; then Find location');
  } catch (err) { status(err.message, true); }
};

// restore state on reload
(async () => {
  try { const s = await api('/api/state'); knownVersion = s.version; setCurves(s.curves); setOverlays(s.overlays); if (s.spec.loaded) await setSpec(s.spec); if (!SOLO && s.tracks.length) setTracks(s.tracks); showDay(s.day); } catch (e) { }
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
