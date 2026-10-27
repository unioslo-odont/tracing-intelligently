'use strict';

const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const colors = ['#74e0c8', '#f5b76b', '#b399f2', '#78b8ef', '#ef8ca5', '#c9da77'];
const state = {
  catalog: null, detail: null, tracing: null, points: {}, selected: null,
  baseline: '', dirty: false, undo: [], redo: [], view: null, zoom: 1,
  busy: false, loading: false, loadSequence: 0, drag: null, toastTimer: null,
  imageReady: false,
};

function clone(value) { return JSON.parse(JSON.stringify(value)); }
function node(tag, attrs = {}) {
  const el = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}
function button(text, className, callback) {
  const el = document.createElement('button');
  el.textContent = text; el.className = className; el.addEventListener('click', callback);
  return el;
}
function span(text, className = '') {
  const el = document.createElement('span'); el.textContent = text; el.className = className; return el;
}
function toast(message, error = false) {
  clearTimeout(state.toastTimer);
  $('toast').textContent = message; $('toast').className = `toast${error ? ' error' : ''}`;
  $('toast').hidden = false;
  state.toastTimer = setTimeout(() => { $('toast').hidden = true; }, error ? 10000 : 4500);
}
async function api(url, payload) {
  const options = payload === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Review-Token': state.catalog.token },
    body: JSON.stringify(payload),
  };
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${response.status}).`);
  }
  return response.json();
}
function original() { return state.detail?.tracings.find((t) => t.number === state.tracing)?.points || {}; }
function outside(point) {
  const d = state.detail;
  return point && !(point[0] >= 0 && point[0] < d.width && point[1] >= 0 && point[1] < d.height);
}
function changed(name) { return JSON.stringify(state.points[name]) !== JSON.stringify(original()[name]); }
function snapshot() { return JSON.stringify({ tracing: state.tracing, points: state.points, notes: $('notes').value }); }
function updateDirty() {
  state.dirty = !!state.detail && snapshot() !== state.baseline;
  renderSaveState(); updateButtons();
}
function checkpoint() {
  state.undo.push(clone(state.points));
  if (state.undo.length > 100) state.undo.shift();
  state.redo = [];
}
function undo(redo = false) {
  if (state.busy || state.loading) return;
  const source = redo ? state.redo : state.undo, target = redo ? state.undo : state.redo;
  if (!source.length) return;
  target.push(clone(state.points)); state.points = source.pop();
  updateDirty(); renderLandmarks(); renderOverlay();
}
function canLeave() {
  if (state.busy || state.loading) return false;
  return !state.dirty || window.confirm('This image has unsaved changes. Discard them and continue?');
}
function filteredItems() {
  if (!state.catalog) return [];
  const search = $('search').value.toLowerCase(), filter = $('filter').value;
  return state.catalog.items.filter((item) => item.id.toLowerCase().includes(search)
    && (filter === 'all' || (filter === 'warning' ? item.warning : item.status === filter)));
}
function renderQueue() {
  const items = state.catalog.items;
  const counts = { pending: 0, draft: 0, accepted: 0, excluded: 0 };
  items.forEach((item) => { counts[item.status]++; });
  $('queue-total').textContent = items.length;
  $('reviewed-count').textContent = counts.accepted + counts.excluded;
  $('counts').textContent = `${counts.accepted} accepted · ${counts.pending} awaiting review`;
  $('progress-fill').style.width = `${items.length ? (counts.accepted + counts.excluded) / items.length * 100 : 0}%`;
  $('queue').replaceChildren();
  filteredItems().forEach((item) => {
    const row = button('', `queue-item${item.id === state.detail?.id ? ' current' : ''}`, () => loadImage(item.id));
    row.setAttribute('aria-current', item.id === state.detail?.id ? 'true' : 'false');
    row.title = `${item.id} · ${item.status}${item.warning ? ' · import warning' : ''}`;
    row.append(span('◈', 'queue-icon'));
    const text = document.createElement('div');
    text.append(span(item.id.replace(/\.[^.]+$/, ''), 'queue-name'),
      span(`${item.tracing_count} tracing${item.tracing_count === 1 ? '' : 's'} · ${item.status}${item.warning ? ' ⚠' : ''}`, 'queue-detail'));
    row.append(text, span('', `status-dot ${item.status}`)); $('queue').append(row);
  });
  if (!filteredItems().length) $('queue').append(span(state.catalog.loaded ? 'No images match this filter.' : 'No dataset open.', 'small muted'));
  updateNavigation(); updateButtons();
}
function updateNavigation() {
  const items = filteredItems(), index = items.findIndex((i) => i.id === state.detail?.id);
  $('image-position').textContent = index >= 0 ? `${index + 1} / ${items.length}` : '';
  $('previous').disabled = state.busy || state.loading || index <= 0;
  $('next').disabled = state.busy || state.loading || index < 0 || index >= items.length - 1;
}
async function navigate(step) {
  const items = filteredItems(), index = items.findIndex((item) => item.id === state.detail?.id);
  const target = items[index + step];
  if (target) await loadImage(target.id);
}
async function loadImage(id, skipPrompt = false) {
  if (!skipPrompt && !canLeave()) return;
  const sequence = ++state.loadSequence;
  state.loading = true; updateButtons(); updateNavigation();
  try {
    const d = await api(`/api/detail?id=${encodeURIComponent(id)}`);
    if (sequence !== state.loadSequence) return;
    state.detail = d;
    const queueItem = state.catalog.items.find((item) => item.id === id);
    if (queueItem) { queueItem.status = d.stale ? 'pending' : d.review?.status || 'pending'; queueItem.revision = d.review?.revision || 0; }
    const review = d.stale ? null : d.review;
    const suggested = d.imported_selection?.numbers?.[0];
    const initial = d.tracings.find((t) => t.number === suggested) || d.tracings[0];
    state.tracing = review?.tracing_number ?? initial?.number ?? null;
    state.points = clone(review?.points ?? initial?.points ?? {});
    state.selected = null; state.undo = []; state.redo = []; state.drag = null;
    $('notes').value = review?.notes || '';
    state.baseline = snapshot(); state.dirty = false;
    $('image-title').textContent = d.id.replace(/\.[^.]+$/, '');
    const calibration = d.pixel_spacing_mm ? `${d.pixel_spacing_mm.toFixed(4)} mm/px (Facad scale)` : 'No pixel calibration';
    $('image-meta').textContent = `${d.width} × ${d.height} px · ${calibration}`;
    state.imageReady = false;
    $('radiograph').setAttribute('href', `/api/image?id=${encodeURIComponent(id)}&v=${d.source_signature}&load=${sequence}`);
    $('radiograph').setAttribute('width', d.width); $('radiograph').setAttribute('height', d.height);
    $('viewer-empty').hidden = true; $('viewer-badge').hidden = false;
    resetDisplay(); fit(); renderTracings(); renderLandmarks(); renderHistory(); renderWarning(); renderQueue(); renderSaveState();
    if (d.stale) toast('Source files changed since the saved review. Choose and review a current tracing before accepting.', true);
  } catch (error) { toast(error.message, true); }
  finally { if (sequence === state.loadSequence) { state.loading = false; updateButtons(); updateNavigation(); } }
}
function chooseTracing(number) {
  if (state.busy || state.loading || number === state.tracing) return;
  const source = state.detail.tracings.find((t) => t.number === number);
  if (!source) return;
  if (state.dirty && !window.confirm('Replace unsaved landmark edits with this source tracing?')) return;
  state.tracing = number; state.points = clone(source.points); state.undo = []; state.redo = [];
  state.selected = null;
  updateDirty(); renderTracings(); renderLandmarks(); renderOverlay(); renderWarning();
}
function renderTracings() {
  $('tracings').replaceChildren();
  $('tracing-count').textContent = `${state.detail.tracings.length} available`;
  state.detail.tracings.forEach((t) => {
    const card = button('', `tracing-card${state.tracing === t.number ? ' active' : ''}`, () => chooseTracing(t.number));
    card.setAttribute('aria-pressed', state.tracing === t.number ? 'true' : 'false');
    const title = span('', 'tracing-name'), swatch = span('', 'tracing-swatch');
    swatch.style.backgroundColor = colors[(t.number - 1) % colors.length];
    title.append(swatch, document.createTextNode(`Tracing ${t.number}`));
    card.append(title, span(`${Object.keys(t.points).length} points · ${t.author || 'Reviewer unspecified'}`, 'tracing-detail'));
    if (t.name) card.append(span(t.name, 'tracing-detail'));
    if (t.out_of_bounds.length || t.warnings.length) card.append(span('⚠ Check source quality', 'tracing-detail'));
    $('tracings').append(card);
  });
  if (!state.detail.tracings.length) $('tracings').append(span('No linked tracings. This image can be excluded.', 'small muted'));
}
function allNames() {
  const names = new Set(Object.keys(state.catalog.landmarks));
  state.detail?.tracings.forEach((t) => (t.marker_names || Object.keys(t.points)).forEach((name) => names.add(name)));
  Object.keys(state.points).forEach((name) => names.add(name));
  return [...names].sort((a, b) => a.localeCompare(b, 'en', { sensitivity: 'variant' }));
}
function renderLandmarks() {
  $('landmark-count').textContent = `${Object.keys(state.points).length} / ${allNames().length}`;
  $('landmark-list').replaceChildren();
  allNames().forEach((name) => {
    const p = state.points[name];
    const row = button('', `landmark-row${name === state.selected ? ' selected' : ''}${!p ? ' missing' : ''}${changed(name) ? ' changed' : ''}${outside(p) ? ' outside' : ''}`, () => selectPoint(name));
    const code = span('', 'point-code'); code.append(span('', 'point-indicator'), document.createTextNode(name));
    const position = p ? `${p[0].toFixed(1)}, ${p[1].toFixed(1)}` : 'Missing · place +';
    row.append(code, span(position, 'coordinates'));
    row.title = state.catalog.landmarks[name] || name;
    row.setAttribute('aria-pressed', name === state.selected ? 'true' : 'false');
    $('landmark-list').append(row);
  });
  renderPointEditor();
}
function selectPoint(name) {
  if (state.busy || state.loading) return;
  state.selected = name; renderLandmarks(); renderOverlay();
}
function renderPointEditor() {
  const name = state.selected, p = state.points[name], source = original()[name];
  $('point-editor').hidden = !name;
  if (!name) return;
  $('point-name').textContent = name;
  $('point-description').textContent = state.catalog.landmarks[name] || 'Landmark name preserved from Facad.';
  $('point-x').value = p ? Number(p[0].toFixed(3)) : '';
  $('point-y').value = p ? Number(p[1].toFixed(3)) : '';
  $('point-x').disabled = $('point-y').disabled = !p || state.busy || state.loading;
  $('focus-point').disabled = !p;
  $('restore-point').disabled = !source || state.busy || state.loading;
  $('remove-point').disabled = !p || state.busy || state.loading;
  $('place-hint').hidden = !!p;
  if (p && source) {
    const distance = Math.hypot(p[0] - source[0], p[1] - source[1]);
    $('point-delta').textContent = `${distance.toFixed(2)} px from source${state.detail.pixel_spacing_mm ? ` · ${(distance * state.detail.pixel_spacing_mm).toFixed(3)} mm` : ''}`;
  } else $('point-delta').textContent = p ? 'Added during review' : 'Unavailable landmarks remain masked in training.';
}
function renderWarning() {
  if (!state.detail) return;
  const warnings = [], d = state.detail;
  const bad = Object.entries(state.points).filter(([, p]) => outside(p)).map(([name]) => name);
  if (bad.length) warnings.push(`${bad.length} points outside the actual image (${bad.join(', ')}). Check image/tracing alignment before accepting.`);
  const source = d.tracings.find((t) => t.number === state.tracing);
  if (source?.warnings.length) warnings.push(...source.warnings);
  const missing = Object.keys(state.catalog.landmarks).filter((name) => !state.points[name]);
  if (missing.length) warnings.push(`${missing.length} configured landmarks missing. Place visible landmarks or leave them masked.`);
  if (d.stale) warnings.push('Saved review refers to older source files. The current source must be reviewed again.');
  if (!d.review && d.imported_selection) {
    const selection = d.imported_selection;
    warnings.push(selection.numbers ? `Previous review suggested tracing(s) ${selection.raw}; choose one reference and save a new decision.` : 'Previous review marked this image excluded. Confirm a new decision before export.');
  }
  $('warning').textContent = warnings.join(' '); $('warning').hidden = !warnings.length;
}
function renderSaveState() {
  const d = state.detail;
  $('save-state').className = `save-state${state.dirty ? ' dirty' : ''}`;
  if (!d) $('save-state').textContent = 'No image selected';
  else if (state.busy) $('save-state').textContent = 'Saving review…';
  else if (state.dirty) $('save-state').textContent = '● Unsaved changes · save a draft to keep them';
  else if (d.review && !d.stale) $('save-state').textContent = `${d.review.status.charAt(0).toUpperCase() + d.review.status.slice(1)} · revision ${d.review.revision} · ${d.review.reviewer}`;
  else $('save-state').textContent = 'Awaiting review · source tracing selected';
  $('viewer-badge').textContent = state.tracing ? `Tracing ${state.tracing}${state.dirty ? ' · edited' : ''}` : 'No tracing';
}
function renderHistory() {
  $('history').replaceChildren();
  if (!state.detail.history.length) { $('history').textContent = 'No saved revisions.'; return; }
  state.detail.history.forEach((entry) => {
    const el = document.createElement('p');
    el.textContent = `r${entry.revision} · ${entry.status} · ${entry.reviewer} · ${new Date(entry.saved_at).toLocaleString()}`;
    $('history').append(el);
  });
}
function updateButtons() {
  const blocked = state.busy || state.loading || !state.detail;
  const noPoints = !Object.keys(state.points).length;
  const bad = state.detail && Object.values(state.points).some(outside);
  $('accept').disabled = blocked || !state.tracing || noPoints || bad || !state.imageReady;
  $('draft').disabled = blocked || !state.tracing;
  $('exclude').disabled = blocked;
  $('undo').disabled = blocked || !state.undo.length;
  $('redo').disabled = blocked || !state.redo.length;
  $('notes').disabled = $('reviewer').disabled = state.busy || state.loading;
  $('export').disabled = state.busy || state.loading || !state.catalog?.items.some((i) => i.status === 'accepted');
  document.querySelectorAll('.tracing-card').forEach((el) => { el.disabled = state.busy || state.loading; });
  if (state.selected) renderPointEditor();
}

function fit() {
  if (!state.detail) return;
  state.view = { x: 0, y: 0, w: state.detail.width, h: state.detail.height };
  state.zoom = 1; applyView();
}
function applyView() {
  if (!state.view) return;
  const v = state.view;
  $('canvas').setAttribute('viewBox', `${v.x} ${v.y} ${v.w} ${v.h}`);
  $('zoom-label').textContent = `${Math.round(state.zoom * 100)}%`;
  renderOverlay();
}
function zoom(factor, anchor) {
  if (!state.view) return;
  const nextZoom = Math.min(20, Math.max(0.5, state.zoom * factor));
  factor = nextZoom / state.zoom;
  const v = state.view, center = anchor || { x: v.x + v.w / 2, y: v.y + v.h / 2 };
  state.view = { x: center.x - (center.x - v.x) / factor, y: center.y - (center.y - v.y) / factor,
    w: v.w / factor, h: v.h / factor };
  state.zoom = nextZoom; applyView();
}
function screenPoint(event) {
  const matrix = $('canvas').getScreenCTM();
  return matrix ? new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse()) : null;
}
function renderOverlay() {
  $('ghost-points').replaceChildren(); $('editable-points').replaceChildren();
  if (!state.detail || !state.view || !$('show-points').checked) return;
  const matrix = $('canvas').getScreenCTM(), unit = matrix?.a ? 1 / matrix.a : state.view.w / 600;
  if ($('compare').checked) state.detail.tracings.forEach((t) => {
    Object.entries(t.points).forEach(([name, [x, y]]) => {
      const circle = node('circle', { cx: x, cy: y, r: unit * 4, fill: 'none',
        stroke: colors[(t.number - 1) % colors.length], 'stroke-width': unit * 1.3, opacity: 0.6 });
      const title = node('title'); title.textContent = `Tracing ${t.number}: ${name}`; circle.append(title);
      $('ghost-points').append(circle);
    });
  });
  if ($('show-source').checked) Object.entries(original()).forEach(([, [x, y]]) => {
    $('ghost-points').append(node('path', { d: `M${x - 4 * unit},${y}h${8 * unit}M${x},${y - 4 * unit}v${8 * unit}`,
      stroke: '#ffffff', 'stroke-width': unit, opacity: 0.6 }));
  });
  Object.entries(state.points).forEach(([name, [x, y]]) => {
    const selected = name === state.selected, group = node('g', { class: `landmark-point${selected ? ' selected' : ''}`, 'data-name': name });
    const color = changed(name) ? '#f5b76b' : '#74e0c8';
    // The transparent hit area is larger than the visible point for precise placement.
    group.append(node('circle', { cx: x, cy: y, r: unit * 11, fill: 'transparent' }));
    if (selected) group.append(node('circle', { cx: x, cy: y, r: unit * 9, fill: 'none', stroke: color, 'stroke-width': unit, opacity: 0.85 }));
    group.append(node('circle', { cx: x, cy: y, r: unit * 3.6, fill: color, stroke: '#10231e', 'stroke-width': unit * 1.2 }));
    if ($('show-labels').checked) {
      const label = node('text', { x: x + unit * 9, y: y - unit * 7, 'font-size': unit * 11, class: 'landmark-label' });
      label.textContent = name; group.append(label);
    }
    $('editable-points').append(group);
  });
}
function clampPoint(p) {
  return [Math.max(0, Math.min(state.detail.width - 1, p[0])), Math.max(0, Math.min(state.detail.height - 1, p[1]))];
}
function setPoint(name, point, keepInside = false) {
  if (state.busy || state.loading || !point.every(Number.isFinite)) return;
  checkpoint(); state.points[name] = keepInside ? clampPoint(point) : point;
  updateDirty(); renderLandmarks(); renderOverlay(); renderWarning();
}
$('canvas').addEventListener('pointerdown', (event) => {
  if (!state.detail || state.busy || state.loading || event.button !== 0) return;
  const p = screenPoint(event); if (!p) return;
  const marker = event.target.closest('[data-name]');
  if (marker) {
    const name = marker.getAttribute('data-name');
    state.selected = name; checkpoint();
    state.drag = { type: 'point', name, start: p, point: [...state.points[name]], pointer: event.pointerId };
    renderLandmarks(); renderOverlay();
  } else if (state.selected && !state.points[state.selected]) {
    if (p.x < 0 || p.y < 0 || p.x >= state.detail.width || p.y >= state.detail.height) return;
    setPoint(state.selected, [p.x, p.y], true);
  } else {
    state.drag = { type: 'pan', clientX: event.clientX, clientY: event.clientY, view: { ...state.view }, pointer: event.pointerId };
  }
  $('canvas').setPointerCapture(event.pointerId);
  $('canvas').focus({ preventScroll: true }); event.preventDefault();
});
$('canvas').addEventListener('pointermove', (event) => {
  const drag = state.drag; if (!drag || drag.pointer !== event.pointerId) return;
  if (drag.type === 'point') {
    const p = screenPoint(event); if (!p) return;
    state.points[drag.name] = clampPoint([drag.point[0] + p.x - drag.start.x, drag.point[1] + p.y - drag.start.y]);
    renderOverlay(); renderPointEditor(); updateDirty();
  } else {
    const matrix = $('canvas').getScreenCTM(); if (!matrix) return;
    state.view = { ...drag.view, x: drag.view.x - (event.clientX - drag.clientX) / matrix.a,
      y: drag.view.y - (event.clientY - drag.clientY) / matrix.d }; applyView();
  }
});
function endDrag(event) {
  if (!state.drag || state.drag.pointer !== event.pointerId) return;
  state.drag = null; renderLandmarks(); renderWarning(); updateButtons();
}
$('canvas').addEventListener('pointerup', endDrag);
$('canvas').addEventListener('pointercancel', endDrag);
$('canvas').addEventListener('lostpointercapture', endDrag);
$('canvas').addEventListener('wheel', (event) => {
  if (!state.detail) return;
  event.preventDefault(); zoom(event.deltaY < 0 ? 1.12 : 1 / 1.12, screenPoint(event));
}, { passive: false });
$('radiograph').addEventListener('load', () => { state.imageReady = true; updateButtons(); });
$('radiograph').addEventListener('error', () => {
  state.imageReady = false; updateButtons(); toast('Image preview failed. Check that the original image can be decoded.', true);
});

function display() {
  const slope = Number($('contrast').value) / 100;
  const intercept = (1 - slope) / 2 + Number($('brightness').value) / 100;
  ['red-brightness', 'green-brightness', 'blue-brightness'].forEach((id) => {
    $(id).setAttribute('slope', slope); $(id).setAttribute('intercept', intercept);
  });
}
function resetDisplay() { $('brightness').value = 0; $('contrast').value = 100; display(); }

async function save(status) {
  if (state.busy || state.loading || !state.detail) return;
  if (status === 'accepted' && !state.imageReady) { toast('Wait for the image preview to load before accepting.', true); return; }
  const reviewer = $('reviewer').value.trim();
  if (!reviewer) { toast('Enter your reviewer name before saving.', true); $('reviewer').focus(); return; }
  const imageId = state.detail.id;
  // Keep the next pending ID before the saved image disappears from a filtered queue.
  const visible = filteredItems(), currentIndex = visible.findIndex((item) => item.id === imageId);
  const forward = [...visible.slice(currentIndex + 1), ...visible.slice(0, currentIndex)];
  const nextPending = forward.find((item) => ['pending', 'draft'].includes(item.status));
  state.busy = true; updateButtons(); renderSaveState();
  try {
    const review = await api('/api/review', { image_id: imageId, revision: state.detail.review?.revision || 0,
      source_signature: state.detail.source_signature, tracing_number: state.tracing,
      points: state.points, status, reviewer, notes: $('notes').value });
    state.detail.review = review; state.detail.stale = false;
    state.detail.history.unshift({ revision: review.revision, status, reviewer, saved_at: review.saved_at });
    const item = state.catalog.items.find((i) => i.id === imageId); item.status = status; item.revision = review.revision;
    state.baseline = snapshot(); state.dirty = false;
    try { localStorage.setItem('openceph-reviewer', reviewer); } catch (_) { /* Optional preference only. */ }
    renderQueue(); renderHistory(); renderWarning();
    toast(status === 'accepted' ? 'Accepted. Corrected landmarks saved for the next round.' : status === 'excluded' ? 'Image excluded from the training round.' : 'Draft saved. You can resume this review later.');
    state.busy = false;
    if (status === 'accepted' && nextPending) await loadImage(nextPending.id, true);
    else if (status === 'accepted') toast('Review saved. No further pending images in this queue.');
  } catch (error) { toast(error.message, true); }
  finally { state.busy = false; updateButtons(); updateNavigation(); renderSaveState(); }
}
async function exportRound() {
  if (state.busy || state.loading || !state.catalog) return;
  if (state.dirty) { toast('Save the current edits as accepted or draft before exporting.', true); return; }
  state.busy = true; updateButtons();
  const previous = $('export').textContent; $('export').textContent = 'Preparing round…';
  try {
    // Stream directly to disk in browsers supporting the File System Access API.
    // The fallback downloads a Blob for browsers without it.
    let handle = null;
    if (window.showSaveFilePicker) {
      try { handle = await window.showSaveFilePicker({ suggestedName: 'curated-training-round.zip',
        types: [{ description: 'Training round', accept: { 'application/zip': ['.zip'] } }] }); }
      catch (error) { if (error.name === 'AbortError') return; throw error; }
    }
    const response = await fetch('/api/export', { method: 'POST', headers: { 'X-Review-Token': state.catalog.token } });
    if (!response.ok) { const data = await response.json(); throw new Error(data.error); }
    if (handle && response.body) {
      const writable = await handle.createWritable(); await response.body.pipeTo(writable);
    } else {
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a'); link.href = url; link.download = 'curated-training-round.zip';
      document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 60000);
    }
    toast('Training round exported: images, corrected Facad references, CSV, and review audit.');
  } catch (error) { toast(error.message || 'Export failed.', true); }
  finally { $('export').textContent = previous; state.busy = false; updateButtons(); renderSaveState(); }
}

$('search').addEventListener('input', renderQueue);
$('filter').addEventListener('change', renderQueue);
$('previous').addEventListener('click', () => navigate(-1));
$('next').addEventListener('click', () => navigate(1));
$('fit').addEventListener('click', fit);
$('zoom-in').addEventListener('click', () => zoom(1.25));
$('zoom-out').addEventListener('click', () => zoom(0.8));
['show-labels', 'show-points', 'compare', 'show-source'].forEach((id) => $(id).addEventListener('change', renderOverlay));
$('undo').addEventListener('click', () => undo()); $('redo').addEventListener('click', () => undo(true));
$('brightness').addEventListener('input', display); $('contrast').addEventListener('input', display);
$('reset-display').addEventListener('click', resetDisplay);
$('notes').addEventListener('input', updateDirty);
['point-x', 'point-y'].forEach((id) => $(id).addEventListener('change', () => {
  if (!state.selected || !state.points[state.selected]) return;
  const x = $('point-x').valueAsNumber, y = $('point-y').valueAsNumber;
  if (!Number.isFinite(x) || !Number.isFinite(y)) { toast('Enter finite X and Y coordinates.', true); renderPointEditor(); return; }
  setPoint(state.selected, [x, y]);
}));
$('restore-point').addEventListener('click', () => {
  const p = original()[state.selected]; if (p) setPoint(state.selected, [...p]);
});
$('remove-point').addEventListener('click', () => {
  if (state.busy || state.loading || !state.points[state.selected]) return;
  checkpoint(); delete state.points[state.selected]; updateDirty(); renderLandmarks(); renderOverlay(); renderWarning();
});
$('focus-point').addEventListener('click', () => {
  const p = state.points[state.selected]; if (!p || !state.detail) return;
  const w = state.detail.width / 4, h = state.detail.height / 4;
  state.view = { x: p[0] - w / 2, y: p[1] - h / 2, w, h }; state.zoom = 4; applyView(); $('canvas').focus();
});
$('accept').addEventListener('click', () => save('accepted'));
$('draft').addEventListener('click', () => save('draft'));
$('exclude').addEventListener('click', () => save('excluded'));
$('export').addEventListener('click', exportRound);
window.addEventListener('beforeunload', (event) => { if (state.dirty || state.busy) { event.preventDefault(); event.returnValue = ''; } });
document.addEventListener('keydown', (event) => {
  if (event.target.closest('input,textarea,select') || $('open-dialog').open || state.busy || state.loading) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); undo(event.shiftKey); return; }
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key.toLowerCase() === 'f') { fit(); event.preventDefault(); }
  if (event.key === '[') navigate(-1);
  if (event.key === ']') navigate(1);
  const direction = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
  const p = state.points[state.selected];
  if (direction && p) { event.preventDefault(); const step = event.shiftKey ? 0.1 : 1;
    setPoint(state.selected, [p[0] + direction[0] * step, p[1] + direction[1] * step], true); }
});
new ResizeObserver(renderOverlay).observe($('viewer'));

function resetWorkspace() {
  // Invalidate any in-flight image load, then return the viewer to its empty state.
  state.loadSequence++; state.loading = false; state.busy = false;
  Object.assign(state, { detail: null, tracing: null, points: {}, selected: null, undo: [], redo: [],
    drag: null, baseline: '', dirty: false, imageReady: false, view: null });
  $('radiograph').removeAttribute('href'); $('canvas').removeAttribute('viewBox');
  $('viewer-empty').hidden = false; $('viewer-badge').hidden = true;
  $('image-title').textContent = 'Select an image';
  $('image-meta').textContent = 'Choose a tracing, refine its landmarks, and accept it for training.';
  $('notes').value = ''; $('warning').hidden = true;
  $('tracings').replaceChildren(); $('tracing-count').textContent = '';
  $('landmark-list').replaceChildren(); $('landmark-count').textContent = '0';
  $('point-editor').hidden = true; $('history').textContent = 'No saved revisions.';
  $('issues').replaceChildren(); $('issues-panel').hidden = true;
  renderOverlay(); renderSaveState();
}
async function showCatalog(catalog) {
  resetWorkspace();
  state.catalog = catalog;
  $('environment').textContent = catalog.demo ? 'SYNTHETIC DEMO' : 'LOCAL WORKSPACE';
  $('viewer-empty-title').textContent = catalog.loaded ? 'Your next reference starts here' : 'No dataset open';
  $('viewer-empty-text').textContent = catalog.loaded ? 'Load an image from the queue to compare and refine its tracings.'
    : 'Choose the images and Facad folders with “Open folders…”.';
  if (catalog.issues.length) {
    $('issues-panel').hidden = false; $('issues-title').textContent = `${catalog.issues.length} import issues`;
    catalog.issues.forEach((issue) => { const li = document.createElement('li'); li.textContent = issue; $('issues').append(li); });
  }
  renderQueue(); updateButtons();
  const initial = catalog.items.find((item) => item.status === 'pending') || catalog.items[0];
  if (initial) await loadImage(initial.id);
  else if (catalog.loaded) toast('No reviewable images found. Check the folders and the import issues.', true);
}

// Folder chooser. The browser cannot reveal real folder paths, so the local server lists folders.
const picker = { target: null, path: '', sequence: 0 };
const folderFields = { images: ['images-folder', 'Images folder'], facad: ['facad-folder', 'Facad folder'],
  output: ['output-folder', 'Review workspace'] };
function dialogStatus(message, error = false) {
  $('open-status').textContent = message; $('open-status').className = `small open-status${error ? ' error' : ''}`;
  $('open-status').hidden = !message;
}
function openDialog() {
  // Always show the dialog (or say why not); any unsaved-edit check happens when the dataset is opened.
  try {
    if (!state.catalog) { toast('The workbench server did not answer. Reload the page, or restart server.py.', true); return; }
    const folders = state.catalog.folders || {};
    Object.entries(folderFields).forEach(([key, [id]]) => { if (folders[key]) $(id).value = folders[key]; });
    closePicker();
    // A server started before this feature was added still serves the new page but cannot open folders.
    dialogStatus('folders' in state.catalog ? '' : 'This server is running older code. Stop it (Ctrl+C) and start server.py again.', !('folders' in state.catalog));
    if (!$('open-dialog').open) $('open-dialog').showModal();
  } catch (error) { toast(`Cannot open the folder dialog: ${error.message}`, true); }
}
function closePicker() { picker.target = null; $('folder-picker').hidden = true; }
async function browse(path) {
  const sequence = ++picker.sequence;
  try {
    const listing = await api('/api/browse', { path });
    if (sequence !== picker.sequence) return;
    picker.path = listing.path; picker.facadFiles = listing.facad_files; dialogStatus('');
    $('picker-path').value = listing.path; $('picker-up').disabled = !listing.parent; picker.parent = listing.parent;
    $('picker-counts').textContent = `${listing.image_files} image${listing.image_files === 1 ? '' : 's'} · ${listing.facad_files} .fcd`;
    $('picker-roots').replaceChildren(...listing.roots.map((root) => button(root, '', () => browse(root))));
    const separator = listing.path.includes('\\') ? '\\' : '/';
    $('picker-list').replaceChildren(...listing.folders.map((name) => button(`📁 ${name}`, '',
      () => browse(listing.path.endsWith(separator) ? listing.path + name : listing.path + separator + name))));
    if (!listing.folders.length) $('picker-list').append(span('No sub-folders.', 'small muted'));
    if (listing.truncated) $('picker-list').append(span('List truncated; type the path instead.', 'small muted'));
  } catch (error) { if (sequence === picker.sequence) dialogStatus(error.message, true); }
}
function startPicker(target) {
  picker.target = target;
  $('picker-title').textContent = `Choose: ${folderFields[target][1]}`;
  $('folder-picker').hidden = false;
  browse($(folderFields[target][0]).value);
}
function selectPickedFolder() {
  if (!picker.target) return;
  const target = picker.target;
  $(folderFields[target][0]).value = picker.path;
  // A flat export keeps images and .fcd files together; offer that folder for both.
  if (target === 'images' && !$('facad-folder').value && picker.facadFiles) $('facad-folder').value = picker.path;
  closePicker();
}
async function openDataset() {
  const body = { images: $('images-folder').value, facad: $('facad-folder').value, output: $('output-folder').value };
  if (!body.images.trim() || !body.facad.trim()) { dialogStatus('Choose both the images folder and the Facad folder.', true); return; }
  if (state.busy || state.loading) { dialogStatus('Wait for the current image to finish loading or saving, then try again.', true); return; }
  if (!canLeave()) { dialogStatus('Cancelled: the current image has unsaved changes.', true); return; }
  $('open-confirm').disabled = true; dialogStatus('Indexing images and tracings… large folders can take a minute.');
  try {
    const catalog = await api('/api/open', body);
    $('open-dialog').close();
    await showCatalog(catalog);
    toast(`${catalog.items.length} images indexed; ${catalog.issues.length} import issues.`);
  } catch (error) { dialogStatus(error.message, true); }
  finally { $('open-confirm').disabled = false; }
}
$('open-folders').addEventListener('click', openDialog);
$('open-cancel').addEventListener('click', () => $('open-dialog').close());
$('open-confirm').addEventListener('click', openDataset);
Object.keys(folderFields).forEach((key) => $(`browse-${key}`).addEventListener('click', () => startPicker(key)));
$('picker-cancel').addEventListener('click', closePicker);
$('picker-select').addEventListener('click', selectPickedFolder);
$('picker-up').addEventListener('click', () => picker.parent && browse(picker.parent));
$('picker-go').addEventListener('click', () => browse($('picker-path').value));
$('picker-path').addEventListener('keydown', (event) => { if (event.key === 'Enter') browse($('picker-path').value); });

async function initialize() {
  updateButtons();
  try {
    const catalog = await api('/api/catalog');
    try { $('reviewer').value = localStorage.getItem('openceph-reviewer') || ''; } catch (_) { /* Optional preference. */ }
    await showCatalog(catalog);
    if (!catalog.loaded) openDialog();
  } catch (error) {
    toast(`Cannot reach the workbench server (${error.message}). Start it with code\\curation\\start_workbench.cmd `
      + 'and use the page it opens at http://127.0.0.1:8765.', true);
  }
}
initialize();
