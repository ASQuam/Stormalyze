/**
 * popup.js — the entire UI.
 *
 * The MV3 action popup is destroyed whenever it loses focus, so it cannot stay open
 * across the selection step and it cannot hold the model. Two things make the
 * "results in the popup" experience work anyway:
 *
 *   - the model lives in an offscreen document, so it stays warm between openings;
 *   - after the crop is classified, background.js calls chrome.action.openPopup()
 *     to bring this popup back up already showing the result.
 *
 * The popup is therefore stateless: on every open it reads `uiState` from
 * chrome.storage.session and renders whatever phase the workflow is in. That also
 * means the last result is still here the next time you click the icon.
 */

const UI_STATE = 'uiState';

const views = {
  idle: document.getElementById('view-idle'),
  working: document.getElementById('view-working'),
  result: document.getElementById('view-result'),
  error: document.getElementById('view-error'),
  info: document.getElementById('view-info'),
  model: document.getElementById('view-model')
};

const ui = {
  analyze: document.getElementById('analyze'),
  again: document.getElementById('again'),
  retry: document.getElementById('retry'),
  status: document.getElementById('status'),
  statusText: document.getElementById('status-text'),
  topLabel: document.getElementById('top-label'),
  headlineNote: document.getElementById('headline-note'),
  uncertain: document.getElementById('uncertain'),
  hazards: document.getElementById('hazards'),
  stormNote: document.getElementById('storm-note'),
  bars: document.getElementById('bars'),
  stats: document.getElementById('stats'),
  errorText: document.getElementById('error-text'),
  thumb: document.getElementById('thumb'),
  sub: document.getElementById('sub'),
  file: document.getElementById('file')
};

/** Remembered so the info/model views' Back buttons return where you came from. */
let currentPhase = 'idle';

const infoBtn = document.getElementById('info-btn');

function show(phase) {
  if (phase !== 'info' && phase !== 'model') currentPhase = phase;
  for (const [name, el] of Object.entries(views)) el.hidden = name !== phase;

  // The same header button toggles role: ⓘ on the main screens, home while reading
  // the info pages, so there is always a one-click way back.
  const inInfo = phase === 'info' || phase === 'model';
  infoBtn.textContent = inInfo ? '⌂' : 'i';
  infoBtn.title = inInfo ? 'Back to Stormalyze' : 'About & disclaimer';
  infoBtn.setAttribute('aria-label', inInfo ? 'Back' : 'About');
  infoBtn.classList.toggle('home', inInfo);
}

function setStatus(state, text) {
  ui.status.className = 'value ' + state;
  ui.statusText.textContent = text;
}

// ------------------------------------------------------------------ render ---

function render(state) {
  if (!state || state.phase === 'idle') {
    ui.thumb.hidden = true;
    show('idle');
    return;
  }

  if (state.thumb) {
    ui.thumb.src = state.thumb;
    ui.thumb.hidden = false;
  }

  if (state.phase === 'working') {
    show('working');
    return;
  }

  if (state.phase === 'error') {
    ui.errorText.textContent = state.error || 'Unknown error.';
    show('error');
    return;
  }

  if (state.phase === 'result') {
    renderResult(state);
    show('result');
  }
}

/** Below this the model is hedging enough that the hazard list needs a caveat. */
const CONFIDENT_ENOUGH = 0.5;

function renderResult(state) {
  const ranked = state.ranked || [];
  const top = ranked[0];
  if (!top) return;

  const info = state.info || {};
  const haz = (typeof HAZARDS !== 'undefined' && HAZARDS[top.name]) || null;

  // --- headline: the storm type, then what it means -------------------------
  ui.topLabel.textContent = top.label;
  ui.headlineNote.textContent = haz ? haz.headline : '';

  // --- uncertainty is surfaced, not hidden ----------------------------------
  // The classifier is wrong often enough that presenting a hazard list with no
  // hedge would overstate what it knows.
  if (top.prob < CONFIDENT_ENOUGH) {
    ui.uncertain.textContent =
      `The model is only ${(top.prob * 100).toFixed(0)}% confident in this ` +
      `classification — treat the list below as one possibility, not a read of the storm.`;
    ui.uncertain.hidden = false;
  } else {
    ui.uncertain.hidden = true;
  }

  // --- the hazard list ------------------------------------------------------
  ui.hazards.innerHTML = '';
  for (const item of (haz ? haz.items : [])) {
    const li = document.createElement('li');
    li.className = 'hazard sev-' + item.severity;

    const dot = document.createElement('span');
    dot.className = 'sev-dot';

    const body = document.createElement('div');

    const name = document.createElement('div');
    name.className = 'hazard-name';
    name.textContent = item.name;

    const detail = document.createElement('div');
    detail.className = 'hazard-detail';
    detail.textContent = item.detail;

    body.append(name, detail);
    li.append(dot, body);
    ui.hazards.appendChild(li);
  }

  ui.stormNote.textContent = haz ? haz.note : '';

  // --- collapsed model detail (for debugging, not the headline) -------------
  ui.bars.innerHTML = '';
  for (const item of ranked) {
    const li = document.createElement('li');
    if (item.index === top.index) li.className = 'top';

    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = item.label;

    const pct = document.createElement('span');
    pct.className = 'pct';
    pct.textContent = (item.prob * 100).toFixed(1) + '%';

    const track = document.createElement('div');
    track.className = 'track';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.width = (item.prob * 100).toFixed(2) + '%';
    track.appendChild(fill);

    li.append(name, pct, track);
    ui.bars.appendChild(li);
  }

  ui.stats.textContent =
    `${state.width}×${state.height} px → ${info.size || 512}×${info.size || 512} · ` +
    `${Math.round(state.ms || 0)} ms · ${info.backend || '?'}` +
    (info.fingerprint ? ` · ${info.fingerprint}` : '');

}

// ------------------------------------------------------------------ actions ---

async function beginCapture(button) {
  button.disabled = true;
  try {
    const res = await chrome.runtime.sendMessage({ type: 'START_CAPTURE' });
    if (res && res.ok === false) throw new Error(res.error);
    window.close(); // the overlay owns the screen now
  } catch (e) {
    await chrome.storage.session.set({
      [UI_STATE]: { phase: 'error', error: String(e?.message || e) }
    });
    render({ phase: 'error', error: String(e?.message || e) });
    button.disabled = false;
  }
}

// About / disclaimer view, and the model-information sub-view inside it.
infoBtn.addEventListener('click', () => {
  const inInfo = !views.info.hidden || !views.model.hidden;
  show(inInfo ? currentPhase : 'info');
});
document.getElementById('info-back').addEventListener('click', () => show(currentPhase));
document.getElementById('model-btn').addEventListener('click', () => show('model'));
document.getElementById('model-back').addEventListener('click', () => show('info'));

ui.analyze.addEventListener('click', () => beginCapture(ui.analyze));
ui.again.addEventListener('click', () => beginCapture(ui.again));
ui.retry.addEventListener('click', () => beginCapture(ui.retry));

// Live updates while the popup happens to be open.
chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === 'STATE_CHANGED') render(msg.state);
  return false;
});

// Manual test hook: same pipeline, arbitrary local image. Handy for comparing a
// browser prediction against tools/predict_python.py on the same file.
ui.file.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  try {
    const dataUrl = await new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = () => reject(new Error('Could not read that file.'));
      r.readAsDataURL(file);
    });

    render({ phase: 'working', thumb: dataUrl });
    show('working');

    const res = await chrome.runtime.sendMessage({ type: 'REGION_SELECTED', dataUrl });
    if (res && res.ok === false) throw new Error(res.error);

    const store = await chrome.storage.session.get(UI_STATE);
    render(store[UI_STATE]);
  } catch (err) {
    render({ phase: 'error', error: String(err?.message || err) });
  } finally {
    e.target.value = '';
  }
});

// -------------------------------------------------------------------- boot ---

async function boot() {
  const store = await chrome.storage.session.get(UI_STATE);
  render(store[UI_STATE]);

  // Model presence check for the idle view.
  try {
    const res = await fetch(chrome.runtime.getURL('model/model.json'));
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const json = await res.json();
    if (!json.modelTopology) throw new Error('not a tfjs layers model');

    setStatus('pending', 'Warming up model…');
    ui.analyze.disabled = false;

    // Ask the offscreen host to load the weights now, so the first snip is fast.
    const ready = await chrome.runtime.sendMessage({ type: 'ENSURE_READY' });
    if (ready?.ok) {
      const i = ready.info || {};
      setStatus('ok', `Model ready ✓ · ${i.backend || 'local'}`);
      // Which model is actually loaded — fingerprint of the weight bytes.
      if (i.fingerprint) {
        const dd = document.getElementById('model-id');
        if (dd) {
          dd.textContent = i.fingerprint;
          dd.title = `converted ${i.convertedAt || '?'} from ${i.source || '?'}`;
        }
      }
    } else {
      setStatus('bad', 'Model failed to load');
      ui.errorText.textContent = ready?.error || 'Unknown error loading the model.';
    }
  } catch (e) {
    setStatus('bad', 'Model missing');
    ui.analyze.disabled = true;
    ui.errorText.textContent =
      'Could not read model/model.json.\n\nConvert your .keras file into the model/ ' +
      'folder and reload the extension.\n\n(' + e.message + ')';
  }
}

ui.analyze.disabled = true;
boot();
