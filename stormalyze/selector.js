/**
 * selector.js — the "snipping tool" overlay. Injected on demand with
 * chrome.scripting.executeScript (activeTab), so it is NOT a always-on content script.
 *
 * It lives entirely inside a closed-ish Shadow DOM so the host page's CSS can never
 * bleed in (and ours can never leak out).
 *
 * Coordinate handling is the important part:
 *   captureVisibleTab() returns an image whose pixel dimensions are
 *   viewportCssPx * devicePixelRatio — and on some zoom / HiDPI / Chrome versions the
 *   ratio is not exactly window.devicePixelRatio. So we never trust devicePixelRatio.
 *   We measure the ratio directly from the captured image:
 *       scaleX = img.naturalWidth  / window.innerWidth
 *       scaleY = img.naturalHeight / window.innerHeight
 *   and use that to map CSS-pixel drag coordinates into screenshot pixels.
 */

(() => {
  // executeScript re-runs this file on every activation; only wire up once.
  if (window.__weatherAiSelectorInstalled) return;
  window.__weatherAiSelectorInstalled = true;

  const MIN_SELECTION_CSS_PX = 8;

  let host = null;      // shadow host element
  let shadow = null;
  let els = null;       // cached shadow elements
  let img = null;       // the screenshot <img>
  let dragging = false;
  let start = { x: 0, y: 0 };
  let rect = { x: 0, y: 0, w: 0, h: 0 };

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.type !== 'SHOW_SELECTOR') return false;

    show(msg.screenshotDataUrl, msg.css)
      .then(() => sendResponse({ ok: true }))
      .catch((e) => sendResponse({ ok: false, error: String(e?.message || e) }));

    return true; // async response
  });

  async function show(dataUrl, css) {
    teardown(); // in case a previous overlay is somehow still up

    host = document.createElement('div');
    host.id = '__weather-ai-selector-host';
    // Inline styles on the host itself: the page cannot override these because
    // nothing outside can select an element we never put a class on.
    host.style.cssText = [
      'all: initial',
      'position: fixed',
      'inset: 0',
      'width: 100%',
      'height: 100%',
      'z-index: 2147483647',
      'cursor: crosshair'
    ].join(';');

    shadow = host.attachShadow({ mode: 'open' });
    shadow.innerHTML =
      '<style>' + css + '</style>' +
      '<div class="root">' +
        '<img class="shot" alt="">' +
        '<div class="dim"></div>' +
        '<div class="sel" hidden><div class="size"></div></div>' +
        '<div class="hint">Drag to select a radar cell &nbsp;·&nbsp; <b>Esc</b> to cancel</div>' +
        '<div class="toast" hidden></div>' +
      '</div>';

    els = {
      shot: shadow.querySelector('.shot'),
      dim: shadow.querySelector('.dim'),
      sel: shadow.querySelector('.sel'),
      size: shadow.querySelector('.size'),
      hint: shadow.querySelector('.hint'),
      toast: shadow.querySelector('.toast')
    };

    (document.documentElement || document.body).appendChild(host);

    // Load the screenshot before we let the user drag, so naturalWidth is known.
    img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error('Failed to decode the captured screenshot.'));
      img.src = dataUrl;
    });
    els.shot.src = dataUrl;

    host.addEventListener('pointerdown', onDown, true);
    window.addEventListener('pointermove', onMove, true);
    window.addEventListener('pointerup', onUp, true);
    window.addEventListener('keydown', onKey, true);
    host.addEventListener('wheel', preventAll, { passive: false, capture: true });
    host.addEventListener('contextmenu', preventAll, true);
  }

  function preventAll(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  function onKey(e) {
    if (e.key === 'Escape') {
      preventAll(e);
      cancel();
    }
  }

  function onDown(e) {
    if (e.button !== 0) return;
    preventAll(e);

    dragging = true;
    start = { x: e.clientX, y: e.clientY };
    rect = { x: e.clientX, y: e.clientY, w: 0, h: 0 };

    els.dim.hidden = true;   // the selection box's giant box-shadow dims instead
    els.sel.hidden = false;
    els.toast.hidden = true;
    draw();
  }

  function onMove(e) {
    if (!dragging) return;
    preventAll(e);

    rect.x = Math.min(start.x, e.clientX);
    rect.y = Math.min(start.y, e.clientY);
    rect.w = Math.abs(e.clientX - start.x);
    rect.h = Math.abs(e.clientY - start.y);
    draw();
  }

  function onUp(e) {
    if (!dragging) return;
    preventAll(e);
    dragging = false;

    if (rect.w < MIN_SELECTION_CSS_PX || rect.h < MIN_SELECTION_CSS_PX) {
      els.sel.hidden = true;
      els.dim.hidden = false;
      toast('Selection too small — drag a larger box.');
      return;
    }

    let payload;
    try {
      payload = crop(rect);
    } catch (err) {
      toast('Crop failed: ' + (err?.message || err));
      els.sel.hidden = true;
      els.dim.hidden = false;
      return;
    }

    teardown();
    chrome.runtime.sendMessage({ type: 'REGION_SELECTED', ...payload }).catch(() => {});
  }

  function draw() {
    const s = els.sel.style;
    s.left = rect.x + 'px';
    s.top = rect.y + 'px';
    s.width = rect.w + 'px';
    s.height = rect.h + 'px';

    // Show the size in *screenshot* pixels — that is what actually feeds the model.
    const { scaleX, scaleY } = scale();
    els.size.textContent =
      Math.round(rect.w * scaleX) + ' × ' + Math.round(rect.h * scaleY) + ' px';

    // Flip the label inside the box when the selection is near the bottom edge.
    els.size.classList.toggle('inside', rect.y + rect.h > window.innerHeight - 28);
  }

  /** Ratio between screenshot pixels and CSS pixels, measured, never assumed. */
  function scale() {
    return {
      scaleX: img.naturalWidth / window.innerWidth,
      scaleY: img.naturalHeight / window.innerHeight
    };
  }

  /**
   * Crop the selected rectangle out of the full-resolution screenshot.
   * Returns a PNG data URL at native screenshot resolution — no resampling happens
   * here, so the only resize in the whole pipeline is the bilinear one in TF.js.
   */
  function crop(r) {
    const { scaleX, scaleY } = scale();

    let sx = Math.round(r.x * scaleX);
    let sy = Math.round(r.y * scaleY);
    let sw = Math.round(r.w * scaleX);
    let sh = Math.round(r.h * scaleY);

    // Clamp to the image so a drag that ends off-screen can't produce an invalid crop.
    sx = Math.max(0, Math.min(sx, img.naturalWidth - 1));
    sy = Math.max(0, Math.min(sy, img.naturalHeight - 1));
    sw = Math.max(1, Math.min(sw, img.naturalWidth - sx));
    sh = Math.max(1, Math.min(sh, img.naturalHeight - sy));

    const canvas = document.createElement('canvas');
    canvas.width = sw;
    canvas.height = sh;

    const ctx = canvas.getContext('2d', { willReadFrequently: false });
    if (!ctx) throw new Error('2D canvas context unavailable.');

    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

    return { dataUrl: canvas.toDataURL('image/png'), width: sw, height: sh };
  }

  function toast(text) {
    els.toast.textContent = text;
    els.toast.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { if (els) els.toast.hidden = true; }, 2200);
  }

  function cancel() {
    teardown();
    chrome.runtime.sendMessage({ type: 'SELECTION_CANCELLED' }).catch(() => {});
  }

  function teardown() {
    window.removeEventListener('pointermove', onMove, true);
    window.removeEventListener('pointerup', onUp, true);
    window.removeEventListener('keydown', onKey, true);

    dragging = false;
    if (host && host.isConnected) host.remove();
    host = null;
    shadow = null;
    els = null;
    img = null;
  }
})();
