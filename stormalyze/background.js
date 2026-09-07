/**
 * background.js — MV3 service worker. Coordinator only.
 *
 * A service worker has NO DOM (no document, no Image, no canvas, no WebGL) and Chrome
 * terminates it when idle, so it neither crops nor runs the model. It:
 *
 *   1. captures the visible tab
 *   2. injects the selection overlay
 *   3. receives the crop, forwards it to the offscreen document for inference
 *   4. stores the result and re-opens the action popup to show it
 *
 * All shared state lives in chrome.storage.session, because this worker can be killed
 * between any two steps.
 */

const OFFSCREEN_PATH = 'offscreen.html';
const UI_STATE = 'uiState';

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.target === 'offscreen') return false; // not ours

  switch (msg?.type) {
    case 'START_CAPTURE':
      startCapture().then(sendResponse, (err) => sendResponse(failure(err)));
      return true;

    case 'REGION_SELECTED':
      handleRegion(msg).then(sendResponse, (err) => sendResponse(failure(err)));
      return true;

    case 'SELECTION_CANCELLED':
      setState({ phase: 'idle' });
      sendResponse({ ok: true });
      return false;

    case 'ENSURE_READY':
      warmup().then(sendResponse, (err) => sendResponse(failure(err)));
      return true;

    default:
      return false;
  }
});

const failure = (err) => ({ ok: false, error: String(err?.message || err) });

// ---------------------------------------------------------------- offscreen ---

let creating = null; // de-dupes concurrent createDocument calls

async function ensureOffscreen() {
  const existing = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT']
  });
  if (existing.length > 0) return;

  if (creating) return creating;

  creating = chrome.offscreen.createDocument({
    url: OFFSCREEN_PATH,
    reasons: ['WORKERS'],
    justification:
      'Keeps the TensorFlow.js model and its compiled WebGL shaders resident between ' +
      'analyses; the action popup is destroyed whenever it loses focus.'
  });

  try {
    await creating;
  } finally {
    creating = null;
  }
}

/**
 * chrome.offscreen.createDocument() resolves as soon as the document exists — which is
 * before its deferred <script type="module"> has run and registered an onMessage
 * listener. Sending work into that window silently resolves `undefined`. So poll a
 * cheap PING until the document answers.
 */
async function waitForOffscreen({ tries = 60, gap = 100 } = {}) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await chrome.runtime.sendMessage({ target: 'offscreen', type: 'PING' });
      if (res?.ok) return;
    } catch {
      // "Receiving end does not exist" while the document is still parsing
    }
    await new Promise((r) => setTimeout(r, gap));
  }
  throw new Error('The inference host did not start. Try reloading the extension.');
}

/** Never let a stuck request leave the UI spinning forever. */
function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(label + ' timed out after ' + Math.round(ms / 1000) + 's.')), ms)
    )
  ]);
}

async function toOffscreen(message, timeoutMs = 180000) {
  await ensureOffscreen();
  await waitForOffscreen();

  const res = await withTimeout(
    chrome.runtime.sendMessage({ target: 'offscreen', ...message }),
    timeoutMs,
    message.type === 'WARMUP' ? 'Loading the model' : 'Inference'
  );

  if (!res) throw new Error('The inference host did not respond.');
  if (res.ok === false) throw new Error(res.error);
  return res;
}

/** Load the model ahead of time so the first analysis isn't waiting on it. */
async function warmup() {
  const res = await toOffscreen({ type: 'WARMUP' });
  return { ok: true, info: res.info };
}

// ------------------------------------------------------------------- state ---

async function setState(state) {
  await chrome.storage.session.set({ [UI_STATE]: state });
  // Update the popup if it happens to be open right now. No listener = harmless.
  chrome.runtime.sendMessage({ type: 'STATE_CHANGED', state }).catch(() => {});
}

async function showPopup() {
  try {
    await chrome.action.openPopup();
    chrome.action.setBadgeText({ text: '' });
  } catch {
    // openPopup() needs Chrome 127+ and can refuse if no window is focused.
    // Fall back to nudging the user via the toolbar badge.
    chrome.action.setBadgeText({ text: '✓' });
    chrome.action.setBadgeBackgroundColor({ color: '#4c8dff' });
  }
}

// ----------------------------------------------------------------- capture ---

async function startCapture() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab || tab.id === undefined) throw new Error('No active tab found.');

  const url = tab.url || '';
  if (/^(chrome|edge|about|devtools|chrome-extension|chrome-untrusted):/i.test(url) ||
      /^https:\/\/chromewebstore\.google\.com/i.test(url) ||
      /^https:\/\/chrome\.google\.com\/webstore/i.test(url)) {
    throw new Error(
      'Chrome blocks extensions on this page (chrome://, the Web Store, devtools). ' +
      'Open a normal web page and try again.'
    );
  }

  // Start loading the model now, in parallel with the user dragging their selection.
  warmup().catch(() => {});

  let screenshotDataUrl;
  try {
    screenshotDataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png' });
  } catch (e) {
    throw new Error(
      'Screenshot capture failed: ' + (e?.message || e) +
      '. Click the extension icon on the tab you want to capture.'
    );
  }
  if (!screenshotDataUrl) throw new Error('Screenshot capture returned no image.');

  const css = await (await fetch(chrome.runtime.getURL('selector.css'))).text();

  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['selector.js'] });
  } catch (e) {
    throw new Error('Could not inject the selection overlay into this page: ' +
                    (e?.message || e));
  }

  const res = await chrome.tabs.sendMessage(tab.id, {
    type: 'SHOW_SELECTOR',
    screenshotDataUrl,
    css
  });

  return res || { ok: true };
}

// --------------------------------------------------------------- inference ---

async function handleRegion(msg) {
  await setState({ phase: 'working', thumb: msg.dataUrl });
  showPopup();

  try {
    const res = await toOffscreen({ type: 'INFER', dataUrl: msg.dataUrl });

    await setState({
      phase: 'result',
      thumb: msg.dataUrl,
      ranked: res.ranked,
      ms: res.ms,
      width: res.width,
      height: res.height,
      info: res.info,
      at: Date.now()
    });
  } catch (e) {
    await setState({ phase: 'error', error: String(e?.message || e), thumb: msg.dataUrl });
  }

  return { ok: true };
}
