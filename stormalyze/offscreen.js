/**
 * offscreen.js — the inference host.
 *
 * Lives in an offscreen document so the loaded model survives popup open/close
 * cycles. Speaks a tiny message protocol with background.js:
 *
 *   { target: 'offscreen', type: 'WARMUP' }              -> { ok, info }
 *   { target: 'offscreen', type: 'INFER', dataUrl }      -> { ok, ranked, probs, ms, info, width, height }
 *
 * Every message carries `target` so the service worker's own onMessage listener
 * can ignore what isn't meant for it (and vice versa).
 */

import { loadModel, predict, getModelInfo } from './inference.js';

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.target !== 'offscreen') return false;

  handle(msg)
    .then(sendResponse)
    .catch((e) => sendResponse({ ok: false, error: String(e?.message || e) }));

  return true; // async
});

async function handle(msg) {
  switch (msg.type) {
    // Liveness probe. background.js polls this after createDocument(), because that
    // call resolves as soon as the document exists — which is BEFORE this deferred
    // module has run and registered the listener above. Without the probe, the first
    // INFER lands in a document that isn't listening yet and silently resolves
    // undefined.
    case 'PING':
      return { ok: true };

    case 'WARMUP': {
      await loadModel();
      return { ok: true, info: getModelInfo() };
    }

    case 'INFER': {
      const image = await decode(msg.dataUrl);
      const res = await predict(image);

      return {
        ok: true,
        probs: res.probs,
        ranked: res.ranked,
        ms: res.ms,
        width: image.naturalWidth,
        height: image.naturalHeight,
        info: getModelInfo()
      };
    }

    default:
      return { ok: false, error: 'Unknown message type: ' + msg.type };
  }
}

function decode(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () =>
      img.naturalWidth && img.naturalHeight
        ? resolve(img)
        : reject(new Error('The cropped region is empty (0 pixels).'));
    img.onerror = () => reject(new Error('Could not decode the cropped image.'));
    img.src = dataUrl;
  });
}
