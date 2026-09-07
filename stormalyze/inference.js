/**
 * inference.js — everything TensorFlow.js. Imported by result.js as an ES module.
 * `tf` is the global from vendor/tf.min.js (loaded by result.html before this module).
 *
 * Design notes
 * ------------
 * - The model is loaded ONCE and memoised in `modelPromise`. Because the result window
 *   is reused for subsequent snips, the 5–40 MB of weights are only fetched once.
 * - The JS /255 rescale is applied ONLY if the exported graph does not already contain
 *   a Rescaling layer. Keras models built with `layers.Rescaling(1./255)` inside the
 *   model carry that layer through conversion, and dividing again in JS would feed the
 *   network values in [0, 1/255] — a classic silent-wrong-answer bug.
 *   Override with FORCE_RESCALE_MODE if you know better.
 */

/**
 * MUST match `train_ds.class_names` exactly — Keras sorts class folders
 * alphabetically, which is why `nonradar_image` lands at index 2.
 */
export const CLASS_NAMES = [
  'bowecho_squall',
  'multicell',
  'nonradar_image',
  'qlcs_squall',
  'single_cell',
  'supercell'
];

/** Pretty names for the UI, index-aligned with CLASS_NAMES. */
export const CLASS_LABELS = [
  'Bow Echo Squall',
  'Multicell',
  'Not Radar Imagery',
  'QLCS Squall',
  'Single Cell',
  'Supercell'
];

export const MODEL_URL = 'model/model.json';

/** 'auto' | 'always' | 'never' — how to handle the [0,255] -> [0,1] rescale in JS. */
export const FORCE_RESCALE_MODE = 'auto';


/** Fallback if the model.json doesn't declare a concrete input size. */
const DEFAULT_SIZE = 512;

let modelPromise = null;
let modelInfo = null;

/**
 * Load (and cache) the TFJS LayersModel. Safe to call repeatedly.
 * @param {(msg: string) => void} [onProgress]
 */
export function loadModel(onProgress = () => {}) {
  if (modelPromise) return modelPromise;

  modelPromise = (async () => {
    onProgress('Starting TensorFlow.js…');
    await tf.ready();

    onProgress('Downloading model…');

    let model;
    try {
      model = await tf.loadLayersModel(MODEL_URL, {
        onProgress: (f) => onProgress('Loading weights… ' + Math.round(f * 100) + '%')
      });
    } catch (e) {
      throw new Error(describeLoadError(e));
    }

    const inShape = model.inputs?.[0]?.shape || [];
    const size = Number.isInteger(inShape[1]) ? inShape[1] : DEFAULT_SIZE;
    const channels = Number.isInteger(inShape[3]) ? inShape[3] : 3;
    const outUnits = model.outputs?.[0]?.shape?.slice(-1)[0] ?? null;

    if (outUnits !== null && outUnits !== CLASS_NAMES.length) {
      throw new Error(
        'Model outputs ' + outUnits + ' values but CLASS_NAMES lists ' +
        CLASS_NAMES.length + '. Fix CLASS_NAMES in inference.js (it must match the ' +
        'alphabetical class_names order Keras used at training time).'
      );
    }

    // Optional provenance written by tools/convert_model.py, surfaced in the popup so
    // you can always tell WHICH model is live — replacing the folder with a fresh
    // download silently swaps model/, and identical-looking predictions from a
    // different model are miserable to debug.
    let meta = null;
    try {
      const r = await fetch('model/model_meta.json');
      if (r.ok) meta = await r.json();
    } catch { /* optional file */ }

    const hasRescaling = containsRescaling(model);
    const rescaleInJs =
      FORCE_RESCALE_MODE === 'always' ? true :
      FORCE_RESCALE_MODE === 'never' ? false :
      !hasRescaling;

    modelInfo = {
      size,
      channels,
      backend: tf.getBackend(),
      hasRescaling,
      rescaleInJs,
      params: model.countParams(),
      fingerprint: meta?.fingerprint || null,
      convertedAt: meta?.converted_at || null,
      source: meta?.source || null
    };

    onProgress('Warming up…');
    // First predict compiles the WebGL shaders. Do it now so the first real snip is fast.
    const warm = tf.tidy(() => model.predict(tf.zeros([1, size, size, channels])));
    await warm.data();   // force the GPU work to actually complete
    warm.dispose();

    // NOTE: do NOT use tf.nextFrame() here. It awaits requestAnimationFrame, which
    // never fires in an offscreen document (the document is never rendered), so the
    // load would hang forever. A macrotask yield is enough.
    await new Promise((r) => setTimeout(r, 0));

    onProgress('Model loaded');
    return model;
  })();

  // A failed load must not poison the cache — let the user retry.
  modelPromise.catch(() => { modelPromise = null; });

  return modelPromise;
}

export function getModelInfo() {
  return modelInfo;
}

/**
 * Run the full preprocessing + inference pipeline on an image element.
 *
 * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap} imageElement  the crop
 * @returns {Promise<{probs:number[], topIndex:number, ranked:Array, ms:number}>}
 */
export async function predict(imageElement) {
  const model = await loadModel();
  const { size, channels, rescaleInJs } = modelInfo;

  const t0 = performance.now();

  // ---- preprocess -------------------------------------------------------
  // tf.tidy disposes every intermediate tensor created inside it except the one
  // that is returned, so nothing leaks on the GPU.
  const input = tf.tidy(() => {
    // [h, w, 3] uint8 -> float32 in [0, 255]
    let x = tf.browser.fromPixels(imageElement, channels).toFloat();

    // Bilinear resize, alignCorners=false, halfPixelCenters=TRUE.
    //
    // That last flag is not cosmetic. TF.js defaults halfPixelCenters to false, which
    // is TF1's `resize_bilinear` sampling grid, while TF2's `tf.image.resize` — the op
    // behind `image_dataset_from_directory` and `layers.Resizing` — always uses
    // half-pixel centres. Leaving it false shifts every sample by half a pixel and
    // makes browser predictions drift from Keras. Verified numerically: with the flag
    // on, TF.js matches tf.image.resize(..., 'bilinear', antialias=False) to ~1e-6.
    x = tf.image.resizeBilinear(x, [size, size], false, true);

    // [0,255] -> [0,1] (skipped when the model already has a Rescaling layer)
    if (rescaleInJs) x = x.div(255);

    // add the batch dimension -> [1, size, size, 3]
    return x.expandDims(0);
  });

  // ---- inference --------------------------------------------------------
  let output, probs;
  try {
    output = model.predict(input);
    if (Array.isArray(output)) output = output[0];
    probs = Array.from(await output.data());
  } catch (e) {
    throw new Error('TensorFlow.js inference failed: ' + (e?.message || e));
  } finally {
    input.dispose();
    if (output && typeof output.dispose === 'function') output.dispose();
  }

  // Safety net: if the exported model ends in logits rather than softmax,
  // normalise here so the UI percentages still mean something.
  probs = ensureProbabilities(probs);

  const ranked = probs
    .map((p, i) => ({ index: i, name: CLASS_NAMES[i], label: CLASS_LABELS[i], prob: p }))
    .sort((a, b) => b.prob - a.prob);

  return {
    probs,
    ranked,
    topIndex: ranked[0].index,
    ms: performance.now() - t0
  };
}

/** True if a Rescaling layer exists anywhere in the model (including nested models). */
function containsRescaling(model) {
  const seen = new Set();

  const walk = (layers) => {
    for (const layer of layers || []) {
      if (!layer || seen.has(layer)) continue;
      seen.add(layer);

      const cls = (layer.getClassName && layer.getClassName()) || '';
      if (cls === 'Rescaling' || cls === 'Normalization') return true;

      if (Array.isArray(layer.layers) && walk(layer.layers)) return true;
    }
    return false;
  };

  return walk(model.layers);
}

/** Accept either softmax output or raw logits. */
function ensureProbabilities(values) {
  const sum = values.reduce((a, b) => a + b, 0);
  const looksLikeProbs =
    values.every((v) => v >= -1e-6 && v <= 1 + 1e-6) && Math.abs(sum - 1) < 1e-3;

  if (looksLikeProbs) return values.map((v) => Math.min(1, Math.max(0, v)));

  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / total);
}

function describeLoadError(e) {
  const msg = String(e?.message || e);

  if (/404|Failed to fetch|Not Found/i.test(msg)) {
    return 'model/model.json (or one of its .bin weight shards) was not found. ' +
           'Copy the converter output into the extension’s model/ folder and reload ' +
           'the extension. — ' + msg;
  }
  if (/Unknown layer|Unknown class|not registered/i.test(msg)) {
    return 'The model contains a layer TensorFlow.js cannot rebuild (often ' +
           'RandomFlip / RandomRotation / RandomZoom data-augmentation layers, or a ' +
           'custom layer). Strip augmentation layers before converting — see README. — ' + msg;
  }
  if (/topology|modelTopology|GraphModel/i.test(msg)) {
    return 'model.json is not a tfjs *layers* model. Convert with ' +
           '--output_format=tfjs_layers_model so tf.loadLayersModel() can read it. — ' + msg;
  }
  return 'Failed to load the model: ' + msg;
}
