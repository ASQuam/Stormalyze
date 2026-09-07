# Stormalyze — Chrome MV3 snipping-tool classifier

Select a rectangle anywhere on the current tab, and a TensorFlow.js port of your Keras
CNN classifies the crop into one of six radar-cell types — entirely inside the browser.
No server, no upload, no network call after install.

> **Your model is already converted and included.** `model/` contains the TF.js build of
> `project_cell_id_model.keras` (1,635,014 params, 6 classes, 6.3 MB). Verified against
> Keras: preprocessing is bit-identical and penultimate-layer activations agree to
> ~1e-6. Skip to §3 to install — §2 is only needed when you retrain.

```
stormalyze/
├── manifest.json          MV3 manifest (activeTab + scripting + storage only)
├── background.js          service worker: capture, inject overlay, route the crop
├── popup.html/.css/.js    the entire UI: analyze, hazards, info, errors
├── hazards.js             hazard reference text, keyed by storm class
├── selector.js/.css       the snipping overlay (injected on demand, Shadow DOM)
├── offscreen.html/.js     hidden document that keeps the model resident
├── inference.js           TF.js: load, preprocess, predict, dispose
├── vendor/tf.min.js       TensorFlow.js 4.22.0, bundled (MV3 forbids CDN scripts)
├── model/                 model.json + weight shards (already populated)
├── icons/
└── tools/
    ├── update_model.bat   Windows one-click: swap in a retrained model
    ├── update_model.py    backup + convert + verify (called by the .bat)
    ├── convert_model.py   .keras -> Keras 2 rebuild -> tfjs_layers_model
    ├── compare_models.py  run two .keras models over the same images, side by side
    └── predict_python.py  reference prediction, for browser-vs-Keras parity
```

---

## 1. Architecture, and the MV3 limits that shaped it

These constraints are real and each one changed a design decision:

| Manifest V3 constraint | Consequence |
|---|---|
| **No remote code.** `script-src 'self'` — you cannot `<script src="https://cdn.jsdelivr.net/…tfjs">`. | TensorFlow.js is vendored into `vendor/tf.min.js` and shipped with the extension. |
| **No `eval` / `new Function`.** MV3 has no `'unsafe-eval'` escape hatch for extension pages. TF.js's default `tf.min.js` contains one `Function("return this")()` global-detection fallback, which CSP blocks — leaving `window.tf` an empty object and throwing *"tf.ready is not a function"*. | `vendor/tf.min.js` is the official **`tf.es2017.min.js`** build, which uses `globalThis` instead and contains no `eval` at all. It is also 400 KB smaller. See §3 before replacing it. |
| **The action popup is destroyed when it loses focus.** It dies the instant the selection overlay appears. | The popup cannot stay open across the selection step, and cannot hold the model. Results still appear *in the popup* because `background.js` calls `chrome.action.openPopup()` (Chrome 127+) once the crop is classified, and the popup rebuilds itself from `chrome.storage.session`. |
| **The service worker has no DOM** — no `document`, no `Image`, no `HTMLCanvasElement`, no WebGL, and Chrome may terminate it after ~30 s idle. | The worker never crops and never runs the model. Cropping happens in the page; inference happens in an **offscreen document**, which has a DOM and WebGL and outlives the popup, so the weights and compiled shaders load exactly once. No state lives in worker globals. |
| **`chrome.offscreen.createDocument()` resolves before the document's deferred module scripts run.** Work sent immediately lands in a document with no listener and silently resolves `undefined`. | `background.js` polls a `PING` message until the offscreen document answers, then sends real work. Every request also has a timeout, so a wedged host surfaces an error instead of an endless spinner. |
| **`requestAnimationFrame` never fires in an offscreen document** — it is never rendered. | `inference.js` must not call `tf.nextFrame()`; it would deadlock model loading forever. It yields with `setTimeout` instead. This one costs an afternoon if you hit it blind. |
| **`chrome.tabs.captureVisibleTab` may only be called from an extension context**, not a content script, and needs `activeTab` (granted by your click on the icon) or a host permission. | Capture happens in `background.js`, triggered by the popup button. |
| **Content scripts can't be injected into `chrome://`, the Web Store, or the PDF viewer.** | `background.js` detects those URLs up front and returns a readable error instead of failing silently. |
| **`chrome.storage.session` has a 10 MB quota** and is the only place the worker, the offscreen document and the popup all share. | The crop (a PNG data URL) is handed over through it, and removed as soon as it's read. |
| **`tf.loadLayersModel` needs `tfjs_layers_model` format** — it cannot read `.keras`, `.h5`, or a `GraphModel`. | Conversion is a separate offline step (`tools/convert_model.py`). |

One thing deliberately *not* used: **React**. The whole UI is two static pages; a
framework would only add build steps.

Permissions are `activeTab`, `scripting`, `storage`, `offscreen` — no `<all_urls>`, no
`tabs`, so Chrome shows no "read your data on all websites" warning.

---

## 2. Re-converting after you retrain

Only needed when the weights change. The output already in `model/` was produced this way.

### The easy way (Windows)

Drag your retrained `.keras` file onto **`tools\update_model.bat`**, or double-click it
and paste the path. It backs up the current model to `model/_previous/`, clears stale
weight shards, converts, verifies the output is loadable, cross-checks the class count
against `CLASS_NAMES` in `inference.js`, and restores the old model automatically if
anything fails.

First run builds a private Python environment in `%USERPROFILE%\.stormalyze-convert-env`
(a few minutes, ~250 MB — TensorFlow only). Later runs reuse it. Your training
environment is untouched.

**Requires Python 3.10–3.12 from python.org.** Two Windows traps the launcher now handles
explicitly:

- **TensorFlow publishes no Python 3.13 packages.** If 3.13 is all you have, the install
  cannot succeed. The launcher looks for 3.12, then 3.11, then 3.10, and says so plainly
  if it finds none.
- **`tensorflowjs` is uninstallable on Windows** (see above). Not used.

If a run dies partway through the install, just run the file again — it detects an
incomplete environment (folder present, packages missing) and finishes the install rather
than assuming it's ready. To start completely fresh, delete
`%USERPROFILE%\.stormalyze-convert-env`.
- **The Microsoft Store build of Python sandboxes file writes**, silently redirecting
  anything under `AppData` into
  `...\Packages\PythonSoftwareFoundation.Python.3.x\LocalCache\Local\...`. A virtual
  environment created there lands somewhere other than where the caller looks for it. The
  environment now goes under `%USERPROFILE%` instead, which isn't redirected.

Then hit reload on the extension card at `chrome://extensions`.

On macOS/Linux the same thing is `python tools/update_model.py path/to/new_model.keras`.

### The manual way

### Why there's a script instead of a one-liner

Two independent problems, both of which `tools/convert_model.py` now handles.

**1. `pip install tensorflowjs` cannot work on Windows.** It depends on
`tensorflow-decision-forests`, which publishes **no Windows wheels at all** — pip exits
with `ResolutionImpossible` and there is no flag that fixes it. So the script writes the
TF.js `layers-model` format itself: a `model.json` plus 4 MiB weight shards. The output is
verified **byte-for-byte identical** to `tensorflowjs_converter`'s (only the informational
`convertedBy` string differs). Nothing but TensorFlow is required.

**2. Keras 3 configs are unreadable by TF.js.** Your model is saved by Keras 3.13.2, and
TF.js parses Keras 2 configs:

- Keras 3 writes `batch_shape`; TF.js expects `batch_input_shape` → `tf.loadLayersModel`
  throws *"An InputLayer should be passed either a `batchInputShape` or an `inputShape`"*.
- Keras 3 writes `dtype` as a `DTypePolicy` object; TF.js expects the string `"float32"`.

Patching the JSON afterwards is brittle, so the script **rebuilds the architecture in
tf_keras and copies the weights across**, producing a genuine Keras 2 graph. It verifies
the weight transfer is exact (max |Δ| must be 0.0) before writing anything, and fails
loudly on any layer type it doesn't recognise rather than silently producing a wrong graph.

```bash
# Python 3.10-3.12 only; TensorFlow has no 3.13 wheels
pip install "tensorflow-cpu==2.17.1" "tf-keras==2.17.0"

cd stormalyze
python tools/convert_model.py /path/to/project_cell_id_model.keras --out model
```

On Linux/macOS you can pass `--use-tfjs-converter` to shell out to the official
`tensorflowjs_converter` instead. There is no reason to — the output is the same.

Read three lines of its output:

- `output units : 6` — must equal the length of `CLASS_NAMES` in `inference.js`.
- `Rescaling layer inside the model: True` — tells the extension to skip the JS ÷255.
- `weight transfer verified, max |Δ| = 0.0` — anything else aborts.

If it reports an unmapped layer type, add it to `convert()` in the script (the mapping
covers Conv2D, SeparableConv2D, Dense, Dropout, Flatten, Activation, BatchNorm, the
pooling layers, Rescaling and Resizing). If your retrained model has augmentation layers,
add `--strip-augmentation`.

### Sanity check the output

`model/` should hold `model.json` plus `group1-shard*of*.bin`. Open `model.json` and
confirm the first layer's config contains **`batch_input_shape`** — if it says
`batch_shape`, the Keras 2 rebuild didn't happen and TF.js will refuse to load it.

### If you change the number of classes

Update `CLASS_NAMES` **and** `CLASS_LABELS` in `inference.js` to match
`print(train_ds.class_names)` exactly. Keras sorts folders alphabetically — that ordering
is the contract between Python and the browser. `inference.js` refuses to load a model
whose output width disagrees with `CLASS_NAMES`, so a mismatch fails loudly, not silently.

---

## 3. Install / run

**TensorFlow.js is already bundled** (`vendor/tf.min.js`, v4.22.0) — there is nothing to
`npm install` to use the extension.

If you ever refresh it, **copy the `es2017` build, not `tf.min.js`**:

```bash
npm install @tensorflow/tfjs@4.22.0
cp node_modules/@tensorflow/tfjs/dist/tf.es2017.min.js vendor/tf.min.js   # <- es2017!
```

The default `dist/tf.min.js` ships a core-js `Function("return this")()` polyfill that
MV3's CSP blocks, which breaks TF.js at load with *"tf.ready is not a function"*. The
`es2017` build uses `globalThis` and has no `eval`. Verify after copying:

```bash
grep -c 'Function("return' vendor/tf.min.js    # must print 0
```

### Load it in Chrome

1. Open `chrome://extensions`.
2. Toggle **Developer mode** — top-right corner of that page.
3. Click **Load unpacked**.
4. Select the `stormalyze` folder (the one containing `manifest.json`).
5. Pin the extension: puzzle-piece icon in the toolbar → pin **Stormalyze**.

After changing any file — including dropping in new model files — click the **↻ reload**
icon on the extension's card.

> **Replacing this folder replaces `model/` too.** Unzipping a fresh copy over your
> install overwrites whatever model you converted in, silently swapping which weights are
> live. Either re-run `tools\update_model.bat` afterwards, or copy your `model/` folder
> back over the new one. The popup shows a **Weights** fingerprint — six hex characters
> of the weight bytes — so you can confirm at a glance which model is loaded; hover it for
> the conversion date and source filename.

---

## 4. Using it

1. Open a page showing radar imagery (any normal `http(s)` page).
2. Click the **Stormalyze** icon. The popup should say **Model found ✓**.
3. Click **Analyze Screen**. The page freezes under a dark overlay.
4. Drag a rectangle around one storm cell. The live size readout shows the crop size in
   *screenshot* pixels. Press **Esc** to bail out.
5. Release. The popup reopens by itself showing the **expected hazards** for the storm
   type it detected. The classification itself is internal — the raw class probabilities
   live under *Model detail*, collapsed.
6. Hit **Analyze Another** to go straight back to the overlay. The model stays resident in
   the offscreen document, so every analysis after the first skips the load entirely
   (measured: 16.2 s for the first, 7.0 s for the second, on a machine with no GPU).

**Performance note.** This is a 512×512-input CNN whose first conv runs at full
resolution, so it is not a lightweight model. On a real GPU expect a few hundred ms per
prediction. With software rendering (no GPU acceleration — the fallback in a VM or with
hardware acceleration disabled in Chrome) it takes several seconds; the popup prints the
actual latency and which backend was used, so you can tell immediately which
case you're in. If it says `cpu` and you expected `webgl`, enable
*Settings → System → Use graphics acceleration when available*.

---

### Hazard output and safety framing

The result view leads with hazards rather than a class label, because "damaging winds,
brief spin-up tornadoes" is more useful than "QLCS Squall, 87%". Three things are
deliberate:

- **Uncertainty is surfaced, not hidden.** Below 50% top-class confidence the popup shows
  a caveat saying so. The model has been confidently wrong before; a hazard list with no
  hedge would overstate what it knows.
- **Every result carries a "not a warning product" disclaimer** plus direct links to NWS
  active alerts and SPC outlooks and mesoscale discussions.
- **An info view** (the ⓘ in the header, which becomes a home button while you're in it)
  states plainly that this is a student project, unaffiliated with NOAA/NWS. Behind a
  **Model information** button inside it: what the network does, what it cannot do,
  measured accuracy, privacy, and the image-file test hook for Python-vs-browser parity
  checks.

Hazard text lives in `hazards.js`, keyed by class name — edit there, not in `popup.js`.

## 5. Data flow, end to end

```
 [1] popup.js  "Analyze Screen" / "Analyze Another"
        │  chrome.runtime.sendMessage({type:'START_CAPTURE'}); popup closes
        ▼
 [2] background.js
        │  warms the offscreen model in parallel with the user's drag
        │  chrome.tabs.captureVisibleTab() -> PNG of the visible viewport,
        │  sized viewportCssPx × devicePixelRatio (e.g. 1512×787 -> 3024×1574)
        │  chrome.scripting.executeScript(['selector.js']) + selector.css text
        ▼
 [3] selector.js  (in the page, Shadow DOM)
        │  draws the frozen screenshot at 100vw×100vh, dims it, tracks the drag
        │
        │  MEASURED scale, never assumed:
        │      scaleX = img.naturalWidth  / window.innerWidth
        │      scaleY = img.naturalHeight / window.innerHeight
        │
        │  canvas.drawImage(shot, sx,sy,sw,sh, 0,0,sw,sh)  -> exact crop, NO resampling
        ▼
 [4] background.js
        │  state = 'working'  ->  chrome.action.openPopup()   (popup reappears)
        │  PING the offscreen doc until it answers, then send INFER
        ▼
 [5] offscreen.js  (hidden document, model already resident)
        │
        │  tf.browser.fromPixels(img, 3).toFloat()      -> [h, w, 3] float32 0..255
        │  tf.image.resizeBilinear(x, [512,512],
        │                          alignCorners=false,
        │                          halfPixelCenters=TRUE) -> [512, 512, 3]
        │  x.div(255)   ONLY if the model has no Rescaling layer  -> [0, 1]
        │  x.expandDims(0)                              -> [1, 512, 512, 3]
        │  model.predict(input)                         -> [1, 6]
        │  input.dispose(); output.dispose()            (numTensors stays flat)
        ▼
 [6] background.js  writes the result to chrome.storage.session
        ▼
 [7] popup.js  renders: top class + confidence, six CSS bars sorted descending,
        crop size, latency, backend. The result persists, so re-opening the popup
        later still shows it until you analyze again.
```

### The two preprocessing details that actually matter

**`halfPixelCenters: true`.** TF.js defaults it to `false`, which is TF1's sampling grid.
TF2's `tf.image.resize` — the op behind `image_dataset_from_directory` and
`layers.Resizing` — always uses half-pixel centres. With the flag off, every sample is
shifted half a pixel and browser predictions drift from Keras. Measured: the flag on
agrees with a TF2 reference grid to ~1e-6; off, it disagrees by over 1.0 on 8-bit pixels.

**The `/255` is conditional.** The model has `layers.Rescaling(1./255)` baked in, and that
layer survives conversion. Dividing again in JS would feed the network `[0, 1/255]` and
quietly ruin every prediction. `inference.js` walks the loaded graph and only rescales in
JS when no `Rescaling`/`Normalization` layer is present. Override with
`FORCE_RESCALE_MODE` at the top of `inference.js`.

---

## 6. Verifying the browser matches Keras

This was already done for the bundled model. Measured across four test images of assorted
sizes (437×311, 700×420, 512×512, 97×63):

| Quantity | Browser vs Keras |
|---|---|
| Resized 512×512 input tensor | **exact** (max Δ = 0.0) |
| Penultimate 128-d activations | max Δ ≈ 1.9e-6 (≈1.2e-7 relative) |
| Output probabilities | max Δ ≈ 1.2e-7 |
| Top-1 class | agreed on every image |

The resize being bit-identical is the meaningful part — it confirms `halfPixelCenters`
matches TF2's grid. The ~1e-6 on activations is ordinary float32 reduction-order noise
between WebGL and CPU TensorFlow.

### Repeating it yourself after retraining

1. Save any test crop as a PNG (right-click the thumbnail in the popup → **Save image
   as…**, or use a training image).
2. Reference prediction in Python:

   ```bash
   python tools/predict_python.py project_cell_id_model.keras test_crop.png
   ```

   This reproduces the JS pipeline exactly: float32 `[0,255]` → `tf.image.resize`
   bilinear `antialias=False` → conditional ÷255 → batch dim.
3. In the popup click **Test with a file** and choose the same PNG.
4. Compare. Agreement to ~1e-3 on each probability is expected and correct.

Interpreting a mismatch:

| Symptom | Cause |
|---|---|
| Probabilities wildly different, browser near-uniform | Double rescaling. The footer says either *"Rescaling layer in model"* or *"rescaled in JS"* — never both. Force it with `FORCE_RESCALE_MODE` in `inference.js`. |
| Right ranking, ~1–5 % off | Resize convention. Confirm `halfPixelCenters` is `true` and Python used `antialias=False`. |
| Ranking permuted consistently | `CLASS_NAMES` order. Must equal `print(train_ds.class_names)` — currently `['bowecho_squall', 'multicell', 'nonradar_image', 'qlcs_squall', 'single_cell', 'supercell']`. |
| Everything says `nonradar_image` | Probably correct — the model has a dedicated reject class and it fires on anything that isn't radar imagery, including UI chrome caught in the crop. Tighten the selection. |

### Comparing two trained models

After a training change (epochs, class weights, augmentation), a validation number won't
tell you what actually moved. `tools/compare_models.py` runs both models over the same
images and reports each one's top class, confidence, and normalised entropy, plus
aggregates and disagreement count:

```bash
python tools/compare_models.py old.keras new.keras path/to/test_images/

# with ground truth from folder names (images/supercell/*.png) or filename prefixes
python tools/compare_models.py old.keras new.keras images/ --labelled --csv out.csv
```

Entropy is the useful column: an undertrained model **hedges** (entropy up, top
confidence down), while a model with a bad decision boundary stays confident and wrong.
Those two failures need opposite fixes, and confidence alone can't distinguish them.

To isolate conversion loss from runtime loss, load the converted model back in Python:
`tfjs.converters.load_keras_model('model/model.json')` and compare to the original.

---

## 7. Debugging

**Where the logs are** — three separate consoles, and picking the wrong one is the most
common time sink:

- `chrome://extensions` → the extension card → **service worker** — for `background.js`,
  and the same card's **offscreen.html** link for `offscreen.js` / `inference.js` / TF.js.
- Right-click inside the popup → **Inspect** — for `popup.js`.
- DevTools on the page you snipped — for `selector.js`.

| Error | Fix |
|---|---|
| Popup says **Model missing** | `model/model.json` isn't there, or is a `GraphModel`. Re-run the converter with `--output_format=tfjs_layers_model` and reload the extension. |
| `Failed to fetch … group1-shard1of3.bin` | Shards weren't copied next to `model.json`, or were renamed. Copy the converter's whole output folder. |
| `tf.ready is not a function` (or `tf.loadLayersModel is not a function`) | The vendored TF.js bundle hit a CSP block and only half-initialised. Check the offscreen document's console for *"Refused to evaluate a string as JavaScript"*, then re-copy the **es2017** build per §3. |
| `An InputLayer should be passed either a batchInputShape or an inputShape` | A Keras 3 config reached TF.js. Reconvert with `convert_model.py` (it rebuilds in Keras 2); check `model.json`'s first layer says `batch_input_shape`. |
| `Model outputs N values but CLASS_NAMES lists M` | Update `CLASS_NAMES` and `CLASS_LABELS` in `inference.js` to match `train_ds.class_names`. |
| `Unknown layer: RandomFlip` (or `RandomRotation`, `RandomZoom`) | Re-run `convert_model.py --strip-augmentation`. |
| `Unknown layer: <your custom layer>` | Register it in JS with `tf.serialization.registerClass`, or refactor it out of the inference graph. |
| `ResolutionImpossible` installing tensorflowjs | Expected on Windows — `tensorflow-decision-forests` has no Windows wheels. You don't need the package; `convert_model.py` writes the format directly. |
| `UNMAPPED LAYER TYPE: X` from the converter | Your retrained model uses a layer the Keras 2 rebuild doesn't know. Add it to `convert()` in `tools/convert_model.py`. |
| **Chrome blocks extensions on this page** | You're on `chrome://…`, the Web Store, or a PDF. Try a normal web page. |
| `Cannot access contents of the page` | `activeTab` is granted per-click. Click the toolbar icon on the tab you want; don't trigger capture from a stale popup. |
| Overlay never appears | The page may have loaded before the extension was installed — reload the tab once. |
| Popup stuck on "Analyzing…" | Every offscreen request has a timeout, so this resolves into a readable error rather than spinning forever. If it times out repeatedly, open the offscreen document's console from the extension card. |
| Popup doesn't reappear after the drag | `chrome.action.openPopup()` needs Chrome 127+ and a focused window. The fallback is a ✓ badge on the toolbar icon — click it and the result is there. |
| `Unable to create WebGLTexture` / GPU OOM | Very large model or a busy GPU. Add `await tf.setBackend('cpu')` before `tf.ready()` in `inference.js` to confirm, then decide (CPU is ~10× slower but reliable). |
| Predictions differ between two snips of the same area | Expected only at the 1e-6 level. Larger drift means the crop rectangles differed — check the px readout during the drag. |
| `The inference host did not start` | The offscreen document failed to load. Reload the extension; check the offscreen console for a CSP or module error. |
| `Extension context invalidated` | You reloaded the extension while an overlay was open. Reload the page. |

---

## 8. Privacy & permissions

Requested permissions, and why each is the minimum:

- **`activeTab`** — lets `captureVisibleTab` see the tab you clicked on, and only after you
  click. This is why there is no `<all_urls>` and no `tabs` permission, and why the
  extension shows no scary "read your data on all websites" warning.
- **`scripting`** — to inject `selector.js` on demand instead of running a content script
  on every page you visit.
- **`storage`** — `chrome.storage.session` only (in-memory, cleared when Chrome closes) to
  pass the crop and the result between the worker, the offscreen document and the popup.
- **`offscreen`** — creates the one hidden document that holds the model in memory. It has
  no network access of its own and is closed when Chrome closes.

The screenshot never leaves the browser process. There is no `fetch` to any origin, no
analytics, and the only network activity is Chrome loading files from the extension folder
itself. `vendor/tf.min.js` is bundled precisely so no CDN request ever happens.
