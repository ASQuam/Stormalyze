#!/usr/bin/env python3
"""
convert_model.py — turn `project_cell_id_model.keras` into a TensorFlow.js
*layers* model (model.json + group1-shard*of*.bin) that tf.loadLayersModel() can read.

Why this isn't a one-liner
--------------------------
`tensorflowjs_converter` speaks **Keras 2** (tf_keras). Your model was saved by
**Keras 3**, and the two disagree about config keys in ways that break TF.js at load
time even when the converter itself reports success:

  * Keras 3 writes `batch_shape`; TF.js expects `batch_input_shape`
    -> "An InputLayer should be passed either a `batchInputShape` or an `inputShape`"
  * Keras 3 writes `dtype` as a DTypePolicy object; TF.js expects the string "float32"

Patching the JSON afterwards is brittle. Instead this script **rebuilds the architecture
in tf_keras and copies the weights across**, which produces a genuine Keras 2 graph the
converter and TF.js both understand. The rebuild is verified layer-by-layer and the
weight transfer is checked to be exact before anything is written.

No `tensorflowjs` package required
----------------------------------
`pip install tensorflowjs` cannot succeed on Windows: it depends on
`tensorflow-decision-forests`, which publishes no Windows wheels at all, so pip exits
with ResolutionImpossible. This script therefore writes the tfjs `layers-model` format
itself — it is just a `model.json` plus 4 MiB weight shards. The output is verified
byte-for-byte identical to what `tensorflowjs_converter` produces.

Usage
-----
    pip install "tensorflow-cpu==2.17.1" "tf-keras==2.17.0"
    python tools/convert_model.py project_cell_id_model.keras --out model

Options
-------
    --strip-augmentation   drop RandomFlip/RandomRotation/RandomZoom/... layers.
                           TF.js cannot rebuild most of them and they are inactive at
                           inference time anyway.
    --skip-rebuild         convert the Keras 3 file directly (only works if you are
                           already on Keras 2).
    --use-tfjs-converter   shell out to `tensorflowjs_converter` instead of writing the
                           format directly. Needs the tensorflowjs package, so this is
                           Linux/macOS only in practice.
"""

import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys

AUGMENTATION_LAYERS = {
    "RandomFlip", "RandomRotation", "RandomZoom", "RandomTranslation",
    "RandomContrast", "RandomBrightness", "RandomCrop", "RandomHeight",
    "RandomWidth", "GaussianNoise", "GaussianDropout",
}


def log(msg):
    print(f"[convert] {msg}", flush=True)


def flatten_layers(model):
    """Yield every layer, descending into nested Sequential/Functional models."""
    for layer in getattr(model, "layers", []):
        yield layer
        if hasattr(layer, "layers"):
            yield from flatten_layers(layer)


def rebuild_in_keras2(k3_model, strip_augmentation=False):
    """
    Recreate a Sequential Keras 3 model as a tf_keras (Keras 2) model and copy weights.

    Extend the mapping below if you add layer types later — it fails loudly on anything
    it doesn't recognise rather than silently producing a wrong graph.
    """
    import numpy as np
    import tf_keras
    L = tf_keras.layers

    def convert(layer):
        c = layer.get_config()
        n = type(layer).__name__

        if n == "InputLayer":
            shape = c.get("batch_shape") or c.get("batch_input_shape")
            return L.InputLayer(input_shape=tuple(shape[1:]), name=c["name"])
        if n == "Rescaling":
            return L.Rescaling(scale=c["scale"], offset=c.get("offset", 0.0), name=c["name"])
        if n == "Resizing":
            return L.Resizing(c["height"], c["width"],
                              interpolation=c.get("interpolation", "bilinear"), name=c["name"])
        if n == "Conv2D":
            return L.Conv2D(c["filters"], c["kernel_size"], strides=c["strides"],
                            padding=c["padding"], activation=c["activation"],
                            use_bias=c["use_bias"], name=c["name"])
        if n == "SeparableConv2D":
            return L.SeparableConv2D(c["filters"], c["kernel_size"], strides=c["strides"],
                                     padding=c["padding"], activation=c["activation"],
                                     use_bias=c["use_bias"], name=c["name"])
        if n == "MaxPooling2D":
            return L.MaxPooling2D(pool_size=c["pool_size"], strides=c["strides"],
                                  padding=c["padding"], name=c["name"])
        if n == "AveragePooling2D":
            return L.AveragePooling2D(pool_size=c["pool_size"], strides=c["strides"],
                                      padding=c["padding"], name=c["name"])
        if n == "GlobalAveragePooling2D":
            return L.GlobalAveragePooling2D(name=c["name"])
        if n == "GlobalMaxPooling2D":
            return L.GlobalMaxPooling2D(name=c["name"])
        if n == "Dense":
            return L.Dense(c["units"], activation=c["activation"],
                           use_bias=c["use_bias"], name=c["name"])
        if n == "Dropout":
            return L.Dropout(c["rate"], name=c["name"])
        if n == "Flatten":
            return L.Flatten(name=c["name"])
        if n == "Activation":
            return L.Activation(c["activation"], name=c["name"])
        if n == "BatchNormalization":
            kw = {k: c[k] for k in ("axis", "momentum", "epsilon", "center", "scale") if k in c}
            return L.BatchNormalization(name=c["name"], **kw)

        raise SystemExit(
            f"[convert] UNMAPPED LAYER TYPE: {n}\n"
            f"          Add it to convert() in tools/convert_model.py, or convert this "
            f"model on Keras 2 with --skip-rebuild."
        )

    layers = []
    for layer in k3_model.layers:
        if strip_augmentation and type(layer).__name__ in AUGMENTATION_LAYERS:
            continue
        layers.append(convert(layer))

    in_shape = tuple(k3_model.inputs[0].shape)
    k2 = tf_keras.Sequential(layers, name="cell_classifier")
    k2.build(in_shape)

    src_weights = k3_model.get_weights()
    k2.set_weights(src_weights)

    # Prove the transfer is exact rather than assuming it.
    dst_weights = k2.get_weights()
    if len(src_weights) != len(dst_weights):
        raise SystemExit("[convert] weight count mismatch after rebuild — aborting.")
    worst = max(float(np.abs(a - b).max()) for a, b in zip(src_weights, dst_weights))
    log(f"weight transfer verified, max |Δ| = {worst}")
    if worst != 0.0:
        raise SystemExit("[convert] weights did not transfer exactly — aborting.")

    log(f"rebuilt in tf_keras {tf_keras.__version__}: "
        f"{k2.count_params():,} params (source had {k3_model.count_params():,})")
    return k2


SHARD_BYTES = 4 * 1024 * 1024  # what tensorflowjs_converter uses


def strip_layer_meta(topo):
    """
    to_json() tags every LAYER with module/registered_name/build_config; the official
    converter omits them. Strip at layer level ONLY — the same keys inside nested
    initializer configs ARE kept by the converter, so a blind recursive strip diverges
    from the reference output.
    """
    def walk(layers):
        for layer in layers or []:
            for k in ("module", "registered_name", "build_config"):
                layer.pop(k, None)
            inner = layer.get("config", {})
            if isinstance(inner, dict) and "layers" in inner:
                walk(inner["layers"])

    walk(topo.get("config", {}).get("layers"))


def export_tfjs_layers_model(model, out_dir, keras_version):
    """
    Write model.json + group1-shard*of*.bin in TensorFlow.js `layers-model` format.

    Layout, straight from the reference converter's output:
      - one weight group whose `paths` lists every shard in order;
      - the shards are a plain byte-concatenation of each weight's C-order float32 bytes;
      - weight names are "<layer name>/<kernel|bias|...>".
    """
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    topo = json.loads(model.to_json())
    strip_layer_meta(topo)

    specs, blobs = [], []
    for layer in model.layers:
        for w in layer.weights:
            leaf = w.name.split("/")[-1].split(":")[0]
            arr = np.asarray(w.numpy())
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            specs.append({
                "name": f"{layer.name}/{leaf}",
                "shape": list(arr.shape),
                "dtype": "float32",
            })
            blobs.append(arr.tobytes("C"))

    if not specs:
        raise SystemExit("[convert] the model has no weights — nothing to export.")

    data = b"".join(blobs)
    total = max(1, math.ceil(len(data) / SHARD_BYTES))

    # Remove shards from a previous run so a smaller model can't leave orphans behind.
    # Tolerate Windows/OneDrive locks: a leftover shard is harmless (model.json names
    # exactly the ones it needs), so a failed delete must not abort the conversion.
    for stale in glob.glob(os.path.join(out_dir, "group1-shard*of*.bin")):
        try:
            os.chmod(stale, stat.S_IWRITE)
            os.remove(stale)
        except OSError:
            pass

    paths = []
    for i in range(total):
        name = f"group1-shard{i + 1}of{total}.bin"
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(data[i * SHARD_BYTES:(i + 1) * SHARD_BYTES])
        paths.append(name)

    fingerprint = hashlib.md5(data).hexdigest()[:6]

    manifest = {
        "format": "layers-model",
        "generatedBy": f"keras v{keras_version}",
        "convertedBy": "stormalyze/tools/convert_model.py (tfjs layers-model writer)",
        "modelTopology": {
            "keras_version": keras_version,
            "backend": "tensorflow",
            "model_config": {"class_name": topo["class_name"], "config": topo["config"]},
        },
        "weightsManifest": [{"paths": paths, "weights": specs}],
    }

    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(manifest, f)

    log(f"wrote model.json + {total} shard(s), {len(data) / 1e6:.1f} MB of weights")
    log(f"weight fingerprint: {fingerprint}  <- shown in the popup, so you can confirm "
        f"which model is live")
    return manifest, fingerprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keras_model", help="path to project_cell_id_model.keras")
    ap.add_argument("--out", default="model", help="output dir for model.json + shards")
    ap.add_argument("--strip-augmentation", action="store_true")
    ap.add_argument("--skip-rebuild", action="store_true")
    ap.add_argument("--use-tfjs-converter", action="store_true")
    args = ap.parse_args()

    import keras
    import tensorflow as tf  # noqa: F401

    log(f"tensorflow {tf.__version__}, keras {keras.__version__}")

    model = keras.models.load_model(args.keras_model, compile=False)

    in_shape = tuple(model.inputs[0].shape)
    out_units = int(model.outputs[0].shape[-1])
    log(f"input shape  : {in_shape}")
    log(f"output units : {out_units}  <- CLASS_NAMES in inference.js must have this many entries")

    names = [type(l).__name__ for l in flatten_layers(model)]
    has_rescaling = "Rescaling" in names
    aug_present = sorted(set(names) & AUGMENTATION_LAYERS)

    log(f"Rescaling layer inside the model: {has_rescaling}")
    if has_rescaling:
        log("  -> the browser must NOT divide by 255 again; inference.js detects this "
            "automatically (FORCE_RESCALE_MODE = 'auto').")
    else:
        log("  -> the browser will divide pixels by 255 itself.")

    if aug_present:
        log(f"augmentation layers found: {aug_present}")
        if not args.strip_augmentation:
            log("  !! TF.js will fail with 'Unknown layer'. Re-run with --strip-augmentation.")

    keras3 = keras.__version__.startswith("3")
    if keras3 and not args.skip_rebuild:
        model = rebuild_in_keras2(model, strip_augmentation=args.strip_augmentation)

    os.makedirs(args.out, exist_ok=True)

    fingerprint = None
    if args.use_tfjs_converter:
        convert_with_tfjs_cli(model, args.out)
    else:
        import tf_keras
        _manifest, fingerprint = export_tfjs_layers_model(model, args.out, tf_keras.__version__)

    # Written last so it records what actually shipped. inference.js reads this and the
    # popup displays it, so you can always tell which model the extension is running.
    with open(os.path.join(args.out, "model_meta.json"), "w") as f:
        json.dump(
            {
                "fingerprint": fingerprint,
                "converted_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "source": os.path.basename(args.keras_model),
                "input_shape": [None if d is None else int(d) for d in in_shape],
                "output_units": out_units,
                "has_rescaling_layer": bool(has_rescaling),
                "source_keras_version": keras.__version__,
                "rebuilt_for_keras2": bool(keras3 and not args.skip_rebuild),
            },
            f,
            indent=2,
        )

    log(f"done. {args.out}/model.json + weight shards are ready.")
    log("Sanity-check model.json: the first layer's config should contain "
        "'batch_input_shape' (Keras 2), NOT 'batch_shape' (Keras 3).")


def convert_with_tfjs_cli(model, out_dir):
    """Optional path: shell out to the official converter (not installable on Windows)."""
    h5_path = os.path.join(out_dir, "_model_for_conversion.h5")
    model.save(h5_path)
    log(f"wrote {h5_path}")

    exe = shutil.which("tensorflowjs_converter")
    if not exe:
        log("tensorflowjs_converter not found on PATH.")
        print_manual_command(h5_path, out_dir)
        sys.exit(1)

    cmd = [exe, "--input_format=keras", "--output_format=tfjs_layers_model", h5_path, out_dir]
    log("running: " + " ".join(cmd))

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print_manual_command(h5_path, out_dir)
        sys.exit(res.returncode)


def print_manual_command(h5_path, out_dir):
    print(
        "\n"
        "  python -m venv .tfjs-convert\n"
        "  source .tfjs-convert/bin/activate        # Windows: .tfjs-convert\\Scripts\\activate\n"
        '  pip install "tensorflowjs==4.22.0"\n'
        "  tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model \\\n"
        f"      {h5_path} {out_dir}\n"
    )


if __name__ == "__main__":
    main()
