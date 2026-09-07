#!/usr/bin/env python3
"""
update_model.py — one command to swap a retrained model into the extension.

Does the boring, easy-to-get-wrong parts:
  1. backs up the current model/ into model/_previous/ so you can roll back
  2. clears stale weight shards (a new model may produce a different shard count)
  3. runs tools/convert_model.py
  4. sanity-checks the result: Keras 2 config, shard files present, output width
  5. cross-checks the model's output width against CLASS_NAMES in inference.js
  6. deletes the big intermediate .h5

Usage
-----
    python tools/update_model.py path/to/new_model.keras

Windows users: just double-click tools/update_model.bat, or drag your .keras file
onto it.
"""

import glob
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time

# TensorFlow's C++ layer chatters on import; we only want this script's output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "model")
BACKUP_DIR = os.path.join(MODEL_DIR, "_previous")
INFERENCE_JS = os.path.join(ROOT, "inference.js")

WEIGHT_GLOBS = ("*.bin",)
MODEL_JSON = "model.json"


def log(msg=""):
    print(msg, flush=True)


def fail(msg):
    log("")
    log("!! " + msg)
    sys.exit(1)


def class_names_from_inference_js():
    """Pull the CLASS_NAMES array out of inference.js without executing it."""
    try:
        src = open(INFERENCE_JS, encoding="utf8").read()
    except OSError:
        return None

    m = re.search(r"export const CLASS_NAMES\s*=\s*\[(.*?)\]", src, re.S)
    if not m:
        return None
    return re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))


def output_units_from_model_json(path):
    """Read the unit count of the last Dense layer in a tfjs layers model."""
    with open(path, encoding="utf8") as f:
        j = json.load(f)

    layers = j["modelTopology"]["model_config"]["config"]["layers"]
    for layer in reversed(layers):
        if "units" in layer.get("config", {}):
            return int(layer["config"]["units"])
    return None


def force_remove(path, tries=5):
    """
    Delete a file that Windows may have locked or flagged read-only.

    OneDrive, Chrome (while the extension is loaded), and the search indexer all hold
    transient handles on files inside a synced folder, so a single unlink can fail with
    WinError 5 / 32 and succeed a moment later.
    """
    for i in range(tries):
        try:
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)   # clear read-only
                os.remove(path)
            return True
        except PermissionError:
            if i == tries - 1:
                return False
            time.sleep(0.4)
        except OSError:
            return False
    return False


def backup_current():
    existing = [os.path.join(MODEL_DIR, MODEL_JSON)] if os.path.exists(
        os.path.join(MODEL_DIR, MODEL_JSON)) else []
    for pattern in WEIGHT_GLOBS:
        existing += glob.glob(os.path.join(MODEL_DIR, pattern))

    if not existing:
        log("no existing model to back up")
        return

    # Deliberately NOT shutil.rmtree(): removing a directory inside a OneDrive-synced
    # folder routinely fails with "Access is denied" because the sync client holds a
    # handle on it. os.replace() overwrites the destination file atomically, so the
    # directory never has to be deleted.
    os.makedirs(BACKUP_DIR, exist_ok=True)

    moved, failed = 0, []
    for src in existing:
        dst = os.path.join(BACKUP_DIR, os.path.basename(src))
        try:
            os.replace(src, dst)
            moved += 1
        except OSError:
            # Fall back to copy-then-delete; if even the delete fails, the convert step
            # will overwrite the original anyway, so this is not fatal.
            try:
                shutil.copy2(src, dst)
                force_remove(src)
                moved += 1
            except OSError:
                failed.append(os.path.basename(src))

    log(f"backed up {moved} file(s) -> model/_previous/")
    if failed:
        log(f"could not back up: {', '.join(failed)} (continuing; they'll be overwritten)")


def check_python():
    """TensorFlow publishes no wheels for 3.13+, and none below 3.10 for current TF."""
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10) or (major, minor) >= (3, 13):
        fail(
            f"This is Python {major}.{minor}, and TensorFlow only publishes packages for "
            f"3.10 - 3.12.\n"
            f"   Install Python 3.12 from python.org (not the Microsoft Store build) and\n"
            f"   run tools/update_model.bat again — it will pick the right interpreter."
        )

    try:
        import tensorflow  # noqa: F401
    except ImportError:
        fail(
            "TensorFlow isn't installed in this interpreter.\n"
            "   Run tools/update_model.bat (Windows), or:\n"
            '     pip install "tensorflow-cpu==2.17.1" "tf-keras==2.17.0"'
        )


def main():
    check_python()

    if len(sys.argv) > 1:
        keras_path = sys.argv[1].strip('"')
    else:
        keras_path = input("Path to your .keras model file: ").strip().strip('"')

    if not keras_path:
        fail("No model path given.")
    if not os.path.isfile(keras_path):
        fail(f"File not found: {keras_path}")
    if not keras_path.lower().endswith(".keras"):
        log(f"warning: {os.path.basename(keras_path)} doesn't end in .keras — continuing anyway")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.sep + "OneDrive" + os.sep in os.path.abspath(MODEL_DIR):
        log("")
        log("note: this folder is inside OneDrive. Sync can lock files mid-write and")
        log("      Chrome reloads the extension on every sync touch. If you hit odd")
        log("      permission errors, move the extension somewhere unsynced, e.g.")
        log(f"      {os.path.join(os.path.expanduser('~'), 'stormalyze')}")

    # (the .bat already printed a banner)
    log(f"source    : {keras_path}")
    log(f"target    : {MODEL_DIR}")
    log("")

    backup_current()
    log("")

    cmd = [sys.executable, os.path.join(HERE, "convert_model.py"), keras_path, "--out", MODEL_DIR]
    res = subprocess.run(cmd)

    if res.returncode != 0:
        log("")
        log("Conversion failed. Restoring the previous model…")
        restore_backup()
        fail("Nothing was changed. See the converter output above.")

    # ---- verify the output ------------------------------------------------
    model_json = os.path.join(MODEL_DIR, MODEL_JSON)
    if not os.path.exists(model_json):
        restore_backup()
        fail("Converter reported success but produced no model.json. Previous model restored.")

    shards = sorted(glob.glob(os.path.join(MODEL_DIR, "*.bin")))
    if not shards:
        restore_backup()
        fail("No weight shards were produced. Previous model restored.")

    with open(model_json, encoding="utf8") as f:
        first_layer = json.load(f)["modelTopology"]["model_config"]["config"]["layers"][0]

    if "batch_shape" in first_layer.get("config", {}):
        restore_backup()
        fail(
            "model.json carries a Keras 3 input config ('batch_shape'). TensorFlow.js "
            "cannot load this. Previous model restored — see README §2."
        )

    # drop the large intermediate file
    h5 = os.path.join(MODEL_DIR, "_model_for_conversion.h5")
    if os.path.exists(h5):
        os.remove(h5)

    units = output_units_from_model_json(model_json)
    names = class_names_from_inference_js()

    log("")
    log("=" * 66)
    log("  Converted OK")
    log("=" * 66)
    log(f"model.json    : {os.path.getsize(model_json) / 1024:.1f} KB")
    log(f"weight shards : {len(shards)} "
        f"({sum(os.path.getsize(s) for s in shards) / 1e6:.1f} MB total)")
    log(f"output units  : {units}")

    if names is not None and units is not None:
        if len(names) == units:
            log(f"CLASS_NAMES   : {len(names)} entries — matches ✓")
            log(f"                {names}")
        else:
            log("")
            log("!! CLASS COUNT MISMATCH")
            log(f"   The model outputs {units} classes but inference.js lists {len(names)}:")
            log(f"   {names}")
            log("")
            log("   Edit CLASS_NAMES and CLASS_LABELS in inference.js to match")
            log("   print(train_ds.class_names) from training, then reload the extension.")
            log("   The extension will refuse to load until these agree.")

    log("")
    log("Next steps:")
    log("  1. Go to chrome://extensions and click the reload icon on the Stormalyze card.")
    log("     (That also discards the old model held in the offscreen document.)")
    log("  2. Click the Stormalyze icon — the popup should say 'Model ready'.")
    log("")
    log("Rollback if something looks wrong: copy the files from model/_previous/")
    log("back into model/ and reload the extension.")


def restore_backup():
    if not os.path.isdir(BACKUP_DIR):
        return
    for src in glob.glob(os.path.join(BACKUP_DIR, "*")):
        try:
            shutil.copy2(src, os.path.join(MODEL_DIR, os.path.basename(src)))
        except OSError as e:
            log(f"could not restore {os.path.basename(src)}: {e}")
    log("previous model restored from model/_previous/")


if __name__ == "__main__":
    main()
