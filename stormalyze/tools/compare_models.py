#!/usr/bin/env python3
"""
compare_models.py — run two Keras models over the same folder of images and
tabulate how their predictions differ.

Built for exactly the question "did that training change actually help?", which is
impossible to answer from a validation number alone once the change involves class
weights, epoch counts, or anything else that shifts calibration.

Reports, per image: each model's top class + confidence, and normalised entropy
(0 = fully confident, 1 = uniform). Then aggregates: how often each model picks each
class, mean confidence, mean entropy, and — if you name the true class in the filename
or use one-class-per-subfolder layout — accuracy.

Usage
-----
    python tools/compare_models.py old.keras new.keras path/to/images/

    # with ground truth: name files like  supercell_0031.png,  or use
    #   images/supercell/*.png   images/multicell/*.png   ...
    python tools/compare_models.py old.keras new.keras images/ --labelled

Preprocessing matches inference.js exactly: float32 [0,255] -> bilinear resize to the
model's input size (antialias=False) -> /255 only if the model has no Rescaling layer.
"""

import argparse
import glob
import os
import sys

CLASS_NAMES = [
    "bowecho_squall",
    "multicell",
    "nonradar_image",
    "qlcs_squall",
    "single_cell",
    "supercell",
]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")


def flatten_layers(model):
    for layer in getattr(model, "layers", []):
        yield layer
        if hasattr(layer, "layers"):
            yield from flatten_layers(layer)


def has_rescaling(model):
    return any(
        type(l).__name__ in ("Rescaling", "Normalization") for l in flatten_layers(model)
    )


def find_images(root):
    if os.path.isfile(root):
        return [root]
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def truth_of(path):
    """Ground truth from the parent folder name, else from a filename prefix."""
    parent = os.path.basename(os.path.dirname(path))
    if parent in CLASS_NAMES:
        return parent
    stem = os.path.basename(path).lower()
    for c in CLASS_NAMES:
        if stem.startswith(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_a")
    ap.add_argument("model_b")
    ap.add_argument("images")
    ap.add_argument("--labelled", action="store_true",
                    help="derive ground truth from folder names or filename prefixes")
    ap.add_argument("--csv", help="also write per-image rows to this CSV")
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf
    import keras

    files = find_images(args.images)
    if not files:
        sys.exit(f"No images found under {args.images}")

    models, infos = {}, {}
    for tag, path in (("A", args.model_a), ("B", args.model_b)):
        m = keras.models.load_model(path, compile=False)
        models[tag] = m
        infos[tag] = {
            "name": os.path.basename(path),
            "rescaling": has_rescaling(m),
            "size": int(m.inputs[0].shape[1]) if m.inputs[0].shape[1] else 512,
            "units": int(m.outputs[0].shape[-1]),
        }
        if infos[tag]["units"] != len(CLASS_NAMES):
            sys.exit(f"{path} outputs {infos[tag]['units']} classes; CLASS_NAMES lists "
                     f"{len(CLASS_NAMES)}. Edit CLASS_NAMES at the top of this script.")

    for tag in ("A", "B"):
        i = infos[tag]
        print(f"model {tag}: {i['name']}  input {i['size']}²  "
              f"Rescaling-in-model={i['rescaling']}")
    print(f"{len(files)} images\n")

    def preprocess(path, tag):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        x = tf.cast(img, tf.float32)
        x = tf.image.resize(x, [infos[tag]["size"]] * 2, method="bilinear", antialias=False)
        if not infos[tag]["rescaling"]:
            x = x / 255.0
        return tf.expand_dims(x, 0)

    def entropy(p):
        p = np.clip(p, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum() / np.log(len(p)))

    rows = []
    width = min(34, max(len(os.path.basename(f)) for f in files))

    header = f"{'image':<{width}} | {'A: top':<16} {'conf':>7} {'H':>6} | {'B: top':<16} {'conf':>7} {'H':>6}"
    if args.labelled:
        header = f"{'truth':<15} " + header
    print(header)
    print("-" * len(header))

    for f in files:
        pa = models["A"].predict(preprocess(f, "A"), verbose=0)[0]
        pb = models["B"].predict(preprocess(f, "B"), verbose=0)[0]
        t = truth_of(f) if args.labelled else None

        row = {
            "file": f, "truth": t,
            "a_top": CLASS_NAMES[int(pa.argmax())], "a_conf": float(pa.max()), "a_H": entropy(pa),
            "b_top": CLASS_NAMES[int(pb.argmax())], "b_conf": float(pb.max()), "b_H": entropy(pb),
        }
        rows.append(row)

        line = (f"{os.path.basename(f)[:width]:<{width}} | "
                f"{row['a_top']:<16} {row['a_conf']*100:6.2f}% {row['a_H']:6.3f} | "
                f"{row['b_top']:<16} {row['b_conf']*100:6.2f}% {row['b_H']:6.3f}")
        if args.labelled:
            line = f"{str(t):<15} " + line
        print(line)

    # ---- aggregates --------------------------------------------------------
    print("\n" + "=" * 66)
    print("  Summary")
    print("=" * 66)

    for tag, key in (("A", "a"), ("B", "b")):
        confs = [r[f"{key}_conf"] for r in rows]
        ents = [r[f"{key}_H"] for r in rows]
        print(f"\nmodel {tag} ({infos[tag]['name']})")
        print(f"  mean confidence : {np.mean(confs)*100:.2f}%")
        print(f"  mean entropy    : {np.mean(ents):.4f}   (0 = certain, 1 = uniform)")
        counts = {}
        for r in rows:
            counts[r[f"{key}_top"]] = counts.get(r[f"{key}_top"], 0) + 1
        spread = ", ".join(f"{c}={n}" for c, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"  predictions     : {spread}")

        if args.labelled:
            known = [r for r in rows if r["truth"]]
            if known:
                acc = sum(r[f"{key}_top"] == r["truth"] for r in known) / len(known)
                print(f"  accuracy        : {acc*100:.1f}%  ({len(known)} labelled)")

    disagree = [r for r in rows if r["a_top"] != r["b_top"]]
    print(f"\nmodels disagree on {len(disagree)}/{len(rows)} images")

    # entropy is the underfitting tell: a model that hasn't converged hedges
    dH = np.mean([r["b_H"] for r in rows]) - np.mean([r["a_H"] for r in rows])
    if dH > 0.02:
        print(f"model B hedges more (entropy +{dH:.3f}) — consistent with undertraining.")
    elif dH < -0.02:
        print(f"model B is more decisive (entropy {dH:.3f}).")
    else:
        print("calibration is broadly unchanged between the two.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
