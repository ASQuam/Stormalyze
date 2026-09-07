#!/usr/bin/env python3
"""
predict_python.py — reference prediction, for verifying the browser matches Keras.

It reproduces EXACTLY what inference.js does:
    decode image -> float32 [0,255] -> tf.image.resize(bilinear, antialias=False)
    -> /255 only if the model has no Rescaling layer -> batch dim -> predict

Usage
-----
    python tools/predict_python.py project_cell_id_model.keras test_crop.png

Then load the same PNG in the extension popup via "Test with a file"
and compare. Agreement to ~1e-3 is expected
(WebGL runs float32 with slightly different reduction order than CPU TF).
"""

import argparse
import sys

CLASS_NAMES = [
    "bowecho_squall",
    "multicell",
    "nonradar_image",
    "qlcs_squall",
    "single_cell",
    "supercell",
]


def flatten_layers(model):
    for layer in getattr(model, "layers", []):
        yield layer
        if hasattr(layer, "layers"):
            yield from flatten_layers(layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keras_model")
    ap.add_argument("image")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf
    import keras

    model = keras.models.load_model(args.keras_model, compile=False)

    has_rescaling = any(
        type(l).__name__ in ("Rescaling", "Normalization") for l in flatten_layers(model)
    )

    raw = tf.io.read_file(args.image)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    x = tf.cast(img, tf.float32)                                   # [0, 255]
    x = tf.image.resize(x, [args.size, args.size], method="bilinear", antialias=False)

    if not has_rescaling:
        x = x / 255.0

    x = tf.expand_dims(x, 0)

    probs = np.asarray(model.predict(x, verbose=0))[0]

    # match inference.js: softmax the output if it isn't already a distribution
    if not (probs.min() >= -1e-6 and probs.max() <= 1 + 1e-6 and abs(probs.sum() - 1) < 1e-3):
        e = np.exp(probs - probs.max())
        probs = e / e.sum()

    order = np.argsort(-probs)

    print(f"\nimage             : {args.image}")
    print(f"model has Rescaling: {has_rescaling}  (JS divides by 255: {not has_rescaling})")
    print(f"input tensor       : {tuple(x.shape)}  range [{float(tf.reduce_min(x)):.4f}, "
          f"{float(tf.reduce_max(x)):.4f}]")
    print("\nprediction:", CLASS_NAMES[int(order[0])], f"{probs[order[0]]*100:.2f}%\n")

    for i in order:
        bar = "█" * int(round(probs[i] * 30))
        print(f"  {CLASS_NAMES[i]:<16} {probs[i]*100:6.2f}%  {bar}")

    print()
    print("raw vector:", ", ".join(f"{p:.6f}" for p in probs))


if __name__ == "__main__":
    sys.exit(main())
