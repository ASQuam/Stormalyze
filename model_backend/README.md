# Stormalyze — Model Training

The training notebook behind [Stormalyze](../stormalyze) — a CNN that classifies radar
reflectivity imagery into 6 storm morphologies. Runs in Google Colab against a hand-labeled
image dataset stored on Google Drive.

## Pipeline

1. **Load data** — `image_dataset_from_directory` on a `train/<class_name>/*.png` folder
   structure, 512×512 images, 80/20 train/validation split
2. **Augment** — a custom `randomize_background` step detects near-black background
   pixels and repaints them a random color each epoch, so the model can't shortcut by
   keying off a plain black background instead of the actual storm structure
3. **Build** — a 5-block CNN from scratch (Conv2D + MaxPooling, 32→512 filters) with
   global average pooling and a dense classification head (see architecture below)
4. **Train** — 30 epochs, Adam optimizer, categorical cross-entropy
5. **Evaluate** — accuracy/loss curves, spot-check predictions on validation images
6. **Export** — saved as `.keras` for downstream conversion to TensorFlow.js

## Model architecture

```
Rescaling(1/255)
→ Conv2D(32)  → MaxPool
→ Conv2D(64)  → MaxPool
→ Conv2D(128) → MaxPool
→ Conv2D(256) → MaxPool
→ Conv2D(512) → MaxPool
→ GlobalAveragePooling2D
→ Dense(128, relu) → Dropout(0.2)
→ Dense(6, softmax)
```

No pretrained backbone — trained end-to-end on a hand-labeled dataset of ~1,000 images
across 6 classes (`bowecho_squall`, `multicell`, `nonradar_image`, `qlcs_squall`,
`single_cell`, `supercell`).

## Requirements

Google Colab (or local Jupyter) with TensorFlow/Keras, scikit-learn, matplotlib, seaborn.
Dataset expected at `drive/MyDrive/cell_id_products/train/`.

## Output

`project_cell_id_model.keras` — converted separately for browser inference (see the
Stormalyze repo's `tools/convert_model.py`).
