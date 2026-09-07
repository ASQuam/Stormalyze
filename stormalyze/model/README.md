# Converted model (already populated)

This folder holds the TensorFlow.js build of `project_cell_id_model.keras`:

```
model/
├── model.json              topology + weight manifest (tfjs_layers_model, Keras 2 config)
├── group1-shard1of2.bin    4.0 MB
└── group1-shard2of2.bin    2.3 MB
```

1,635,014 parameters, input `(None, 512, 512, 3)`, 6 softmax outputs, with
`Rescaling(1/255)` baked in — so the extension does **not** divide by 255 again.

To regenerate after retraining:

```bash
python tools/convert_model.py project_cell_id_model.keras --out model
```

Then click the reload icon on the extension card at `chrome://extensions`.
Don't put the `.keras` file here — the browser can't read it.
