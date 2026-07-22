"""
Gesto — convert trained Keras model to TFLite, WITHOUT rebuilding the
architecture (which risks set_weights misalignment) and WITHOUT Flex ops.

Strategy: convert the ORIGINAL loaded model directly. To avoid the dynamic
TensorList/While ops that LSTMs normally emit (which need the Flex delegate),
we set the LSTM layers' `unroll` flag ON THE LOADED MODEL IN PLACE via a
clone, preserving exact weights by cloning rather than manually rebuilding.

Then it SELF-VERIFIES: runs the same input through the Keras model and the
converted TFLite model and refuses to save if they disagree.

Run:
    python convert_tflite.py --model gesto_model.h5 --labels labels.json --out gesto.tflite
"""

import json
import argparse

import numpy as np
import tensorflow as tf


def _force_unroll(layer):
    """Clone-config hook: turn on unroll for LSTM layers so conversion stays
    on core ops, without touching weights or structure."""
    cfg = layer.get_config()
    if layer.__class__.__name__ == "LSTM":
        cfg["unroll"] = True
    return layer.__class__.from_config(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gesto_model.h5")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--out", default="gesto.tflite")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    frames = int(model.input_shape[1])
    dim = int(model.input_shape[2])
    n_classes = int(model.output_shape[1])
    print(f"Loaded model: [{frames}, {dim}] -> {n_classes} classes")

    # Clone the model with unroll forced on LSTMs, then copy weights EXACTLY.
    # clone_model preserves layer order/structure, so set_weights aligns 1:1.
    unrolled = tf.keras.models.clone_model(model, clone_function=_force_unroll)
    unrolled.set_weights(model.get_weights())

    # sanity: Keras-vs-Keras (original vs unrolled) must already match
    x = np.random.random((1, frames, dim)).astype(np.float32)
    a = model.predict(x, verbose=0)[0]
    b = unrolled.predict(x, verbose=0)[0]
    diff_keras = np.abs(a - b).max()
    print(f"Keras original-vs-unrolled max diff: {diff_keras:.6f}")
    if diff_keras > 1e-4:
        raise RuntimeError(
            "Unrolled clone doesn't match original — aborting. "
            "The architecture clone changed the math.")

    converter = tf.lite.TFLiteConverter.from_keras_model(unrolled)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # DO NOT enable optimizations by default: quantization is a common cause
    # of large keras-vs-tflite drift for LSTMs. Keep it float32.
    tflite_model = converter.convert()

    # ---- SELF-VERIFY against the original Keras model ----
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outp = interp.get_output_details()[0]

    max_diff = 0.0
    for _ in range(5):
        xt = np.random.random((1, frames, dim)).astype(np.float32)
        k = model.predict(xt, verbose=0)[0]
        interp.set_tensor(inp["index"], xt)
        interp.invoke()
        t = interp.get_tensor(outp["index"])[0]
        max_diff = max(max_diff, float(np.abs(k - t).max()))

    print(f"Keras-vs-TFLite max diff over 5 inputs: {max_diff:.6f}")
    if max_diff > 0.01:
        raise RuntimeError(
            f"CONVERSION IS BROKEN: max diff {max_diff:.4f} (should be <0.01). "
            f"NOT saving. The tflite would misclassify. Tell me this number.")

    with open(args.out, "wb") as f:
        f.write(tflite_model)
    print(f"VERIFIED OK — saved {args.out} ({len(tflite_model)/1024:.1f} KB)")

    # companion meta
    try:
        meta = json.load(open(args.labels))
    except Exception:
        meta = {}
    meta.update({"frames": frames, "feature_dim": dim, "n_classes": n_classes})
    out_meta = args.out.rsplit(".", 1)[0] + "_meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta -> {out_meta}")


if __name__ == "__main__":
    main()