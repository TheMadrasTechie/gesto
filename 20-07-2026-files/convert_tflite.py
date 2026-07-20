"""
convert_tflite.py — convert a trained Gesto model (.keras) to TFLite.

Works for BOTH the static model and the LSTM model. For LSTM models, plain
tf.lite conversion can silently produce a graph that computes different numbers
than Keras (no error — just wrong output). The fix, already validated on the
sign-language project: clone the model forcing unroll=True on every LSTM layer,
convert that, then verify the TFLite output matches Keras on random inputs.

Usage:
    python convert_tflite.py artifacts_lstm/lstm_model.keras
    python convert_tflite.py artifacts_static/static_model.keras
    python convert_tflite.py artifacts_lstm/lstm_model.keras --tolerance 1e-4
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def force_unroll(layer):
    """clone_function hook: rebuild LSTM layers with unroll=True, pass others
    through unchanged. Same architecture/weights — only the LSTM execution mode
    changes, which is what makes TFLite conversion numerically faithful."""
    if isinstance(layer, layers.LSTM):
        cfg = layer.get_config()
        cfg["unroll"] = True
        return layers.LSTM.from_config(cfg)
    return layer.__class__.from_config(layer.get_config())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="Path to a .keras model")
    ap.add_argument("--out", default=None, help="Output .tflite path")
    ap.add_argument("--num_checks", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()

    src = Path(args.model_path)
    out = Path(args.out) if args.out else src.with_suffix(".tflite")
    model = keras.models.load_model(src)

    has_lstm = any(isinstance(l, layers.LSTM) for l in model.layers)
    if has_lstm:
        print("LSTM detected — cloning with unroll=True for faithful conversion.")
        clone = models.clone_model(model, clone_function=force_unroll)
        clone.set_weights(model.get_weights())
        convert_model = clone
    else:
        convert_model = model

    converter = tf.lite.TFLiteConverter.from_keras_model(convert_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,      # some LSTM ops need this
    ]
    tflite = converter.convert()
    out.write_bytes(tflite)
    print(f"Wrote {out} ({len(tflite)} bytes)")

    # ---- self-verification: TFLite vs Keras on random inputs ----
    shape = list(model.inputs[0].shape)
    shape[0] = 1
    shape = [d if d is not None else 1 for d in shape]

    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]; outp = interp.get_output_details()[0]

    max_diff = 0.0
    for _ in range(args.num_checks):
        x = np.random.rand(*shape).astype(np.float32)
        k = model.predict(x, verbose=0)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        t = interp.get_tensor(outp["index"])
        max_diff = max(max_diff, float(np.max(np.abs(k - t))))

    print(f"Max Keras-vs-TFLite diff over {args.num_checks} checks: {max_diff:.2e}")
    if max_diff <= args.tolerance:
        print("PASS — conversion is numerically faithful.")
    else:
        print(f"WARNING — diff exceeds tolerance {args.tolerance:.0e}. "
              "Do not trust this .tflite; check the conversion.")


if __name__ == "__main__":
    main()
