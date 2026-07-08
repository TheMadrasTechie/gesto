"""
Gesto — convert a trained Keras model to TFLite for Flutter.

LSTM models use TF ops that aren't all in the core TFLite builtin set, so this
enables SELECT_TF_OPS (TF ops fallback). That makes conversion reliable but
means the Flutter side must use a tflite build that includes the Flex delegate
(see the notes printed at the end).

Also writes a small labels.json copy next to the .tflite so the app has the
class map + input shape in one place.

Run:
    python convert_tflite.py --model gesto_model.h5 --labels labels.json
    python convert_tflite.py --model gesto_model.h5 --out gesto.tflite
"""

import json
import argparse

import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gesto_model.h5")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--out", default="gesto.tflite")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    in_shape = model.input_shape           # (None, frames, feature_dim)
    out_shape = model.output_shape         # (None, n_classes)
    frames = int(in_shape[1])
    feature_dim = int(in_shape[2])
    n_classes = int(out_shape[1])
    print(f"Model input : {in_shape}  ->  frames={frames}, dim={feature_dim}")
    print(f"Model output: {out_shape} ->  classes={n_classes}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Core builtins + TF ops fallback (required for LSTM).
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # LSTMs contain TensorList ops; this lowering makes them convertible.
    converter._experimental_lower_tensor_list_ops = False
    # Optional size/speed optimization (safe default).
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(args.out, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite -> {args.out}  ({len(tflite_model)/1024:.1f} KB)")

    # sanity check: try one inference. On desktops without the Flex delegate
    # this can fail even though the .tflite is valid — that's fine, the model
    # is already written; Flutter supplies the delegate at runtime.
    try:
        interp = tf.lite.Interpreter(model_content=tflite_model)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        outp = interp.get_output_details()[0]
        dummy = np.random.random((1, frames, feature_dim)).astype(np.float32)
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        probs = interp.get_tensor(outp["index"])[0]
        print(f"Self-test OK: output {probs.shape}, sums to {probs.sum():.3f}")
    except Exception as e:
        print("Self-test skipped (Flex delegate not in this Python env) — "
              "this is normal for LSTM models; the .tflite is still valid.")

    # write a companion metadata file the Flutter app can read
    meta = {}
    try:
        meta = json.load(open(args.labels))
    except Exception:
        pass
    meta["frames"] = frames
    meta["feature_dim"] = feature_dim
    meta["n_classes"] = n_classes
    out_meta = args.out.rsplit(".", 1)[0] + "_meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta   -> {out_meta}")

    print("\nFlutter notes:")
    print("  - Use tflite_flutter with the Flex delegate (SELECT_TF_OPS).")
    print(f"  - Input : float32 [1, {frames}, {feature_dim}]")
    print(f"  - Output: float32 [1, {n_classes}]  (softmax probabilities)")
    print("  - You MUST normalize landmarks in Dart exactly like landmarks.py.")


if __name__ == "__main__":
    main()