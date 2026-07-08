"""
Gesto — convert a trained Keras model to TFLite for Flutter.

IMPORTANT CHANGE from the SELECT_TF_OPS approach: LSTM layers by default
convert to a dynamic WHILE + FlexTensorListReserve/Stack graph, which only
runs via the TF Select ops (Flex) delegate. tflite_flutter does not bundle
that delegate, so a model converted with SELECT_TF_OPS will fail to load
on-device in Flutter with "Select TensorFlow op(s) not supported."

Fix: since `frames` is fixed (e.g. 30), we rebuild the identical
architecture with unroll=True on every LSTM layer. Keras then statically
unrolls the recurrence at graph-build time instead of using a dynamic
While loop -- same math, same weights (loaded from the existing .h5, no
retraining), but the resulting .tflite only needs core TFLite builtin ops.
No Flex delegate required anywhere in Flutter.

Also writes a small labels.json copy next to the .tflite so the app has the
class map + input shape in one place.

Run:
    python convert_tflite.py --model gesto_model.h5 --labels labels.json
    python convert_tflite.py --model gesto_model.h5 --labels labels.json --out gesto.tflite
"""

import json
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking


def build_unrolled_model(frames, feature_dim, n_classes):
    """Same architecture as gesto_labeller_train.py's build_model, but with
    unroll=True on every LSTM layer so conversion avoids Flex ops."""
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(frames, feature_dim)),
        LSTM(64, return_sequences=True, activation="tanh", unroll=True),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation="tanh", unroll=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation="tanh", unroll=True),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gesto_model.h5")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--out", default="gesto.tflite")
    args = ap.parse_args()

    meta = json.load(open(args.labels))
    frames = int(meta["frames"])
    feature_dim = int(meta["feature_dim"])
    n_classes = len(meta["labels"])
    print(f"From labels.json: frames={frames}, dim={feature_dim}, classes={n_classes}")

    old_model = load_model(args.model)
    in_shape = old_model.input_shape
    out_shape = old_model.output_shape
    print(f"Original model input : {in_shape}")
    print(f"Original model output: {out_shape}")
    if int(in_shape[1]) != frames or int(in_shape[2]) != feature_dim:
        raise RuntimeError(
            f"labels.json (frames={frames}, dim={feature_dim}) doesn't match "
            f"the model's actual input shape {in_shape} — check you passed "
            f"the right labels.json for this model."
        )

    # Rebuild with unroll=True and copy the trained weights over.
    model = build_unrolled_model(frames, feature_dim, n_classes)
    model.set_weights(old_model.get_weights())

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Deliberately NOT enabling SELECT_TF_OPS here -- unroll=True keeps this
    # to core TFLITE_BUILTINS only, which is what makes it work in Flutter
    # without a Flex delegate.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(args.out, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite -> {args.out}  ({len(tflite_model)/1024:.1f} KB)")

    # Self-test: this should now succeed in a plain (non-Flex) interpreter,
    # proving the Flutter side will be able to load it too.
    try:
        interp = tf.lite.Interpreter(model_content=tflite_model)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        outp = interp.get_output_details()[0]
        dummy = np.random.random((1, frames, feature_dim)).astype(np.float32)
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        probs = interp.get_tensor(outp["index"])[0]
        print(f"Self-test OK (no Flex delegate needed): output {probs.shape}, "
              f"sums to {probs.sum():.3f}")
    except Exception as e:
        print(f"Self-test FAILED: {e}")
        print("If this still mentions Flex/Select ops, something in the "
              "architecture above doesn't match your original model closely "
              "enough -- check gesto_labeller_train.py's build_model for any "
              "differences (extra layers, different units, etc).")

    # write a companion metadata file the Flutter app can read
    out_meta = args.out.rsplit(".", 1)[0] + "_meta.json"
    meta["frames"] = frames
    meta["feature_dim"] = feature_dim
    meta["n_classes"] = n_classes
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta   -> {out_meta}")

    print("\nFlutter notes:")
    print("  - Plain tflite_flutter Interpreter.fromAsset() works -- no Flex delegate needed.")
    print(f"  - Input : float32 [1, {frames}, {feature_dim}]")
    print(f"  - Output: float32 [1, {n_classes}]  (softmax probabilities)")
    print("  - You MUST normalize landmarks in Dart exactly like landmarks.py (wrist-relative, scale-normalised).")


if __name__ == "__main__":
    main()