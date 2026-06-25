"""
Gesto — convert a Keras .h5 gesture model to TFLite for mobile.

Older .h5 files (legacy Keras) don't deserialize in Keras 3. This rebuilds the
exact architecture in code and loads the weights, which is robust across TF
versions, then converts to TFLite (with Flex ops, required for LSTM).

Usage:
    python convert_to_tflite.py model.h5 gesto.tflite [num_frames] [num_classes]

num_frames/num_classes default to 30 and are auto-read from the .h5 if possible.
"""

import sys
import json
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


def read_shape_from_h5(path):
    """Pull (num_frames, num_features, num_classes) from the stored config."""
    with h5py.File(path, "r") as f:
        cfg = f.attrs.get("model_config")
    if isinstance(cfg, bytes):
        cfg = cfg.decode("utf-8")
    cfg = json.loads(cfg)
    layers = cfg["config"]["layers"]
    frames, feats, classes = 30, 63, None
    for l in layers:
        bis = l["config"].get("batch_input_shape")
        if bis and len(bis) == 3:
            frames, feats = bis[1], bis[2]
        if l["class_name"] == "Dense":
            classes = l["config"]["units"]  # last Dense wins = output classes
    return frames, feats, classes


def build_model(num_frames, num_features, num_classes):
    """Exact Gesto/alphabet architecture: LSTM 64-128-64 + Dense 64-32-classes."""
    return Sequential([
        Input(shape=(num_frames, num_features)),
        LSTM(64, return_sequences=True, activation="relu"),
        LSTM(128, return_sequences=True, activation="relu"),
        LSTM(64, return_sequences=False, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "gesto_model.h5"
    dst = sys.argv[2] if len(sys.argv) > 2 else "gesto.tflite"

    frames, feats, classes = read_shape_from_h5(src)
    if len(sys.argv) > 3:
        frames = int(sys.argv[3])
    if len(sys.argv) > 4:
        classes = int(sys.argv[4])
    print(f"Architecture: frames={frames} features={feats} classes={classes}")

    model = build_model(frames, feats, classes)
    model.load_weights(src)
    print("Weights loaded into rebuilt model.")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    with open(dst, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote {dst}  ({len(tflite_model)/1024:.1f} KB)")

    # verify TFLite == Keras on the same input
    x = np.random.random((1, frames, feats)).astype(np.float32)
    keras_out = model.predict(x, verbose=0)[0]
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    tfl_out = interp.get_tensor(out["index"])[0]
    diff = float(np.max(np.abs(keras_out - tfl_out)))
    print(f"Keras vs TFLite max diff: {diff:.2e} "
          f"({'MATCH' if diff < 1e-4 else 'CHECK'})")
    print(f"argmax  keras={keras_out.argmax()}  tflite={tfl_out.argmax()}")


if __name__ == "__main__":
    main()