"""
Gesto — train (variable length).

Loads variable-length samples from gesture_data/, resamples each to a fixed
length (TARGET_LEN frames), and trains an LSTM. Because every sample is
resampled to the same length, the model is speed-invariant: slow and fast
versions of a gesture look the same.

Saves:
    gesto_model.h5
    labels.json     (index -> class name / number)

Run:
    python train.py
"""

import os
import json

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from gesto_common import resample_sequence, TARGET_LEN, FEATURE_DIM

DATA_PATH = "gesture_data"
MODEL_PATH = "gesto_model.h5"
LABELS_PATH = "labels.json"
EPOCHS = 300
BATCH_SIZE = 16


def load_data():
    """Load every <class>/<n>.npy (variable length) and resample to TARGET_LEN."""
    classes = sorted(
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    )
    if not classes:
        raise RuntimeError(f"No class folders found in '{DATA_PATH}'.")

    label_map = {name: idx for idx, name in enumerate(classes)}
    X, y = [], []

    for name in classes:
        class_dir = os.path.join(DATA_PATH, name)
        files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
        for f in files:
            arr = np.load(os.path.join(class_dir, f))
            if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
                print(f"  skipping {f} in '{name}': shape {arr.shape}")
                continue
            X.append(resample_sequence(arr, TARGET_LEN))   # -> (TARGET_LEN, 63)
            y.append(label_map[name])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"Loaded {len(X)} samples across {len(classes)} classes: {classes}")
    print(f"X shape: {X.shape}")
    return X, y, label_map


def build_model(n_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, activation="tanh",
             input_shape=(TARGET_LEN, FEATURE_DIM)),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation="tanh"),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation="tanh"),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    return model


def main():
    X, y, label_map = load_data()
    n_classes = len(label_map)
    if n_classes < 2:
        raise RuntimeError("Need at least 2 classes to train.")

    y_cat = to_categorical(y, num_classes=n_classes).astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}   Validation: {X_val.shape[0]}")

    model = build_model(n_classes)
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_categorical_accuracy", patience=40,
        restore_best_weights=True, mode="max",
    )
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop])

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {acc:.3f}")

    model.save(MODEL_PATH)
    index_to_name = {idx: name for name, idx in label_map.items()}
    with open(LABELS_PATH, "w") as f:
        json.dump(index_to_name, f, indent=2)
    print(f"Saved model  -> {MODEL_PATH}")
    print(f"Saved labels -> {LABELS_PATH}")


if __name__ == "__main__":
    main()
