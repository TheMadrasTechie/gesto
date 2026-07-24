"""Dataset loading."""
import numpy as np
import pytest

from gesto.data import class_summary, load_sequence, load_static


def _write(project, mode, label, arrays):
    d = project / "data" / mode / label
    d.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(arrays):
        np.save(d / f"{i}.npy", arr.astype(np.float32))


def test_load_static(tmp_path):
    _write(tmp_path, "static", "A", [np.random.rand(63) for _ in range(4)])
    _write(tmp_path, "static", "B", [np.random.rand(63) for _ in range(3)])
    X, y, labels = load_static(tmp_path, "hands_one")
    assert X.shape == (7, 63)
    assert labels == ["A", "B"]
    assert class_summary(y, labels) == {"A": 4, "B": 3}


def test_load_static_rejects_wrong_dimension(tmp_path):
    _write(tmp_path, "static", "A", [np.random.rand(132)])
    with pytest.raises(ValueError, match="expects 63"):
        load_static(tmp_path, "hands_one")


def test_load_sequence_trims_and_skips(tmp_path):
    _write(tmp_path, "sequence", "A",
           [np.random.rand(40, 63), np.random.rand(30, 63),
            np.random.rand(12, 63)])          # last one is too short
    X, y, labels = load_sequence(tmp_path, "hands_one", seq_len=30)
    assert X.shape == (2, 30, 63)             # trimmed to 30, short one dropped


def test_missing_mode_points_at_the_other_one(tmp_path):
    _write(tmp_path, "sequence", "A", [np.random.rand(30, 63)])
    with pytest.raises(FileNotFoundError, match="sequence data instead"):
        load_static(tmp_path, "hands_one")
