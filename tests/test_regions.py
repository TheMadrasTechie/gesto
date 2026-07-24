"""Extraction, normalization and dimensions."""
from types import SimpleNamespace

import numpy as np
import pytest

from gesto.regions import (REGION_INFO, REGION_KEYS, extract, feature_dim,
                           normalize)


def _lm(x, y, z, v=0.9):
    return SimpleNamespace(x=x, y=y, z=z, visibility=v)


def _hand(seed=0):
    rng = np.random.RandomState(seed)
    return SimpleNamespace(landmark=[_lm(*rng.rand(3)) for _ in range(21)])


def _pose(seed=0):
    rng = np.random.RandomState(seed)
    return SimpleNamespace(landmark=[_lm(*rng.rand(3), rng.rand())
                                     for _ in range(33)])


def _result(pose=None, left=None, right=None):
    return SimpleNamespace(pose_landmarks=pose, left_hand_landmarks=left,
                           right_hand_landmarks=right)


@pytest.mark.parametrize("region", REGION_KEYS)
def test_extract_matches_declared_dimension(region):
    res = _result(pose=_pose(1), left=_hand(2), right=_hand(3))
    vec = extract(res, region)
    assert vec is not None
    assert vec.shape == (feature_dim(region),)


@pytest.mark.parametrize("region", REGION_KEYS)
def test_normalize_preserves_shape(region):
    vec = np.random.rand(feature_dim(region)).astype(np.float32)
    assert normalize(vec, region).shape == vec.shape


@pytest.mark.parametrize("region", REGION_KEYS)
def test_extract_returns_none_when_nothing_detected(region):
    assert extract(_result(), region) is None


def test_one_hand_prefers_right_but_falls_back_to_left():
    right_only = extract(_result(right=_hand(5)), "hands_one")
    left_only = extract(_result(left=_hand(5)), "hands_one")
    # same landmarks either way: one-hand mode uses whichever hand is present
    assert np.allclose(right_only, left_only)


def test_normalization_is_translation_invariant():
    # shifting a hand should not change its normalized representation
    rng = np.random.RandomState(0)
    pts = rng.rand(21, 3).astype(np.float32)
    shifted = pts + np.array([0.2, -0.1, 0.05], np.float32)
    a = normalize(pts.reshape(-1), "hands_one")
    b = normalize(shifted.reshape(-1), "hands_one")
    assert np.allclose(a, b, atol=1e-5)


def test_normalization_is_scale_invariant():
    rng = np.random.RandomState(1)
    pts = rng.rand(21, 3).astype(np.float32)
    a = normalize(pts.reshape(-1), "hands_one")
    b = normalize((pts * 2.5).reshape(-1), "hands_one")
    assert np.allclose(a, b, atol=1e-5)


def test_unknown_region_rejected():
    with pytest.raises(ValueError, match="Unknown region"):
        feature_dim("elbows")
