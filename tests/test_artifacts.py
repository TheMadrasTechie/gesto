"""Artifact layout and versioning."""
import pytest
from gesto import artifacts as A


def test_first_run_uses_plain_region_name(tmp_path):
    run = A.new_run(tmp_path / "artifacts", "static", "pose")
    assert run.name == "pose"
    assert run.parent.name == "static"


def test_runs_version_instead_of_overwriting(tmp_path):
    root = tmp_path / "artifacts"
    names = [A.new_run(root, "static", "pose").name for _ in range(3)]
    assert names == ["pose", "pose_2", "pose_3"]


def test_modes_and_regions_are_independent(tmp_path):
    root = tmp_path / "artifacts"
    A.new_run(root, "static", "pose")
    A.new_run(root, "sequence", "pose")
    A.new_run(root, "static", "legs")
    assert [p.name for p in A.list_runs(root, "static", "pose")] == ["pose"]
    assert [p.name for p in A.list_runs(root, "sequence", "pose")] == ["pose"]
    assert [p.name for p in A.list_runs(root, "static", "legs")] == ["legs"]


def test_resolve_defaults_to_newest(tmp_path):
    root = tmp_path / "artifacts"
    for _ in range(3):
        A.new_run(root, "static", "pose")
    assert A.resolve(root, "static", "pose").name == "pose_3"
    assert A.resolve(root, "static", "pose", 1).name == "pose"
    assert A.resolve(root, "static", "pose", 2).name == "pose_2"


def test_resolve_missing_model_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train one first"):
        A.resolve(tmp_path / "artifacts", "static", "pose")


def test_region_prefix_is_not_confused_with_version(tmp_path):
    root = tmp_path / "artifacts"
    A.new_run(root, "static", "pose")
    A.new_run(root, "static", "pose_alt")
    assert [p.name for p in A.list_runs(root, "static", "pose")] == ["pose"]


def test_bad_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        A.new_run(tmp_path, "nonsense", "pose")
