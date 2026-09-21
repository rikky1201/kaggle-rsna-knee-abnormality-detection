"""
src/attention_dataset.py（SliceFeatureStore）のユニットテスト。
`kaggle_dinov2_extract_slices_benchmark`が生成するnpz形式を合成データで再現して検証する。

Usage:
    python tests/test_attention_dataset.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention_dataset import SliceFeatureStore

EMBED_DIM = 8


def _build_synthetic_npz(path: Path, seed: int = 0) -> dict:
    """study A: series 2件(3,5スライス), study B: series 1件(4スライス) を持つ合成npzを作る。"""
    rng = np.random.default_rng(seed)

    slice_counts = [3, 5, 4]  # series0(studyA), series1(studyA), series2(studyB)
    slice_features = np.concatenate([rng.random((n, EMBED_DIM)).astype(np.float32) for n in slice_counts], axis=0)
    slice_to_series = np.concatenate([np.full(n, i, dtype=np.int32) for i, n in enumerate(slice_counts)])

    series_study_uid = np.array(["studyA", "studyA", "studyB"])
    series_instance_uid = np.array(["s0", "s1", "s2"])
    series_anatomical_plane = np.array(["Sagittal", "Coronal", "Axial"])
    series_fluid_sensitive = np.array([0, 1, 0], dtype=np.int32)
    series_fat_suppression = np.array([1, 0, 1], dtype=np.int32)

    np.savez(
        path,
        slice_features=slice_features,
        slice_to_series=slice_to_series,
        series_study_uid=series_study_uid,
        series_instance_uid=series_instance_uid,
        series_anatomical_plane=series_anatomical_plane,
        series_fluid_sensitive=series_fluid_sensitive,
        series_fat_suppression=series_fat_suppression,
    )
    return {"slice_counts": slice_counts, "slice_features": slice_features}


def test_load_and_query_study():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "synthetic.npz"
        built = _build_synthetic_npz(npz_path)

        store = SliceFeatureStore(npz_path)
        assert store.embed_dim == EMBED_DIM
        assert store.has_study("studyA")
        assert store.has_study("studyB")
        assert not store.has_study("studyC")

        series_a = store.get_study_series("studyA")
        assert len(series_a) == 2
        assert series_a[0]["slice_features"].shape == (3, EMBED_DIM)
        assert series_a[1]["slice_features"].shape == (5, EMBED_DIM)
        assert series_a[0]["anatomical_plane"] == "Sagittal"
        assert series_a[1]["fluid_sensitive"] == 1
        assert series_a[1]["fat_suppression"] == 0

        series_b = store.get_study_series("studyB")
        assert len(series_b) == 1
        assert series_b[0]["slice_features"].shape == (4, EMBED_DIM)

        # 実際の値も一致しているか（offset計算が正しいか）確認
        import numpy as _np
        _np.testing.assert_allclose(series_a[0]["slice_features"].numpy(), built["slice_features"][0:3])
        _np.testing.assert_allclose(series_a[1]["slice_features"].numpy(), built["slice_features"][3:8])
        _np.testing.assert_allclose(series_b[0]["slice_features"].numpy(), built["slice_features"][8:12])


def test_missing_study_raises_keyerror():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "synthetic.npz"
        _build_synthetic_npz(npz_path)
        store = SliceFeatureStore(npz_path)
        try:
            store.get_study_series("nonexistent")
            raise AssertionError("KeyErrorが発生しませんでした")
        except KeyError:
            pass


def test_non_contiguous_slice_to_series_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "broken.npz"
        rng = np.random.default_rng(0)
        slice_features = rng.random((4, EMBED_DIM)).astype(np.float32)
        # series0とseries1のスライスが交互に混ざっている（連続区間の前提が崩れている）壊れたデータ
        slice_to_series = np.array([0, 1, 0, 1], dtype=np.int32)
        np.savez(
            npz_path,
            slice_features=slice_features,
            slice_to_series=slice_to_series,
            series_study_uid=np.array(["A", "B"]),
            series_instance_uid=np.array(["s0", "s1"]),
            series_anatomical_plane=np.array(["Sagittal", "Coronal"]),
            series_fluid_sensitive=np.array([0, 0], dtype=np.int32),
            series_fat_suppression=np.array([0, 0], dtype=np.int32),
        )
        try:
            SliceFeatureStore(npz_path)
            raise AssertionError("AssertionErrorが発生しませんでした")
        except AssertionError:
            pass


def _run_all():
    import inspect
    import sys as _sys

    module = _sys.modules[__name__]
    tests = [f for name, f in inspect.getmembers(module, inspect.isfunction) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
