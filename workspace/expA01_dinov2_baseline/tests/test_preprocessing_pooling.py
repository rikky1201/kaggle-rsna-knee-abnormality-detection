"""
preprocessing.py / pooling.py のユニットテスト（合成データ）。
実DICOMデータはKaggle Notebook側でのみ扱うため、ここでは形状・値域・境界条件のみ検証する。

Usage:
    python -m pytest tests/test_preprocessing_pooling.py -v
    または
    python tests/test_preprocessing_pooling.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pooling import (
    N_SLOTS,
    SLOT_KEYS,
    pool_series_to_study,
    pool_slices_to_series,
    slot_key_for_series,
    study_feature_dim,
)
from src.preprocessing import MAX_SLICES, percentile_clip_normalize, preprocess_series, subsample_slices


def test_percentile_clip_normalize_range():
    rng = np.random.default_rng(0)
    volume = rng.normal(1000, 200, size=(20, 64, 64)).astype(np.float32)
    out = percentile_clip_normalize(volume)
    assert out.shape == volume.shape
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6
    assert out.dtype == np.float32


def test_percentile_clip_normalize_rejects_constant_volume():
    volume = np.full((10, 32, 32), 500.0, dtype=np.float32)
    try:
        percentile_clip_normalize(volume)
        raise AssertionError("定数volumeでValueErrorが発生しなかった")
    except ValueError:
        pass


def test_subsample_slices_under_limit_unchanged():
    volume = np.zeros((30, 16, 16), dtype=np.float32)
    out = subsample_slices(volume, max_slices=MAX_SLICES)
    assert out.shape[0] == 30  # 60枚以下はそのまま


def test_subsample_slices_over_limit_reduces_to_max():
    volume = np.arange(100 * 8 * 8, dtype=np.float32).reshape(100, 8, 8)
    out = subsample_slices(volume, max_slices=MAX_SLICES)
    assert out.shape[0] == MAX_SLICES
    # 最初と最後のスライスが保持されていること（均等間隔で両端を含む）
    assert np.array_equal(out[0], volume[0])
    assert np.array_equal(out[-1], volume[-1])


def test_preprocess_series_pipeline():
    rng = np.random.default_rng(1)
    volume = rng.normal(1000, 200, size=(120, 32, 32)).astype(np.float32)
    out = preprocess_series(volume)
    assert out.shape[0] == MAX_SLICES  # 間引き適用
    assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6  # 正規化適用


def test_pool_slices_to_series_shape_and_values():
    feat_dim = 384
    n_slices = 45
    rng = np.random.default_rng(2)
    slice_features = rng.normal(size=(n_slices, feat_dim)).astype(np.float32)
    out = pool_slices_to_series(slice_features)
    assert out.shape == (2 * feat_dim,)
    np.testing.assert_allclose(out[:feat_dim], slice_features.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(out[feat_dim:], slice_features.max(axis=0), rtol=1e-5)


def test_pool_slices_to_series_rejects_empty():
    try:
        pool_slices_to_series(np.zeros((0, 10), dtype=np.float32))
        raise AssertionError("空配列でValueErrorが発生しなかった")
    except ValueError:
        pass


def test_slot_key_for_series_known_and_unknown():
    assert slot_key_for_series("Sagittal", 1) == "Sagittal_1"
    assert slot_key_for_series("Coronal", 0) == "Coronal_0"
    assert slot_key_for_series("Oblique", 1) == "other"  # 未知の平面は other


def test_slot_keys_count():
    assert N_SLOTS == 7
    assert len(SLOT_KEYS) == 7
    assert SLOT_KEYS[-1] == "other"


def test_pool_series_to_study_shape():
    feat_dim2 = 768  # 2*384
    rng = np.random.default_rng(3)

    series_list = [
        {"feature": rng.normal(size=feat_dim2).astype(np.float32), "anatomical_plane": "Sagittal",
         "fluid_sensitive": 1, "fat_suppression": 1},
        {"feature": rng.normal(size=feat_dim2).astype(np.float32), "anatomical_plane": "Coronal",
         "fluid_sensitive": 0, "fat_suppression": 0},
        {"feature": rng.normal(size=feat_dim2).astype(np.float32), "anatomical_plane": "Oblique",
         "fluid_sensitive": 1, "fat_suppression": 1},  # -> other スロット
    ]
    out = pool_series_to_study(series_list)
    expected_dim = study_feature_dim(feat_dim2)
    assert out.shape == (expected_dim,)
    assert expected_dim == 7 * (feat_dim2 + 1)


def test_pool_series_to_study_empty_slots_are_zero():
    feat_dim2 = 16
    rng = np.random.default_rng(4)
    series_list = [
        {"feature": rng.normal(size=feat_dim2).astype(np.float32), "anatomical_plane": "Sagittal",
         "fluid_sensitive": 1, "fat_suppression": 1},
    ]
    out = pool_series_to_study(series_list)
    # "Sagittal_1" は SLOT_KEYS[1] (Sagittal_0, Sagittal_1, Coronal_0, ...)
    slot_size = feat_dim2 + 1
    sagittal1_idx = SLOT_KEYS.index("Sagittal_1")
    coronal0_idx = SLOT_KEYS.index("Coronal_0")

    sagittal1_slot = out[sagittal1_idx * slot_size: (sagittal1_idx + 1) * slot_size]
    coronal0_slot = out[coronal0_idx * slot_size: (coronal0_idx + 1) * slot_size]

    assert not np.allclose(sagittal1_slot[:-1], 0)  # データがあるスロットは非ゼロ
    assert np.allclose(coronal0_slot, 0)  # データが無いスロットは特徴もfat_suppressionもゼロ


def test_pool_series_to_study_multi_series_same_slot_averaged():
    feat_dim2 = 4
    series_list = [
        {"feature": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), "anatomical_plane": "Axial",
         "fluid_sensitive": 0, "fat_suppression": 1},
        {"feature": np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32), "anatomical_plane": "Axial",
         "fluid_sensitive": 0, "fat_suppression": 0},
    ]
    out = pool_series_to_study(series_list)
    slot_size = feat_dim2 + 1
    axial0_idx = SLOT_KEYS.index("Axial_0")
    slot = out[axial0_idx * slot_size: (axial0_idx + 1) * slot_size]
    np.testing.assert_allclose(slot[:-1], [2.0, 2.0, 2.0, 2.0])  # (1+3)/2
    assert slot[-1] == 0.5  # fat_suppressionの平均 (1+0)/2


def test_pool_series_to_study_rejects_dim_mismatch():
    series_list = [
        {"feature": np.zeros(8, dtype=np.float32), "anatomical_plane": "Sagittal",
         "fluid_sensitive": 1, "fat_suppression": 0},
        {"feature": np.zeros(4, dtype=np.float32), "anatomical_plane": "Coronal",
         "fluid_sensitive": 0, "fat_suppression": 0},
    ]
    try:
        pool_series_to_study(series_list)
        raise AssertionError("次元不一致でValueErrorが発生しなかった")
    except ValueError:
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
