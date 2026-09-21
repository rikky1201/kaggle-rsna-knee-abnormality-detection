"""
src/attention_model.py（AttentionStudyModel / pool_series_to_study_torch）のユニットテスト。

Usage:
    python tests/test_attention_model.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention_model import AttentionStudyModel, pool_series_to_study_torch
from src.pooling import study_feature_dim

EMBED_DIM = 8
NUM_CLASSES = 12


def _synthetic_series_list(n_series=3, min_slices=2, max_slices=6, seed=0):
    torch.manual_seed(seed)
    planes = ["Sagittal", "Coronal", "Axial"]
    series_list = []
    for i in range(n_series):
        n_slices = torch.randint(min_slices, max_slices + 1, (1,)).item()
        series_list.append({
            "slice_features": torch.randn(n_slices, EMBED_DIM),
            "anatomical_plane": planes[i % len(planes)],
            "fluid_sensitive": i % 2,
            "fat_suppression": i % 2,
        })
    return series_list


def test_pool_series_to_study_torch_output_dim():
    device = torch.device("cpu")
    entries = [{"feature": torch.randn(EMBED_DIM), "anatomical_plane": "Sagittal",
                "fluid_sensitive": 0, "fat_suppression": 1}]
    out = pool_series_to_study_torch(entries, EMBED_DIM, device)
    assert out.shape == (study_feature_dim(EMBED_DIM),)


def test_pool_series_to_study_torch_empty_raises():
    try:
        pool_series_to_study_torch([], EMBED_DIM, torch.device("cpu"))
        raise AssertionError("ValueErrorが発生しませんでした")
    except ValueError:
        pass


def test_forward_study_output_shape():
    model = AttentionStudyModel(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES)
    series_list = _synthetic_series_list()
    logits = model.forward_study(series_list, torch.device("cpu"))
    assert logits.shape == (NUM_CLASSES,)


def test_forward_study_gradient_flows_end_to_end():
    """attention側(V,U,w)にもLinearProbeHead側にも勾配が流れることを確認する。"""
    model = AttentionStudyModel(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES)
    series_list = _synthetic_series_list(seed=1)
    logits = model.forward_study(series_list, torch.device("cpu"))
    loss = logits.sum()
    loss.backward()

    assert model.attn_pool.V.weight.grad is not None and model.attn_pool.V.weight.grad.abs().sum() > 0
    assert model.attn_pool.U.weight.grad is not None and model.attn_pool.U.weight.grad.abs().sum() > 0
    assert model.attn_pool.w.weight.grad is not None and model.attn_pool.w.weight.grad.abs().sum() > 0
    assert model.head.linear.weight.grad is not None and model.head.linear.weight.grad.abs().sum() > 0


def test_forward_study_varying_series_count():
    """study間でseries数・スライス数が異なっても（ragged）問題なく動くことを確認する。"""
    model = AttentionStudyModel(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES)
    for n_series, seed in [(1, 10), (3, 11), (7, 12)]:
        series_list = _synthetic_series_list(n_series=n_series, seed=seed)
        logits = model.forward_study(series_list, torch.device("cpu"))
        assert logits.shape == (NUM_CLASSES,)


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
