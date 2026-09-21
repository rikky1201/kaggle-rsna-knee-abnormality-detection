"""
src/attention_pooling.py（Gated Attention pooling）のユニットテスト（合成データ）。

Usage:
    python tests/test_attention_pooling.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention_pooling import GatedAttentionPooling


def test_output_shape():
    torch.manual_seed(0)
    pool = GatedAttentionPooling(embed_dim=16, hidden_dim=8)
    h = torch.randn(5, 16)
    z, a = pool(h)
    assert z.shape == (16,)
    assert a.shape == (5,)


def test_attention_weights_sum_to_one():
    torch.manual_seed(1)
    pool = GatedAttentionPooling(embed_dim=8, hidden_dim=4)
    h = torch.randn(10, 8)
    _, a = pool(h)
    assert torch.allclose(a.sum(), torch.tensor(1.0), atol=1e-5)
    assert (a >= 0).all()


def test_single_slice_passthrough():
    """スライスが1枚だけの場合、attention重みは1.0でzはhそのものになるはず。"""
    torch.manual_seed(2)
    pool = GatedAttentionPooling(embed_dim=8, hidden_dim=4)
    h = torch.randn(1, 8)
    z, a = pool(h)
    assert torch.allclose(a, torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(z, h[0], atol=1e-5)


def test_gradient_flows_to_parameters():
    torch.manual_seed(3)
    pool = GatedAttentionPooling(embed_dim=8, hidden_dim=4)
    h = torch.randn(6, 8, requires_grad=True)
    z, _ = pool(h)
    loss = z.sum()
    loss.backward()
    assert pool.V.weight.grad is not None and pool.V.weight.grad.abs().sum() > 0
    assert pool.U.weight.grad is not None and pool.U.weight.grad.abs().sum() > 0
    assert pool.w.weight.grad is not None and pool.w.weight.grad.abs().sum() > 0
    assert h.grad is not None


def test_rejects_wrong_ndim():
    pool = GatedAttentionPooling(embed_dim=8)
    try:
        pool(torch.randn(8))
        raise AssertionError("ValueErrorが発生しませんでした")
    except ValueError:
        pass


def test_rejects_empty_slices():
    pool = GatedAttentionPooling(embed_dim=8)
    try:
        pool(torch.zeros(0, 8))
        raise AssertionError("ValueErrorが発生しませんでした")
    except ValueError:
        pass


def test_permutation_invariance():
    """スライスの順序を入れ替えてもzは（浮動小数点誤差の範囲で）変わらないはず。"""
    torch.manual_seed(4)
    pool = GatedAttentionPooling(embed_dim=8, hidden_dim=4)
    h = torch.randn(7, 8)
    perm = torch.randperm(7)
    z1, _ = pool(h)
    z2, _ = pool(h[perm])
    assert torch.allclose(z1, z2, atol=1e-5)


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
