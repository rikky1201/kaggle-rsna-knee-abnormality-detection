"""
src/metrics.py（bootstrap信頼区間ユーティリティ）のユニットテスト（合成データ）。

Usage:
    python tests/test_metrics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import bootstrap_label_aucs, bootstrap_paired_delta, compute_label_aucs, summarize_ci

LABELS = ["A", "B"]


def _synthetic_data(n=58, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(n, len(LABELS))).astype(np.float32)
    y_prob = rng.random((n, len(LABELS))).astype(np.float32)
    return y_true, y_prob


def test_bootstrap_shape():
    y_true, y_prob = _synthetic_data()
    boot = bootstrap_label_aucs(y_true, y_prob, LABELS, n_bootstrap=100, seed=1)
    assert boot.shape == (100, len(LABELS))


def test_bootstrap_reproducible_with_same_seed():
    y_true, y_prob = _synthetic_data()
    boot1 = bootstrap_label_aucs(y_true, y_prob, LABELS, n_bootstrap=50, seed=42)
    boot2 = bootstrap_label_aucs(y_true, y_prob, LABELS, n_bootstrap=50, seed=42)
    assert np.array_equal(boot1, boot2, equal_nan=True)


def test_summarize_ci_contains_macro_and_is_ordered():
    y_true, y_prob = _synthetic_data()
    boot = bootstrap_label_aucs(y_true, y_prob, LABELS, n_bootstrap=500, seed=2)
    summary = summarize_ci(boot, LABELS)
    assert "MACRO" in summary
    for label in [*LABELS, "MACRO"]:
        s = summary[label]
        assert s["ci_low"] <= s["mean"] <= s["ci_high"]


def test_summarize_ci_matches_point_estimate_roughly():
    y_true, y_prob = _synthetic_data(n=200, seed=3)  # 標本数を増やしてbootstrap平均を安定させる
    point = compute_label_aucs(y_true, y_prob, LABELS)
    boot = bootstrap_label_aucs(y_true, y_prob, LABELS, n_bootstrap=1000, seed=3)
    summary = summarize_ci(boot, LABELS)
    for label in LABELS:
        assert abs(summary[label]["mean"] - point[label]) < 0.05


def test_paired_delta_is_zero_when_models_identical():
    y_true, y_prob = _synthetic_data()
    delta = bootstrap_paired_delta(y_true, y_prob, y_prob, LABELS, n_bootstrap=50, seed=4)
    valid = delta[~np.isnan(delta)]
    assert np.allclose(valid, 0.0)


def test_degenerate_label_produces_nan():
    y_true, y_prob = _synthetic_data()
    y_true[:, 0] = 1  # ラベルAを全陽性にして単一クラス化
    aucs = compute_label_aucs(y_true, y_prob, LABELS)
    assert np.isnan(aucs["A"])
    boot = bootstrap_label_aucs(y_true, y_prob, LABELS, n_bootstrap=50, seed=5)
    assert np.isnan(boot[:, 0]).all()


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
