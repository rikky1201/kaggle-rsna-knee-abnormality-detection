"""
gold OOF評価の指標計算 + bootstrap信頼区間。

方針（ユーザー指示、2026-09-16）: 「今後の比較検証はすべてこの信頼区間を見てから
提出するかどうか判断する」。RadImageNetルーティング検証でgold58件のOOF改善
（Effusion +0.111, Synovitis +0.125）が非公開test setには再現されなかったことを受け、
n=58というサンプルサイズでの評価がどれだけ不確実性を持つかを可視化する。

リサンプリング単位は「study」（12ラベルの相関構造を保つため、ラベルごとに独立に
リサンプリングしない）。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_label_aucs(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> dict[str, float]:
    """ラベルごとのAUCを計算する。片方のクラスしか無いラベルはNaN。"""
    aucs = {}
    for i, label in enumerate(labels):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            aucs[label] = float("nan")
            continue
        aucs[label] = float(roc_auc_score(yt, y_prob[:, i]))
    return aucs


def bootstrap_label_aucs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """study単位で重複ありリサンプリングし、リサンプルごとにラベル別AUCを計算する。

    Args:
        y_true: (n_studies, n_labels) の真値（0/1）
        y_prob: (n_studies, n_labels) の予測確率
        labels: ラベル名リスト（列順と対応）
        n_bootstrap: リサンプリング回数
        seed: 乱数シード（再現性のため固定）

    Returns:
        (n_bootstrap, n_labels) の配列。該当リサンプルでラベルが単一クラスのみの場合はNaN。
    """
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    result = np.full((n_bootstrap, len(labels)), np.nan, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt_b, yp_b = y_true[idx], y_prob[idx]
        for i in range(len(labels)):
            yt_col = yt_b[:, i]
            if len(np.unique(yt_col)) < 2:
                continue
            result[b, i] = roc_auc_score(yt_col, yp_b[:, i])
    return result


def summarize_ci(bootstrap_aucs: np.ndarray, labels: list[str], ci: float = 0.95) -> dict:
    """bootstrap_label_aucs()の出力から、ラベル別 + MACRO の点推定・信頼区間をまとめる。

    MACROは各リサンプルでラベル別AUCをnanmeanしたもの（単一クラスで欠損したラベルは
    そのリサンプルのmacro計算から除外。train.py本体のcompute_label_aucs+nanmeanと同じ規約）。
    """
    alpha = (1 - ci) / 2
    summary = {}
    for i, label in enumerate(labels):
        col = bootstrap_aucs[:, i]
        col_valid = col[~np.isnan(col)]
        if len(col_valid) == 0:
            summary[label] = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_valid": 0}
            continue
        summary[label] = {
            "mean": float(np.mean(col_valid)),
            "ci_low": float(np.percentile(col_valid, 100 * alpha)),
            "ci_high": float(np.percentile(col_valid, 100 * (1 - alpha))),
            "n_valid": int(len(col_valid)),
        }

    macro_per_bootstrap = np.nanmean(bootstrap_aucs, axis=1)
    summary["MACRO"] = {
        "mean": float(np.mean(macro_per_bootstrap)),
        "ci_low": float(np.percentile(macro_per_bootstrap, 100 * alpha)),
        "ci_high": float(np.percentile(macro_per_bootstrap, 100 * (1 - alpha))),
        "n_valid": int(len(macro_per_bootstrap)),
    }
    return summary


def bootstrap_paired_delta(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    labels: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """モデルA→Bの差分（B-A）を、同一リサンプルインデックスを両者に使う対応ありbootstrapで計算する。

    2モデルの比較（例: DINOv2単体 vs ルーティング版）で「差が統計的に意味を持つか」を見るために使う。
    通常の2つの独立したCIを別々に見て重なり判定するより、対応ありの方が検出力が高く正しい。

    Returns:
        (n_bootstrap, n_labels) の配列（B側AUC - A側AUC）。単一クラスで計算不可の場合はNaN。
    """
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    delta = np.full((n_bootstrap, len(labels)), np.nan, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt_col_all = y_true[idx]
        for i in range(len(labels)):
            yt_col = yt_col_all[:, i]
            if len(np.unique(yt_col)) < 2:
                continue
            auc_a = roc_auc_score(yt_col, y_prob_a[idx, i])
            auc_b = roc_auc_score(yt_col, y_prob_b[idx, i])
            delta[b, i] = auc_b - auc_a
    return delta


def print_ci_table(summary: dict, labels: list[str], title: str = "") -> None:
    if title:
        print(title)
    print(f"{'label':<20}{'mean':>10}{'95% CI low':>14}{'95% CI high':>14}")
    for label in [*labels, "MACRO"]:
        s = summary[label]
        if np.isnan(s["mean"]):
            print(f"{label:<20}{'NaN':>10}")
            continue
        print(f"{label:<20}{s['mean']:>10.4f}{s['ci_low']:>14.4f}{s['ci_high']:>14.4f}")
