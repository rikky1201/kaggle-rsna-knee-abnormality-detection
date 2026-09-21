"""
gold OOF予測（`compute_oof_auc.py` / `compute_oof_auc_radimagenet.py`が保存したnpz）を使い、
対応ありbootstrapで2モデル間の差が統計的に意味を持つかを判定する。

方針（ユーザー指示、2026-09-16）: 「今後の比較検証はすべてこの信頼区間を見てから
提出するかどうか判断する」。単純なbackbone比較に加え、実際に提出した
「Effusion・SynovitisのみRadImageNet、残りDINOv2」というルーティング版が、
DINOv2単体と比べてgold58件のOOF上でどれだけ有意な差だったかを事後検証する
（実LBスコアが全く変化しなかった理由の答え合わせ）。

Usage:
    python bootstrap_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
from src.metrics import bootstrap_paired_delta, summarize_ci  # noqa: E402

N_BOOTSTRAP = 1000
ROUTED_LABELS_FROM_B = {"Effusion", "Synovitis"}  # 実際の提出(v003)と同じ割り当て


def load_oof(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "study_uids": data["study_uids"],
        "y_true": data["y_true"],
        "y_prob": data["y_prob"],
        "labels": list(data["labels"]),
    }


def print_delta_table(summary: dict, labels: list[str], name_a: str, name_b: str) -> None:
    print(f"{'label':<20}{'mean delta':>12}{'95% CI low':>14}{'95% CI high':>14}{'有意?':>8}")
    for label in [*labels, "MACRO"]:
        s = summary[label]
        if np.isnan(s["mean"]):
            print(f"{label:<20}{'NaN':>12}")
            continue
        significant = "*" if (s["ci_low"] > 0 or s["ci_high"] < 0) else ""
        print(f"{label:<20}{s['mean']:>+12.4f}{s['ci_low']:>+14.4f}{s['ci_high']:>+14.4f}{significant:>8}")
    print(f"(delta = {name_b} - {name_a}。95%CIが0をまたがない場合のみ '*' で有意と表示)")


def compare(name_a: str, y_prob_a: np.ndarray, name_b: str, y_prob_b: np.ndarray, y_true: np.ndarray, labels: list[str]) -> None:
    print("=" * 90)
    print(f"[{name_a} vs {name_b}] 対応ありbootstrap（n_bootstrap={N_BOOTSTRAP}, seed=42, n=58）")
    print("=" * 90)
    delta = bootstrap_paired_delta(y_true, y_prob_a, y_prob_b, labels, n_bootstrap=N_BOOTSTRAP, seed=42)
    summary = summarize_ci(delta, labels)
    print_delta_table(summary, labels, name_a, name_b)
    print()


def main() -> None:
    dinov2 = load_oof(EXP_DIR / "results" / "expA01_linear_probe" / "oof_predictions.npz")
    radimagenet = load_oof(EXP_DIR / "results" / "expA01_linear_probe_radimagenet" / "oof_predictions.npz")

    if list(dinov2["study_uids"]) != list(radimagenet["study_uids"]):
        raise AssertionError("2モデルのOOF study_uidsの並びが一致しません（同じgold58件・同じ順序である前提）")
    if dinov2["labels"] != radimagenet["labels"]:
        raise AssertionError("2モデルのラベル順が一致しません")

    labels = dinov2["labels"]
    y_true = dinov2["y_true"]

    # 1) DINOv2 vs RadImageNet 単体同士の比較
    compare("DINOv2", dinov2["y_prob"], "RadImageNet", radimagenet["y_prob"], y_true, labels)

    # 2) 実際に提出したルーティング版（Effusion/SynovitisのみRadImageNet、残りDINOv2）との比較
    routed_prob = dinov2["y_prob"].copy()
    for i, label in enumerate(labels):
        if label in ROUTED_LABELS_FROM_B:
            routed_prob[:, i] = radimagenet["y_prob"][:, i]
    compare("DINOv2単体(v002)", dinov2["y_prob"], "ルーティング版(v003)", routed_prob, y_true, labels)

    print("[まとめ] '*' が付いたラベル/MACROのみ、gold58件のOOF上で統計的に有意な差があったと言える。")
    print("それ以外は95%信頼区間が0をまたいでおり、『差があるように見えても偶然の範囲』と解釈すべき。")


if __name__ == "__main__":
    main()
