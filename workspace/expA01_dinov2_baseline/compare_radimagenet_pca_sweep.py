"""
RadImageNet特徴（28,679次元）をPCAで削減した場合の、次元数ごとのgold-only OOF比較。

背景: `compare_backbones_gold_only.py`でDINOv2(10,759次元) vs RadImageNet(28,679次元)を
同じL2正則化強度(C=1.0)で比較したところRadImageNetが劣後したが、次元数が約2.7倍あるため
「バックボーンの実力差」と「次元数に対する正則化不足による過学習」が交絡している疑いがあった。
ユーザー要望はPCAで768次元（DINOv2のembed_dimと同じ）に削減して再比較することだったが、
gold 58件・fold内train ~41-45件という制約上、768成分の抽出は数学的に不可能
（PCA成分数の上限はtrain件数未満）。そのため実行可能な範囲でn_componentsをスイープし、
次元削減が結果にどう影響するかを見る（ユーザー承認済み: 2026-09-16）。

設計: PCAは各foldのtrain側のみでfit（CLAUDE.mdの前処理原則: testの情報を使わない）。
標準化→PCA→(PCA後の再標準化)→LogisticRegression(L2, C=1.0)という順序は
`compare_backbones_gold_only.py`と揃える。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
from src.model import FeatureStandardizer  # noqa: E402
from compare_backbones_gold_only import FOLDS_CSV, LABELS, load_aligned  # noqa: E402

import pandas as pd  # noqa: E402

RADIMAGENET_NPZ = EXP_DIR / "kaggle_radimagenet_gold_extract" / "output" / "gold58_features_radimagenet_resnet50.npz"
N_COMPONENTS_SWEEP = [5, 10, 20, 30, 40]
FOCUS_LABELS = ["Synovitis", "Effusion", "Medial OA", "Baker's"]

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_oof_with_pca(features: np.ndarray, labels: np.ndarray, folds: np.ndarray, n_components: int | None) -> dict[str, float]:
    n_folds = int(folds.max()) + 1
    oof_probs = np.full_like(labels, np.nan, dtype=np.float32)

    for fold in range(n_folds):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        scaler = FeatureStandardizer()
        scaler.fit(features[train_idx])
        train_feat = scaler.transform(features[train_idx])
        val_feat = scaler.transform(features[val_idx])

        if n_components is not None:
            max_allowed = min(len(train_idx), train_feat.shape[1]) - 1
            k = min(n_components, max_allowed)
            pca = PCA(n_components=k, svd_solver="full", whiten=True, random_state=42)
            train_feat = pca.fit_transform(train_feat)
            val_feat = pca.transform(val_feat)

        for j, label in enumerate(LABELS):
            y_train = labels[train_idx, j]
            if len(np.unique(y_train)) < 2:
                oof_probs[val_idx, j] = 0.5
                continue
            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=5000, class_weight="balanced")
            clf.fit(train_feat, y_train)
            oof_probs[val_idx, j] = clf.predict_proba(val_feat)[:, 1]

    label_aucs = {}
    for j, label in enumerate(LABELS):
        yt = labels[:, j]
        if len(np.unique(yt)) < 2:
            label_aucs[label] = float("nan")
            continue
        label_aucs[label] = float(roc_auc_score(yt, oof_probs[:, j]))
    return label_aucs


def main() -> None:
    gold_df = pd.read_csv(FOLDS_CSV)
    features, labels, folds, uids = load_aligned(RADIMAGENET_NPZ, gold_df)
    print(f"n={len(uids)}, raw feature_dim={features.shape[1]}")
    print(f"fold別train件数: {[int((folds != f).sum()) for f in range(int(folds.max()) + 1)]}")

    settings = [("no-PCA(28679)", None)] + [(f"PCA={k}", k) for k in N_COMPONENTS_SWEEP]

    results = {}
    for name, k in settings:
        results[name] = run_oof_with_pca(features, labels, folds, k)

    print()
    print("=" * 100)
    header = f"{'label':<20}" + "".join(f"{name:<16}" for name, _ in settings)
    print(header)
    print("=" * 100)
    for label in LABELS:
        marker = " *" if label in FOCUS_LABELS else "  "
        row = f"{label:<18}{marker}"
        for name, _ in settings:
            v = results[name][label]
            row += f"{v:<16.4f}" if not np.isnan(v) else f"{'NaN':<16}"
        print(row)

    print("-" * 100)
    macro_row = f"{'MACRO':<20}"
    for name, _ in settings:
        macro = float(np.nanmean(list(results[name].values())))
        macro_row += f"{macro:<16.4f}"
    print(macro_row)

    print()
    print("[DINOv2-base 参考値（compare_backbones_gold_only.py の再掲、no-PCA, 10759次元）]")
    print("  MACRO=0.5907 / Synovitis=0.5603 / Effusion=0.6658 / Medial OA=0.8031 / Baker's=0.5580")


if __name__ == "__main__":
    main()
