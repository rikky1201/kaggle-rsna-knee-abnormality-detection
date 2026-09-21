"""
gold 58件のみ（疑似ラベル不使用）での DINOv2 vs RadImageNet ResNet50 バックボーン比較。

意思決定用の小規模検証（ユーザー指示、2026-09-16）。全4,407件への本格適用はこの結果を見てから
別途指示される。pseudo-labelには一切触れない。

設計判断（train.py本体とは異なる点、意図的）:
- 学習データはgold 58件のみ（各fold train ~41-45件、val 13-17件）と極端に少数かつ
  高次元（DINOv2:10759次元, RadImageNet:28679次元）のp>>n設定のため、train.py本体の
  AdamW+多epoch学習ヘッド（正則化なしの素朴な線形層）をそのまま使うと大きく過学習しうる。
  代わりに標準的なlinear-probe評価でよく使われる `LogisticRegression(penalty="l2")`
  （scikit-learn既定のL2正則化）をラベルごとに独立に学習する。epoch数・early stopping等の
  追加ハイパーパラメータ選択（それ自体がバイアス源になりうる、前回発見したfold平均の楽観バイアスと同種の問題）
  を避けるための意図的な選択
- 標準化・fold分割はtrain.py本体と同じ設計（trainのみの統計量でfit、pooled OOF評価）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
from src.model import FeatureStandardizer  # noqa: E402

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

FOLDS_CSV = EXP_DIR.parent / "fold" / "v1_multilabel_gold58" / "folds.csv"
BACKBONES = {
    "DINOv2-base": EXP_DIR / "kaggle_radimagenet_gold_extract" / "output" / "gold58_features_dinov2_base.npz",
    "RadImageNet-ResNet50": EXP_DIR / "kaggle_radimagenet_gold_extract" / "output" / "gold58_features_radimagenet_resnet50.npz",
}


def load_aligned(features_npz: Path, gold_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """features_npz を gold_df の順序に合わせて並べ替える。抽出失敗で欠けたstudyがあれば除外して警告する。"""
    data = np.load(features_npz)
    feat_uids = list(data["study_uids"])
    feat_map = {uid: i for i, uid in enumerate(feat_uids)}

    missing = [u for u in gold_df["StudyInstanceUID"] if u not in feat_map]
    if missing:
        print(f"  [警告] {features_npz.name}: 抽出できなかったstudyを除外します ({len(missing)}件): {missing}")

    keep_mask = gold_df["StudyInstanceUID"].isin(feat_map)
    kept_df = gold_df[keep_mask].reset_index(drop=True)
    idx = [feat_map[u] for u in kept_df["StudyInstanceUID"]]
    features = data["features"][idx]
    labels = kept_df[LABELS].to_numpy(dtype=np.float32)
    folds = kept_df["fold"].to_numpy()
    return features, labels, folds, kept_df["StudyInstanceUID"].to_numpy()


def run_gold_only_oof(features: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> dict[str, float]:
    n_folds = int(folds.max()) + 1
    oof_probs = np.full_like(labels, np.nan, dtype=np.float32)

    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for fold in range(n_folds):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        scaler = FeatureStandardizer()
        scaler.fit(features[train_idx])
        train_feat = scaler.transform(features[train_idx])
        val_feat = scaler.transform(features[val_idx])

        for j, label in enumerate(LABELS):
            y_train = labels[train_idx, j]
            if len(np.unique(y_train)) < 2:
                print(f"  [警告] fold{fold} label={label}: train側が単一クラスのためこのfold・labelはフォールバック(0.5)")
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
    if len(gold_df) != 58:
        raise AssertionError(f"gold件数が58と一致しません: 実際={len(gold_df)}")

    results = {}
    for backbone_name, npz_path in BACKBONES.items():
        if not npz_path.exists():
            raise FileNotFoundError(f"{backbone_name} の特徴ファイルがありません: {npz_path}")
        print(f"=== {backbone_name} ===")
        features, labels, folds, uids = load_aligned(npz_path, gold_df)
        print(f"  n={len(uids)}, feature_dim={features.shape[1]}")
        label_aucs = run_gold_only_oof(features, labels, folds)
        results[backbone_name] = label_aucs

    # ── 比較表
    print()
    print("=" * 90)
    print(f"{'label':<20}" + "".join(f"{name:<24}" for name in results) + "delta(RadImageNet-DINOv2)")
    print("=" * 90)
    deltas = {}
    for label in LABELS:
        row = f"{label:<20}"
        vals = []
        for name in results:
            v = results[name][label]
            vals.append(v)
            row += f"{v:<24.4f}" if not np.isnan(v) else f"{'NaN':<24}"
        delta = vals[1] - vals[0] if not (np.isnan(vals[0]) or np.isnan(vals[1])) else float("nan")
        deltas[label] = delta
        row += f"{delta:+.4f}" if not np.isnan(delta) else "NaN"
        print(row)

    macro = {name: float(np.nanmean(list(v.values()))) for name, v in results.items()}
    print("-" * 90)
    names = list(results.keys())
    macro_row = f"{'MACRO':<20}" + "".join(f"{macro[n]:<24.4f}" for n in names)
    macro_row += f"{macro[names[1]] - macro[names[0]]:+.4f}"
    print(macro_row)

    print()
    print("[注目ラベル]")
    for label in ["Synovitis", "Effusion", "Medial OA", "Baker's"]:
        print(f"  {label}: DINOv2={results['DINOv2-base'][label]:.4f} -> RadImageNet={results['RadImageNet-ResNet50'][label]:.4f} (delta={deltas[label]:+.4f})")


if __name__ == "__main__":
    main()
