"""
学習データ準備: labels + DINOv2特徴 + fold割り当てを結合し、
ソフトターゲット・pos_weightを計算する。

設計方針: KAGGLE_DIRECTION.md「モデル設計方針」§2+6 損失関数
- gold: y_j をそのまま使用
- pseudo: pred=1 → soft_target = PPV_j, pred=0 → soft_target = 1 - NPV_j
- pos_weight π_j = (陰性数/陽性数) は gold 58件の分布から算出
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


@dataclass
class TrainingData:
    study_uids: np.ndarray       # (N,)
    features: np.ndarray         # (N, feat_dim)
    hard_labels: np.ndarray      # (N, 12) — gold実測値 or pseudo予測値（0/1）。検証用の真値として使うのは gold 行のみ
    soft_targets: np.ndarray     # (N, 12) — 学習に使うソフトターゲット
    label_source: np.ndarray     # (N,) — "gold" / "pseudo"
    pos_weight: np.ndarray       # (12,) — gold 58件の分布から算出
    fold: np.ndarray             # (N,) — gold行のみ fold番号 (0..n_splits-1)、pseudo行は -1


def load_training_data(
    labels_csv: Path,
    features_npz: Path,
    reliability_csv: Path,
    folds_csv: Path,
) -> TrainingData:
    logger.info("labels を読み込みます: %s", labels_csv)
    labels_df = pd.read_csv(labels_csv)

    logger.info("features を読み込みます: %s", features_npz)
    feat_data = np.load(features_npz)
    feat_df = pd.DataFrame({"StudyInstanceUID": feat_data["study_uids"]})
    feat_df["_feat_idx"] = np.arange(len(feat_df))

    # labels(4407) と features(4405) を inner join する
    # （破損DICOM 2件は features 側に存在しないため自動的に除外される）
    merged = labels_df.merge(feat_df, on="StudyInstanceUID", how="inner")
    n_dropped = len(labels_df) - len(merged)
    logger.info("labels(%d) と features(%d) を結合: %d 行（%d 行が features 欠損のため除外）",
                len(labels_df), len(feat_df), len(merged), n_dropped)
    if n_dropped != 2:
        raise AssertionError(f"除外件数が想定(2件)と一致しません: 実際={n_dropped}")

    gold_count = (merged["label_source"] == "gold").sum()
    if gold_count != 58:
        raise AssertionError(f"結合後の gold 件数が58と一致しません: 実際={gold_count}")

    logger.info("reliability を読み込みます: %s", reliability_csv)
    reliability_df = pd.read_csv(reliability_csv).set_index("label")
    ppv = reliability_df.loc[LABELS, "ppv"].to_numpy()
    npv = reliability_df.loc[LABELS, "npv"].to_numpy()

    logger.info("folds を読み込みます: %s", folds_csv)
    folds_df = pd.read_csv(folds_csv)[["StudyInstanceUID", "fold"]]
    merged = merged.merge(folds_df, on="StudyInstanceUID", how="left")
    merged["fold"] = merged["fold"].fillna(-1).astype(int)  # pseudo行は-1（全foldで常にtrain側）

    hard_labels = merged[LABELS].to_numpy(dtype=np.float32)
    is_gold = (merged["label_source"] == "gold").to_numpy()

    # ソフトターゲット計算
    soft_targets = hard_labels.copy()
    pseudo_mask = ~is_gold
    # pseudo行: pred==1 -> PPV, pred==0 -> 1-NPV （ラベルごとにベクトル化）
    soft_targets[pseudo_mask] = np.where(
        hard_labels[pseudo_mask] == 1,
        ppv[np.newaxis, :],
        1.0 - npv[np.newaxis, :],
    )

    # pos_weight: gold 58件の分布から算出
    gold_labels = hard_labels[is_gold]
    pos_count = gold_labels.sum(axis=0)
    neg_count = len(gold_labels) - pos_count
    if (pos_count == 0).any():
        zero_labels = [LABELS[i] for i in np.where(pos_count == 0)[0]]
        raise ValueError(f"gold内で陽性が0件のラベルがあります（pos_weight計算不可）: {zero_labels}")
    pos_weight = (neg_count / pos_count).astype(np.float32)
    logger.info("pos_weight (gold 58件から算出):\n%s",
                "\n".join(f"  {l}: {w:.3f}" for l, w in zip(LABELS, pos_weight)))

    feat_indices = merged["_feat_idx"].to_numpy()
    features = feat_data["features"][feat_indices]

    return TrainingData(
        study_uids=merged["StudyInstanceUID"].to_numpy(),
        features=features.astype(np.float32),
        hard_labels=hard_labels,
        soft_targets=soft_targets.astype(np.float32),
        label_source=merged["label_source"].to_numpy(),
        pos_weight=pos_weight,
        fold=merged["fold"].to_numpy(),
    )


@dataclass
class AttentionTrainingData:
    """`TrainingData`のattention pooling版（固定サイズのstudy特徴 `features` を持たない）。

    特徴アクセスは`AttentionStudyModel`が`SliceFeatureStore`経由でstudy_uidから直接行うため、
    ここではlabel/fold関連の情報のみを保持する。
    """

    study_uids: np.ndarray       # (N,)
    hard_labels: np.ndarray      # (N, 12)
    soft_targets: np.ndarray     # (N, 12)
    label_source: np.ndarray     # (N,) — "gold" / "pseudo"
    pos_weight: np.ndarray       # (12,)
    fold: np.ndarray             # (N,)


def load_training_data_slices(
    labels_csv: Path,
    reliability_csv: Path,
    folds_csv: Path,
    available_study_uids: set[str],
) -> AttentionTrainingData:
    """`load_training_data`のattention pooling版。固定次元の特徴npzの代わりに、
    `SliceFeatureStore`が保持しているstudy_uid集合（`available_study_uids`）との
    inner joinで結合する。

    labels/reliability/foldsの読み込み・ソフトターゲット・pos_weight計算のロジックは
    `load_training_data`と同一（study単位mean+max版との公平な比較のため、ラベル側の
    前処理は変えない）。
    """
    logger.info("labels を読み込みます: %s", labels_csv)
    labels_df = pd.read_csv(labels_csv)

    merged = labels_df[labels_df["StudyInstanceUID"].isin(available_study_uids)].copy()
    n_dropped = len(labels_df) - len(merged)
    logger.info("labels(%d) と slice特徴ストア(%d studies) を結合: %d 行（%d 行が特徴欠損のため除外）",
                len(labels_df), len(available_study_uids), len(merged), n_dropped)

    gold_count = (merged["label_source"] == "gold").sum()
    if gold_count != 58:
        raise AssertionError(f"結合後の gold 件数が58と一致しません: 実際={gold_count}")

    logger.info("reliability を読み込みます: %s", reliability_csv)
    reliability_df = pd.read_csv(reliability_csv).set_index("label")
    ppv = reliability_df.loc[LABELS, "ppv"].to_numpy()
    npv = reliability_df.loc[LABELS, "npv"].to_numpy()

    logger.info("folds を読み込みます: %s", folds_csv)
    folds_df = pd.read_csv(folds_csv)[["StudyInstanceUID", "fold"]]
    merged = merged.merge(folds_df, on="StudyInstanceUID", how="left")
    merged["fold"] = merged["fold"].fillna(-1).astype(int)

    hard_labels = merged[LABELS].to_numpy(dtype=np.float32)
    is_gold = (merged["label_source"] == "gold").to_numpy()

    soft_targets = hard_labels.copy()
    pseudo_mask = ~is_gold
    soft_targets[pseudo_mask] = np.where(
        hard_labels[pseudo_mask] == 1,
        ppv[np.newaxis, :],
        1.0 - npv[np.newaxis, :],
    )

    gold_labels = hard_labels[is_gold]
    pos_count = gold_labels.sum(axis=0)
    neg_count = len(gold_labels) - pos_count
    if (pos_count == 0).any():
        zero_labels = [LABELS[i] for i in np.where(pos_count == 0)[0]]
        raise ValueError(f"gold内で陽性が0件のラベルがあります（pos_weight計算不可）: {zero_labels}")
    pos_weight = (neg_count / pos_count).astype(np.float32)

    return AttentionTrainingData(
        study_uids=merged["StudyInstanceUID"].to_numpy(),
        hard_labels=hard_labels,
        soft_targets=soft_targets.astype(np.float32),
        label_source=merged["label_source"].to_numpy(),
        pos_weight=pos_weight,
        fold=merged["fold"].to_numpy(),
    )


def split_fold(data: TrainingData | AttentionTrainingData, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """指定fold番号について (train_indices, val_indices) を返す。

    train = gold(held-out fold以外) + pseudo全件
    val   = gold(held-out fold)のみ
    """
    is_val = (data.fold == fold)
    is_gold = (data.label_source == "gold")
    val_mask = is_val & is_gold
    train_mask = ~val_mask
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    if len(val_idx) == 0:
        raise ValueError(f"fold={fold} の検証データが0件です")
    return train_idx, val_idx
