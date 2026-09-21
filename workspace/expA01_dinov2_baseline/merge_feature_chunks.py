"""
Kaggle Notebookでチャンク単位（200件ずつ）保存された DINOv2-base 特徴量
（`data/features_chunks/features_chunk_*.npz`）を1本の npz に統合する。

Usage:
    python merge_feature_chunks.py
"""

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
CHUNKS_DIR = EXP_DIR / "data" / "features_chunks"
OUTPUT_PATH = EXP_DIR / "data" / "dinov2_base_features_4405.npz"

EXPECTED_N = 4405
EXPECTED_DIM = 10759


def main() -> None:
    chunk_files = sorted(CHUNKS_DIR.glob("features_chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"チャンクファイルが見つかりません: {CHUNKS_DIR}")
    logger.info("%d 個のチャンクファイルを統合します", len(chunk_files))

    all_uids = []
    all_feats = []
    for f in chunk_files:
        data = np.load(f)
        all_uids.append(data["study_uids"])
        all_feats.append(data["features"])

    study_uids = np.concatenate(all_uids, axis=0)
    features = np.concatenate(all_feats, axis=0)

    if len(study_uids) != EXPECTED_N:
        raise AssertionError(f"件数が一致しません: expected={EXPECTED_N}, actual={len(study_uids)}")
    if features.shape[1] != EXPECTED_DIM:
        raise AssertionError(f"特徴次元が一致しません: expected={EXPECTED_DIM}, actual={features.shape[1]}")
    if len(set(study_uids.tolist())) != len(study_uids):
        raise AssertionError("StudyInstanceUID に重複があります")

    np.savez_compressed(OUTPUT_PATH, study_uids=study_uids, features=features.astype(np.float32))
    logger.info("統合ファイルを保存しました: %s (n=%d, dim=%d)", OUTPUT_PATH, len(study_uids), features.shape[1])


if __name__ == "__main__":
    main()
