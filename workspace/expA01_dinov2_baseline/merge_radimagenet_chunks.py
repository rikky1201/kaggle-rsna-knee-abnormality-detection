"""
Kaggle Notebookでチャンク単位（200件ずつ）保存された RadImageNet ResNet50 特徴量
（`kaggle_radimagenet_full_extract/output/features_chunk_*.npz`）を1本の npz に統合する。
`merge_feature_chunks.py`（DINOv2版）と同じロジックだが、件数・次元はextraction_manifest.json
から動的に読み取る（DINOv2側と同じ2件が失敗するとは限らないため、決め打ちにしない）。

Usage:
    python merge_radimagenet_chunks.py
"""

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
CHUNKS_DIR = EXP_DIR / "kaggle_radimagenet_full_extract" / "output"
MANIFEST_PATH = CHUNKS_DIR / "extraction_manifest.json"
OUTPUT_DIR = EXP_DIR / "data"


def main() -> None:
    chunk_files = sorted(CHUNKS_DIR.glob("features_chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"チャンクファイルが見つかりません: {CHUNKS_DIR}")
    logger.info("%d 個のチャンクファイルを統合します", len(chunk_files))

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    logger.info("manifest: n_total=%d n_done=%d n_failed=%d failed_uids=%s",
                manifest["n_total"], manifest["n_done"], manifest["n_failed"],
                [f["study_uid"] for f in manifest["failed_uids"]])

    all_uids = []
    all_feats = []
    for f in chunk_files:
        data = np.load(f)
        all_uids.append(data["study_uids"])
        all_feats.append(data["features"])

    study_uids = np.concatenate(all_uids, axis=0)
    features = np.concatenate(all_feats, axis=0)

    if len(study_uids) != manifest["n_done"]:
        raise AssertionError(f"件数がmanifestと一致しません: manifest.n_done={manifest['n_done']}, actual={len(study_uids)}")
    if len(set(study_uids.tolist())) != len(study_uids):
        raise AssertionError("StudyInstanceUID に重複があります")

    output_path = OUTPUT_DIR / f"radimagenet_resnet50_features_{len(study_uids)}.npz"
    np.savez_compressed(output_path, study_uids=study_uids, features=features.astype(np.float32))
    logger.info("統合ファイルを保存しました: %s (n=%d, dim=%d)", output_path, len(study_uids), features.shape[1])


if __name__ == "__main__":
    main()
