"""
Kaggle Notebookでチャンク単位（200件ずつ）保存されたスライス単位DINOv2特徴
（`data/slice_features_chunks/slice_features_chunk_*.npz`）を1本のnpzに統合する。

各チャンク内の `slice_to_series` はチャンクローカルな通し番号のため、統合時に
グローバルなseries通し番号へオフセットし直す。

副産物として、各チャンクに含まれる study_features（mean+max pooling, 再計算値）を
既存の `data/dinov2_base_features_4405.npz`（study単位本番実行の結果）と突き合わせ、
再抽出が前回と一致することを検証する（クロスチェック）。

Usage:
    python merge_slice_feature_chunks.py
"""

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
CHUNKS_DIR = EXP_DIR / "data" / "slice_features_chunks"
OUTPUT_PATH = EXP_DIR / "data" / "dinov2_base_slice_features_4407.npz"
EXISTING_STUDY_FEATURES_PATH = EXP_DIR / "data" / "dinov2_base_features_4405.npz"

EXPECTED_N_STUDIES = 4407  # 2026-09-19再抽出では旧DICOM読み込み失敗2件も成功し全件揃った（下記参照）
EXPECTED_EMBED_DIM = 768
CROSS_CHECK_ATOL = 1e-4


def main() -> None:
    chunk_files = sorted(CHUNKS_DIR.glob("slice_features_chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"チャンクファイルが見つかりません: {CHUNKS_DIR}")
    logger.info("%d 個のチャンクファイルを統合します", len(chunk_files))

    all_slice_features = []
    all_slice_to_series = []
    all_series_study_uid = []
    all_series_instance_uid = []
    all_series_anatomical_plane = []
    all_series_fluid_sensitive = []
    all_series_fat_suppression = []
    all_study_uids = []
    all_study_features = []

    series_offset = 0
    for f in chunk_files:
        data = np.load(f)
        n_series_this_chunk = len(data["series_study_uid"])

        all_slice_features.append(data["slice_features"])
        all_slice_to_series.append(data["slice_to_series"].astype(np.int64) + series_offset)
        all_series_study_uid.append(data["series_study_uid"])
        all_series_instance_uid.append(data["series_instance_uid"])
        all_series_anatomical_plane.append(data["series_anatomical_plane"])
        all_series_fluid_sensitive.append(data["series_fluid_sensitive"])
        all_series_fat_suppression.append(data["series_fat_suppression"])
        all_study_uids.append(data["study_uids"])
        all_study_features.append(data["study_features"])

        series_offset += n_series_this_chunk

    slice_features = np.concatenate(all_slice_features, axis=0).astype(np.float32)
    slice_to_series = np.concatenate(all_slice_to_series, axis=0).astype(np.int32)
    series_study_uid = np.concatenate(all_series_study_uid, axis=0)
    series_instance_uid = np.concatenate(all_series_instance_uid, axis=0)
    series_anatomical_plane = np.concatenate(all_series_anatomical_plane, axis=0)
    series_fluid_sensitive = np.concatenate(all_series_fluid_sensitive, axis=0)
    series_fat_suppression = np.concatenate(all_series_fat_suppression, axis=0)
    study_uids = np.concatenate(all_study_uids, axis=0)
    study_features = np.concatenate(all_study_features, axis=0).astype(np.float32)

    if len(study_uids) != EXPECTED_N_STUDIES:
        raise AssertionError(f"study件数が一致しません: expected={EXPECTED_N_STUDIES}, actual={len(study_uids)}")
    if slice_features.shape[1] != EXPECTED_EMBED_DIM:
        raise AssertionError(f"スライス特徴次元が一致しません: expected={EXPECTED_EMBED_DIM}, actual={slice_features.shape[1]}")
    if len(set(study_uids.tolist())) != len(study_uids):
        raise AssertionError("StudyInstanceUID に重複があります")
    if slice_to_series.max() != len(series_study_uid) - 1:
        raise AssertionError("slice_to_series のグローバルoffset変換に不整合があります")

    # ── クロスチェック: 再抽出したstudy_featuresが既存のstudy単位本番実行結果と一致するか
    if EXISTING_STUDY_FEATURES_PATH.exists():
        existing = np.load(EXISTING_STUDY_FEATURES_PATH)
        existing_uid_to_idx = {uid: i for i, uid in enumerate(existing["study_uids"].tolist())}
        common_uids = [u for u in study_uids.tolist() if u in existing_uid_to_idx]
        if not common_uids:
            logger.warning("既存study特徴との共通study_uidがありません。クロスチェックをスキップします")
        else:
            new_uid_to_idx = {uid: i for i, uid in enumerate(study_uids.tolist())}
            new_idx = [new_uid_to_idx[u] for u in common_uids]
            old_idx = [existing_uid_to_idx[u] for u in common_uids]
            diff = np.abs(study_features[new_idx] - existing["features"][old_idx])
            max_diff = float(diff.max())
            logger.info(
                "クロスチェック: 共通%d件のstudy特徴 最大絶対差=%.6g（既存 dinov2_base_features_4405.npz との比較）",
                len(common_uids), max_diff,
            )
            if max_diff > CROSS_CHECK_ATOL:
                raise AssertionError(
                    f"再抽出したstudy特徴が既存の結果と乖離しています（max_diff={max_diff:.6g} > {CROSS_CHECK_ATOL}）。"
                    "抽出パイプラインに差分が無いか確認してください。"
                )
    else:
        logger.warning("既存study特徴ファイルが見つからないためクロスチェックをスキップします: %s", EXISTING_STUDY_FEATURES_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        slice_features=slice_features,
        slice_to_series=slice_to_series,
        series_study_uid=series_study_uid,
        series_instance_uid=series_instance_uid,
        series_anatomical_plane=series_anatomical_plane,
        series_fluid_sensitive=series_fluid_sensitive,
        series_fat_suppression=series_fat_suppression,
        study_uids=study_uids,
        study_features=study_features,
    )
    logger.info(
        "統合ファイルを保存しました: %s (n_studies=%d, n_series=%d, n_slices=%d, embed_dim=%d)",
        OUTPUT_PATH, len(study_uids), len(series_study_uid), slice_features.shape[0], slice_features.shape[1],
    )


if __name__ == "__main__":
    main()
