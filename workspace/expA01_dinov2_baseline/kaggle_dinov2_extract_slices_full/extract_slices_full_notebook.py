# %% [markdown]
# # DINOv2-base（凍結）スライス単位特徴抽出 — 全4,407件 本番実行
#
# attention pooling実装のため、study単位に集約する前の「スライスごとのDINOv2特徴ベクトル」を
# series単位でまとめて保存する（`kaggle_dinov2_extract_full/`はstudy単位の最終ベクトルしか
# 保存しておらず、再実行が必要になった反省を踏まえた再発防止。ユーザー指示、2026-09-19）。
#
# 方針（`kaggle_dinov2_extract_slices_benchmark/`での100件計測結果を踏まえる）:
# - モデル: DINOv2-base（既存の study 単位特徴抽出と同一設定）
# - 推定所要時間: 7.31s/件 × 4,407 ≈ 約9時間
# - 推定ストレージ: 約2.25GB（スライス単位特徴、圧縮後）
#
# **チャンク単位（CHUNK_SIZE件ごと）で `/kaggle/working/slice_features_chunk_XXXX.npz` に保存**する
# （`kaggle_dinov2_extract_full/`と同じ設計: npzは追記不可、1チャンクの失敗が既存チャンクを
# 破壊しない）。各チャンク内では、スライス単位特徴（ragged array）に加えて、既存の
# study単位mean+max特徴も同時に計算して保存する（再抽出結果が既存の
# `data/dinov2_base_features_4405.npz` と一致するかのクロスチェック用。計算コストはほぼゼロ）。
#
# per-study処理失敗時は既存の本番実行と同じ方針（Exceptionを広く捕捉しログ＋manifestに記録して
# スキップ。4,407件規模のバッチジョブで1件のために全体を失わないため）。

# %%
import json
import logging
import sys as _sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPUが利用できません。kernel-metadata.json の enable_gpu / Accelerator設定を確認してください。")
logger.info("GPU: %s", torch.cuda.get_device_name(0))


# %%
def _find_by_marker(root: Path, marker_relpath: str) -> Path:
    matches = list(root.rglob(marker_relpath))
    if not matches:
        listing = [str(p.relative_to(root)) for p in root.rglob("*")][:200]
        raise FileNotFoundError(f"'{marker_relpath}' が {root} 配下に見つかりません。中身:\n  " + "\n  ".join(listing))
    return matches[0].parent


INPUT_ROOT = Path("/kaggle/input")
COMP_DIR = _find_by_marker(INPUT_ROOT, "train_series.csv")
ASSETS_DIR = _find_by_marker(INPUT_ROOT, "preprocessing.py")
logger.info("competition dir: %s", COMP_DIR)
logger.info("assets dir: %s", ASSETS_DIR)

_sys.path.append(str(ASSETS_DIR))
from feature_extraction import Dinov2FeatureExtractor  # noqa: E402
from pooling import pool_series_to_study, pool_slices_to_series  # noqa: E402
from preprocessing import preprocess_series  # noqa: E402

OUTPUT_DIR = Path("/kaggle/working")

# %%
# ── 推論パラメータ
MODEL_SIZE = "base"
BATCH_SIZE = 32
CHUNK_SIZE = 200  # この件数ごとに slice_features_chunk_XXXX.npz を保存


# %%
def _read_dicom_slice(path: Path) -> tuple[int, np.ndarray]:
    ds = pydicom.dcmread(str(path))
    return int(ds.InstanceNumber), ds.pixel_array.astype(np.float32)


def load_series_volume(series_dir: Path) -> np.ndarray:
    """v1方式（逐次読み込み）。既存の本番実行（`kaggle_dinov2_extract_full/`）と同じ理由で採用。"""
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"DICOMファイルが見つかりません: {series_dir}")
    slices = [_read_dicom_slice(f) for f in dcm_files]
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices], axis=0)


def process_study(study_uid: str, series_df: pd.DataFrame, extractor: Dinov2FeatureExtractor):
    """戻り値: (study_feature, per_series_records)

    per_series_records: 各要素が {slice_features, series_uid, anatomical_plane,
        fluid_sensitive, fat_suppression}
    """
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    if len(study_series) == 0:
        raise ValueError(f"train_series.csv に {study_uid} のseriesがありません")

    series_entries = []
    per_series_records = []
    for _, row in study_series.iterrows():
        series_dir = COMP_DIR / "train_series" / study_uid / row["SeriesInstanceUID"]
        volume = load_series_volume(series_dir)
        processed = preprocess_series(volume)
        slice_features = extractor.extract(processed, batch_size=BATCH_SIZE)  # (n_slices, embed_dim)
        series_feat = pool_slices_to_series(slice_features)
        series_entries.append({
            "feature": series_feat,
            "anatomical_plane": row["Anatomical_Plane"],
            "fluid_sensitive": int(row["Fluid_Sensitive"]),
            "fat_suppression": int(row["Fat_Suppression"]),
        })
        per_series_records.append({
            "slice_features": slice_features,
            "series_uid": row["SeriesInstanceUID"],
            "anatomical_plane": row["Anatomical_Plane"],
            "fluid_sensitive": int(row["Fluid_Sensitive"]),
            "fat_suppression": int(row["Fat_Suppression"]),
        })
    study_feature = pool_series_to_study(series_entries)
    return study_feature, per_series_records


# %%
train_df = pd.read_csv(COMP_DIR / "train.csv")
series_df = pd.read_csv(COMP_DIR / "train_series.csv")
all_study_uids = train_df["StudyInstanceUID"].tolist()
logger.info("対象件数: %d 件 / model=DINOv2-%s / chunk_size=%d", len(all_study_uids), MODEL_SIZE, CHUNK_SIZE)

logger.info("モデルをロードします: DINOv2-%s", MODEL_SIZE)
t_load_start = time.time()
extractor = Dinov2FeatureExtractor(size=MODEL_SIZE)
logger.info("モデルロード完了: %.1fs (embed_dim=%d)", time.time() - t_load_start, extractor.embed_dim)

# %%
n_done = 0
n_failed = 0
failed_uids: list[dict] = []
t_all_start = time.time()

for chunk_start in range(0, len(all_study_uids), CHUNK_SIZE):
    chunk_uids = all_study_uids[chunk_start: chunk_start + CHUNK_SIZE]
    chunk_no = chunk_start // CHUNK_SIZE + 1
    n_chunks = (len(all_study_uids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info("[chunk %d/%d] %d件を処理中... (累計 %d/%d)", chunk_no, n_chunks, len(chunk_uids), chunk_start, len(all_study_uids))

    t_chunk_start = time.time()

    # ── チャンクローカルの蓄積（ragged array形式）
    chunk_slice_features = []
    chunk_slice_to_series = []  # チャンクローカルのseries通し番号
    chunk_series_study_uid = []
    chunk_series_instance_uid = []
    chunk_series_anatomical_plane = []
    chunk_series_fluid_sensitive = []
    chunk_series_fat_suppression = []
    chunk_study_features = []
    chunk_study_uids = []
    series_counter = 0  # チャンク内でリセット（merge時にグローバルoffsetへ変換する）

    for study_uid in chunk_uids:
        try:
            study_feature, per_series_records = process_study(study_uid, series_df, extractor)
        except Exception as e:  # noqa: BLE001 — 4,407件規模のバッチジョブで1件のために全体を
            # 落とすのは非効率なため、詳細をログに残した上で当該件のみスキップする
            # （fail-fastの原則＝握りつぶさないことは、ログでの可視化とmanifestへの記録で担保する）
            n_failed += 1
            failed_uids.append({"study_uid": study_uid, "error": str(e)})
            logger.warning("study処理失敗（スキップ）: %s | %s\n%s", study_uid, e, traceback.format_exc())
            continue

        chunk_study_features.append(study_feature)
        chunk_study_uids.append(study_uid)
        for rec in per_series_records:
            n_slices = rec["slice_features"].shape[0]
            chunk_slice_features.append(rec["slice_features"])
            chunk_slice_to_series.append(np.full(n_slices, series_counter, dtype=np.int32))
            chunk_series_study_uid.append(study_uid)
            chunk_series_instance_uid.append(rec["series_uid"])
            chunk_series_anatomical_plane.append(rec["anatomical_plane"])
            chunk_series_fluid_sensitive.append(rec["fluid_sensitive"])
            chunk_series_fat_suppression.append(rec["fat_suppression"])
            series_counter += 1
        n_done += 1

    if chunk_study_uids:
        chunk_path = OUTPUT_DIR / f"slice_features_chunk_{chunk_no:04d}.npz"
        np.savez_compressed(
            chunk_path,
            slice_features=np.concatenate(chunk_slice_features, axis=0).astype(np.float32),
            slice_to_series=np.concatenate(chunk_slice_to_series, axis=0),
            series_study_uid=np.array(chunk_series_study_uid),
            series_instance_uid=np.array(chunk_series_instance_uid),
            series_anatomical_plane=np.array(chunk_series_anatomical_plane),
            series_fluid_sensitive=np.array(chunk_series_fluid_sensitive, dtype=np.int32),
            series_fat_suppression=np.array(chunk_series_fat_suppression, dtype=np.int32),
            study_uids=np.array(chunk_study_uids),
            study_features=np.stack(chunk_study_features, axis=0).astype(np.float32),
        )

    chunk_sec = time.time() - t_chunk_start
    elapsed = time.time() - t_all_start
    processed_so_far = chunk_start + len(chunk_uids)
    eta_sec = (elapsed / processed_so_far) * (len(all_study_uids) - processed_so_far) if processed_so_far else 0
    logger.info(
        "[chunk %d/%d] 完了: %.1fs（1件 %.2fs）| 累計成功 %d件 | 失敗 %d件 | ETA %.1f分",
        chunk_no, n_chunks, chunk_sec, chunk_sec / len(chunk_uids), n_done, n_failed, eta_sec / 60,
    )

t_all_total = time.time() - t_all_start
logger.info("全件完了: 成功 %d件 / 失敗 %d件 | 総時間 %.1f分", n_done, n_failed, t_all_total / 60)

# %%
manifest = {
    "model_size": MODEL_SIZE,
    "embed_dim": extractor.embed_dim,
    "chunk_size": CHUNK_SIZE,
    "n_total": len(all_study_uids),
    "n_done": n_done,
    "n_failed": n_failed,
    "failed_uids": failed_uids,
    "total_sec": t_all_total,
    "mean_sec_per_study": t_all_total / len(all_study_uids),
}
manifest_path = OUTPUT_DIR / "slice_extraction_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
logger.info("manifestを保存しました: %s", manifest_path)
print(json.dumps({k: v for k, v in manifest.items() if k != "failed_uids"}, indent=2))
