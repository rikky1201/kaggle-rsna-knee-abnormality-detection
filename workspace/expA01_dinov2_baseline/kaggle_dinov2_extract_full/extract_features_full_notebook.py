# %% [markdown]
# # DINOv2-base（凍結）特徴抽出 — 全4,407件 本番実行
#
# `kaggle_dinov2_extract/` でのタイミング計測結果を踏まえた最終方針:
# - モデル: **DINOv2-base**（small/baseの速度差はごくわずかで表現力優先。50〜100件計測: 平均4.87s/件）
# - DICOM読み込み: **逐次（v1方式）**。ThreadPoolExecutorでの並列化はsmall改善・base悪化と
#   結果が不一致だったため、再現性の高いシンプルな逐次実装のままユーザー判断で採用
# - 推定所要時間: 4.87s × 4,407 ≈ **約6時間**
#
# **チャンク単位（CHUNK_SIZE件ごと）で `/kaggle/working/features_chunk_XXXX.npz` に保存**する。
# npzは追記できないため、pseudo-label生成時のCSV追記とは異なりチャンクごとに別ファイルとして
# 保存する（1チャンクの失敗が既存チャンクを破壊しない設計）。

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
CHUNK_SIZE = 200  # この件数ごとに features_chunk_XXXX.npz を保存

# %%
def _read_dicom_slice(path: Path) -> tuple[int, np.ndarray]:
    ds = pydicom.dcmread(str(path))
    return int(ds.InstanceNumber), ds.pixel_array.astype(np.float32)


def load_series_volume(series_dir: Path) -> np.ndarray:
    """v1方式（逐次読み込み）。並列化(v2)はタイミング計測で改善/悪化が混在し
    再現性が不確かだったため、シンプルな逐次実装を採用する。"""
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"DICOMファイルが見つかりません: {series_dir}")
    slices = [_read_dicom_slice(f) for f in dcm_files]
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices], axis=0)


def process_study(study_uid: str, series_df: pd.DataFrame, extractor: Dinov2FeatureExtractor) -> np.ndarray:
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    if len(study_series) == 0:
        raise ValueError(f"train_series.csv に {study_uid} のseriesがありません")

    series_entries = []
    for _, row in study_series.iterrows():
        series_dir = COMP_DIR / "train_series" / study_uid / row["SeriesInstanceUID"]
        volume = load_series_volume(series_dir)
        processed = preprocess_series(volume)
        slice_features = extractor.extract(processed, batch_size=BATCH_SIZE)
        series_feat = pool_slices_to_series(slice_features)
        series_entries.append({
            "feature": series_feat,
            "anatomical_plane": row["Anatomical_Plane"],
            "fluid_sensitive": int(row["Fluid_Sensitive"]),
            "fat_suppression": int(row["Fat_Suppression"]),
        })
    return pool_series_to_study(series_entries)


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
    chunk_features = []
    chunk_study_uids = []
    for study_uid in chunk_uids:
        try:
            feat = process_study(study_uid, series_df, extractor)
        except Exception as e:  # noqa: BLE001 — 4,407件規模のバッチジョブで1件のために全体を
            # 落とすのは非効率なため、詳細をログに残した上で当該件のみスキップする
            # （fail-fastの原則＝握りつぶさないことは、ログでの可視化とmanifestへの記録で担保する）
            n_failed += 1
            failed_uids.append({"study_uid": study_uid, "error": str(e)})
            logger.warning("study処理失敗（スキップ）: %s | %s\n%s", study_uid, e, traceback.format_exc())
            continue
        chunk_features.append(feat)
        chunk_study_uids.append(study_uid)

    if chunk_features:
        chunk_path = OUTPUT_DIR / f"features_chunk_{chunk_no:04d}.npz"
        np.savez_compressed(
            chunk_path,
            study_uids=np.array(chunk_study_uids),
            features=np.stack(chunk_features, axis=0).astype(np.float32),
        )
        n_done += len(chunk_features)

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
manifest_path = OUTPUT_DIR / "extraction_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
logger.info("manifestを保存しました: %s", manifest_path)
print(json.dumps({k: v for k, v in manifest.items() if k != "failed_uids"}, indent=2))
