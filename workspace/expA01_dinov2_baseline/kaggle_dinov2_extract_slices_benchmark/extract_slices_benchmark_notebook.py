# %% [markdown]
# # DINOv2（凍結）スライス単位特徴抽出 — 50〜100件での時間・容量計測
#
# attention pooling実装のため、study単位に集約する前の「スライスごとのDINOv2特徴ベクトル」を
# series単位でまとめてディスクに保存する版を新規実装する（ユーザー指示、2026-09-16）。
# 前回の`kaggle_dinov2_extract_full/`はstudy単位の最終ベクトルしか保存しておらず、
# attention pooling実装のために再抽出が必要になった反省を踏まえ、今回は集約前の
# 中間出力（スライス単位ベクトル）を必ずキャッシュする。
#
# 本Notebookは**時間・保存容量の計測のみ**（100件）。全4,407件への本実行は
# 別途 `kaggle_dinov2_extract_slices_full/` で行う。
#
# 保存形式（ragged array）: studyごとに series数・スライス数が可変のため、
# 全スライス特徴を1つの配列に連結し、どのseries/studyに属するかを示す
# インデックス配列を併せて保存する。

# %%
import json
import logging
import sys as _sys
import time
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
N_SAMPLE = 100
SEED = 42  # kaggle_dinov2_extract/ の初回ベンチマークと同一シード（比較可能にするため）
BATCH_SIZE = 32
MODEL_SIZE = "base"


# %%
def _read_dicom_slice(path: Path) -> tuple[int, np.ndarray]:
    ds = pydicom.dcmread(str(path))
    return int(ds.InstanceNumber), ds.pixel_array.astype(np.float32)


def load_series_volume(series_dir: Path) -> np.ndarray:
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"DICOMファイルが見つかりません: {series_dir}")
    slices = [_read_dicom_slice(f) for f in dcm_files]
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices], axis=0)


def process_study(study_uid: str, series_df: pd.DataFrame, extractor: Dinov2FeatureExtractor):
    """戻り値: (study_feature, per_series_records)
    per_series_records: 各要素が {slice_features, series_uid, anatomical_plane, fluid_sensitive, fat_suppression}
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
rng = np.random.default_rng(SEED)
all_study_uids = train_df["StudyInstanceUID"].unique()
sample_study_uids = rng.choice(all_study_uids, size=min(N_SAMPLE, len(all_study_uids)), replace=False)
logger.info("計測サンプル数: %d", len(sample_study_uids))

extractor = Dinov2FeatureExtractor(size=MODEL_SIZE)
logger.info("DINOv2-%s ロード完了: embed_dim=%d", MODEL_SIZE, extractor.embed_dim)

# %%
# ── ragged array形式でスライス単位特徴を蓄積
all_slice_features = []  # 最後にconcatenate
slice_to_series_idx = []  # 各スライスがどのseries(通し番号)に属するか
series_study_uid = []
series_instance_uid = []
series_anatomical_plane = []
series_fluid_sensitive = []
series_fat_suppression = []
study_features = []
study_uids_done = []

series_counter = 0
times = []
n_slices_per_study = []
n_series_per_study = []

for i, study_uid in enumerate(sample_study_uids, start=1):
    t0 = time.time()
    study_feature, per_series_records = process_study(study_uid, series_df, extractor)
    elapsed = time.time() - t0
    times.append(elapsed)

    study_features.append(study_feature)
    study_uids_done.append(study_uid)

    n_slices_this_study = 0
    for rec in per_series_records:
        n_slices = rec["slice_features"].shape[0]
        n_slices_this_study += n_slices
        all_slice_features.append(rec["slice_features"])
        slice_to_series_idx.append(np.full(n_slices, series_counter, dtype=np.int32))
        series_study_uid.append(study_uid)
        series_instance_uid.append(rec["series_uid"])
        series_anatomical_plane.append(rec["anatomical_plane"])
        series_fluid_sensitive.append(rec["fluid_sensitive"])
        series_fat_suppression.append(rec["fat_suppression"])
        series_counter += 1

    n_slices_per_study.append(n_slices_this_study)
    n_series_per_study.append(len(per_series_records))

    if i % 20 == 0:
        logger.info("%d/%d 件処理済み (直近平均 %.2fs/件)", i, len(sample_study_uids), np.mean(times[-20:]))

# %%
slice_features_arr = np.concatenate(all_slice_features, axis=0).astype(np.float32)
slice_to_series_arr = np.concatenate(slice_to_series_idx, axis=0)
study_features_arr = np.stack(study_features, axis=0).astype(np.float32)

out_path = OUTPUT_DIR / "slice_features_benchmark.npz"
np.savez_compressed(
    out_path,
    slice_features=slice_features_arr,
    slice_to_series=slice_to_series_arr,
    series_study_uid=np.array(series_study_uid),
    series_instance_uid=np.array(series_instance_uid),
    series_anatomical_plane=np.array(series_anatomical_plane),
    series_fluid_sensitive=np.array(series_fluid_sensitive, dtype=np.int32),
    series_fat_suppression=np.array(series_fat_suppression, dtype=np.int32),
    study_uids=np.array(study_uids_done),
    study_features=study_features_arr,
)
file_size_bytes = out_path.stat().st_size

# %%
times_arr = np.array(times)
n_slices_arr = np.array(n_slices_per_study)
n_series_arr = np.array(n_series_per_study)

result = {
    "n_studies": len(sample_study_uids),
    "n_series_total": series_counter,
    "n_slices_total": int(slice_features_arr.shape[0]),
    "embed_dim": extractor.embed_dim,
    "mean_slices_per_study": float(n_slices_arr.mean()),
    "mean_series_per_study": float(n_series_arr.mean()),
    "mean_sec_per_study": float(times_arr.mean()),
    "median_sec_per_study": float(np.median(times_arr)),
    "total_extraction_sec": float(times_arr.sum()),
    "estimated_full_4407_hours": float(times_arr.mean() * 4407 / 3600),
    "benchmark_file_size_bytes": int(file_size_bytes),
    "benchmark_file_size_mb": float(file_size_bytes / 1e6),
    "estimated_full_4407_slice_storage_gb": float(file_size_bytes / 1e9 * (4407 / len(sample_study_uids))),
}
logger.info("計測結果: %s", json.dumps(result, indent=2))

with open(OUTPUT_DIR / "slice_benchmark_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, indent=2))
