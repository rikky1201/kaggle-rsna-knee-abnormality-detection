# %% [markdown]
# # DINOv2（凍結）特徴抽出 — 50〜100件での時間計測
#
# `src/preprocessing.py`（percentile clip正規化 + スライス間引き）、
# `src/pooling.py`（slice→series→study pooling）、`src/feature_extraction.py`
# （DINOv2ラッパー）を private Kaggle Dataset 経由で import し、実際の DICOM に対して
# study特徴ベクトルを構築する。
#
# **今回はタイミング計測のみ**（small/base 両サイズ）。全4,407件への本実行はユーザーの
# 指示を待ってから別途行う。
#
# 判明したデータ構造（`kaggle_explore_data/` での探索結果）:
# - `/kaggle/input/competitions/<comp-slug>/train_series/<StudyUID>/<SeriesUID>/<SOPUID>.dcm`
# - ファイル名は SOPInstanceUID（順不同）→ **InstanceNumber でソートが必要**
# - MR, 512x512, uint16, MONOCHROME2
#
# ## v2: DICOM読み込みの並列化
# v1（逐次読み込み）の計測結果: small 平均4.58s/件・全件推定5.60h, base 平均4.87s/件・
# 全件推定5.97h。small/baseの速度差がわずか6%だったことから、DICOM I/O（ネットワーク
# ストレージへの逐次アクセス）が律速と判断。`ThreadPoolExecutor`（NUM_IO_WORKERS=8）で
# 1 series 内のDICOM読み込みを並列化し、同じ100件・同じシードで再計測してv1と比較する。

# %%
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "pydicom"], check=True)

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
# ── パス解決（マウント階層は変わりうるため、実ファイルを再帰的に探す方式で統一する）
def _find_by_marker(root: Path, marker_relpath: str) -> Path:
    """root配下を再帰探索し、marker_relpath（例 'train_series.csv'）を含むディレクトリを返す。"""
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
from pooling import pool_series_to_study, pool_slices_to_series, study_feature_dim  # noqa: E402
from preprocessing import preprocess_series  # noqa: E402

OUTPUT_DIR = Path("/kaggle/working")

# %%
# ── パラメータ
N_SAMPLE = 100  # 時間計測のサンプル数（50〜100件）
SEED = 42
BATCH_SIZE = 32
MODEL_SIZES = ["small", "base"]
NUM_IO_WORKERS = 8  # DICOM読み込みの並列度（I/Oバウンドな処理のためスレッド並列で十分）

# %%
# ── DICOM series 読み込み（InstanceNumberでソート）
# 初回計測（逐次読み込み）ではDICOM I/Oが律速だったため、ThreadPoolExecutorで
# ファイル読み込みを並列化する（pydicom.dcmread はファイルI/O待ちが主体でGILを解放するため、
# スレッド並列でも効果が見込める）
from concurrent.futures import ThreadPoolExecutor


def _read_dicom_slice(path: Path) -> tuple[int, np.ndarray]:
    ds = pydicom.dcmread(str(path))
    return int(ds.InstanceNumber), ds.pixel_array.astype(np.float32)


def load_series_volume(series_dir: Path, num_workers: int = NUM_IO_WORKERS) -> np.ndarray:
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"DICOMファイルが見つかりません: {series_dir}")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        slices = list(executor.map(_read_dicom_slice, dcm_files))
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
# ── サンプル抽出（再現性のため固定シード）
train_df = pd.read_csv(COMP_DIR / "train.csv")
series_df = pd.read_csv(COMP_DIR / "train_series.csv")

rng = np.random.default_rng(SEED)
all_study_uids = train_df["StudyInstanceUID"].unique()
sample_size = min(N_SAMPLE, len(all_study_uids))
sample_study_uids = rng.choice(all_study_uids, size=sample_size, replace=False)
logger.info("時間計測サンプル数: %d / 全体 %d 件", sample_size, len(all_study_uids))

# %%
# ── モデルサイズごとに計測
benchmark_results = {}
sample_features = {}  # 次元確認用に各サイズ最初の3件だけ保存

for size in MODEL_SIZES:
    logger.info("=== DINOv2-%s のロード ===", size)
    t_load_start = time.time()
    extractor = Dinov2FeatureExtractor(size=size)
    load_sec = time.time() - t_load_start
    logger.info("ロード完了: %.1fs (embed_dim=%d)", load_sec, extractor.embed_dim)

    times = []
    feats_preview = []
    for i, study_uid in enumerate(sample_study_uids, start=1):
        t0 = time.time()
        feat = process_study(study_uid, series_df, extractor)
        elapsed = time.time() - t0
        times.append(elapsed)
        if i <= 3:
            feats_preview.append({"study_uid": study_uid, "feature_dim": int(feat.shape[0]),
                                   "feature_head": feat[:5].tolist()})
        if i % 20 == 0:
            logger.info("[%s] %d/%d 件処理済み (直近平均 %.2fs/件)", size, i, sample_size, np.mean(times[-20:]))

    times_arr = np.array(times)
    result = {
        "model_size": size,
        "io_parallel_workers": NUM_IO_WORKERS,
        "embed_dim": extractor.embed_dim,
        "study_feature_dim": study_feature_dim(2 * extractor.embed_dim),
        "n_studies": sample_size,
        "model_load_sec": load_sec,
        "total_extraction_sec": float(times_arr.sum()),
        "mean_sec_per_study": float(times_arr.mean()),
        "median_sec_per_study": float(np.median(times_arr)),
        "min_sec_per_study": float(times_arr.min()),
        "max_sec_per_study": float(times_arr.max()),
        "estimated_full_4407_hours": float(times_arr.mean() * 4407 / 3600),
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }
    benchmark_results[size] = result
    sample_features[size] = feats_preview

    logger.info(
        "=== DINOv2-%s 結果: 平均 %.2fs/件, 全4407件推定 %.2f時間 ===",
        size, result["mean_sec_per_study"], result["estimated_full_4407_hours"],
    )

    del extractor
    torch.cuda.empty_cache()

# %%
# ── 結果保存
output = {"benchmark_results": benchmark_results, "sample_features_preview": sample_features,
          "sample_study_uids": sample_study_uids.tolist()}
output_path = OUTPUT_DIR / "timing_benchmark.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
logger.info("計測結果を保存しました: %s", output_path)
print(json.dumps(benchmark_results, indent=2))
