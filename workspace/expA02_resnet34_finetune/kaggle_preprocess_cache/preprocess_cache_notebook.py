# %% [markdown]
# # 前処理済み画像キャッシュ生成（GPU不要, CPUのみ）
#
# ResNet34 fine-tuningをRunPodへ移す際、生DICOM全件（約1.36TB）を転送するのは非現実的
# （帯域・pod課金時間の両方で割に合わない）と判断し、代わりに**前処理済み画像
# （percentile-clip正規化 + slice間引き + 224x224リサイズ + uint8量子化）**をキャッシュして
# 転送する設計に変更（ユーザー指示、2026-09-24）。
#
# 現行の`kaggle_train/train_notebook.py`は毎epochごとに DICOM読み込み→前処理→224リサイズ
# を繰り返しているが、この前処理自体はepoch間で変化しない決定論的処理のため、1回だけ
# 実行してキャッシュすれば以降のepochで再計算不要になる（RunPod移行に関わらずKaggle継続でも
# 高速化の副次効果あり）。
#
# 保存形式は`kaggle_dinov2_extract_slices_full`のスライス単位npzと同じ構造（images配列が
# DINOv2特徴の代わりに前処理済み画像そのものになる点のみ異なる）。GPU不要のためCPUのみで
# 実行し、Kaggle週間GPU予算を一切消費しない。
#
# 本Notebookは`N_SAMPLE`で全件/ベンチマークを切り替える（まず少数件で時間計測してから
# 全件に切り替える運用。既存の DINOv2 slice抽出ベンチマークと同じ流儀）。

# %%
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


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

import sys as _sys

_sys.path.append(str(ASSETS_DIR))
from preprocessing import preprocess_series  # noqa: E402

OUTPUT_DIR = Path("/kaggle/working")

# %%
# ── パラメータ（train_notebook.py と完全に同一の値を使うこと。ズレるとキャッシュと
#    ライブ処理で学習条件が変わってしまう）
N_SAMPLE = None  # ベンチマーク用。本番実行時は None にして全件処理する
SEED = 42
INPUT_SIZE = 224
FINETUNE_MAX_SLICES_PER_SERIES = 16
CHUNK_SIZE = 200

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


def resize_to_uint8(images_01: np.ndarray, size: int) -> np.ndarray:
    """images_01: (n, H, W) float32 [0,1] → (n, size, size) uint8 [0,255]。

    train_notebook.py の `F.interpolate(..., mode="bilinear")` と挙動を揃えるため、
    ここもPIL(BILINEAR)ではなくOpenCV(INTER_LINEAR)を使う（PILのBILINEARはtorchの
    bilinear実装と厳密には異なるアンチエイリアス処理をするため、torch側と挙動を揃える）。
    """
    import cv2

    out = np.empty((len(images_01), size, size), dtype=np.uint8)
    for i, img in enumerate(images_01):
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        out[i] = np.clip(resized * 255.0, 0, 255).astype(np.uint8)
    return out


def process_study(study_uid: str, series_df: pd.DataFrame):
    """戻り値: per_series_records（各要素が {images(uint8), series_uid, anatomical_plane,
    fluid_sensitive, fat_suppression}）"""
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    if len(study_series) == 0:
        raise ValueError(f"train_series.csv に {study_uid} のseriesがありません")

    records = []
    for _, row in study_series.iterrows():
        series_dir = COMP_DIR / "train_series" / study_uid / row["SeriesInstanceUID"]
        volume = load_series_volume(series_dir)
        processed = preprocess_series(volume, max_slices=FINETUNE_MAX_SLICES_PER_SERIES)  # (n,H,W) float32 [0,1]
        images_u8 = resize_to_uint8(processed, INPUT_SIZE)
        records.append({
            "images": images_u8,
            "series_uid": row["SeriesInstanceUID"],
            "anatomical_plane": row["Anatomical_Plane"],
            "fluid_sensitive": int(row["Fluid_Sensitive"]),
            "fat_suppression": int(row["Fat_Suppression"]),
        })
    return records


# %%
train_df = pd.read_csv(COMP_DIR / "train.csv")
series_df = pd.read_csv(COMP_DIR / "train_series.csv")
all_study_uids = train_df["StudyInstanceUID"].tolist()

if N_SAMPLE is not None:
    rng = np.random.default_rng(SEED)
    all_study_uids = rng.choice(all_study_uids, size=min(N_SAMPLE, len(all_study_uids)), replace=False).tolist()

logger.info("対象件数: %d件 / INPUT_SIZE=%d / FINETUNE_MAX_SLICES_PER_SERIES=%d / chunk_size=%d",
            len(all_study_uids), INPUT_SIZE, FINETUNE_MAX_SLICES_PER_SERIES, CHUNK_SIZE)

# %%
n_done, n_failed = 0, 0
failed_uids: list[dict] = []
t_all_start = time.time()

for chunk_start in range(0, len(all_study_uids), CHUNK_SIZE):
    chunk_uids = all_study_uids[chunk_start: chunk_start + CHUNK_SIZE]
    chunk_no = chunk_start // CHUNK_SIZE + 1
    n_chunks = (len(all_study_uids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info("[chunk %d/%d] %d件を処理中...", chunk_no, n_chunks, len(chunk_uids))
    t_chunk_start = time.time()

    chunk_images = []
    chunk_slice_to_series = []
    chunk_series_study_uid = []
    chunk_series_instance_uid = []
    chunk_series_anatomical_plane = []
    chunk_series_fluid_sensitive = []
    chunk_series_fat_suppression = []
    chunk_study_uids = []
    series_counter = 0

    for study_uid in chunk_uids:
        try:
            records = process_study(study_uid, series_df)
        except Exception as e:  # noqa: BLE001 — 4,407件規模のバッチジョブで1件のために全体を落とさない（本番実行と同じ方針）
            n_failed += 1
            failed_uids.append({"study_uid": study_uid, "error": str(e)})
            logger.warning("study処理失敗（スキップ）: %s | %s\n%s", study_uid, e, traceback.format_exc())
            continue

        chunk_study_uids.append(study_uid)
        for rec in records:
            n_slices = rec["images"].shape[0]
            chunk_images.append(rec["images"])
            chunk_slice_to_series.append(np.full(n_slices, series_counter, dtype=np.int32))
            chunk_series_study_uid.append(study_uid)
            chunk_series_instance_uid.append(rec["series_uid"])
            chunk_series_anatomical_plane.append(rec["anatomical_plane"])
            chunk_series_fluid_sensitive.append(rec["fluid_sensitive"])
            chunk_series_fat_suppression.append(rec["fat_suppression"])
            series_counter += 1
        n_done += 1

    if chunk_study_uids:
        chunk_path = OUTPUT_DIR / f"preprocess_cache_chunk_{chunk_no:04d}.npz"
        np.savez_compressed(
            chunk_path,
            images=np.concatenate(chunk_images, axis=0),
            slice_to_series=np.concatenate(chunk_slice_to_series, axis=0),
            series_study_uid=np.array(chunk_series_study_uid),
            series_instance_uid=np.array(chunk_series_instance_uid),
            series_anatomical_plane=np.array(chunk_series_anatomical_plane),
            series_fluid_sensitive=np.array(chunk_series_fluid_sensitive, dtype=np.int32),
            series_fat_suppression=np.array(chunk_series_fat_suppression, dtype=np.int32),
            study_uids=np.array(chunk_study_uids),
        )
        file_size_mb = chunk_path.stat().st_size / 1e6
        logger.info("[chunk %d/%d] 保存: %s (%.1fMB)", chunk_no, n_chunks, chunk_path.name, file_size_mb)

    chunk_sec = time.time() - t_chunk_start
    elapsed = time.time() - t_all_start
    processed_so_far = chunk_start + len(chunk_uids)
    eta_sec = (elapsed / processed_so_far) * (len(all_study_uids) - processed_so_far) if processed_so_far else 0
    logger.info(
        "[chunk %d/%d] 完了: %.1fs（1件 %.2fs）| 累計成功 %d件 | 失敗 %d件 | ETA %.1f分",
        chunk_no, n_chunks, chunk_sec, chunk_sec / len(chunk_uids), n_done, n_failed, eta_sec / 60,
    )

t_all_total = time.time() - t_all_start

# %%
manifest = {
    "n_sample_param": N_SAMPLE,
    "input_size": INPUT_SIZE,
    "finetune_max_slices_per_series": FINETUNE_MAX_SLICES_PER_SERIES,
    "chunk_size": CHUNK_SIZE,
    "n_total": len(all_study_uids),
    "n_done": n_done,
    "n_failed": n_failed,
    "failed_uids": failed_uids,
    "total_sec": t_all_total,
    "mean_sec_per_study": t_all_total / len(all_study_uids) if all_study_uids else 0,
    "estimated_full_4407_hours": (t_all_total / len(all_study_uids)) * 4407 / 3600 if all_study_uids else 0,
}
with open(OUTPUT_DIR / "preprocess_cache_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
logger.info("manifest: %s", json.dumps({k: v for k, v in manifest.items() if k != "failed_uids"}, indent=2))
print(json.dumps({k: v for k, v in manifest.items() if k != "failed_uids"}, indent=2))
