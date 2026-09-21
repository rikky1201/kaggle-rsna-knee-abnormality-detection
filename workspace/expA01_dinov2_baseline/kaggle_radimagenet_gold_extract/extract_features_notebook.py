# %% [markdown]
# # gold 58件限定: DINOv2 vs RadImageNet ResNet50 バックボーン比較用特徴抽出
#
# 疑似ラベル4,349件には一切触れない小規模な意思決定用の検証。gold labelの58件のみを対象に、
# 既存のDINOv2特徴抽出パイプライン（前処理・pooling）はそのままで、バックボーンだけ
# RadImageNet ResNet50（医療画像事前学習）に差し替えた場合の特徴を新たに抽出する。
# DINOv2側は既にキャッシュ済み特徴（`data/dinov2_base_features_4405.npz`）があるが、
# 本Notebookでは比較対象を完全に揃えるため両バックボーンとも同じ58件・同じ実行環境で
# 再抽出する（DICOM読み込み・前処理コードのバージョン差による揺らぎを排除するため）。
#
# 対象studyの一覧: `folds_gold58.csv`（`workspace/fold/v1_multilabel_gold58/folds.csv`と同一、
# StudyInstanceUID + 12ラベル + fold番号）

# %%
import json
import logging
import sys
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
WEIGHTS_DIR = _find_by_marker(INPUT_ROOT, "RadImageNet-ResNet50_notop.pth")
logger.info("competition dir: %s", COMP_DIR)
logger.info("assets dir: %s", ASSETS_DIR)
logger.info("radimagenet weights dir: %s", WEIGHTS_DIR)

sys.path.append(str(ASSETS_DIR))
from feature_extraction import Dinov2FeatureExtractor  # noqa: E402
from pooling import pool_series_to_study, pool_slices_to_series  # noqa: E402
from preprocessing import preprocess_series  # noqa: E402
from radimagenet_feature_extraction import RadImageNetFeatureExtractor  # noqa: E402

OUTPUT_DIR = Path("/kaggle/working")
BATCH_SIZE = 32

# %%
# ── DICOM読み込み（InstanceNumberでソート。逐次読み込み: 58件のみなので並列化不要）


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


def extract_study_feature(study_uid: str, series_df: pd.DataFrame, extractor) -> np.ndarray:
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
gold_df = pd.read_csv(ASSETS_DIR / "folds_gold58.csv")
series_df = pd.read_csv(COMP_DIR / "train_series.csv")
study_uids = gold_df["StudyInstanceUID"].tolist()
if len(study_uids) != 58:
    raise AssertionError(f"gold件数が58と一致しません: 実際={len(study_uids)}")
logger.info("対象study数: %d", len(study_uids))

# %%
# ── バックボーンごとに抽出（失敗studyはmanifestに記録してスキップ。1件の異常で全体を落とさない）
backbones = {
    "dinov2_base": lambda: Dinov2FeatureExtractor(size="base"),
    "radimagenet_resnet50": lambda: RadImageNetFeatureExtractor(
        weights_path=str(WEIGHTS_DIR / "RadImageNet-ResNet50_notop.pth")
    ),
}

failed_studies: dict[str, dict[str, str]] = {}

for backbone_name, build_extractor in backbones.items():
    logger.info("=== backbone=%s のロード ===", backbone_name)
    extractor = build_extractor()
    logger.info("embed_dim=%d", extractor.embed_dim)

    feats = []
    ok_uids = []
    for i, study_uid in enumerate(study_uids, start=1):
        try:
            feat = extract_study_feature(study_uid, series_df, extractor)
        except Exception as e:  # noqa: BLE001 — 破損DICOM等1件の異常で全体を落とさないため広く捕捉（学習側extract jobと同じ方針）
            logger.exception("[%s][%d/%d] %s の特徴抽出に失敗", backbone_name, i, len(study_uids), study_uid)
            failed_studies.setdefault(backbone_name, {})[study_uid] = f"{type(e).__name__}: {e}"
            continue
        feats.append(feat)
        ok_uids.append(study_uid)
        if i % 10 == 0:
            logger.info("[%s] %d/%d 件処理済み", backbone_name, i, len(study_uids))

    feats_arr = np.stack(feats, axis=0).astype(np.float32)
    out_path = OUTPUT_DIR / f"gold58_features_{backbone_name}.npz"
    np.savez(out_path, study_uids=np.array(ok_uids), features=feats_arr)
    logger.info("=== backbone=%s 完了: %d/%d 件成功, feature_dim=%d, 保存先=%s ===",
                backbone_name, len(ok_uids), len(study_uids), feats_arr.shape[1], out_path)

    del extractor
    torch.cuda.empty_cache()

if failed_studies:
    with open(OUTPUT_DIR / "extraction_failures.json", "w", encoding="utf-8") as f:
        json.dump(failed_studies, f, ensure_ascii=False, indent=2)
    logger.warning("失敗studyがありました: %s", failed_studies)
else:
    logger.info("失敗studyはありませんでした（58/58件成功、両backbone）")

print("完了")
