# %% [markdown]
# # RSNA Knee Abnormality Detection — 提出用推論Notebook（ResNet34 fine-tuning版, v005）
#
# Code Competition submission（`enable_internet: false` 必須）。
#
# expA02: DINOv2凍結+linear probe（v002, LB 0.688）に対し、バックボーンをfine-tuning
# する方向で検証（ユーザー指示）。ResNet34(ImageNet事前学習)を全層fine-tuning。
# Kaggle T4で3epoch（週間GPU予算切れで中断）→ RunPod(RTX4090, 前処理キャッシュ経由)で
# epoch3〜9を追加学習し、train_lossの収束を検知して自動停止。
#
# Source: workspace/expA02_resnet34_finetune/final_ckpt/best.ckpt (epoch7,
#   gold45+pseudo4349で学習, val macro AUC=0.6151)
#
# 依存データセット:
# - `rikky1201/rsna-knee-dinov2-baseline-assets`: preprocessing.py + 学習済み
#   `full_model_resnet34_finetune.ckpt`（ImageNet事前学習重みは不要。fine-tuning済みの
#   全重みがckptに含まれるため、enable_internet:false下でも追加ダウンロード不要）
# - competition_sources: test.csv, test_series.csv, test_series/ のDICOM本体

# %%
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPUが利用できません。kernel-metadata.json の enable_gpu / Accelerator設定を確認してください。")
logger.info("GPU: %s", torch.cuda.get_device_name(0))


# %%
def _find_by_marker(root: Path, marker_relpath: str, max_depth: int = 5) -> Path:
    import os

    skip_dirnames = {"train_series", "test_series"}
    root_depth = len(root.parts)

    if root.exists():
        for dirpath, dirnames, _filenames in os.walk(root):
            dirpath_p = Path(dirpath)
            depth = len(dirpath_p.parts) - root_depth
            if depth >= max_depth:
                dirnames.clear()
                continue
            dirnames[:] = [d for d in dirnames if d not in skip_dirnames]
            if (dirpath_p / marker_relpath).exists():
                return dirpath_p

    logger.warning("深さ%d以内で '%s' が見つからなかったため、フォールバックで全体を再帰探索します（時間がかかる可能性）", max_depth, marker_relpath)
    matches = list(root.rglob(marker_relpath))
    if not matches:
        listing = [str(p.relative_to(root)) for p in root.rglob("*")][:200]
        raise FileNotFoundError(f"'{marker_relpath}' が {root} 配下に見つかりません。中身:\n  " + "\n  ".join(listing))
    return matches[0].parent


INPUT_ROOT = Path("/kaggle/input")
COMP_DIR = _find_by_marker(INPUT_ROOT, "test_series.csv")
ASSETS_DIR = _find_by_marker(INPUT_ROOT, "full_model_resnet34_finetune.ckpt")
logger.info("competition dir: %s", COMP_DIR)
logger.info("assets dir: %s", ASSETS_DIR)

sys.path.append(str(ASSETS_DIR))
from preprocessing import preprocess_series  # noqa: E402

OUTPUT_CSV = Path("/kaggle/working/submission.csv")

# %%
# ── パラメータ（学習時と完全に同一の値を使う）
INPUT_SIZE = 224
FINETUNE_MAX_SLICES_PER_SERIES = 16
IMG_BATCH_SIZE = 8
N_SLOTS = 7
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ANATOMICAL_PLANES = ["Sagittal", "Coronal", "Axial"]
SLOT_KEYS = [f"{p}_{f}" for p in ANATOMICAL_PLANES for f in [0, 1]] + ["other"]


def slot_key_for_series(plane: str, fluid_sensitive: int) -> str:
    key = f"{plane}_{int(fluid_sensitive)}"
    return key if key in SLOT_KEYS[:-1] else "other"


# %%
# ── モデルロード: ImageNet事前学習は使わない（fine-tuning済みの全重みがckptに入っている）
device = torch.device("cuda")
from torchvision.models import resnet34

ckpt = torch.load(ASSETS_DIR / "full_model_resnet34_finetune.ckpt", map_location="cpu", weights_only=False)
LABELS = ckpt["labels"]
logger.info("labels (%d): %s", len(LABELS), LABELS)
logger.info("学習時epoch=%d, val_macro_auc=%s", ckpt["epoch"], ckpt.get("best_macro_auc"))

backbone = resnet34(weights=None)
backbone.fc = nn.Identity()
backbone.load_state_dict(ckpt["backbone_state_dict"])
backbone = backbone.to(device).eval()

embed_dim = 512
study_feat_dim = N_SLOTS * (2 * embed_dim + 1)
head = nn.Linear(study_feat_dim, len(LABELS)).to(device)
head.load_state_dict(ckpt["head_state_dict"])
head.eval()

# ── フォールバック: 破損DICOM等で特徴抽出に失敗したstudyは、head部分に直接ゼロベクトルを
#    通した予測で代替する（series_listが空だとforward_studyが成立しないため、
#    poolingをバイパスする設計。他backbone版の提出Notebookと同じ考え方）
FALLBACK_STUDY_FEATURE = torch.zeros(1, study_feat_dim, device=device)


def predict_fallback() -> np.ndarray:
    with torch.no_grad():
        logits = head(FALLBACK_STUDY_FEATURE)
    return torch.sigmoid(logits).cpu().numpy()[0]


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


def imagenet_preprocess(images_01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=images_01.dtype, device=images_01.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=images_01.dtype, device=images_01.device).view(1, 3, 1, 1)
    return (images_01 - mean) / std


def forward_slices(images_01: np.ndarray) -> torch.Tensor:
    feats = []
    images_t = torch.from_numpy(images_01).float().to(device)
    with torch.no_grad():
        for i in range(0, len(images_t), IMG_BATCH_SIZE):
            batch = images_t[i: i + IMG_BATCH_SIZE].unsqueeze(1)
            batch = F.interpolate(batch, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
            batch = batch.repeat(1, 3, 1, 1)
            batch = imagenet_preprocess(batch)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pooled = backbone(batch)
            feats.append(pooled.float())
    return torch.cat(feats, dim=0)


def predict_study(study_uid: str, series_df: pd.DataFrame) -> np.ndarray:
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    if len(study_series) == 0:
        raise ValueError(f"test_series.csv に {study_uid} のseriesがありません")

    slot_features: dict[str, list[torch.Tensor]] = {k: [] for k in SLOT_KEYS}
    slot_fat_suppression: dict[str, list[float]] = {k: [] for k in SLOT_KEYS}

    for _, row in study_series.iterrows():
        series_dir = COMP_DIR / "test_series" / study_uid / row["SeriesInstanceUID"]
        volume = load_series_volume(series_dir)
        processed = preprocess_series(volume, max_slices=FINETUNE_MAX_SLICES_PER_SERIES)
        slice_feats = forward_slices(processed)
        series_feat = torch.cat([slice_feats.mean(dim=0), slice_feats.max(dim=0).values])

        key = slot_key_for_series(row["Anatomical_Plane"], int(row["Fluid_Sensitive"]))
        slot_features[key].append(series_feat)
        slot_fat_suppression[key].append(float(row["Fat_Suppression"]))

    slot_vectors = []
    for key in SLOT_KEYS:
        feats = slot_features[key]
        if feats:
            pooled = torch.stack(feats, dim=0).mean(dim=0)
            fat_supp = sum(slot_fat_suppression[key]) / len(slot_fat_suppression[key])
        else:
            pooled = torch.zeros(2 * embed_dim, device=device)
            fat_supp = 0.0
        fat_t = torch.tensor([fat_supp], device=device, dtype=pooled.dtype)
        slot_vectors.append(torch.cat([pooled, fat_t]))

    study_feat = torch.cat(slot_vectors)
    with torch.no_grad():
        logits = head(study_feat.unsqueeze(0)).squeeze(0)
    return torch.sigmoid(logits).cpu().numpy()


# %%
test_df = pd.read_csv(COMP_DIR / "test.csv")
test_series_df = pd.read_csv(COMP_DIR / "test_series.csv")
logger.info("test件数: %d", len(test_df))

rows = []
failed_studies: dict[str, str] = {}
for i, study_uid in enumerate(test_df["StudyInstanceUID"], start=1):
    logger.info("[%d/%d] %s を処理中...", i, len(test_df), study_uid)
    try:
        probs = predict_study(study_uid, test_series_df)
    except Exception as e:  # noqa: BLE001 — 破損DICOM/欠損メタデータ等の外部データ起因の障害を1件のみ無効化するため広く捕捉
        logger.exception("[%d/%d] %s の推論に失敗。フォールバック予測（ゼロ特徴）で代替します", i, len(test_df), study_uid)
        failed_studies[study_uid] = f"{type(e).__name__}: {e}"
        probs = predict_fallback()

    row = {"StudyInstanceUID": study_uid}
    for j, label in enumerate(LABELS):
        row[label] = float(probs[j])
    rows.append(row)

if failed_studies:
    logger.warning("%d/%d件のstudyでフォールバック予測を使用しました: %s", len(failed_studies), len(test_df), failed_studies)
    with open("/kaggle/working/inference_failures.json", "w") as f:
        json.dump(failed_studies, f, indent=2, ensure_ascii=False)

submission = pd.DataFrame(rows, columns=["StudyInstanceUID", *LABELS])

# %%
# ── 提出前フォーマット検証（fail-fast）
assert len(submission) == len(test_df), f"行数不一致: submission={len(submission)}, test={len(test_df)}"
assert list(submission.columns) == ["StudyInstanceUID", *LABELS], f"列名/順序が不正: {list(submission.columns)}"
assert submission[LABELS].isna().sum().sum() == 0, "欠損値が含まれています"
assert (submission[LABELS].to_numpy() >= 0).all() and (submission[LABELS].to_numpy() <= 1).all(), "値域[0,1]を超えています"
assert set(submission["StudyInstanceUID"]) == set(test_df["StudyInstanceUID"]), "StudyInstanceUIDが一致しません"

submission.to_csv(OUTPUT_CSV, index=False)
logger.info("submission.csv を保存しました: %s (%d行)", OUTPUT_CSV, len(submission))
print(submission.head())
