# %% [markdown]
# # DINOv2 部分fine-tuning（後半4層のみ凍結解除）— 1epoch相当の時間・GPUメモリ計測
#
# `kaggle_dinov2_finetune_benchmark/`（全パラメータ凍結解除, 86.6M params）との比較用。
# DINOv2-base（ViT, 12 encoder layer）のうち、**embeddings + 前半8層を凍結、
# 後半4層(encoder.layer[8:12]) + 最終layernormのみ凍結解除**する
# （"後半数層のみ" の具体化。約1/3の層を学習対象にする中間的な選択）。
#
# 凍結した層は `requires_grad_(False)` にするだけでよい: forward時、frozen層の出力の
# 手前まではrequires_grad=Falseの葉ノードのみで構成されるため、autogradは自動的に
# それらの活性化を保持しない（グラフノードが作られない）。full fine-tuning版のような
# 明示的なメモリ対策（AMP以外）は不要で、単純に凍結解除する層数を減らすだけで
# 活性化メモリも比例して削減される見込み、というのがこの計測の狙い。
#
# その他の設計（slice間引き, gradient accumulation, AMP, OOMハンドリング）は
# `kaggle_dinov2_finetune_benchmark/`と完全に同一にし、変数を「凍結範囲」だけに絞る。

# %%
import gc
import json
import logging
import time
import traceback
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
logger.info("GPU total memory: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)


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
# ── ベンチマークパラメータ
N_SAMPLE = 100
SEED = 42  # 過去のベンチマークと同一シード（比較可能にするため）
MODEL_SIZE = "base"
MODEL_ID = "facebook/dinov2-base"
IMG_BATCH_SIZE = 8
FINETUNE_MAX_SLICES_PER_SERIES = 16  # full fine-tuning版と同一条件で比較するため揃える
ACCUM_STUDIES = 8
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
NUM_CLASSES = 12
N_SLOTS = 7
N_UNFROZEN_LAYERS = 4  # DINOv2-base(12層)のうち後半何層を凍結解除するか

# %%
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
backbone = AutoModel.from_pretrained(MODEL_ID).to("cuda")
backbone.train()

n_total_layers = len(backbone.encoder.layer)
if N_UNFROZEN_LAYERS > n_total_layers:
    raise ValueError(f"N_UNFROZEN_LAYERS={N_UNFROZEN_LAYERS} が総層数{n_total_layers}を超えています")

# 全パラメータをまず凍結し、後半N_UNFROZEN_LAYERS層 + 最終layernormのみ解凍する
for p in backbone.parameters():
    p.requires_grad_(False)
for layer in backbone.encoder.layer[n_total_layers - N_UNFROZEN_LAYERS:]:
    for p in layer.parameters():
        p.requires_grad_(True)
for p in backbone.layernorm.parameters():
    p.requires_grad_(True)

n_backbone_params_total = sum(p.numel() for p in backbone.parameters())
n_backbone_params_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
embed_dim = backbone.config.hidden_size
logger.info(
    "DINOv2-%s ロード完了: embed_dim=%d, 総層数=%d, 凍結解除層数=%d, trainable_params=%.2fM / total=%.1fM",
    MODEL_SIZE, embed_dim, n_total_layers, N_UNFROZEN_LAYERS,
    n_backbone_params_trainable / 1e6, n_backbone_params_total / 1e6,
)

study_feat_dim = N_SLOTS * (2 * embed_dim + 1)
head = nn.Linear(study_feat_dim, NUM_CLASSES).to("cuda")
n_head_params = sum(p.numel() for p in head.parameters())
logger.info("分類ヘッド: input_dim=%d, params=%d", study_feat_dim, n_head_params)

optimizer = torch.optim.AdamW([
    {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
    {"params": head.parameters(), "lr": LR_HEAD},
])
grad_scaler = torch.amp.GradScaler("cuda")

ANATOMICAL_PLANES = ["Sagittal", "Coronal", "Axial"]
SLOT_KEYS = [f"{p}_{f}" for p in ANATOMICAL_PLANES for f in [0, 1]] + ["other"]


def slot_key_for_series(plane: str, fluid_sensitive: int) -> str:
    key = f"{plane}_{int(fluid_sensitive)}"
    return key if key in SLOT_KEYS[:-1] else "other"


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


def forward_slices_with_grad(images: np.ndarray) -> torch.Tensor:
    images_rgb = np.repeat(images[:, :, :, None], 3, axis=3)
    feats = []
    for i in range(0, len(images_rgb), IMG_BATCH_SIZE):
        batch = list(images_rgb[i: i + IMG_BATCH_SIZE])
        inputs = processor(images=batch, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = backbone(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
        feats.append(cls.float())
    return torch.cat(feats, dim=0)


def forward_study(study_uid: str, series_df: pd.DataFrame) -> torch.Tensor:
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    if len(study_series) == 0:
        raise ValueError(f"train_series.csv に {study_uid} のseriesがありません")

    slot_features: dict[str, list[torch.Tensor]] = {k: [] for k in SLOT_KEYS}
    slot_fat_suppression: dict[str, list[float]] = {k: [] for k in SLOT_KEYS}

    for _, row in study_series.iterrows():
        series_dir = COMP_DIR / "train_series" / study_uid / row["SeriesInstanceUID"]
        volume = load_series_volume(series_dir)
        processed = preprocess_series(volume, max_slices=FINETUNE_MAX_SLICES_PER_SERIES)
        slice_feats = forward_slices_with_grad(processed)
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
            pooled = torch.zeros(2 * embed_dim, device="cuda")
            fat_supp = 0.0
        fat_t = torch.tensor([fat_supp], device="cuda", dtype=pooled.dtype)
        slot_vectors.append(torch.cat([pooled, fat_t]))

    study_feat = torch.cat(slot_vectors)
    return head(study_feat.unsqueeze(0)).squeeze(0)


# %%
train_df = pd.read_csv(COMP_DIR / "train.csv")
series_df = pd.read_csv(COMP_DIR / "train_series.csv")
rng = np.random.default_rng(SEED)
all_study_uids = train_df["StudyInstanceUID"].unique()
sample_study_uids = rng.choice(all_study_uids, size=min(N_SAMPLE, len(all_study_uids)), replace=False)
logger.info(
    "計測サンプル数: %d / N_UNFROZEN_LAYERS=%d / FINETUNE_MAX_SLICES_PER_SERIES=%d / ACCUM_STUDIES=%d",
    len(sample_study_uids), N_UNFROZEN_LAYERS, FINETUNE_MAX_SLICES_PER_SERIES, ACCUM_STUDIES,
)

torch.cuda.reset_peak_memory_stats()

# %%
times = []
oom_studies: list[dict] = []
n_success = 0
optimizer.zero_grad()

t_all_start = time.time()
for i, study_uid in enumerate(sample_study_uids, start=1):
    t0 = time.time()
    try:
        logits = forward_study(study_uid, series_df)
        dummy_target = torch.randint(0, 2, (NUM_CLASSES,), device="cuda").float()
        loss = F.binary_cross_entropy_with_logits(logits, dummy_target) / ACCUM_STUDIES
        grad_scaler.scale(loss).backward()
        n_success += 1
    except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001-exempt — 具体的な例外型のみ捕捉。回復可能（次studyで再試行）
        logger.warning("OOM（スキップ）: %s | %s", study_uid, e)
        oom_studies.append({"study_uid": study_uid, "error": str(e)})
        optimizer.zero_grad(set_to_none=True)
        gc.collect()  # 参照サイクルの明示解放（full fine-tuning版で判明したリーク対策）
        torch.cuda.empty_cache()
        continue

    if n_success % ACCUM_STUDIES == 0:
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad(set_to_none=True)

    elapsed = time.time() - t0
    times.append(elapsed)
    if i % 10 == 0:
        logger.info(
            "%d/%d 件処理済み (直近平均 %.2fs/件, peak_mem=%.2fGB)",
            i, len(sample_study_uids), np.mean(times[-10:]), torch.cuda.max_memory_allocated() / 1e9,
        )

if n_success % ACCUM_STUDIES != 0:
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)

t_all_total = time.time() - t_all_start

# %%
times_arr = np.array(times)
peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

WEEKLY_GPU_BUDGET_HOURS = 30.0
N_TRAIN_FULL = 4390
N_FOLDS = 4

sec_per_study = float(times_arr.mean())
epoch_sec_per_fold = sec_per_study * N_TRAIN_FULL
epoch_hours_per_fold = epoch_sec_per_fold / 3600
epochs_per_fold_in_budget = WEEKLY_GPU_BUDGET_HOURS / (epoch_hours_per_fold * N_FOLDS)
epochs_single_fold_in_budget = WEEKLY_GPU_BUDGET_HOURS / epoch_hours_per_fold

result = {
    "candidate": "A_dinov2_partial_unfreeze_last4",
    "n_sample": len(sample_study_uids),
    "n_success": n_success,
    "n_oom": len(oom_studies),
    "oom_studies": oom_studies,
    "embed_dim": embed_dim,
    "n_total_layers": n_total_layers,
    "n_unfrozen_layers": N_UNFROZEN_LAYERS,
    "backbone_total_params_m": n_backbone_params_total / 1e6,
    "backbone_trainable_params_m": n_backbone_params_trainable / 1e6,
    "finetune_max_slices_per_series": FINETUNE_MAX_SLICES_PER_SERIES,
    "img_batch_size": IMG_BATCH_SIZE,
    "accum_studies": ACCUM_STUDIES,
    "mean_sec_per_study": sec_per_study,
    "median_sec_per_study": float(np.median(times_arr)) if len(times_arr) else None,
    "min_sec_per_study": float(times_arr.min()) if len(times_arr) else None,
    "max_sec_per_study": float(times_arr.max()) if len(times_arr) else None,
    "total_benchmark_sec": t_all_total,
    "peak_gpu_memory_gb": peak_mem_gb,
    "total_gpu_memory_gb": total_mem_gb,
    "peak_memory_utilization_pct": peak_mem_gb / total_mem_gb * 100,
    "estimate": {
        "n_train_studies_per_fold_assumed": N_TRAIN_FULL,
        "epoch_hours_per_fold": epoch_hours_per_fold,
        "epoch_hours_all_4folds": epoch_hours_per_fold * N_FOLDS,
        "weekly_gpu_budget_hours": WEEKLY_GPU_BUDGET_HOURS,
        "epochs_per_fold_if_4fold_equal_split": epochs_per_fold_in_budget,
        "epochs_if_single_fold_only": epochs_single_fold_in_budget,
    },
}
logger.info("計測結果: %s", json.dumps(result, indent=2, ensure_ascii=False))

with open(OUTPUT_DIR / "partial_finetune_benchmark_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, indent=2, ensure_ascii=False))
