# %% [markdown]
# # ResNet-34(ImageNet) フルfine-tuning — 本番学習（1 push = 1 epoch）
#
# expA01のDINOv2凍結+linear probe（LB 0.688）に対し、バックボーンをfine-tuningする
# 実験（ユーザー指示、2026-09-20）。5候補のベンチマーク比較（`expA01_dinov2_baseline/
# kaggle_*_finetune_benchmark/`）でResNet-34(ImageNet事前学習, 全層解凍)を採用。
#
# fold戦略: 4fold CVだと週間30h予算で1.83epoch/foldしか回らないため、単一train/val splitに
# 集中する（ユーザー判断）。`folds_gold58.csv`のfold=0（gold13件）をval、残りgold45件+
# pseudo4,349件=4,394件をtrainに使う。
#
# **Kaggleセッション制約（1セッション実行上限 約9時間）への対応**: 1epoch ≈ 4.1時間
# （ベンチマーク実績）のため、**1回のpushで1epochだけ**学習し、`last.ckpt`を出力する。
# 次回pushでは`kernel_sources`に自分自身（このkernel）を追加することで、前回の出力
# （`/kaggle/input/<このkernelのslug>/last.ckpt`）を読み込んで学習を再開する
# （Kaggle Notebookの自己参照chaining）。目標epoch数（TARGET_EPOCHS）に達するまで
# 手動で複数回push→完了確認→再pushを繰り返す運用。
#
# pooling・損失関数・前処理はexpA01のパイプラインを踏襲し、バックボーンのfine-tuning
# 効果だけを見る（変数を増やさない）。

# %%
import gc
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPUが利用できません。kernel-metadata.json の enable_gpu / Accelerator設定を確認してください。")
logger.info("GPU: %s", torch.cuda.get_device_name(0))


# %%
def _find_by_marker(root: Path, marker_relpath: str, required: bool = True) -> Path | None:
    matches = list(root.rglob(marker_relpath))
    if not matches:
        if not required:
            return None
        listing = [str(p.relative_to(root)) for p in root.rglob("*")][:200]
        raise FileNotFoundError(f"'{marker_relpath}' が {root} 配下に見つかりません。中身:\n  " + "\n  ".join(listing))
    return matches[0].parent


INPUT_ROOT = Path("/kaggle/input")
COMP_DIR = _find_by_marker(INPUT_ROOT, "train_series.csv")
ASSETS_DIR = _find_by_marker(INPUT_ROOT, "preprocessing.py")
# 前回pushの出力（last.ckpt）。初回pushはkernel_sourcesが空なので見つからずNoneになる
RESUME_DIR = _find_by_marker(INPUT_ROOT, "last.ckpt", required=False)
logger.info("competition dir: %s", COMP_DIR)
logger.info("assets dir: %s", ASSETS_DIR)
logger.info("resume dir: %s", RESUME_DIR if RESUME_DIR else "(なし。新規学習として開始)")

import sys as _sys

_sys.path.append(str(ASSETS_DIR))
from preprocessing import preprocess_series  # noqa: E402

OUTPUT_DIR = Path("/kaggle/working")

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# %%
# ── 学習パラメータ（ファイル冒頭の定数で管理。Kaggle実行のためconfig.yamlではなくここで管理する
#    ——kernel push はこのファイル単体しか実行環境に持ち込めないため、他のKaggle実行notebook
#    (ベンチマーク・提出notebook)と同じ規約に揃えている）
TARGET_EPOCHS = 7  # 週間30h予算からの見積もり（ResNet34ベンチマーク実績: 4.10h/epoch）
VAL_FOLD = 0  # folds_gold58.csv の fold=0（13件）をvalに固定
INPUT_SIZE = 224
IMG_BATCH_SIZE = 8
FINETUNE_MAX_SLICES_PER_SERIES = 16  # ベンチマークと同一条件
ACCUM_STUDIES = 8
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
NUM_CLASSES = 12
N_SLOTS = 7
SEED = 42
DEBUG_MAX_TRAIN_STUDIES = None  # スモークテスト用（小さい数値にすると学習件数を絞れる）。本番はNone
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ANATOMICAL_PLANES = ["Sagittal", "Coronal", "Axial"]
SLOT_KEYS = [f"{p}_{f}" for p in ANATOMICAL_PLANES for f in [0, 1]] + ["other"]


def slot_key_for_series(plane: str, fluid_sensitive: int) -> str:
    key = f"{plane}_{int(fluid_sensitive)}"
    return key if key in SLOT_KEYS[:-1] else "other"


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

# %%
# ── データ読み込み: labels + reliability(PPV/NPV) + fold定義（train_series.csv/train.csvは
#    競技データ本体、他3ファイルはKaggle assetsデータセットに追加済み）
labels_df = pd.read_csv(ASSETS_DIR / "train_full_labels_4407.csv")
reliability_df = pd.read_csv(ASSETS_DIR / "pseudo_label_reliability.csv").set_index("label")
folds_df = pd.read_csv(ASSETS_DIR / "folds_gold58.csv")[["StudyInstanceUID", "fold"]]

merged = labels_df.merge(folds_df, on="StudyInstanceUID", how="left")
merged["fold"] = merged["fold"].fillna(-1).astype(int)  # pseudo行は-1

is_gold = (merged["label_source"] == "gold").to_numpy()
is_val = (merged["fold"].to_numpy() == VAL_FOLD) & is_gold
is_train = ~is_val

train_df_labels = merged[is_train].reset_index(drop=True)
val_df_labels = merged[is_val].reset_index(drop=True)
if DEBUG_MAX_TRAIN_STUDIES is not None:
    train_df_labels = train_df_labels.sample(n=DEBUG_MAX_TRAIN_STUDIES, random_state=SEED).reset_index(drop=True)
logger.info("train=%d件（gold%d+pseudo%d）, val=%d件（gold, fold=%d）",
            len(train_df_labels), (train_df_labels["label_source"] == "gold").sum(),
            (train_df_labels["label_source"] == "pseudo").sum(), len(val_df_labels), VAL_FOLD)

ppv = reliability_df.loc[LABELS, "ppv"].to_numpy()
npv = reliability_df.loc[LABELS, "npv"].to_numpy()


def compute_soft_targets(df: pd.DataFrame) -> np.ndarray:
    hard = df[LABELS].to_numpy(dtype=np.float32)
    is_gold_row = (df["label_source"] == "gold").to_numpy()
    soft = hard.copy()
    pseudo_mask = ~is_gold_row
    soft[pseudo_mask] = np.where(hard[pseudo_mask] == 1, ppv[np.newaxis, :], 1.0 - npv[np.newaxis, :])
    return soft.astype(np.float32)


train_soft_targets = compute_soft_targets(train_df_labels)
val_hard_labels = val_df_labels[LABELS].to_numpy(dtype=np.float32)

# pos_weight: gold全58件の分布から算出（expA01と同一のロジック。fold分割に依存させない）
gold_all = merged[is_gold]
gold_labels_arr = gold_all[LABELS].to_numpy(dtype=np.float32)
pos_count = gold_labels_arr.sum(axis=0)
neg_count = len(gold_labels_arr) - pos_count
pos_weight_arr = (neg_count / pos_count).astype(np.float32)
logger.info("pos_weight: %s", dict(zip(LABELS, pos_weight_arr.round(3))))

# %%
# ── モデル: ResNet-34(ImageNet) + 7スロットpooling + linear head
from torchvision.models import ResNet34_Weights, resnet34

backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()
backbone = backbone.to("cuda")
for p in backbone.parameters():
    p.requires_grad_(True)

embed_dim = 512
study_feat_dim = N_SLOTS * (2 * embed_dim + 1)
head = nn.Linear(study_feat_dim, NUM_CLASSES).to("cuda")

pos_weight = torch.tensor(pos_weight_arr, dtype=torch.float32, device="cuda")


class SoftTargetBCELoss(nn.Module):
    """expA01 `src/loss.py` と同一のロジック（ラベル軸sum→バッチ平均）。

    ここではstudy単位（バッチ次元なし, shape=(12,)）で1件ずつ呼ぶため、dim=-1で
    ラベル軸を指定する（バッチ次元があってもなくても正しく最終軸に対して合計される）。
    """

    def __init__(self, pw: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pw)

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        per_element = F.binary_cross_entropy_with_logits(logits, soft_targets, pos_weight=self.pos_weight, reduction="none")
        return per_element.sum(dim=-1)


criterion = SoftTargetBCELoss(pos_weight).to("cuda")
optimizer = torch.optim.AdamW([
    {"params": backbone.parameters(), "lr": LR_BACKBONE},
    {"params": head.parameters(), "lr": LR_HEAD},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TARGET_EPOCHS, eta_min=1e-6)
grad_scaler = torch.amp.GradScaler("cuda")

start_epoch = 0
best_macro_auc = -float("inf")
history: list[dict] = []

if RESUME_DIR is not None:
    # best.ckpt は改善epochでしか書き込まれないため、kernel自己参照chainingでは
    # 「今回改善しなければ前回のbest.ckptがこのバージョンの出力に含まれず、次回参照時に
    # 消失する」問題がある。resume開始時に必ずこのセッションの出力へコピーしておき、
    # 後段で改善した場合のみ上書きすることで、chainを通じて常に持ち越されるようにする
    resume_best_path = RESUME_DIR / "best.ckpt"
    if resume_best_path.exists():
        import shutil
        shutil.copy(resume_best_path, OUTPUT_DIR / "best.ckpt")
        logger.info("前回のbest.ckptをこのセッションの出力へ持ち越しました")
    ckpt = torch.load(RESUME_DIR / "last.ckpt", map_location="cuda", weights_only=False)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_macro_auc = ckpt["best_macro_auc"]
    history = ckpt["history"]
    logger.info("チェックポイントから再開します: epoch %d から（best_macro_auc=%.4f）", start_epoch, best_macro_auc)
else:
    logger.info("チェックポイントなし。epoch 0 から新規学習を開始します")

if start_epoch >= TARGET_EPOCHS:
    raise RuntimeError(f"start_epoch({start_epoch}) が TARGET_EPOCHS({TARGET_EPOCHS}) に達しています。既に学習完了済みです。")


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


def forward_slices(images: np.ndarray, train_mode: bool) -> torch.Tensor:
    feats = []
    images_t = torch.from_numpy(images).float().to("cuda")
    grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with grad_ctx:
        for i in range(0, len(images_t), IMG_BATCH_SIZE):
            batch = images_t[i: i + IMG_BATCH_SIZE].unsqueeze(1)
            batch = F.interpolate(batch, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
            batch = batch.repeat(1, 3, 1, 1)
            batch = imagenet_preprocess(batch)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pooled = backbone(batch)
            feats.append(pooled.float())
    return torch.cat(feats, dim=0)


def forward_study(study_uid: str, series_df: pd.DataFrame, train_mode: bool) -> torch.Tensor:
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    if len(study_series) == 0:
        raise ValueError(f"train_series.csv に {study_uid} のseriesがありません")

    slot_features: dict[str, list[torch.Tensor]] = {k: [] for k in SLOT_KEYS}
    slot_fat_suppression: dict[str, list[float]] = {k: [] for k in SLOT_KEYS}

    for _, row in study_series.iterrows():
        series_dir = COMP_DIR / "train_series" / study_uid / row["SeriesInstanceUID"]
        volume = load_series_volume(series_dir)
        processed = preprocess_series(volume, max_slices=FINETUNE_MAX_SLICES_PER_SERIES)
        slice_feats = forward_slices(processed, train_mode)
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


def compute_label_aucs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    aucs = {}
    for i, label in enumerate(LABELS):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            aucs[label] = float("nan")
            continue
        aucs[label] = float(roc_auc_score(yt, y_prob[:, i]))
    return aucs


# %%
train_series_df = pd.read_csv(COMP_DIR / "train_series.csv")
epoch = start_epoch

logger.info("=== epoch %d/%d 開始 ===", epoch, TARGET_EPOCHS - 1)
t_epoch_start = time.time()

train_uids = train_df_labels["StudyInstanceUID"].to_numpy()
rng = np.random.default_rng(SEED + epoch)  # epochごとにシャッフル順を変える
perm = rng.permutation(len(train_uids))

backbone.train()
head.train()
optimizer.zero_grad()
n_success, n_oom = 0, 0
train_losses = []
oom_studies: list[dict] = []

for step, idx in enumerate(perm, start=1):
    study_uid = train_uids[idx]
    try:
        logits = forward_study(study_uid, train_series_df, train_mode=True)
        target = torch.from_numpy(train_soft_targets[idx]).float().to("cuda")
        loss = criterion(logits, target) / ACCUM_STUDIES
        grad_scaler.scale(loss).backward()
        n_success += 1
        train_losses.append(loss.item() * ACCUM_STUDIES)
    except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001-exempt — 具体的な例外型のみ捕捉。回復可能
        logger.warning("OOM（スキップ）: %s | %s", study_uid, e)
        oom_studies.append({"study_uid": str(study_uid), "error": str(e)})
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        n_oom += 1
        continue

    if n_success % ACCUM_STUDIES == 0:
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if step % 200 == 0:
        elapsed = time.time() - t_epoch_start
        logger.info(
            "[epoch %d] %d/%d件処理済み | 直近平均loss=%.4f | 経過%.1f分 | ETA%.1f分",
            epoch, step, len(perm), np.mean(train_losses[-200:]), elapsed / 60,
            (elapsed / step) * (len(perm) - step) / 60,
        )

if n_success % ACCUM_STUDIES != 0:
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)

scheduler.step()
t_epoch_sec = time.time() - t_epoch_start
train_loss_mean = float(np.mean(train_losses)) if train_losses else float("nan")
logger.info("=== epoch %d train完了: %.1f分 | 成功%d件 / OOM%d件 | train_loss=%.4f ===",
            epoch, t_epoch_sec / 60, n_success, n_oom, train_loss_mean)

# %%
# ── 検証（val, gold fold=%d の%d件）
backbone.eval()
head.eval()
val_uids = val_df_labels["StudyInstanceUID"].to_numpy()
val_probs = np.zeros((len(val_uids), NUM_CLASSES), dtype=np.float32)
for i, study_uid in enumerate(val_uids):
    logits = forward_study(study_uid, train_series_df, train_mode=False)
    val_probs[i] = torch.sigmoid(logits).detach().cpu().numpy()

label_aucs = compute_label_aucs(val_hard_labels, val_probs)
macro_auc = float(np.nanmean(list(label_aucs.values())))
logger.info("=== epoch %d val macro AUC = %.4f ===", epoch, macro_auc)
logger.info("ラベル別AUC: %s", {k: round(v, 4) for k, v in label_aucs.items()})

# %%
history.append({
    "epoch": epoch, "train_loss": train_loss_mean, "val_macro_auc": macro_auc,
    "label_aucs": label_aucs, "n_success": n_success, "n_oom": n_oom,
    "epoch_sec": t_epoch_sec, "lr": scheduler.get_last_lr()[0],
    "timestamp": datetime.now().isoformat(),
})

is_best = macro_auc > best_macro_auc
if is_best:
    best_macro_auc = macro_auc

ckpt_state = {
    "epoch": epoch,
    "backbone_state_dict": backbone.state_dict(),
    "head_state_dict": head.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "grad_scaler_state_dict": grad_scaler.state_dict(),
    "best_macro_auc": best_macro_auc,
    "history": history,
    "labels": LABELS,
    "embed_dim": embed_dim,
    "num_classes": NUM_CLASSES,
}
torch.save(ckpt_state, OUTPUT_DIR / "last.ckpt")
if is_best:
    torch.save(ckpt_state, OUTPUT_DIR / "best.ckpt")
    logger.info("best.ckpt を更新しました（macro_auc=%.4f）", macro_auc)

with open(OUTPUT_DIR / "training_log.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)
if oom_studies:
    with open(OUTPUT_DIR / f"oom_studies_epoch{epoch}.json", "w", encoding="utf-8") as f:
        json.dump(oom_studies, f, ensure_ascii=False, indent=2)

remaining = TARGET_EPOCHS - (epoch + 1)
logger.info("=== epoch %d/%d 完了。残りepoch数: %d ===", epoch, TARGET_EPOCHS - 1, remaining)
if remaining > 0:
    logger.info(
        "次のepochを回すには: kernel-metadata.jsonのkernel_sourcesにこのkernel自身を追加してから再push。"
        "(このNotebookは/kaggle/input/<このkernelのslug>/last.ckpt を自動検出して再開します)"
    )
else:
    logger.info("目標epoch数(%d)に到達しました。best_macro_auc=%.4f", TARGET_EPOCHS, best_macro_auc)

print(json.dumps({
    "epoch": epoch, "train_loss": train_loss_mean, "val_macro_auc": macro_auc,
    "best_macro_auc": best_macro_auc, "n_oom": n_oom, "remaining_epochs": remaining,
}, indent=2))
