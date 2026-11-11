"""
ResNet34(ImageNet) フルfine-tuning — RunPod本番学習（前処理キャッシュ版）。

Kaggle版(`kaggle_train/train_notebook.py`)との違い:
- DICOM読み込み・前処理を毎epoch行わず、事前生成済みの前処理キャッシュ（224x224 uint8,
  `kaggle_preprocess_cache/`で生成）を読み込むだけ。DICOM I/Oが完全に消える
- Kaggle Notebookのセッション時間上限（実質約9時間）が無いため「1 push = 1 epoch」制約も無く、
  複数epochを1プロセスで連続実行する
- CLI側の自動停止フラグ(`--stop-after`)がrunpodctlから廃止されているため、
  `--max-hours`でスクリプト自身が経過時間を監視し、超過したら安全に epoch境界で停止する
  （呼び出し側は別途 `runpodctl pod stop` を必ず実行すること。README `tools/runpod/README.md` §8参照）
- 学習再開はKaggle版と同じ`last.ckpt`フォーマットをそのまま読み書きする（epoch0-2はKaggle上で
  実行済みのため、epoch2の`last.ckpt`を`--resume-ckpt`で渡して epoch3 から再開する）

Usage:
    python train_runpod.py \
        --cache-dir /workspace/cache --assets-dir /workspace/assets \
        --ckpt-dir /workspace/ckpt --resume-ckpt /workspace/resume/last.ckpt \
        --target-epochs 7 --max-hours 20 \
        --kaggle-ckpt-dataset rikky1201/resnet34-finetune-runpod-ckpt
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
VAL_FOLD = 0
INPUT_SIZE = 224
ACCUM_STUDIES = 8
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
NUM_CLASSES = 12
N_SLOTS = 7
SEED = 42
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


class ImageCacheStore:
    """`kaggle_preprocess_cache/`が生成したチャンクnpz群を読み込み、study_uid単位で
    series群（前処理済み画像+メタデータ）を引けるようにする。全チャンクをメモリに
    展開する（RunPod podは十分なRAMを持つ前提。SliceFeatureStore(expA01)と同じ設計）。
    """

    def __init__(self, cache_dir: Path):
        chunk_files = sorted(cache_dir.glob("preprocess_cache_chunk_*.npz"))
        if not chunk_files:
            raise FileNotFoundError(f"キャッシュチャンクが見つかりません: {cache_dir}")
        logger.info("%d 個のキャッシュチャンクを読み込みます", len(chunk_files))

        # ── 2パス読み込み（メモリ節約）: 全チャンクをPythonリストに保持してから
        #    np.concatenate すると、一時的に「元データ+結合後データ」の2倍のメモリが必要になり、
        #    低RAM環境（RunPodのGPU tierによってはcgroup上限が32GB程度）でOOM killされる
        #    （2026-09-25実機確認: 86GB RAMのpodでは問題なかったが35GB RAMのpodでOOM killされた）。
        #    そのため1パス目でスライス総数だけ数え、2パス目で事前確保した配列に直接書き込む
        #    （ピークメモリは「最終配列(約13GB) + チャンク1個分(数百MB)」に抑えられる）
        logger.info("1パス目: スライス総数をカウント中...")
        chunk_slice_counts = []
        for f in chunk_files:
            data = np.load(f)
            chunk_slice_counts.append(data["images"].shape[0])
            del data
        total_slices = sum(chunk_slice_counts)
        embed_size = int(np.load(chunk_files[0])["images"].shape[1])
        self.images = np.empty((total_slices, embed_size, embed_size), dtype=np.uint8)
        logger.info("2パス目: %d スライス分の配列へ書き込み中...", total_slices)

        all_slice_to_series = []
        all_series_study_uid, all_series_instance_uid = [], []
        all_series_anatomical_plane, all_series_fluid_sensitive, all_series_fat_suppression = [], [], []
        series_offset = 0
        slice_offset = 0

        for f, n_slices_this_chunk in zip(chunk_files, chunk_slice_counts):
            data = np.load(f)
            n_series_this_chunk = len(data["series_study_uid"])
            self.images[slice_offset:slice_offset + n_slices_this_chunk] = data["images"]
            slice_offset += n_slices_this_chunk
            all_slice_to_series.append(data["slice_to_series"].astype(np.int64) + series_offset)
            all_series_study_uid.append(data["series_study_uid"])
            all_series_instance_uid.append(data["series_instance_uid"])
            all_series_anatomical_plane.append(data["series_anatomical_plane"])
            all_series_fluid_sensitive.append(data["series_fluid_sensitive"])
            all_series_fat_suppression.append(data["series_fat_suppression"])
            series_offset += n_series_this_chunk
            del data

        slice_to_series = np.concatenate(all_slice_to_series, axis=0).astype(np.int32)
        self.series_study_uid = np.concatenate(all_series_study_uid, axis=0)
        self.series_instance_uid = np.concatenate(all_series_instance_uid, axis=0)
        self.series_anatomical_plane = np.concatenate(all_series_anatomical_plane, axis=0)
        self.series_fluid_sensitive = np.concatenate(all_series_fluid_sensitive, axis=0)
        self.series_fat_suppression = np.concatenate(all_series_fat_suppression, axis=0)

        n_series = len(self.series_study_uid)
        counts = np.bincount(slice_to_series, minlength=n_series)
        if counts.sum() != len(slice_to_series):
            raise AssertionError("slice_to_seriesの値がseries数の範囲外です（データ破損の可能性）")
        offsets = np.concatenate([[0], np.cumsum(counts)])
        expected = np.repeat(np.arange(n_series), counts)
        if not np.array_equal(slice_to_series, expected):
            raise AssertionError("slice_to_seriesがseries単位で連続していません（キャッシュ生成側の前提が崩れています）")
        self._series_slice_range = list(zip(offsets[:-1].tolist(), offsets[1:].tolist()))

        self.study_to_series: dict[str, list[int]] = {}
        for series_idx, study_uid in enumerate(self.series_study_uid):
            self.study_to_series.setdefault(str(study_uid), []).append(series_idx)

        logger.info("キャッシュ読み込み完了: study=%d, series=%d, slice=%d",
                    len(self.study_to_series), n_series, self.images.shape[0])

    def get_study_series(self, study_uid: str) -> list[dict]:
        series_idx_list = self.study_to_series.get(study_uid)
        if not series_idx_list:
            raise KeyError(f"study_uid={study_uid} がキャッシュに存在しません")
        out = []
        for idx in series_idx_list:
            start, end = self._series_slice_range[idx]
            out.append({
                "images": self.images[start:end],  # (n_slices, 224, 224) uint8
                "anatomical_plane": str(self.series_anatomical_plane[idx]),
                "fluid_sensitive": int(self.series_fluid_sensitive[idx]),
                "fat_suppression": int(self.series_fat_suppression[idx]),
            })
        return out


def imagenet_preprocess(images_01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=images_01.dtype, device=images_01.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=images_01.dtype, device=images_01.device).view(1, 3, 1, 1)
    return (images_01 - mean) / std


def push_ckpt_to_kaggle(ckpt_dir: Path, dataset_slug: str, message: str) -> None:
    """durability insurance: 学習完了/中断のたびにckptをKaggle datasetへも退避する
    （README `tools/runpod/README.md` の「ckptと良モデルはVolumeだけでなくKaggle(/HF)にも逃がす」原則）。
    失敗しても学習は継続する（ネットワーク一時失敗等はログのみ残しリトライは次epochに委ねる）。

    初回呼び出し時（dataset-metadata.jsonが無い）は `kaggle datasets create` で新規作成し、
    以降は `kaggle datasets version` で上書きする。
    """
    meta_path = ckpt_dir / "dataset-metadata.json"
    is_first = not meta_path.exists()
    if is_first:
        meta_path.write_text(json.dumps({
            "title": "ResNet34 finetune runpod ckpt",
            "id": dataset_slug,
            "subtitle": "Running checkpoint from RunPod ResNet34 finetune",
            "description": "last.ckpt/best.ckpt/training_log.json, updated every epoch during RunPod training.",
            "isPrivate": True,
            "licenses": [{"name": "CC0-1.0"}],
            "keywords": [], "collaborators": [], "data": [],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = ["kaggle", "datasets", "create" if is_first else "version", "-p", str(ckpt_dir), "--dir-mode", "skip"]
    if not is_first:
        cmd += ["-m", message]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        logger.info("ckptをKaggle dataset(%s)へpushしました", dataset_slug)
    except subprocess.CalledProcessError as e:
        logger.warning("Kaggleへのckpt push失敗（学習は継続）: %s", e.stderr[-500:] if e.stderr else e)
    except subprocess.TimeoutExpired:
        logger.warning("Kaggleへのckpt pushがタイムアウト（学習は継続）")


def compute_label_aucs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    aucs = {}
    for i, label in enumerate(LABELS):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            aucs[label] = float("nan")
            continue
        aucs[label] = float(roc_auc_score(yt, y_prob[:, i]))
    return aucs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--assets-dir", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--resume-ckpt", type=Path, default=None)
    ap.add_argument("--target-epochs", type=int, default=7)
    ap.add_argument("--max-hours", type=float, default=None, help="累計経過時間がこれを超えたらepoch境界で安全停止する")
    ap.add_argument("--kaggle-ckpt-dataset", type=str, default=None, help="指定時、epoch完了ごとにckptをこのKaggle datasetへpush")
    ap.add_argument("--debug-max-train-studies", type=int, default=None, help="スモークテスト用。学習件数を絞る")
    ap.add_argument("--convergence-patience", type=int, default=None,
                     help="指定時、train_lossが--convergence-min-delta以上改善しないepochがこの回数連続したら収束とみなし安全停止する")
    ap.add_argument("--convergence-min-delta", type=float, default=0.02,
                     help="train_lossがこれ未満の改善幅なら「改善なし」とみなす（--convergence-patience用）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUが利用できません")
    logger.info("GPU: %s", torch.cuda.get_device_name(0))

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    # ── データ読み込み
    store = ImageCacheStore(args.cache_dir)
    labels_df = pd.read_csv(args.assets_dir / "train_full_labels_4407.csv")
    reliability_df = pd.read_csv(args.assets_dir / "pseudo_label_reliability.csv").set_index("label")
    folds_df = pd.read_csv(args.assets_dir / "folds_gold58.csv")[["StudyInstanceUID", "fold"]]

    merged = labels_df.merge(folds_df, on="StudyInstanceUID", how="left")
    merged["fold"] = merged["fold"].fillna(-1).astype(int)
    merged = merged[merged["StudyInstanceUID"].isin(store.study_to_series.keys())].reset_index(drop=True)

    is_gold = (merged["label_source"] == "gold").to_numpy()
    is_val = (merged["fold"].to_numpy() == VAL_FOLD) & is_gold
    is_train = ~is_val

    train_df_labels = merged[is_train].reset_index(drop=True)
    val_df_labels = merged[is_val].reset_index(drop=True)
    if args.debug_max_train_studies is not None:
        train_df_labels = train_df_labels.sample(n=args.debug_max_train_studies, random_state=SEED).reset_index(drop=True)
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

    gold_all = merged[is_gold]
    gold_labels_arr = gold_all[LABELS].to_numpy(dtype=np.float32)
    pos_count = gold_labels_arr.sum(axis=0)
    neg_count = len(gold_labels_arr) - pos_count
    pos_weight_arr = (neg_count / pos_count).astype(np.float32)

    # ── モデル
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.target_epochs, eta_min=1e-6)
    grad_scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    best_macro_auc = -float("inf")
    history: list[dict] = []

    resume_path = args.resume_ckpt
    if resume_path is None:
        auto = args.ckpt_dir / "last.ckpt"
        resume_path = auto if auto.exists() else None
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cuda", weights_only=False)
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        head.load_state_dict(ckpt["head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_macro_auc = ckpt["best_macro_auc"]
        history = ckpt["history"]
        logger.info("チェックポイントから再開: %s → epoch %d から（best_macro_auc=%.4f）", resume_path, start_epoch, best_macro_auc)
    else:
        logger.info("チェックポイントなし。epoch 0 から新規学習を開始します")

    def forward_slices(images_u8: np.ndarray, train_mode: bool) -> torch.Tensor:
        t = torch.from_numpy(images_u8).to("cuda").float().unsqueeze(1) / 255.0  # (n,1,224,224)
        t = t.repeat(1, 3, 1, 1)
        t = imagenet_preprocess(t)
        grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with grad_ctx, torch.amp.autocast("cuda", dtype=torch.float16):
            feats = backbone(t)
        return feats.float()

    def forward_study(study_uid: str, train_mode: bool) -> torch.Tensor:
        series_list = store.get_study_series(study_uid)
        slot_features: dict[str, list[torch.Tensor]] = {k: [] for k in SLOT_KEYS}
        slot_fat_suppression: dict[str, list[float]] = {k: [] for k in SLOT_KEYS}

        for s in series_list:
            slice_feats = forward_slices(s["images"], train_mode)
            series_feat = torch.cat([slice_feats.mean(dim=0), slice_feats.max(dim=0).values])
            key = slot_key_for_series(s["anatomical_plane"], s["fluid_sensitive"])
            slot_features[key].append(series_feat)
            slot_fat_suppression[key].append(float(s["fat_suppression"]))

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

    train_uids = train_df_labels["StudyInstanceUID"].to_numpy()
    t_run_start = time.time()

    for epoch in range(start_epoch, args.target_epochs):
        logger.info("=== epoch %d/%d 開始 ===", epoch, args.target_epochs - 1)
        t_epoch_start = time.time()

        rng = np.random.default_rng(SEED + epoch)
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
                logits = forward_study(study_uid, train_mode=True)
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

            if step % 500 == 0:
                elapsed = time.time() - t_epoch_start
                logger.info(
                    "[epoch %d] %d/%d件処理済み | 直近平均loss=%.4f | 経過%.1f分 | ETA%.1f分",
                    epoch, step, len(perm), np.mean(train_losses[-500:]), elapsed / 60,
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

        # ── 検証
        backbone.eval()
        head.eval()
        val_uids = val_df_labels["StudyInstanceUID"].to_numpy()
        val_probs = np.zeros((len(val_uids), NUM_CLASSES), dtype=np.float32)
        for i, study_uid in enumerate(val_uids):
            logits = forward_study(study_uid, train_mode=False)
            val_probs[i] = torch.sigmoid(logits).detach().cpu().numpy()

        label_aucs = compute_label_aucs(val_hard_labels, val_probs)
        macro_auc = float(np.nanmean(list(label_aucs.values())))
        logger.info("=== epoch %d val macro AUC = %.4f ===", epoch, macro_auc)

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
        torch.save(ckpt_state, args.ckpt_dir / "last.ckpt")
        if is_best:
            torch.save(ckpt_state, args.ckpt_dir / "best.ckpt")
            logger.info("best.ckpt を更新しました（macro_auc=%.4f）", macro_auc)

        with open(args.ckpt_dir / "training_log.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        if oom_studies:
            with open(args.ckpt_dir / f"oom_studies_epoch{epoch}.json", "w", encoding="utf-8") as f:
                json.dump(oom_studies, f, ensure_ascii=False, indent=2)

        if args.kaggle_ckpt_dataset:
            push_ckpt_to_kaggle(args.ckpt_dir, args.kaggle_ckpt_dataset, f"epoch{epoch} macro_auc={macro_auc:.4f}")

        total_elapsed_h = (time.time() - t_run_start) / 3600
        logger.info("累計経過時間: %.2fh", total_elapsed_h)
        if args.max_hours is not None and total_elapsed_h >= args.max_hours:
            logger.info("--max-hours(%.1fh)に到達したため安全停止します。残りepochは次回起動時に再開してください", args.max_hours)
            break

        if args.convergence_patience is not None and len(history) > args.convergence_patience:
            recent_losses = [h["train_loss"] for h in history[-(args.convergence_patience + 1):]]
            best_recent = min(recent_losses[:-1])  # patience窓の直前までの最小値
            improvements = [best_recent - loss for loss in recent_losses[1:]]
            if max(improvements) < args.convergence_min_delta:
                logger.info(
                    "train_lossが直近%depoch、%.4f未満の改善幅（収束基準）にとどまったため安全停止します（%s）",
                    args.convergence_patience, args.convergence_min_delta,
                    [round(h["train_loss"], 4) for h in history[-(args.convergence_patience + 1):]],
                )
                break

    logger.info("=== 学習ループ終了。best_macro_auc=%.4f ===", best_macro_auc)
    print(json.dumps({"final_epoch": history[-1]["epoch"] if history else start_epoch - 1,
                       "best_macro_auc": best_macro_auc}, indent=2))


if __name__ == "__main__":
    main()
