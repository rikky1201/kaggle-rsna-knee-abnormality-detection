"""
linear probe 分類ヘッドの学習エントリポイント。

キャッシュ済みDINOv2特徴（data/dinov2_base_features_4405.npz）のみを使う。
DICOM・DINOv2本体には一切アクセスしない。

Usage:
    bash run.sh data.fold=0
    bash run.sh data.fold=1 trainer.max_epochs=50
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_prep import LABELS, load_training_data, split_fold  # noqa: E402
from src.dataset import build_dataloader  # noqa: E402
from src.loss import SoftTargetBCELoss  # noqa: E402
from src.model import FeatureStandardizer, LinearProbeHead  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_output_dir(output_dir_root: Path, experiment_name: str, fold: int) -> Path:
    """同名の実験ディレクトリが存在する場合は _001, _002 とナンバリングする。"""
    base = output_dir_root / experiment_name / f"fold{fold}"
    if not base.exists():
        return base
    i = 1
    while True:
        candidate = output_dir_root / f"{experiment_name}_{i:03d}" / f"fold{fold}"
        if not candidate.exists():
            return candidate
        i += 1


def setup_logging(output_dir: Path) -> None:
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)
    logger.info("Log file: %s", log_file)


def compute_label_aucs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """ラベルごとのAUCを計算する。片方のクラスしか無いラベルはNaN（警告ログ）。"""
    aucs = {}
    for i, label in enumerate(LABELS):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            logger.warning("fold検証セットでラベル '%s' が単一クラスのみのためAUC計算不可（NaN扱い）", label)
            aucs[label] = float("nan")
            continue
        aucs[label] = float(roc_auc_score(yt, y_prob[:, i]))
    return aucs


def plot_training_curve(history: list[dict], output_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["val_macro_auc"] for h in history], label="val macro AUC", color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Validation macro AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    cli_overrides = OmegaConf.from_dotlist(sys.argv[1:])
    base_cfg = OmegaConf.load(EXP_DIR / "config.yaml")
    cfg = OmegaConf.merge(base_cfg, cli_overrides)

    fold = int(cfg.data.fold)
    output_dir = resolve_output_dir(EXP_DIR / cfg.paths.output_dir, cfg.experiment.name, fold)
    setup_logging(output_dir)
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))

    shutil.copy(EXP_DIR / "config.yaml", output_dir / "config.yaml")
    seed_everything(int(cfg.seed))

    device = torch.device("cuda" if (cfg.trainer.accelerator == "auto" and torch.cuda.is_available()) else "cpu")
    use_amp = (cfg.trainer.precision == "16-mixed") and (device.type == "cuda")
    logger.info("device=%s, AMP=%s", device, use_amp)

    data = load_training_data(
        labels_csv=(EXP_DIR / cfg.data.labels_csv).resolve(),
        features_npz=(EXP_DIR / cfg.data.features_npz).resolve(),
        reliability_csv=(EXP_DIR / cfg.data.reliability_csv).resolve(),
        folds_csv=(EXP_DIR / cfg.data.folds_csv).resolve(),
    )
    train_idx, val_idx = split_fold(data, fold)
    logger.info("fold=%d: train=%d件（gold+pseudo）, val=%d件（gold held-out のみ）", fold, len(train_idx), len(val_idx))

    scaler_feat = FeatureStandardizer()
    if cfg.model.standardize:
        scaler_feat.fit(data.features[train_idx])
        train_features = scaler_feat.transform(data.features[train_idx])
        val_features = scaler_feat.transform(data.features[val_idx])
    else:
        train_features = data.features[train_idx]
        val_features = data.features[val_idx]

    train_loader = build_dataloader(
        train_features, data.soft_targets[train_idx],
        batch_size=int(cfg.trainer.batch_size), shuffle=True, num_workers=int(cfg.data.num_workers),
    )
    val_loader = build_dataloader(
        val_features, data.soft_targets[val_idx],
        batch_size=int(cfg.trainer.batch_size), shuffle=False, num_workers=int(cfg.data.num_workers),
    )
    val_hard_labels = data.hard_labels[val_idx]

    model = LinearProbeHead(int(cfg.model.input_dim), int(cfg.model.num_classes)).to(device)
    pos_weight = torch.tensor(data.pos_weight, dtype=torch.float32, device=device)
    criterion = SoftTargetBCELoss(pos_weight).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg.trainer.max_epochs), eta_min=float(cfg.scheduler.min_lr),
    )
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    best_macro_auc = -float("inf")
    epochs_since_improve = 0
    history: list[dict] = []

    ckpt_path = cfg.trainer.ckpt_path
    if ckpt_path:
        logger.info("チェックポイントから再開します: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_macro_auc = ckpt["best_macro_auc"]
        history = ckpt["history"]
        logger.info("epoch %d から再開（best_macro_auc=%.4f）", start_epoch, best_macro_auc)

    for epoch in range(start_epoch, int(cfg.trainer.max_epochs)):
        model.train()
        train_losses = []
        for feats, targets in train_loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(feats)
                loss = criterion(logits, targets)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        val_logits_all = []
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(device), targets.to(device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits = model(feats)
                    loss = criterion(logits, targets)
                val_losses.append(loss.item())
                val_logits_all.append(logits.float().cpu().numpy())

        val_probs = 1.0 / (1.0 + np.exp(-np.concatenate(val_logits_all, axis=0)))
        label_aucs = compute_label_aucs(val_hard_labels, val_probs)
        macro_auc = float(np.nanmean(list(label_aucs.values())))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_macro_auc": macro_auc, "label_aucs": label_aucs,
            "lr": scheduler.get_last_lr()[0],
        })
        logger.info(
            "[epoch %3d/%d] train_loss=%.4f val_loss=%.4f val_macro_auc=%.4f",
            epoch, int(cfg.trainer.max_epochs), train_loss, val_loss, macro_auc,
        )

        ckpt_state = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_macro_auc": best_macro_auc, "history": history,
        }
        torch.save(ckpt_state, output_dir / "last.ckpt")

        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            epochs_since_improve = 0
            torch.save(ckpt_state, output_dir / "best.ckpt")
        else:
            epochs_since_improve += 1

        with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if epochs_since_improve >= int(cfg.trainer.early_stopping_patience):
            logger.info("early stopping: %d epoch改善なし（patience=%d）", epochs_since_improve, cfg.trainer.early_stopping_patience)
            break

    plot_training_curve(history, output_dir / "training_curve.png")
    logger.info("=== fold %d 完了: best_val_macro_auc=%.4f ===", fold, best_macro_auc)
    print(json.dumps({"fold": fold, "best_val_macro_auc": best_macro_auc}, indent=2))


if __name__ == "__main__":
    main()
