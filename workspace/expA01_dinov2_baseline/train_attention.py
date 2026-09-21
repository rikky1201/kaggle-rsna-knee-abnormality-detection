"""
attention pooling版 分類ヘッドの学習エントリポイント。

`train.py`（study単位mean+max特徴, data/dinov2_base_features_4405.npz）との違いは
slice→series集約のみ: `GatedAttentionPooling`（学習可能）を使う（series→studyの7スロット
集約は変更しない）。比較を公平にするため、損失関数・pos_weight・ソフトターゲット・
fold設計・optimizer/schedulerはmean+max版と同一にしている。

キャッシュ済みスライス単位DINOv2特徴（data/dinov2_base_slice_features_4407.npz）のみを使う。
DICOM・DINOv2本体には一切アクセスしない。

Usage:
    bash run_attention.sh data.fold=0
    bash run_attention.sh data.fold=1 trainer.max_epochs=50
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.attention_dataset import SliceFeatureStore  # noqa: E402
from src.attention_model import AttentionStudyModel  # noqa: E402
from src.data_prep import LABELS, load_training_data_slices, split_fold  # noqa: E402
from src.loss import SoftTargetBCELoss  # noqa: E402
from src.model import FeatureStandardizer  # noqa: E402

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


def iterate_batches(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator) -> list[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    return [idx[start:start + batch_size] for start in range(0, n, batch_size)]


def run_epoch(
    model: AttentionStudyModel,
    study_uids: np.ndarray,
    soft_targets: np.ndarray,
    store: SliceFeatureStore,
    standardized_features: np.ndarray | None,
    device: torch.device,
    use_amp: bool,
    criterion: SoftTargetBCELoss,
    optimizer: torch.optim.Optimizer | None,
    grad_scaler: torch.amp.GradScaler | None,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """1 epoch分を実行する。optimizer が None なら評価モード（勾配計算なし）。

    Returns: (mean_loss, 全study分のlogits(n, 12))
    """
    is_train = optimizer is not None
    model.train(is_train)

    n = len(study_uids)
    all_logits = np.zeros((n, len(LABELS)), dtype=np.float32)
    losses = []

    for batch_idx in iterate_batches(n, batch_size, shuffle=is_train, rng=rng):
        with torch.set_grad_enabled(is_train):
            if is_train:
                optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                batch_logits = []
                for i in batch_idx:
                    series_list = store.get_study_series(str(study_uids[i]), features_override=standardized_features)
                    batch_logits.append(model.forward_study(series_list, device))
                logits = torch.stack(batch_logits, dim=0)
                targets = torch.from_numpy(soft_targets[batch_idx]).float().to(device)
                loss = criterion(logits, targets)

            if is_train:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

        all_logits[batch_idx] = logits.detach().float().cpu().numpy()
        losses.append(loss.item())

    return float(np.mean(losses)), all_logits


def main() -> None:
    cli_overrides = OmegaConf.from_dotlist(sys.argv[1:])
    base_cfg = OmegaConf.load(EXP_DIR / "config_attention.yaml")
    cfg = OmegaConf.merge(base_cfg, cli_overrides)

    fold = int(cfg.data.fold)
    output_dir = resolve_output_dir(EXP_DIR / cfg.paths.output_dir, cfg.experiment.name, fold)
    setup_logging(output_dir)
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))

    shutil.copy(EXP_DIR / "config_attention.yaml", output_dir / "config_attention.yaml")
    seed_everything(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))

    device = torch.device("cuda" if (cfg.trainer.accelerator == "auto" and torch.cuda.is_available()) else "cpu")
    use_amp = (cfg.trainer.precision == "16-mixed") and (device.type == "cuda")
    logger.info("device=%s, AMP=%s", device, use_amp)

    logger.info("スライス特徴ストアを読み込みます: %s", cfg.data.slice_features_npz)
    store = SliceFeatureStore((EXP_DIR / cfg.data.slice_features_npz).resolve())
    available_study_uids = set(store.study_to_series.keys())
    logger.info("ストア内study数: %d, series数: %d, slice数: %d, embed_dim: %d",
                len(available_study_uids), len(store.series_study_uid), store.slice_features.shape[0], store.embed_dim)

    data = load_training_data_slices(
        labels_csv=(EXP_DIR / cfg.data.labels_csv).resolve(),
        reliability_csv=(EXP_DIR / cfg.data.reliability_csv).resolve(),
        folds_csv=(EXP_DIR / cfg.data.folds_csv).resolve(),
        available_study_uids=available_study_uids,
    )
    train_idx, val_idx = split_fold(data, fold)
    logger.info("fold=%d: train=%d件（gold+pseudo）, val=%d件（gold held-out のみ）", fold, len(train_idx), len(val_idx))

    train_study_uids = data.study_uids[train_idx]
    val_study_uids = data.study_uids[val_idx]

    standardized_features = None
    if cfg.model.standardize:
        logger.info("train側studyのスライスのみで標準化統計量を算出します")
        train_slice_mask = store.slice_mask_for_studies(train_study_uids.tolist())
        scaler = FeatureStandardizer()
        scaler.fit(store.slice_features[train_slice_mask])
        standardized_features = scaler.transform(store.slice_features).astype(np.float32)
    else:
        scaler = None

    model = AttentionStudyModel(
        embed_dim=int(cfg.model.embed_dim), num_classes=int(cfg.model.num_classes),
        attn_hidden_dim=int(cfg.model.attn_hidden_dim),
    ).to(device)
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

    val_hard_labels = data.hard_labels[val_idx]
    val_soft_targets = data.soft_targets[val_idx]
    train_soft_targets = data.soft_targets[train_idx]
    batch_size = int(cfg.trainer.batch_size)

    for epoch in range(start_epoch, int(cfg.trainer.max_epochs)):
        train_loss, _ = run_epoch(
            model, train_study_uids, train_soft_targets, store, standardized_features,
            device, use_amp, criterion, optimizer, grad_scaler, batch_size, rng,
        )
        scheduler.step()

        val_loss, val_logits = run_epoch(
            model, val_study_uids, val_soft_targets, store, standardized_features,
            device, use_amp, criterion, None, None, batch_size, rng,
        )
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        label_aucs = compute_label_aucs(val_hard_labels, val_probs)
        macro_auc = float(np.nanmean(list(label_aucs.values())))

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
            "scaler_mean": scaler.mean_ if scaler is not None else None,
            "scaler_std": scaler.std_ if scaler is not None else None,
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
