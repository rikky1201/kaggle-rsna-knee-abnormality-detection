"""
提出用「フルデータモデル」の学習: gold全58件 + pseudo全件（検証データを割かない）。

4fold CVの目的は「このパイプラインが機能するか」の検証であり、実際の提出には
全ラベル付きデータを使い切ったモデルを使うのが筋が良い（CVは既に完了・記録済み:
平均val macro AUC=0.7501）。固定epoch数は4fold CVの best_epoch の中央値
（0, 5, 5, 11 → 中央値5）を採用する（検証データが無いため early stopping できない）。

推論に必要な標準化統計量（train全件のmean/std）もチェックポイントに含めて保存する。

Usage:
    bash run.sh --full
    または
    python train_full.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_prep import LABELS, load_training_data  # noqa: E402
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


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"train_full_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def main() -> None:
    cli_overrides = OmegaConf.from_dotlist([a for a in sys.argv[1:] if a != "--full"])
    base_cfg = OmegaConf.load(EXP_DIR / "config.yaml")
    cfg = OmegaConf.merge(base_cfg, cli_overrides)

    output_dir = EXP_DIR / cfg.paths.output_dir / cfg.experiment.name / "full"
    setup_logging(output_dir)
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))
    shutil.copy(EXP_DIR / "config.yaml", output_dir / "config.yaml")
    seed_everything(int(cfg.seed))

    device = torch.device("cuda" if (cfg.trainer.accelerator == "auto" and torch.cuda.is_available()) else "cpu")
    use_amp = (cfg.trainer.precision == "16-mixed") and (device.type == "cuda")
    fixed_epochs = int(cfg.trainer.full_fixed_epochs)
    logger.info("device=%s, AMP=%s, fixed_epochs=%d", device, use_amp, fixed_epochs)

    data = load_training_data(
        labels_csv=(EXP_DIR / cfg.data.labels_csv).resolve(),
        features_npz=(EXP_DIR / cfg.data.features_npz).resolve(),
        reliability_csv=(EXP_DIR / cfg.data.reliability_csv).resolve(),
        folds_csv=(EXP_DIR / cfg.data.folds_csv).resolve(),
    )
    logger.info("学習データ全件: %d件（gold+pseudo, 検証データなし）", len(data.study_uids))

    scaler = FeatureStandardizer()
    if cfg.model.standardize:
        scaler.fit(data.features)
        train_features = scaler.transform(data.features)
    else:
        train_features = data.features

    train_loader = build_dataloader(
        train_features, data.soft_targets,
        batch_size=int(cfg.trainer.batch_size), shuffle=True, num_workers=int(cfg.data.num_workers),
    )

    model = LinearProbeHead(int(cfg.model.input_dim), int(cfg.model.num_classes)).to(device)
    pos_weight = torch.tensor(data.pos_weight, dtype=torch.float32, device=device)
    criterion = SoftTargetBCELoss(pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay))
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = []
    for epoch in range(fixed_epochs):
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
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss})
        logger.info("[epoch %d/%d] train_loss=%.4f", epoch, fixed_epochs, train_loss)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.std_,
        "standardize": bool(cfg.model.standardize),
        "labels": LABELS,
        "input_dim": int(cfg.model.input_dim),
        "num_classes": int(cfg.model.num_classes),
        "fixed_epochs": fixed_epochs,
    }
    torch.save(ckpt, output_dir / "full_model.ckpt")
    with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    logger.info("フルデータモデルを保存しました: %s", output_dir / "full_model.ckpt")


if __name__ == "__main__":
    main()
