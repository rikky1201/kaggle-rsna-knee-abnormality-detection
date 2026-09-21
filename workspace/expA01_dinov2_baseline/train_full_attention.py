"""
提出用「フルデータモデル」の学習（attention pooling版）: gold全58件 + pseudo全件
（検証データを割かない）。

4fold CVの目的は「このパイプラインが機能するか」の検証であり、実際の提出には
全ラベル付きデータを使い切ったモデルを使うのが筋が良い（CVは既に完了・記録済み:
pooled OOF macro AUC=0.7369, mean+max版との有意差なし）。固定epoch数は4fold CVの
best_epoch の中央値（19, 5, 27, 8 → 中央値13.5 → 14）を採用する
（検証データが無いため early stopping できない）。

推論に必要な標準化統計量（train全件のスライスから算出したmean/std）もチェックポイントに
含めて保存する。

Usage:
    python train_full_attention.py
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
from src.attention_dataset import SliceFeatureStore  # noqa: E402
from src.attention_model import AttentionStudyModel  # noqa: E402
from src.data_prep import LABELS, load_training_data_slices  # noqa: E402
from src.loss import SoftTargetBCELoss  # noqa: E402
from src.model import FeatureStandardizer  # noqa: E402
from src.pooling import study_feature_dim  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

FIXED_EPOCHS = 14  # 4fold CVのbest_epoch中央値（19,5,27,8 → 13.5 → 14）


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


def iterate_batches(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    return [idx[start:start + batch_size] for start in range(0, n, batch_size)]


def main() -> None:
    cli_overrides = OmegaConf.from_dotlist(sys.argv[1:])
    base_cfg = OmegaConf.load(EXP_DIR / "config_attention.yaml")
    cfg = OmegaConf.merge(base_cfg, cli_overrides)

    output_dir = EXP_DIR / cfg.paths.output_dir / cfg.experiment.name / "full"
    setup_logging(output_dir)
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))
    shutil.copy(EXP_DIR / "config_attention.yaml", output_dir / "config_attention.yaml")
    seed_everything(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))

    device = torch.device("cuda" if (cfg.trainer.accelerator == "auto" and torch.cuda.is_available()) else "cpu")
    use_amp = (cfg.trainer.precision == "16-mixed") and (device.type == "cuda")
    logger.info("device=%s, AMP=%s, fixed_epochs=%d", device, use_amp, FIXED_EPOCHS)

    store = SliceFeatureStore((EXP_DIR / cfg.data.slice_features_npz).resolve())
    available_study_uids = set(store.study_to_series.keys())

    data = load_training_data_slices(
        labels_csv=(EXP_DIR / cfg.data.labels_csv).resolve(),
        reliability_csv=(EXP_DIR / cfg.data.reliability_csv).resolve(),
        folds_csv=(EXP_DIR / cfg.data.folds_csv).resolve(),
        available_study_uids=available_study_uids,
    )
    logger.info("学習データ全件: %d件（gold+pseudo, 検証データなし）", len(data.study_uids))

    scaler = FeatureStandardizer()
    if cfg.model.standardize:
        scaler.fit(store.slice_features)  # 検証データを割かないため全スライスで算出
        standardized_features = scaler.transform(store.slice_features).astype(np.float32)
    else:
        standardized_features = None

    model = AttentionStudyModel(
        embed_dim=int(cfg.model.embed_dim), num_classes=int(cfg.model.num_classes),
        attn_hidden_dim=int(cfg.model.attn_hidden_dim),
    ).to(device)
    pos_weight = torch.tensor(data.pos_weight, dtype=torch.float32, device=device)
    criterion = SoftTargetBCELoss(pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay))
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    batch_size = int(cfg.trainer.batch_size)
    n = len(data.study_uids)
    history = []
    for epoch in range(FIXED_EPOCHS):
        model.train()
        train_losses = []
        for batch_idx in iterate_batches(n, batch_size, rng):
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                batch_logits = []
                for i in batch_idx:
                    series_list = store.get_study_series(str(data.study_uids[i]), features_override=standardized_features)
                    batch_logits.append(model.forward_study(series_list, device))
                logits = torch.stack(batch_logits, dim=0)
                targets = torch.from_numpy(data.soft_targets[batch_idx]).float().to(device)
                loss = criterion(logits, targets)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss})
        logger.info("[epoch %d/%d] train_loss=%.4f", epoch, FIXED_EPOCHS, train_loss)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.std_,
        "standardize": bool(cfg.model.standardize),
        "labels": LABELS,
        "embed_dim": int(cfg.model.embed_dim),
        "num_classes": int(cfg.model.num_classes),
        "attn_hidden_dim": int(cfg.model.attn_hidden_dim),
        "study_feature_dim": study_feature_dim(int(cfg.model.embed_dim)),
        "fixed_epochs": FIXED_EPOCHS,
    }
    torch.save(ckpt, output_dir / "full_model_attention.ckpt")
    with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    logger.info("フルデータモデルを保存しました: %s", output_dir / "full_model_attention.ckpt")


if __name__ == "__main__":
    main()
