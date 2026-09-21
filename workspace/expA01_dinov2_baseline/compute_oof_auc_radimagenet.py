"""
RadImageNet版のgold OOF pooled macro AUC算出（`compute_oof_auc.py`のDINOv2版と対になるスクリプト）。

本番と同じ「58 gold + 4,349 pseudo」の学習データ量・同一fold分割で、DINOv2と条件を揃えて
比較する（ユーザー指示、2026-09-16）。train.py本番学習（AdamW多epoch, early stopping）を
`experiment.name=expA01_linear_probe_radimagenet`, `data.features_npz=data/radimagenet_resnet50_features_4405.npz`,
`model.input_dim=28679` で実行済みの前提（4fold分）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
from src.data_prep import LABELS, load_training_data, split_fold  # noqa: E402
from src.metrics import bootstrap_label_aucs, print_ci_table, summarize_ci  # noqa: E402
from src.model import FeatureStandardizer, LinearProbeHead  # noqa: E402

N_FOLDS = 4
N_BOOTSTRAP = 1000
EXPERIMENT_NAME = "expA01_linear_probe_radimagenet"
FEATURES_NPZ = "data/radimagenet_resnet50_features_4405.npz"
INPUT_DIM = 28679
OOF_OUTPUT_PATH = EXP_DIR / "results" / EXPERIMENT_NAME / "oof_predictions.npz"


def main() -> None:
    cfg = OmegaConf.load(EXP_DIR / "config.yaml")
    data = load_training_data(
        labels_csv=(EXP_DIR / cfg.data.labels_csv).resolve(),
        features_npz=(EXP_DIR / FEATURES_NPZ).resolve(),
        reliability_csv=(EXP_DIR / cfg.data.reliability_csv).resolve(),
        folds_csv=(EXP_DIR / cfg.data.folds_csv).resolve(),
    )

    n_gold = int((data.label_source == "gold").sum())
    oof_probs = np.full((len(data.study_uids), len(LABELS)), np.nan, dtype=np.float32)
    oof_mask = np.zeros(len(data.study_uids), dtype=bool)

    fold_macro_aucs = {}
    device = torch.device("cpu")

    for fold in range(N_FOLDS):
        train_idx, val_idx = split_fold(data, fold)

        scaler = FeatureStandardizer()
        if cfg.model.standardize:
            scaler.fit(data.features[train_idx])
            val_features = scaler.transform(data.features[val_idx])
        else:
            val_features = data.features[val_idx]

        ckpt_path = EXP_DIR / cfg.paths.output_dir / EXPERIMENT_NAME / f"fold{fold}" / "best.ckpt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = LinearProbeHead(INPUT_DIM, int(cfg.model.num_classes)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            logits = model(torch.from_numpy(val_features).float())
            probs = torch.sigmoid(logits).numpy()

        oof_probs[val_idx] = probs
        oof_mask[val_idx] = True

        yt = data.hard_labels[val_idx]
        aucs = {}
        for i, label in enumerate(LABELS):
            if len(np.unique(yt[:, i])) < 2:
                aucs[label] = float("nan")
                continue
            aucs[label] = float(roc_auc_score(yt[:, i], probs[:, i]))
        fold_macro_auc = float(np.nanmean(list(aucs.values())))
        fold_macro_aucs[fold] = (fold_macro_auc, aucs, len(val_idx))

    if oof_mask.sum() != n_gold:
        raise AssertionError(f"OOFが埋まったサンプル数が gold件数と不一致: oof={oof_mask.sum()}, gold={n_gold}")

    print("=" * 70)
    print("[RadImageNet-ResNet50] fold別 macro AUC")
    print("=" * 70)
    for fold, (macro_auc, aucs, n_val) in fold_macro_aucs.items():
        print(f"  fold{fold} (n={n_val:2d}): macro AUC = {macro_auc:.4f}")
    simple_avg = float(np.mean([v[0] for v in fold_macro_aucs.values()]))
    print(f"  --- fold単純平均 = {simple_avg:.4f}")

    print()
    print("=" * 70)
    print("[RadImageNet-ResNet50] ラベル別 fold別 AUC")
    print("=" * 70)
    header = f"{'label':<20}" + "".join(f"fold{f:<10}" for f in range(N_FOLDS))
    print(header)
    for i, label in enumerate(LABELS):
        row = f"{label:<20}"
        for fold in range(N_FOLDS):
            v = fold_macro_aucs[fold][1][label]
            row += f"{v:<14.4f}" if not np.isnan(v) else f"{'NaN':<14}"
        print(row)

    print()
    print("=" * 70)
    print("[RadImageNet-ResNet50] gold OOF全体 (n=58) pooled macro AUC")
    print("=" * 70)
    oof_y = data.hard_labels[oof_mask]
    oof_p = oof_probs[oof_mask]
    oof_label_aucs = {}
    for i, label in enumerate(LABELS):
        if len(np.unique(oof_y[:, i])) < 2:
            oof_label_aucs[label] = float("nan")
            continue
        oof_label_aucs[label] = float(roc_auc_score(oof_y[:, i], oof_p[:, i]))
    for label, v in oof_label_aucs.items():
        print(f"  {label:<20}: {v:.4f}" if not np.isnan(v) else f"  {label:<20}: NaN")
    oof_macro_auc = float(np.nanmean(list(oof_label_aucs.values())))
    print(f"  --- gold OOF macro AUC (n={int(oof_mask.sum())}) = {oof_macro_auc:.4f}")

    print()
    print("[比較] DINOv2 pooled OOF macro AUC (既存, data/dinov2_base_features_4405.npz) = 0.7375")
    print(f"       RadImageNet-ResNet50 pooled OOF macro AUC = {oof_macro_auc:.4f}")
    print(f"       delta (RadImageNet - DINOv2) = {oof_macro_auc - 0.7375:+.4f}")

    # ── OOF予測値を保存（bootstrap比較・再解析用。再学習せずに済むように）
    oof_study_uids = data.study_uids[oof_mask]
    OOF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OOF_OUTPUT_PATH, study_uids=oof_study_uids, y_true=oof_y, y_prob=oof_p, labels=np.array(LABELS))
    print(f"\nOOF予測値を保存しました: {OOF_OUTPUT_PATH}")

    # ── bootstrap 95%信頼区間（study単位で重複ありリサンプリング, n_bootstrap=1000）
    print()
    print("=" * 70)
    print(f"[RadImageNet-ResNet50] gold OOF bootstrap 95%信頼区間（n_bootstrap={N_BOOTSTRAP}, seed=42）")
    print("=" * 70)
    boot = bootstrap_label_aucs(oof_y, oof_p, LABELS, n_bootstrap=N_BOOTSTRAP, seed=42)
    summary = summarize_ci(boot, LABELS)
    print_ci_table(summary, LABELS)


if __name__ == "__main__":
    main()
