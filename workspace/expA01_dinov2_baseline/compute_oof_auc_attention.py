"""
attention pooling版の gold pooled OOF macro AUC + bootstrap信頼区間を計算する。

`compute_oof_auc.py`（mean+max版）と同じ手順（各foldのbest.ckptで該当foldのval(gold
held-out)を推論し、58件全体をpoolしてラベルごとにAUC計算→平均）をattention版に適用する。
さらに、mean+max版の保存済みOOF（`results/expA01_linear_probe/oof_predictions.npz`）と
対応ありbootstrap（`bootstrap_paired_delta`）で比較し、改善が統計的に有意か判定する。
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
from src.attention_dataset import SliceFeatureStore  # noqa: E402
from src.attention_model import AttentionStudyModel  # noqa: E402
from src.data_prep import LABELS, load_training_data_slices, split_fold  # noqa: E402
from src.metrics import bootstrap_label_aucs, bootstrap_paired_delta, print_ci_table, summarize_ci  # noqa: E402

N_FOLDS = 4
N_BOOTSTRAP = 1000
OOF_OUTPUT_PATH = EXP_DIR / "results" / "expA01_attention_probe" / "oof_predictions.npz"
BASELINE_OOF_PATH = EXP_DIR / "results" / "expA01_linear_probe" / "oof_predictions.npz"


def main() -> None:
    cfg = OmegaConf.load(EXP_DIR / "config_attention.yaml")
    device = torch.device("cpu")

    store = SliceFeatureStore((EXP_DIR / cfg.data.slice_features_npz).resolve())
    available_study_uids = set(store.study_to_series.keys())

    data = load_training_data_slices(
        labels_csv=(EXP_DIR / cfg.data.labels_csv).resolve(),
        reliability_csv=(EXP_DIR / cfg.data.reliability_csv).resolve(),
        folds_csv=(EXP_DIR / cfg.data.folds_csv).resolve(),
        available_study_uids=available_study_uids,
    )

    n_gold = int((data.label_source == "gold").sum())
    oof_probs = np.full((len(data.study_uids), len(LABELS)), np.nan, dtype=np.float32)
    oof_mask = np.zeros(len(data.study_uids), dtype=bool)

    fold_macro_aucs = {}

    for fold in range(N_FOLDS):
        train_idx, val_idx = split_fold(data, fold)
        val_study_uids = data.study_uids[val_idx]

        ckpt_path = EXP_DIR / cfg.paths.output_dir / cfg.experiment.name / f"fold{fold}" / "best.ckpt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = AttentionStudyModel(
            embed_dim=int(cfg.model.embed_dim), num_classes=int(cfg.model.num_classes),
            attn_hidden_dim=int(cfg.model.attn_hidden_dim),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        scaler_mean, scaler_std = ckpt["scaler_mean"], ckpt["scaler_std"]
        if scaler_mean is not None:
            standardized_features = ((store.slice_features - scaler_mean) / scaler_std).astype(np.float32)
        else:
            standardized_features = None

        probs_list = []
        with torch.no_grad():
            for study_uid in val_study_uids:
                series_list = store.get_study_series(str(study_uid), features_override=standardized_features)
                logits = model.forward_study(series_list, device)
                probs_list.append(torch.sigmoid(logits).numpy())
        probs = np.stack(probs_list, axis=0)

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
    print("[attention pooling版] fold別 macro AUC（各foldのbest.ckptで該当val_idxを推論した結果）")
    print("=" * 70)
    for fold, (macro_auc, aucs, n_val) in fold_macro_aucs.items():
        print(f"  fold{fold} (n={n_val:2d}): macro AUC = {macro_auc:.4f}")
    simple_avg = float(np.mean([v[0] for v in fold_macro_aucs.values()]))
    print(f"  --- fold単純平均 = {simple_avg:.4f}")

    print()
    print("=" * 70)
    print("gold OOF全体（58件をpoolしてラベルごとにAUC計算 → macro平均）")
    print("=" * 70)
    oof_y = data.hard_labels[oof_mask]
    oof_p = oof_probs[oof_mask]
    oof_study_uids = data.study_uids[oof_mask]
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

    OOF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OOF_OUTPUT_PATH, study_uids=oof_study_uids, y_true=oof_y, y_prob=oof_p, labels=np.array(LABELS))
    print(f"\nOOF予測値を保存しました: {OOF_OUTPUT_PATH}")

    print()
    print("=" * 70)
    print(f"[attention pooling版] gold OOF bootstrap 95%信頼区間（n_bootstrap={N_BOOTSTRAP}, seed=42）")
    print("=" * 70)
    boot = bootstrap_label_aucs(oof_y, oof_p, LABELS, n_bootstrap=N_BOOTSTRAP, seed=42)
    summary = summarize_ci(boot, LABELS)
    print_ci_table(summary, LABELS)

    # ── mean+max版との対応あり比較
    if not BASELINE_OOF_PATH.exists():
        print(f"\n[警告] baseline OOFファイルが見つからないため比較をスキップします: {BASELINE_OOF_PATH}")
        return

    baseline = np.load(BASELINE_OOF_PATH, allow_pickle=True)
    baseline_uid_to_idx = {u: i for i, u in enumerate(baseline["study_uids"].tolist())}
    attn_uid_to_idx = {u: i for i, u in enumerate(oof_study_uids.tolist())}
    common_uids = [u for u in oof_study_uids.tolist() if u in baseline_uid_to_idx]
    if len(common_uids) != len(oof_study_uids) or len(common_uids) != len(baseline["study_uids"]):
        raise AssertionError(
            f"attention版とbaseline版のgold study_uid集合が一致しません: "
            f"attn={len(oof_study_uids)}, baseline={len(baseline['study_uids'])}, common={len(common_uids)}"
        )

    order = common_uids  # attn側の並び順をそのまま正準順とする
    baseline_y = baseline["y_true"][[baseline_uid_to_idx[u] for u in order]]
    baseline_p = baseline["y_prob"][[baseline_uid_to_idx[u] for u in order]]
    attn_y = oof_y[[attn_uid_to_idx[u] for u in order]]
    attn_p = oof_p[[attn_uid_to_idx[u] for u in order]]
    if not np.array_equal(baseline_y, attn_y):
        raise AssertionError("baseline版とattention版でy_trueが一致しません（study_uid整列に誤りがある可能性）")

    print()
    print("=" * 70)
    print(f"[比較] attention pooling版 - mean+max版 の対応ありbootstrap差分（n_bootstrap={N_BOOTSTRAP}, seed=42）")
    print("=" * 70)
    delta = bootstrap_paired_delta(attn_y, baseline_p, attn_p, LABELS, n_bootstrap=N_BOOTSTRAP, seed=42)
    delta_summary = summarize_ci(delta, LABELS)
    print_ci_table(delta_summary, LABELS, title="正の値 = attention pooling版が優位（95%CIが0をまたがなければ有意）")


if __name__ == "__main__":
    main()
