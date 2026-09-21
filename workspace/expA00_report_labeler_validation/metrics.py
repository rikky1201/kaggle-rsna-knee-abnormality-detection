"""
report-labeler 予測結果の評価指標計算。

`run_llm_labeling.py`（Claude）と Kaggle上の候補モデルベンチマーク（Qwen等）の両方が
このモジュールを import して同一ロジックで評価する（「単一の真実源」）。
モデル間の比較でメトリクス定義がズレることを防ぐ。
"""

from sklearn.metrics import precision_recall_fscore_support


def compute_metrics(df, labels: list) -> dict:
    """df は `true_{label}` / `pred_{label}` 列（0/1）を持つ DataFrame。"""
    per_label = {}
    for label in labels:
        y_true = df[f"true_{label}"].astype(int)
        y_pred = df[f"pred_{label}"].astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        accuracy = (y_true == y_pred).mean()
        per_label[label] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_positive_true": int(y_true.sum()),
            "n_positive_pred": int(y_pred.sum()),
        }

    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(labels)
    macro_precision = sum(v["precision"] for v in per_label.values()) / len(labels)
    macro_recall = sum(v["recall"] for v in per_label.values()) / len(labels)
    macro_accuracy = sum(v["accuracy"] for v in per_label.values()) / len(labels)

    exact_match = (
        (df[[f"true_{l}" for l in labels]].values == df[[f"pred_{l}" for l in labels]].values)
        .all(axis=1)
        .mean()
    )

    return {
        "per_label": per_label,
        "macro": {
            "accuracy": float(macro_accuracy),
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "exact_match_rate": float(exact_match),
        "n_studies": len(df),
    }
