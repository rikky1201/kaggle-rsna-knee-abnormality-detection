"""
58件の gold label に対する report-labeler（Qwen2.5-14B-Instruct-AWQ + vLLM + guided
decoding, v2_fewshot プロンプト）の予測から、ラベルごとの PPV / NPV を算出する。

KAGGLE_DIRECTION.md の損失関数設計（ソフトターゲット変換）で使う値:
  疑似ラベル ŷ_j=1 → ỹ_j = PPV_j
  疑似ラベル ŷ_j=0 → ỹ_j = 1 - NPV_j

Usage:
    python compute_label_reliability.py
"""

import logging
from pathlib import Path

import pandas as pd

from prompt_template import LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
VALIDATION_PRED_CSV = EXP_DIR / "data" / "qwen_v2guided_validation_predictions.csv"
OUTPUT_CSV = EXP_DIR / "data" / "pseudo_label_reliability.csv"


def main() -> None:
    logger.info("検証予測を読み込みます: %s", VALIDATION_PRED_CSV)
    df = pd.read_csv(VALIDATION_PRED_CSV)

    rows = []
    for label in LABELS:
        y_true = df[f"true_{label}"].astype(int)
        y_pred = df[f"pred_{label}"].astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        if tp + fp == 0:
            raise AssertionError(f"{label}: 予測陽性が0件のため PPV が計算できません")
        if tn + fn == 0:
            raise AssertionError(f"{label}: 予測陰性が0件のため NPV が計算できません")

        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)

        rows.append({
            "label": label, "ppv": ppv, "npv": npv,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_pred_pos": tp + fp, "n_pred_neg": tn + fn,
        })
        logger.info("%-18s PPV=%.3f (n_pos=%d)  NPV=%.3f (n_neg=%d)", label, ppv, tp + fp, npv, tn + fn)

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False)
    logger.info("信頼性テーブルを保存しました: %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()
