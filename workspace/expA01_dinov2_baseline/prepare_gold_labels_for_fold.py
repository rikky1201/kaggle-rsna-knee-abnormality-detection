"""
train_full_labels_4407.csv から gold label の58件のみを抽出し、
workspace/fold/generate_folds_multilabel.py の入力用CSVを作る。

Usage:
    python prepare_gold_labels_for_fold.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
LABELS_CSV = EXP_DIR.parent / "expA00_report_labeler_validation" / "data" / "train_full_labels_4407.csv"
OUTPUT_CSV = EXP_DIR / "data" / "gold_58_labels.csv"


def main() -> None:
    df = pd.read_csv(LABELS_CSV)
    gold = df[df["label_source"] == "gold"].drop(columns=["label_source"])
    if len(gold) != 58:
        raise AssertionError(f"gold行数が58と一致しません（実際: {len(gold)}）")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(OUTPUT_CSV, index=False)
    logger.info("gold labelを保存しました: %s (%d行)", OUTPUT_CSV, len(gold))


if __name__ == "__main__":
    main()
