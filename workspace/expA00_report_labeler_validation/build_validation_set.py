"""
train.csv から12ラベル全てが確定している行のみを抽出し、
report-labeler（LLM）の検証用データセット data/validation_58.csv を作る。

Usage:
    python build_validation_set.py
"""

import logging
from pathlib import Path

import pandas as pd

from prompt_template import LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
TRAIN_CSV = EXP_DIR.parent.parent / "datasets" / "rsna-knee-abnormality-detection" / "train.csv"
OUTPUT_CSV = EXP_DIR / "data" / "validation_58.csv"


def main() -> None:
    logger.info("train.csv を読み込みます: %s", TRAIN_CSV)
    df = pd.read_csv(TRAIN_CSV)

    missing_cols = [c for c in ["StudyInstanceUID", "Report", *LABELS] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"train.csv に想定した列がありません: {missing_cols}")

    # 12ラベル全てが埋まっている行のみを抽出する（一部だけ欠損している行は対象外）
    fully_labeled = df.dropna(subset=LABELS, how="any")
    logger.info("12ラベル全て確定している行数: %d / %d", len(fully_labeled), len(df))

    if len(fully_labeled) != 58:
        raise AssertionError(
            f"想定していた 58 件と一致しません（実際: {len(fully_labeled)} 件）。"
            "train.csv が更新された可能性があるため、抽出条件を見直すこと。"
        )

    out_cols = ["StudyInstanceUID", "Report", *LABELS]
    result = fully_labeled[out_cols].copy()
    for label in LABELS:
        result[label] = result[label].astype(int)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    logger.info("検証用データセットを保存しました: %s (%d 行)", OUTPUT_CSV, len(result))


if __name__ == "__main__":
    main()
