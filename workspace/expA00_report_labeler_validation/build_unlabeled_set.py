"""
train.csv から12ラベルが全て欠損している行（= gold label が無く、report-labeler で
疑似ラベルを生成する対象）を抽出し、data/unlabeled_4349.csv を作る。

validation_58.csv（12ラベル全確定）の対極。build_validation_set.py と対になるスクリプト。

Usage:
    python build_unlabeled_set.py
"""

import logging
from pathlib import Path

import pandas as pd

from prompt_template import LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
TRAIN_CSV = EXP_DIR.parent.parent / "datasets" / "rsna-knee-abnormality-detection" / "train.csv"
OUTPUT_CSV = EXP_DIR / "data" / "unlabeled_4349.csv"


def main() -> None:
    logger.info("train.csv を読み込みます: %s", TRAIN_CSV)
    df = pd.read_csv(TRAIN_CSV)

    missing_cols = [c for c in ["StudyInstanceUID", "Report", *LABELS] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"train.csv に想定した列がありません: {missing_cols}")

    # 12ラベルが全て欠損している行のみを抽出する（1つでもラベルがある行は対象外。
    # 実際には「全確定」か「全欠損」のどちらかしか無いことを確認済みだが、念のため厳密に絞る）
    unlabeled = df[df[LABELS].isna().all(axis=1)]
    logger.info("12ラベル全欠損の行数: %d / %d", len(unlabeled), len(df))

    if len(unlabeled) != 4349:
        raise AssertionError(
            f"想定していた 4349 件と一致しません（実際: {len(unlabeled)} 件）。"
            "train.csv が更新された可能性があるため、抽出条件を見直すこと。"
        )

    out_cols = ["StudyInstanceUID", "Report"]
    result = unlabeled[out_cols].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    logger.info("未ラベルデータセットを保存しました: %s (%d 行)", OUTPUT_CSV, len(result))


if __name__ == "__main__":
    main()
