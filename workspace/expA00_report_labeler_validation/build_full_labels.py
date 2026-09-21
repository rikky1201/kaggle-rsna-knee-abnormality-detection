"""
58件の gold label（`data/validation_58.csv`）と 4,349件の疑似ラベル
（`data/pseudo_labels_4349.csv`, Qwen2.5-14B-Instruct-AWQ + vLLM + guided decoding,
v2_fewshot プロンプト）を統合し、train.csv 全4,407 studyに対応するラベルセットを作る。

`label_source` 列（"gold" / "pseudo"）を必ず残す。fold設計・学習時にサンプル重み付けや
gold/pseudo別の検証を行う可能性があるため、統合後もソースを区別できるようにする。

Usage:
    python build_full_labels.py
"""

import logging
from pathlib import Path

import pandas as pd

from prompt_template import LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
GOLD_CSV = EXP_DIR / "data" / "validation_58.csv"
PSEUDO_CSV = EXP_DIR / "data" / "pseudo_labels_4349.csv"
OUTPUT_CSV = EXP_DIR / "data" / "train_full_labels_4407.csv"


def main() -> None:
    logger.info("gold label を読み込みます: %s", GOLD_CSV)
    gold = pd.read_csv(GOLD_CSV)[["StudyInstanceUID", *LABELS]].copy()
    gold["label_source"] = "gold"

    logger.info("疑似ラベルを読み込みます: %s", PSEUDO_CSV)
    pseudo = pd.read_csv(PSEUDO_CSV)[["StudyInstanceUID", *LABELS]].copy()
    pseudo["label_source"] = "pseudo"

    overlap = set(gold["StudyInstanceUID"]) & set(pseudo["StudyInstanceUID"])
    if overlap:
        raise AssertionError(f"gold と pseudo に重複する StudyInstanceUID があります: {len(overlap)}件")

    combined = pd.concat([gold, pseudo], ignore_index=True)
    if len(combined) != 4407:
        raise AssertionError(f"統合後の件数が4407と一致しません（実際: {len(combined)}件）")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    logger.info(
        "統合ラベルセットを保存しました: %s (%d行, gold=%d / pseudo=%d)",
        OUTPUT_CSV, len(combined), len(gold), len(pseudo),
    )


if __name__ == "__main__":
    main()
