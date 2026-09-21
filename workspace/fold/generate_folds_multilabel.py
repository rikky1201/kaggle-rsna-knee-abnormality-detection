#!/usr/bin/env python3
"""マルチラベル対応 fold 割り当て生成スクリプト（MultilabelStratifiedKFold）.

`generate_folds.py` の sklearn Stratified*KFold は単一ラベルの層化にしか対応していない。
12ラベル同時分類のようなマルチラベルタスクでは `iterstrat.MultilabelStratifiedKFold`
（`pip install iterative-stratification`）が必要なため、専用スクリプトとして
`generate_folds.py` と同階層に置く（README.md の「専用スクリプトを同階層に追加する」規約）。

出力形式・バージョニング規約は README.md / generate_folds.py と共通。

使い方:
    python generate_folds_multilabel.py --input ../expA01_dinov2_baseline/data/gold_58_labels.csv \\
        --version v1_multilabel_gold58 --id-col StudyInstanceUID \\
        --target-cols "ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture" \\
        --n-splits 4
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

log = logging.getLogger(__name__)

FOLD_ROOT = Path(__file__).resolve().parent


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"generate_folds_{datetime.now():%Y%m%d_%H%M%S}.log"

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

    log.info(f"Log file: {log_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MultilabelStratifiedKFold で fold 割り当てを生成する")
    parser.add_argument("--input", required=True, help="入力 CSV のパス")
    parser.add_argument("--version", required=True, help="バージョン名（出力先ディレクトリ名）")
    parser.add_argument("--id-col", required=True, help="サンプルを一意に識別するキー列名")
    parser.add_argument("--target-cols", required=True, help="stratify対象のラベル列名（カンマ区切り）")
    parser.add_argument("--n-splits", type=int, default=4, help="fold数（デフォルト: 4）")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（デフォルト: 42）")
    return parser.parse_args()


def log_fold_distribution(df: pd.DataFrame, target_cols: list[str]) -> None:
    log.info(f"fold ごとの件数:\n{df['fold'].value_counts().sort_index().to_string()}")
    dist = df.groupby("fold")[target_cols].mean()
    log.info(f"fold ごとのラベル陽性率:\n{dist.round(4).to_string()}")
    for col in target_cols:
        counts = df.groupby("fold")[col].sum()
        if (counts == 0).any():
            log.warning(f"'{col}': 陽性件数が0のfoldがある:\n{counts.to_string()}")


def main() -> None:
    args = parse_args()
    target_cols = [c.strip() for c in args.target_cols.split(",")]

    out_dir = FOLD_ROOT / args.version
    out_csv = out_dir / "folds.csv"
    if out_csv.exists():
        raise SystemExit(f"{out_csv} は既に存在する。上書きせず新しい --version を指定すること")

    setup_logging(out_dir)
    log.info(f"Args: {vars(args)}")

    df = pd.read_csv(args.input)
    log.info(f"入力: {args.input} ({len(df)} 行)")

    for col in [args.id_col, *target_cols]:
        if col not in df.columns:
            raise SystemExit(f"列 '{col}' が入力CSVに存在しない。列一覧: {list(df.columns)}")
    if df[args.id_col].duplicated().any():
        raise SystemExit(f"キー列 '{args.id_col}' に重複がある")

    X = np.zeros((len(df), 1))  # MultilabelStratifiedKFold は X の中身を見ないダミー
    y = df[target_cols].to_numpy()

    splitter = MultilabelStratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(splitter.split(X, y)):
        df.loc[df.index[val_idx], "fold"] = fold
    assert (df["fold"] >= 0).all(), "fold未割り当ての行がある"

    log_fold_distribution(df, target_cols)

    keep_cols = [args.id_col, *target_cols, "fold"]
    df[keep_cols].to_csv(out_csv, index=False)
    log.info(f"保存: {out_csv}")
    log.info(f"config.yaml には data.folds_csv: workspace/fold/{args.version}/folds.csv を指定する")


if __name__ == "__main__":
    main()
