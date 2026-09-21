#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# assets (preprocessing.py/pooling.py/feature_extraction.py) は既にpush済み・変更なしのため
# ここでは再アップロードしない（dataset_sources で参照するのみ）

python3 -m jupytext --to ipynb --set-kernel python3 extract_slices_full_notebook.py

KAGGLE="python3 -m kaggle"
$KAGGLE kernels push -p . --accelerator NvidiaTeslaT4

cat <<EOF

NEXT STEPS:
  1. 実行状況確認: python3 -m kaggle kernels status rikky1201/dinov2-slice-feature-extraction-full
  2. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/dinov2-slice-feature-extraction-full -p ./output
  （推定所要時間: 約9時間。ベンチマーク: 100件, 平均7.31s/件, 推定ストレージ約2.25GB）
EOF
