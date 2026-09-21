#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# enable_internet:false のため pip install は一切できない。
# transformers / pydicom は Kaggle 標準 Python GPU イメージにプリインストール済みの前提で進める

python3 -m jupytext --to ipynb --set-kernel python3 inference_notebook_attention.py

KAGGLE="python3 -m kaggle"
$KAGGLE kernels push -p . --accelerator NvidiaTeslaT4

cat <<EOF

NEXT STEPS:
  1. 実行状況確認: python3 -m kaggle kernels status rikky1201/rsna-knee-v4-expa01-attention-pooling-submission
  2. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/rsna-knee-v4-expa01-attention-pooling-submission -p ./output
  3. submission.csv の中身を検証してから、Kaggle UI で Notebook を開き Save & Run All → Submit to Competition（手動）
EOF
