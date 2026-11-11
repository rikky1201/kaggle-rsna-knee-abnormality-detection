#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m jupytext --to ipynb --set-kernel python3 preprocess_cache_notebook.py

KAGGLE="python3 -m kaggle"
$KAGGLE kernels push -p .

cat <<EOF

NEXT STEPS:
  1. 実行状況確認: python3 -m kaggle kernels status rikky1201/resnet34-preprocess-cache
  2. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/resnet34-preprocess-cache -p ./output
EOF
