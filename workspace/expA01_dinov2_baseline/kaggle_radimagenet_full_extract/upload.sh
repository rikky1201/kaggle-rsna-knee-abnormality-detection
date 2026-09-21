#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# assets (preprocessing.py/pooling.py/radimagenet_feature_extraction.py) と
# radimagenet-resnet50-weights は既にpush済み・変更なしのため、ここでは再アップロードしない
# （dataset_sources で参照するのみ）

python3 -m jupytext --to ipynb --set-kernel python3 extract_features_full_notebook.py

KAGGLE="python3 -m kaggle"
$KAGGLE kernels push -p . --accelerator NvidiaTeslaT4

cat <<EOF

NEXT STEPS:
  1. 実行状況確認: python3 -m kaggle kernels status rikky1201/radimagenet-resnet50-feature-extraction-full
  2. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/radimagenet-resnet50-feature-extraction-full -p ./output
  （推定所要時間: 約4時間、要実測）
EOF
