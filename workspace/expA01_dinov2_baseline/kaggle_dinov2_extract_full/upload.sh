#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# assets (preprocessing.py/pooling.py/feature_extraction.py) は
# kaggle_dinov2_extract/ 側で既に rikky1201/rsna-knee-dinov2-baseline-assets として
# push 済み・変更なしのため、ここでは再アップロードしない（dataset_sources で参照するのみ）

python3 -m jupytext --to ipynb --set-kernel python3 extract_features_full_notebook.py

KAGGLE="python3 -m kaggle"
$KAGGLE kernels push -p . --accelerator NvidiaTeslaT4

cat <<EOF

NEXT STEPS:
  1. 実行状況確認: python3 -m kaggle kernels status rikky1201/dinov2-base-feature-extraction-full
  2. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/dinov2-base-feature-extraction-full -p ./output
  （推定所要時間: 約6時間）
EOF
