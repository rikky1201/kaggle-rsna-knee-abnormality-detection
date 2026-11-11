#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# enable_internet:false のため pip install は一切できない。
# torchvision の resnet34 は weights=None（ImageNet事前学習をダウンロードしない）で構築し、
# fine-tuning済みの全重みをckptからload_state_dictするため、追加ダウンロードは不要

python3 -m jupytext --to ipynb --set-kernel python3 inference_notebook_resnet34.py

KAGGLE="python3 -m kaggle"
$KAGGLE kernels push -p . --accelerator NvidiaTeslaT4

cat <<EOF

NEXT STEPS:
  1. 実行状況確認: python3 -m kaggle kernels status rikky1201/rsna-knee-v5-expa02-resnet34-finetune-submission
  2. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/rsna-knee-v5-expa02-resnet34-finetune-submission -p ./output
  3. submission.csv の中身を検証してから、Kaggle UI で Notebook を開き Save & Run All → Submit to Competition（手動）
EOF
