#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m jupytext --to ipynb --set-kernel python3 train_notebook.py

KAGGLE="python3 -m kaggle"
$KAGGLE kernels push -p . --accelerator NvidiaTeslaT4

cat <<EOF

NEXT STEPS:
  1. 実行状況確認: python3 -m kaggle kernels status rikky1201/resnet34-finetune-production
  2. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/resnet34-finetune-production -p ./output
  3. 次のepochを回す場合: kernel-metadata.json の kernel_sources に
     "rikky1201/resnet34-finetune-production" を追加してから再度このスクリプトを実行
     （前回出力のlast.ckptを自動検出して再開する）
EOF
