#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

KAGGLE="python3 -m kaggle"

wait_ready() {
  local slug="$1"
  echo "Dataset処理完了を待機中... ($slug)"
  for i in $(seq 1 30); do
    status=$($KAGGLE datasets status "$slug" 2>&1)
    echo "  [$i/30] status: $status"
    [ "$status" = "ready" ] && return 0
    sleep 10
  done
}

# 1) assets dataset 更新（preprocessing/pooling/feature_extraction + radimagenet実装 + gold58 fold csv）
ASSETS_SLUG="rikky1201/rsna-knee-dinov2-baseline-assets"
if $KAGGLE datasets status "$ASSETS_SLUG" >/dev/null 2>&1; then
  $KAGGLE datasets version -p kaggle_dataset_assets -m "add radimagenet_feature_extraction.py + folds_gold58.csv"
else
  $KAGGLE datasets create -p kaggle_dataset_assets
fi
wait_ready "$ASSETS_SLUG"

# 2) RadImageNet weights dataset 新規作成
WEIGHTS_SLUG="rikky1201/radimagenet-resnet50-weights"
if $KAGGLE datasets status "$WEIGHTS_SLUG" >/dev/null 2>&1; then
  $KAGGLE datasets version -p kaggle_dataset_weights -m "update weights"
else
  $KAGGLE datasets create -p kaggle_dataset_weights
fi
wait_ready "$WEIGHTS_SLUG"

# 3) jupytext変換 + Notebook push（T4 GPU明示指定）
python3 -m jupytext --to ipynb --set-kernel python3 extract_features_notebook.py
$KAGGLE kernels push -p . --accelerator NvidiaTeslaT4

cat <<EOF

NEXT STEPS:
  1. 実行状況確認: python3 -m kaggle kernels status rikky1201/rsna-knee-gold58-backbone-comparison-extract
  2. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/rsna-knee-gold58-backbone-comparison-extract -p ./output
EOF
