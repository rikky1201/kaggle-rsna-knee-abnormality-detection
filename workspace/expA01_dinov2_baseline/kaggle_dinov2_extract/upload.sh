#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

EXP_DIR=".."  # workspace/expA01_dinov2_baseline/
KAGGLE="python3 -m kaggle"
DATASET_SLUG="rikky1201/rsna-knee-dinov2-baseline-assets"

# 1) assets をフラットに配置（サブフォルダを作ると --dir-mode でアーカイブ化される問題を避ける）
mkdir -p kaggle_dataset
cp "$EXP_DIR/src/preprocessing.py" kaggle_dataset/
cp "$EXP_DIR/src/pooling.py" kaggle_dataset/
cp "$EXP_DIR/src/feature_extraction.py" kaggle_dataset/
cp dataset-metadata.json kaggle_dataset/

# 2) Dataset 作成 or バージョン更新
if $KAGGLE datasets status "$DATASET_SLUG" >/dev/null 2>&1; then
  $KAGGLE datasets version -p kaggle_dataset -m "update preprocessing/pooling/feature_extraction"
else
  $KAGGLE datasets create -p kaggle_dataset
fi

# 2.5) Dataset処理完了待ち
echo "Dataset処理完了を待機中..."
for i in $(seq 1 30); do
  status=$($KAGGLE datasets status "$DATASET_SLUG" 2>&1)
  echo "  [$i/30] status: $status"
  [ "$status" = "ready" ] && break
  sleep 10
done

# 3) jupytext で .py -> .ipynb 変換（kernelspec必須。無いとpapermill実行が即エラー）
python3 -m jupytext --to ipynb --set-kernel python3 extract_features_notebook.py

# 4) Notebook push（T4 GPU明示指定。P100だとtorchの互換性問題で失敗するため）
$KAGGLE kernels push -p . --accelerator NvidiaTeslaT4

cat <<EOF

NEXT STEPS:
  1. 実行状況確認: python3 -m kaggle kernels status rikky1201/dinov2-feature-extraction-timing-benchmark
  2. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/dinov2-feature-extraction-timing-benchmark -p ./output
EOF
