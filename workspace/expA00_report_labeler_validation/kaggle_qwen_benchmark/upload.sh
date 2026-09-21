#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

EXP_DIR=".."  # workspace/expA00_report_labeler_validation/

# 1) assets を kaggle_dataset/ にフラットにコピー（サブフォルダを作ると kaggle CLI の
#    --dir-mode zip/tar でアーカイブ化されて Notebook 側から読みにくくなるため、
#    ディレクトリを作らずファイルを直置きする）
mkdir -p kaggle_dataset
cp "$EXP_DIR/prompt_template.py" kaggle_dataset/
cp "$EXP_DIR/metrics.py" kaggle_dataset/
cp "$EXP_DIR/data/validation_58.csv" kaggle_dataset/
cp dataset-metadata.json kaggle_dataset/

# 2) Dataset 作成 or バージョン更新
KAGGLE="python3 -m kaggle"
DATASET_SLUG="rikky1201/rsna-knee-report-labeler-assets"
if $KAGGLE datasets status "$DATASET_SLUG" >/dev/null 2>&1; then
  $KAGGLE datasets version -p kaggle_dataset -m "update prompt_template/metrics/validation_58"
else
  $KAGGLE datasets create -p kaggle_dataset
fi

# 3) jupytext で .py -> .ipynb 変換（kernelspec を明示しないと Kaggle の papermill 実行が
#    "No kernel name found in notebook" で落ちる）
python3 -m jupytext --to ipynb --set-kernel python3 benchmark_notebook.py

# 4) Notebook push（push すると Kaggle 側でバッチ実行される。手動 Run All は不要）
#    --accelerator は明示的に NvidiaTeslaT4 を指定する。デフォルト割当が Tesla P100
#    (sm_60/Pascal) になることがあり、現在の Kaggle 環境の torch ビルドは sm_70 以降しか
#    サポートしないため、P100 だと bitsandbytes 呼び出し等で "Kernel died" になる
$KAGGLE kernels push -p . --accelerator NvidiaTeslaT4

cat <<EOF

NEXT STEPS:
  1. データセット処理待ち（初回は1〜2分）: python3 -m kaggle datasets status $DATASET_SLUG
  2. 実行状況確認: python3 -m kaggle kernels status rikky1201/qwen2-5-14b-report-labeler-benchmark
  3. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/qwen2-5-14b-report-labeler-benchmark -p ./output
EOF
