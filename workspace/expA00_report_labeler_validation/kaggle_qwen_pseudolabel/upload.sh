#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

EXP_DIR=".."  # workspace/expA00_report_labeler_validation/
KAGGLE="python3 -m kaggle"
DATASET_SLUG="rikky1201/rsna-knee-report-labeler-assets"

# 1) assets を kaggle_dataset/ にフラットにコピー。
#    dataset version は毎回の内容で完全に置き換わるため、既存の validation_58.csv /
#    metrics.py も含めて全部揃えて version する（unlabeled_4349.csv だけでは
#    kaggle_qwen_benchmark 側が参照するファイルが消えてしまう）
mkdir -p kaggle_dataset
cp "$EXP_DIR/prompt_template.py" kaggle_dataset/
cp "$EXP_DIR/metrics.py" kaggle_dataset/
cp "$EXP_DIR/data/validation_58.csv" kaggle_dataset/
cp "$EXP_DIR/data/unlabeled_4349.csv" kaggle_dataset/
cp "$EXP_DIR/kaggle_qwen_benchmark/dataset-metadata.json" kaggle_dataset/

# 2) Dataset バージョン更新（既に存在する前提。無ければ作成）
if $KAGGLE datasets status "$DATASET_SLUG" >/dev/null 2>&1; then
  $KAGGLE datasets version -p kaggle_dataset -m "add unlabeled_4349.csv for pseudo-label generation"
else
  $KAGGLE datasets create -p kaggle_dataset
fi

# 2.5) Dataset の処理完了を待つ（"ready"になる前にkernelをpushすると、
#      Notebook側で /kaggle/input/ に新しいファイルがまだマウントされず
#      FileNotFoundError になるハマりどころがあった）
echo "Dataset処理完了を待機中..."
for i in $(seq 1 30); do
  status=$($KAGGLE datasets status "$DATASET_SLUG" 2>&1)
  echo "  [$i/30] status: $status"
  [ "$status" = "ready" ] && break
  sleep 10
done

# 3) jupytext で .py -> .ipynb 変換
python3 -m jupytext --to ipynb --set-kernel python3 pseudolabel_notebook.py

# 4) Notebook push（T4 GPU明示指定。P100だとtorchの互換性問題で失敗するため）
$KAGGLE kernels push -p . --accelerator NvidiaTeslaT4

cat <<EOF

NEXT STEPS:
  1. データセット処理待ち（1〜2分）: python3 -m kaggle datasets status $DATASET_SLUG
  2. 実行状況確認: python3 -m kaggle kernels status rikky1201/qwen2-5-14b-report-pseudolabel-generation
  3. 完了後の出力取得: python3 -m kaggle kernels output rikky1201/qwen2-5-14b-report-pseudolabel-generation -p ./output
  （推定所要時間: 約3.5〜4時間。途中経過はログに逐次チャンク単位で記録される）
EOF
