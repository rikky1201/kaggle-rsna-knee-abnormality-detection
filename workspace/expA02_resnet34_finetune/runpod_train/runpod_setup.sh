#!/usr/bin/env bash
# RunPod pod上で実行するセットアップ+学習起動スクリプト。
# 前提: /root/.kaggle/kaggle.json が既に配置済み（scpで注入。README `tools/runpod/README.md` §9.5参照）
#
# Usage（pod上のWeb Terminal / SSHで）:
#   bash runpod_setup.sh
set -euo pipefail

WORKDIR=/workspace
CACHE_DIR="$WORKDIR/cache"
ASSETS_DIR="$WORKDIR/assets"
RESUME_DIR="$WORKDIR/resume"
CKPT_DIR="$WORKDIR/ckpt"
mkdir -p "$CACHE_DIR" "$ASSETS_DIR" "$RESUME_DIR" "$CKPT_DIR"

echo "[1/5] 依存パッケージインストール（kaggle CLI + pandas/scikit-learn, torch/torchvisionはtemplateに同梱済み）"
pip install -q --break-system-packages kaggle pandas scikit-learn  # PEP668: runpod-torch templateは--break-system-packages必須

echo "[2/5] assets（labels/reliability/folds csv）ダウンロード"
python -m kaggle datasets download -d rikky1201/rsna-knee-dinov2-baseline-assets -p "$ASSETS_DIR" --unzip

echo "[3/5] 前処理キャッシュ（8データセットに分割済み）ダウンロード"
# ★一部データセットでbulk zip download(`datasets download -d`)が404で失敗する事例を確認
#   （`datasets files`/`datasets status`は正常なのにbulk downloadだけ404。原因不明だが
#   個別ファイル指定(`-f`)なら成功するため、こちらを標準方式にする）
for slug in rsna-knee-resnet34-preprocess-cache rsna-knee-resnet34-cache-b2 rsna-knee-resnet34-cache-b3 \
            rsna-knee-resnet34-cache-b4 rsna-knee-resnet34-cache-b5 rsna-knee-resnet34-cache-b6 \
            rsna-knee-resnet34-cache-b7 rsna-knee-resnet34-cache-b8; do
  echo "  - $slug"
  files=$(python -m kaggle datasets files "rikky1201/$slug" --csv | tail -n +2 | cut -d',' -f1)
  for f in $files; do
    [ -f "$CACHE_DIR/$f" ] && continue  # 既にあればスキップ（再実行時の再ダウンロード防止）
    python -m kaggle datasets download -d "rikky1201/$slug" -f "$f" -p "$CACHE_DIR"
  done
done
echo "  チャンク数: $(ls "$CACHE_DIR"/preprocess_cache_chunk_*.npz | wc -l)（期待値23）"

echo "[4/5] epoch2再開用チェックポイントダウンロード"
python -m kaggle datasets download -d rikky1201/resnet34-finetune-resume-epoch2 -p "$RESUME_DIR" --unzip

echo "[5/5] 学習開始"
# ★nohupだけだとssh切断でSIGHUP、setsidで完全デタッチ + </dev/null必須（README §11参照）
# 追加引数は "$@" で渡す。例:
#   スモークテスト: bash runpod_setup.sh --target-epochs 1 --debug-max-train-studies 20
#   本番:           bash runpod_setup.sh --target-epochs 7 --max-hours 20 --kaggle-ckpt-dataset rikky1201/resnet34-finetune-runpod-ckpt
setsid python train_runpod.py \
  --cache-dir "$CACHE_DIR" --assets-dir "$ASSETS_DIR" \
  --ckpt-dir "$CKPT_DIR" --resume-ckpt "$RESUME_DIR/last.ckpt" \
  "$@" \
  > "$WORKDIR/train.log" 2>&1 </dev/null &

echo "起動しました。ログ確認: tail -f $WORKDIR/train.log"
echo "プロセス確認: pgrep -fa train_runpod.py"
