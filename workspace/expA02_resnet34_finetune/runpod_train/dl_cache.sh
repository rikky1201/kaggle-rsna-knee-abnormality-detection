#!/usr/bin/env bash
set -euo pipefail
CACHE_DIR=/workspace/cache
mkdir -p "$CACHE_DIR"
for slug in rsna-knee-resnet34-preprocess-cache rsna-knee-resnet34-cache-b2 rsna-knee-resnet34-cache-b3 \
            rsna-knee-resnet34-cache-b4 rsna-knee-resnet34-cache-b5 rsna-knee-resnet34-cache-b6 \
            rsna-knee-resnet34-cache-b7 rsna-knee-resnet34-cache-b8; do
  echo "  - $slug"
  files=$(python -m kaggle datasets files "rikky1201/$slug" --csv | tail -n +2 | cut -d',' -f1)
  for f in $files; do
    [ -f "$CACHE_DIR/$f" ] && continue
    python -m kaggle datasets download -d "rikky1201/$slug" -f "$f" -p "$CACHE_DIR"
  done
done
echo "チャンク数: $(ls "$CACHE_DIR"/preprocess_cache_chunk_*.npz | wc -l)"
