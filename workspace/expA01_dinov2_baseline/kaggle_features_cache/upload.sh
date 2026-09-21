#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

EXP_DIR=".."  # workspace/expA01_dinov2_baseline/
KAGGLE="python3 -m kaggle"
DATASET_SLUG="rikky1201/rsna-knee-dinov2-base-features"

mkdir -p kaggle_dataset
cp "$EXP_DIR/data/dinov2_base_features_4405.npz" kaggle_dataset/
cp "$EXP_DIR/data/extraction_manifest.json" kaggle_dataset/
cp dataset-metadata.json kaggle_dataset/

if $KAGGLE datasets status "$DATASET_SLUG" >/dev/null 2>&1; then
  $KAGGLE datasets version -p kaggle_dataset -m "update dinov2-base features (4405/4407)"
else
  $KAGGLE datasets create -p kaggle_dataset
fi

cat <<EOF

Dataset: https://www.kaggle.com/datasets/$DATASET_SLUG
今後の学習Notebookは kernel-metadata.json の dataset_sources に
"$DATASET_SLUG" を追加すれば、DINOv2特徴の再抽出なしで学習に使える。
EOF
