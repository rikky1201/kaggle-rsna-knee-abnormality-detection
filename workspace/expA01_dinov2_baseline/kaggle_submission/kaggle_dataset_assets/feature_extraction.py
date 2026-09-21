"""
DINOv2（凍結）による特徴抽出。

DICOM読み込み・series単位のvolume構築はKaggle Notebook側（実データ環境）で行うため、
ここでは「前処理済みのnumpy画像バッチをモデルに通して特徴を得る」部分をラップする。

設計方針: KAGGLE_DIRECTION.md「モデル設計方針」§5+8 バックボーン
- DINOv2（small または base サイズ。Efficiencyトラック考慮でlarge/giantは避ける）
- 全パラメータ凍結（linear probe）。分類ヘッドのみ学習（本ファイルは特徴抽出のみ担当）
"""

from __future__ import annotations

import numpy as np
import torch

DINOV2_MODEL_IDS = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
}
DINOV2_EMBED_DIMS = {
    "small": 384,
    "base": 768,
}


class Dinov2FeatureExtractor:
    """凍結DINOv2で画像バッチ → CLSトークン特徴ベクトルを抽出するラッパー。"""

    def __init__(self, size: str = "base", device: str | None = None, model_path: str | None = None):
        """
        Args:
            size: "small" または "base"（embed_dim の決定に使用）
            model_path: 指定時はこのローカルパスから重みをロードする（Code Competition の
                enable_internet:false 環境向け。事前に `save_pretrained()` で保存したディレクトリを渡す）。
                None の場合は HuggingFace Hub から `facebook/dinov2-{size}` を取得する（要インターネット）
        """
        if size not in DINOV2_MODEL_IDS:
            raise ValueError(f"size は {list(DINOV2_MODEL_IDS)} のいずれかである必要があります: {size}")

        from transformers import AutoImageProcessor, AutoModel

        model_id = model_path if model_path is not None else DINOV2_MODEL_IDS[size]
        self.size = size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = DINOV2_EMBED_DIMS[size]

    @torch.no_grad()
    def extract(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """images: (N, H, W) の float32 配列（グレースケール, 値域 [0, 1] 想定）。

        Returns:
            (N, embed_dim) の特徴ベクトル（CLSトークン）
        """
        if images.ndim != 3:
            raise ValueError(f"images は (N, H, W) の3次元配列である必要があります: shape={images.shape}")

        # グレースケール → 3チャンネル化（DINOv2はRGB入力想定）。channels-last (N,H,W,3)
        images_rgb = np.repeat(images[:, :, :, None], 3, axis=3)

        features = []
        for i in range(0, len(images_rgb), batch_size):
            batch = list(images_rgb[i: i + batch_size])
            # 既に percentile_clip_normalize で [0,1] にスケーリング済みのため
            # do_rescale=False で二重リスケールを防ぐ（processor側は mean/std 正規化とresizeのみ行う）
            inputs = self.processor(images=batch, return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls_features = outputs.last_hidden_state[:, 0, :]
            features.append(cls_features.cpu().numpy())

        return np.concatenate(features, axis=0).astype(np.float32)
