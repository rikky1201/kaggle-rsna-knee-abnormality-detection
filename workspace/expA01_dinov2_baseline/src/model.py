"""
分類ヘッド: 7スロット分のstudy特徴ベクトル → 12ラベルのlogits。

設計方針: KAGGLE_DIRECTION.md「モデル設計方針」§5+8 バックボーン
「全パラメータ凍結（linear probe）。分類ヘッドのみ学習」に従い、
DINOv2側は既にキャッシュ済み特徴として固定されているため、ここでは
線形層1本（+ 入力の標準化）のみを学習対象とする。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FeatureStandardizer:
    """trainデータの統計量で正規化する（CLAUDE.mdの前処理原則: testの情報を使わない）。"""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "FeatureStandardizer":
        self.mean_ = features.mean(axis=0)
        self.std_ = features.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1e-6  # 定数特徴でゼロ割りしないためのフロア
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("fit() を先に呼び出してください")
        return (features - self.mean_) / self.std_


class LinearProbeHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
