"""
ソフトターゲット + クラス不均衡対応の損失関数。

設計方針: KAGGLE_DIRECTION.md「モデル設計方針」§2+6
  ℓ_j = -[ π_j・ỹ_j・log(p_j) + (1-ỹ_j)・log(1-p_j) ]
  L = Σ_{j=1}^{12} ℓ_j

これは `F.binary_cross_entropy_with_logits(..., pos_weight=π)` の要素ごとの定義と
完全に一致する（pos_weightは正例項 y・log(p) にのみ乗算され、負例項には掛からない）。
ラベル軸(12)を sum してから バッチ軸を mean するために reduction='none' を使う
（デフォルトの 'mean' は全要素(batch×12)で割ってしまい L=Σ_j ℓ_j の定義とズレるため）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SoftTargetBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        if pos_weight.ndim != 1:
            raise ValueError(f"pos_weight は1次元である必要があります: shape={pos_weight.shape}")
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != soft_targets.shape:
            raise ValueError(f"logits と soft_targets の shape が一致しません: {logits.shape} vs {soft_targets.shape}")
        per_element = F.binary_cross_entropy_with_logits(
            logits, soft_targets, pos_weight=self.pos_weight, reduction="none",
        )  # (batch, 12)
        per_sample = per_element.sum(dim=1)  # L = Σ_j ℓ_j（サンプルごと）
        return per_sample.mean()  # バッチ平均（SGD用のスカラー損失）
