"""
Gated Attention pooling（Ilse et al. 2018, "Attention-based Deep Multiple Instance Learning"）
による slice→series 集約。既存の `pooling.pool_slices_to_series`（mean+max concat, 固定・
学習不要）を置き換える学習可能なpooling層（ユーザー指示、2026-09-16）。

    e_i = w^T ( tanh(V h_i) ⊙ sigmoid(U h_i) )
    a_i = softmax(e_i)  （全スライスで正規化、合計1）
    z   = Σ a_i h_i

series→studyの7スロット集約（`pooling.pool_series_to_study`）は変更しない。

設計判断:
- まずは12ラベル共有の単一のV, U, w（`pool_slices_to_series`と同じ「1 series → 1ベクトル」
  という粒度）から始める。ラベルごとに分けるのは効果確認後（ユーザー指示）
- 出力次元はmean+max concat(2*embed_dim)ではなく embed_dim のまま（Σ a_i h_i は
  h_iと同じ次元の加重和のため）。study特徴次元は 7*(embed_dim+1) に変わる
  （mean+max版の 7*(2*embed_dim+1) より小さくなる）
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GatedAttentionPooling(nn.Module):
    """(n_slices, embed_dim) のスライス特徴 → (embed_dim,) のseries特徴に集約する。

    V, U, w は12ラベル共有（呼び出し側でstudy全体・全series・全ラベルに対し同一インスタンスを使う）。
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.U = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            h: (n_slices, embed_dim) 1 series分のスライス特徴（n_slices>=1）

        Returns:
            z: (embed_dim,) 加重和で集約されたseries特徴
            a: (n_slices,) 各スライスのattention重み（合計1）
        """
        if h.ndim != 2:
            raise ValueError(f"h は (n_slices, embed_dim) の2次元テンソルである必要があります: shape={tuple(h.shape)}")
        if h.shape[0] == 0:
            raise ValueError("h が空です（n_slices=0）")

        gate = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))  # (n_slices, hidden_dim)
        e = self.w(gate).squeeze(-1)  # (n_slices,)
        a = torch.softmax(e, dim=0)  # (n_slices,)
        z = (a.unsqueeze(-1) * h).sum(dim=0)  # (embed_dim,)
        return z, a
