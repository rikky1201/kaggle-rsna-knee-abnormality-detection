"""
Attention pooling版の学習モデル（`src/attention_model.py`のflat import版。Kaggle Dataset
アセットはパッケージ構造を持たないため、相対importを絶対importに変更したもの）。

slice→series は `GatedAttentionPooling`（学習可能）、series→study は既存の7スロット集約
（`pooling.pool_series_to_study`）と同じスロット割り当てロジックのtorch版
（`pool_series_to_study_torch`、勾配がattention側まで流れるようにするため）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from attention_pooling import GatedAttentionPooling
from model import LinearProbeHead
from pooling import SLOT_KEYS, slot_key_for_series, study_feature_dim


def pool_series_to_study_torch(series_list: list[dict], feat_dim: int, device: torch.device) -> Tensor:
    """`pooling.pool_series_to_study`と同じスロット割り当てロジックのtorch版（勾配を保持する）。"""
    if not series_list:
        raise ValueError("series_list が空です")

    dtype = series_list[0]["feature"].dtype
    slot_features: dict[str, list[Tensor]] = {k: [] for k in SLOT_KEYS}
    slot_fat_suppression: dict[str, list[float]] = {k: [] for k in SLOT_KEYS}

    for s in series_list:
        key = slot_key_for_series(s["anatomical_plane"], s["fluid_sensitive"])
        slot_features[key].append(s["feature"])
        slot_fat_suppression[key].append(float(s["fat_suppression"]))

    slot_vectors = []
    for key in SLOT_KEYS:
        feats = slot_features[key]
        if feats:
            pooled = torch.stack(feats, dim=0).mean(dim=0)
            fat_supp = sum(slot_fat_suppression[key]) / len(slot_fat_suppression[key])
        else:
            pooled = torch.zeros(feat_dim, device=device, dtype=dtype)
            fat_supp = 0.0
        fat_t = torch.tensor([fat_supp], device=device, dtype=pooled.dtype)
        slot_vectors.append(torch.cat([pooled, fat_t]))

    return torch.cat(slot_vectors)


class AttentionStudyModel(nn.Module):
    """1 studyのseries群（スライス特徴+メタデータ）→ 12ラベルのlogits。"""

    def __init__(self, embed_dim: int, num_classes: int, attn_hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_pool = GatedAttentionPooling(embed_dim, attn_hidden_dim)
        input_dim = study_feature_dim(embed_dim)
        self.head = LinearProbeHead(input_dim, num_classes)

    def forward_study(self, series_list: list[dict], device: torch.device) -> Tensor:
        """
        Args:
            series_list: 各要素 {"slice_features": Tensor(n_slices, embed_dim), "anatomical_plane": str,
                "fluid_sensitive": int, "fat_suppression": int}

        Returns:
            (num_classes,) logits
        """
        if not series_list:
            raise ValueError("series_list が空です（studyにseriesが1件も無い）")

        pooled_entries = []
        for s in series_list:
            z, _ = self.attn_pool(s["slice_features"].to(device))
            pooled_entries.append({
                "feature": z,
                "anatomical_plane": s["anatomical_plane"],
                "fluid_sensitive": s["fluid_sensitive"],
                "fat_suppression": s["fat_suppression"],
            })
        study_feat = pool_series_to_study_torch(pooled_entries, self.embed_dim, device)
        return self.head(study_feat.unsqueeze(0)).squeeze(0)
