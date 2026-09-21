"""
キャッシュ済み特徴を使う軽量Dataset/DataLoader。

DICOM・DINOv2には一切アクセスしない（特徴は事前抽出済みnpzから読み込み済みの
numpy配列としてメモリ上に載せる。4,405件×10,759次元でも約190MBで十分小さい）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_dataloader(
    features: np.ndarray,
    soft_targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(soft_targets).float(),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=False,
    )
