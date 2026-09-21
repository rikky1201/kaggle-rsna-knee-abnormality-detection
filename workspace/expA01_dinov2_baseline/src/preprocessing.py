"""
volume前処理: percentile clip正規化 + スライス間引き。

DICOM読み込み自体はKaggle Notebook側（実データがアタッチされた環境）で行うため、
ここでは numpy 配列 (n_slices, H, W) を受け取る純粋関数として実装する。
ローカルでも合成データでユニットテスト可能（`tests/test_preprocessing.py`）。

設計方針: KAGGLE_DIRECTION.md「モデル設計方針」§3 前処理
- 強度正規化: study(volume)ごとに percentile clip（0.5%〜99.5%）→ [0,1] スケーリング
- スライス数: 60枚を超える series のみ均等間隔で間引いて60枚に抑える。
  60枚以下（大多数）はそのまま使用（mean/maxプーリングが可変長を吸収するため）
"""

import numpy as np

DEFAULT_CLIP_LOW = 0.5
DEFAULT_CLIP_HIGH = 99.5
MAX_SLICES = 60


def percentile_clip_normalize(
    volume: np.ndarray, low: float = DEFAULT_CLIP_LOW, high: float = DEFAULT_CLIP_HIGH
) -> np.ndarray:
    """study(volume)ごとに percentile clip で強度正規化し、[0, 1] にスケーリングする。

    Args:
        volume: (n_slices, H, W) の numpy 配列（1 series = 1 volume として扱う）
        low, high: クリップするパーセンタイル（%）

    Returns:
        (n_slices, H, W) の float32 配列、値域 [0, 1]
    """
    if volume.ndim != 3:
        raise ValueError(f"volume は (n_slices, H, W) の3次元配列である必要があります: shape={volume.shape}")

    lo, hi = np.percentile(volume, [low, high])
    if hi <= lo:
        raise ValueError(f"percentile範囲が不正です（定数画像の可能性）: low={lo}, high={hi}")

    clipped = np.clip(volume, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return normalized.astype(np.float32)


def subsample_slices(volume: np.ndarray, max_slices: int = MAX_SLICES) -> np.ndarray:
    """max_slices を超える series のみ均等間隔で間引く。max_slices 以下はそのまま返す。

    Args:
        volume: (n_slices, H, W) の numpy 配列
        max_slices: 上限スライス数

    Returns:
        (min(n_slices, max_slices), H, W) の配列
    """
    if volume.ndim != 3:
        raise ValueError(f"volume は (n_slices, H, W) の3次元配列である必要があります: shape={volume.shape}")

    n_slices = volume.shape[0]
    if n_slices <= max_slices:
        return volume

    # 均等間隔でインデックスを選ぶ（両端を含む）
    indices = np.linspace(0, n_slices - 1, max_slices).round().astype(int)
    indices = np.unique(indices)  # 稀な重複（n_slicesが小さい場合等）を除去
    return volume[indices]


def preprocess_series(
    volume: np.ndarray,
    low: float = DEFAULT_CLIP_LOW,
    high: float = DEFAULT_CLIP_HIGH,
    max_slices: int = MAX_SLICES,
) -> np.ndarray:
    """1 series 分の前処理を通しで実行する: 正規化 → スライス間引き。

    percentileは間引き前の全スライスから計算する（間引き後だと統計が不安定になるため、
    正規化を先に行う）。
    """
    normalized = percentile_clip_normalize(volume, low=low, high=high)
    return subsample_slices(normalized, max_slices=max_slices)
