"""
Pooling: slice → series → study の2段階プーリング。

設計方針: KAGGLE_DIRECTION.md「モデル設計方針」§4 Pooling
- スライス→series: mean pooling と max pooling を計算し concat
- series→study: Anatomical_Plane（3種）× Fluid_Sensitive（2種）で6スロット +
  その他1スロット = 計7スロット。該当 series が無いスロットはゼロベクトル。
  Fat_Suppression はスロット軸に使わず、各スロットの特徴ベクトルに 0/1 の
  スカラーとして追加結合する

設計判断（仕様に明記が無かった点）:
- 同一スロットに複数 series が属する場合（例: 同じ平面・信号種別のseriesが複数枚存在）は
  series特徴の単純平均で1本のベクトルに集約する
- 同様に Fat_Suppression スカラーも、同一スロット内で複数 series の値が混在する場合は
  平均値（0〜1の連続値になりうる）とする
- 空スロットの Fat_Suppression スカラーは 0 とする
"""

from typing import TypedDict

import numpy as np

ANATOMICAL_PLANES = ["Sagittal", "Coronal", "Axial"]
FLUID_SENSITIVE_VALUES = [0, 1]
OTHER_SLOT = "other"
# Anatomical_Plane × Fluid_Sensitive の6スロット + その他1スロット = 計7スロット
SLOT_KEYS = [f"{p}_{f}" for p in ANATOMICAL_PLANES for f in FLUID_SENSITIVE_VALUES] + [OTHER_SLOT]
N_SLOTS = len(SLOT_KEYS)


class SeriesEntry(TypedDict):
    feature: np.ndarray  # (2*feat_dim,) — pool_slices_to_series() の出力
    anatomical_plane: str
    fluid_sensitive: int
    fat_suppression: int


def pool_slices_to_series(slice_features: np.ndarray) -> np.ndarray:
    """slice特徴 (n_slices, feat_dim) → series特徴 (2*feat_dim,)（mean, max を concat）。"""
    if slice_features.ndim != 2:
        raise ValueError(f"slice_features は (n_slices, feat_dim) の2次元配列である必要があります: shape={slice_features.shape}")
    if slice_features.shape[0] == 0:
        raise ValueError("slice_features が空です（n_slices=0）")

    mean_pooled = slice_features.mean(axis=0)
    max_pooled = slice_features.max(axis=0)
    return np.concatenate([mean_pooled, max_pooled]).astype(np.float32)


def slot_key_for_series(anatomical_plane: str, fluid_sensitive: int) -> str:
    """Anatomical_Plane × Fluid_Sensitive からスロットキーを決定する。
    未知の平面・値の場合は "other" スロットに割り当てる。
    """
    key = f"{anatomical_plane}_{int(fluid_sensitive)}"
    return key if key in SLOT_KEYS[:-1] else OTHER_SLOT


def pool_series_to_study(series_list: list[SeriesEntry], feat_dim2: int | None = None) -> np.ndarray:
    """1 study 内の series 群を7スロットに割り当てて集約し、study特徴ベクトルを作る。

    Args:
        series_list: 各要素が SeriesEntry（feature, anatomical_plane, fluid_sensitive,
            fat_suppression）の辞書。1 study 分の全 series を渡す
        feat_dim2: series特徴の次元（= 2*feat_dim）。series_list が空の場合に
            ゼロベクトルのみを返すために必要（通常は series_list から自動推定できる）

    Returns:
        (N_SLOTS * (feat_dim2 + 1),) の study 特徴ベクトル
        （各スロット: [pooled_feature(feat_dim2次元), fat_suppression_scalar(1次元)]）
    """
    if series_list:
        feat_dim2 = series_list[0]["feature"].shape[0]
    elif feat_dim2 is None:
        raise ValueError("series_list が空の場合は feat_dim2 を明示的に指定してください")

    slot_features: dict[str, list[np.ndarray]] = {k: [] for k in SLOT_KEYS}
    slot_fat_suppression: dict[str, list[int]] = {k: [] for k in SLOT_KEYS}

    for s in series_list:
        if s["feature"].shape[0] != feat_dim2:
            raise ValueError(f"series特徴の次元が揃っていません: expected={feat_dim2}, actual={s['feature'].shape[0]}")
        key = slot_key_for_series(s["anatomical_plane"], s["fluid_sensitive"])
        slot_features[key].append(s["feature"])
        slot_fat_suppression[key].append(int(s["fat_suppression"]))

    slot_vectors = []
    for key in SLOT_KEYS:
        feats = slot_features[key]
        if feats:
            pooled = np.mean(np.stack(feats, axis=0), axis=0)
            fat_supp = float(np.mean(slot_fat_suppression[key]))
        else:
            pooled = np.zeros(feat_dim2, dtype=np.float32)
            fat_supp = 0.0
        slot_vectors.append(np.concatenate([pooled, [fat_supp]]).astype(np.float32))

    return np.concatenate(slot_vectors)


def study_feature_dim(feat_dim2: int) -> int:
    """pool_series_to_study() が返すベクトルの次元数を計算する（事前にモデル入力次元を知りたい場合用）。"""
    return N_SLOTS * (feat_dim2 + 1)
