"""
スライス単位DINOv2特徴（ragged array形式のnpz、`kaggle_dinov2_extract_slices_*`が生成）を
読み込み、study_uid単位でseries群（スライス特徴+メタデータ）を引けるようにするストア。

npzフォーマット（`kaggle_dinov2_extract_slices_benchmark/extract_slices_benchmark_notebook.py`
および本番版が保存する形式）:
    slice_features: (n_slices_total, embed_dim) float32
    slice_to_series: (n_slices_total,) int32 — 各スライスがどのseries(通し番号)に属するか
        （抽出時にseries単位で連続してappendしているため、通し番号ごとに連続区間になっている前提）
    series_study_uid: (n_series_total,) str
    series_instance_uid: (n_series_total,) str
    series_anatomical_plane: (n_series_total,) str
    series_fluid_sensitive: (n_series_total,) int
    series_fat_suppression: (n_series_total,) int
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SliceFeatureStore:
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=True)
        self.slice_features = data["slice_features"]  # (n_slices_total, embed_dim)
        self.embed_dim = self.slice_features.shape[1]

        slice_to_series = data["slice_to_series"]
        self.series_study_uid = data["series_study_uid"]
        self.series_anatomical_plane = data["series_anatomical_plane"]
        self.series_fluid_sensitive = data["series_fluid_sensitive"]
        self.series_fat_suppression = data["series_fat_suppression"]

        n_series = len(self.series_study_uid)
        counts = np.bincount(slice_to_series, minlength=n_series)
        if counts.sum() != len(slice_to_series):
            raise AssertionError("slice_to_seriesの値がseries数の範囲外です（データ破損の可能性）")
        offsets = np.concatenate([[0], np.cumsum(counts)])
        # 抽出時にseries単位で連続してappendしている前提のため、実際に連続区間になっているか検証する
        expected = np.repeat(np.arange(n_series), counts)
        if not np.array_equal(slice_to_series, expected):
            raise AssertionError(
                "slice_to_seriesがseries単位で連続していません（抽出時の前提が崩れています）。"
                "np.bincountによるoffset計算が使えないため、抽出コードを確認してください。"
            )
        self._series_slice_range = list(zip(offsets[:-1].tolist(), offsets[1:].tolist()))

        self.study_to_series: dict[str, list[int]] = {}
        for series_idx, study_uid in enumerate(self.series_study_uid):
            self.study_to_series.setdefault(str(study_uid), []).append(series_idx)

    def has_study(self, study_uid: str) -> bool:
        return study_uid in self.study_to_series

    def slice_mask_for_studies(self, study_uids) -> np.ndarray:
        """指定したstudy_uid群に属するスライスをTrueとする、全スライス分のbool配列を返す。

        fold単位の標準化（train側study群だけの統計量を使う）でスライスを選別するために使う。
        """
        study_uid_set = set(study_uids)
        mask = np.zeros(self.slice_features.shape[0], dtype=bool)
        for series_idx, study_uid in enumerate(self.series_study_uid):
            if str(study_uid) in study_uid_set:
                start, end = self._series_slice_range[series_idx]
                mask[start:end] = True
        return mask

    def get_study_series(self, study_uid: str, features_override: np.ndarray | None = None) -> list[dict]:
        """study_uidに属する全series分の {slice_features(Tensor), anatomical_plane, fluid_sensitive,
        fat_suppression} を返す。

        Args:
            features_override: 指定した場合、self.slice_features の代わりにこの配列（同じ
                shape・スライス順）からスライス特徴を取り出す。fold単位で標準化した配列を
                使う場合に使用する（train.pyのFeatureStandardizerと同じ「trainの統計量で
                正規化」原則を、study単位ではなくslice単位の特徴に適用するため）。
        """
        series_idx_list = self.study_to_series.get(study_uid)
        if not series_idx_list:
            raise KeyError(f"study_uid={study_uid} がストアに存在しません")

        source = self.slice_features if features_override is None else features_override
        if source.shape != self.slice_features.shape:
            raise ValueError(
                f"features_override の shape が self.slice_features と一致しません: "
                f"{source.shape} vs {self.slice_features.shape}"
            )

        out = []
        for idx in series_idx_list:
            start, end = self._series_slice_range[idx]
            out.append({
                "slice_features": torch.from_numpy(source[start:end]).float(),
                "anatomical_plane": str(self.series_anatomical_plane[idx]),
                "fluid_sensitive": int(self.series_fluid_sensitive[idx]),
                "fat_suppression": int(self.series_fat_suppression[idx]),
            })
        return out
