# %% [markdown]
# # RSNA Knee Abnormality Detection — 提出用推論Notebook（attention pooling版, v004）
#
# Code Competition submission（`enable_internet: false` 必須）。
#
# v002（DINOv2-base + mean/max pooling, 単体）に対し、slice→series集約をGated Attention
# （学習可能なpooling層）に置き換えたバリアント。2026-09-19のCV検証（4fold, gold OOF
# bootstrap信頼区間）では mean+max版との統計的有意差は無かった
# （pooled OOF macro AUC: attention 0.7369 vs mean+max 0.7375, 対応ありbootstrap差分
# MACRO=-0.0009, 95%CI[-0.0499,0.0459]）が、非公開test setでの実際の挙動を確認するため
# 提出する（ユーザー指示）。v003のRadImageNetルーティングは今回含めない
# （こちらも有意差なしと確認済みのため、比較をシンプルに保つ）。
#
# Source: workspace/expA01_dinov2_baseline/results/expA01_attention_probe/full/full_model_attention.ckpt
#   （gold全58件+pseudo全件で学習, fixed_epochs=14, CV: pooled OOF macro AUC=0.7369）
#
# 依存データセット:
# - `rikky1201/rsna-knee-dinov2-baseline-assets`: preprocessing.py/pooling.py/
#   feature_extraction.py(DINOv2)/model.py/attention_pooling.py/attention_model.py +
#   学習済み `full_model_attention.ckpt`
# - `rikky1201/dinov2-base-weights-offline`: facebook/dinov2-base のオフライン重み
# - competition_sources: test.csv, test_series.csv, test_series/ のDICOM本体

# %%
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPUが利用できません。kernel-metadata.json の enable_gpu / Accelerator設定を確認してください。")
logger.info("GPU: %s", torch.cuda.get_device_name(0))


# %%
# ── パス解決（マウント階層を仮定せず実ファイルを探索する。過去のトラブルシュートの教訓。
# os.walkを深さ制限付き・既知の巨大ディレクトリを枝刈りしながら探索することで高速化する）
def _find_by_marker(root: Path, marker_relpath: str, max_depth: int = 5) -> Path:
    import os

    skip_dirnames = {"train_series", "test_series"}
    root_depth = len(root.parts)

    if root.exists():
        for dirpath, dirnames, _filenames in os.walk(root):
            dirpath_p = Path(dirpath)
            depth = len(dirpath_p.parts) - root_depth
            if depth >= max_depth:
                dirnames.clear()
                continue
            dirnames[:] = [d for d in dirnames if d not in skip_dirnames]
            if (dirpath_p / marker_relpath).exists():
                return dirpath_p

    logger.warning("深さ%d以内で '%s' が見つからなかったため、フォールバックで全体を再帰探索します（時間がかかる可能性）", max_depth, marker_relpath)
    matches = list(root.rglob(marker_relpath))
    if not matches:
        listing = [str(p.relative_to(root)) for p in root.rglob("*")][:200]
        raise FileNotFoundError(f"'{marker_relpath}' が {root} 配下に見つかりません。中身:\n  " + "\n  ".join(listing))
    return matches[0].parent


INPUT_ROOT = Path("/kaggle/input")
COMP_DIR = _find_by_marker(INPUT_ROOT, "test_series.csv")
ASSETS_DIR = _find_by_marker(INPUT_ROOT, "full_model_attention.ckpt")
DINOV2_WEIGHTS_DIR = _find_by_marker(INPUT_ROOT, "model.safetensors")
logger.info("competition dir: %s", COMP_DIR)
logger.info("assets dir: %s", ASSETS_DIR)
logger.info("dinov2 weights dir: %s", DINOV2_WEIGHTS_DIR)

sys.path.append(str(ASSETS_DIR))
from attention_model import AttentionStudyModel  # noqa: E402
from feature_extraction import Dinov2FeatureExtractor  # noqa: E402
from preprocessing import preprocess_series  # noqa: E402

OUTPUT_CSV = Path("/kaggle/working/submission.csv")

# %%
# ── 学習済みチェックポイントのロード
# weights_only=False: 自作チェックポイント（numpy配列のscaler統計量を含む）のため明示的に許可する
device = torch.device("cuda")

ckpt = torch.load(ASSETS_DIR / "full_model_attention.ckpt", map_location="cpu", weights_only=False)
LABELS = ckpt["labels"]
logger.info("labels (%d): %s", len(LABELS), LABELS)

model = AttentionStudyModel(
    embed_dim=int(ckpt["embed_dim"]), num_classes=int(ckpt["num_classes"]),
    attn_hidden_dim=int(ckpt["attn_hidden_dim"]),
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

SCALER_MEAN = ckpt["scaler_mean"]  # (embed_dim,) — slice単位DINOv2特徴の標準化統計量（train全件から算出）
SCALER_STD = ckpt["scaler_std"]
STANDARDIZE = bool(ckpt["standardize"])

dinov2_extractor = Dinov2FeatureExtractor(size="base", model_path=str(DINOV2_WEIGHTS_DIR), device="cuda")
logger.info("DINOv2-base（オフライン重み）ロード完了: embed_dim=%d", dinov2_extractor.embed_dim)

BATCH_SIZE = 32

# %%
# ── フォールバック: 破損DICOM等で特徴抽出に失敗したstudyは、モデルのhead部分に
# 直接ゼロベクトル（study_feature_dim次元）を通した予測で代替する（series_listが
# 空だとforward_studyがValueErrorで落ちるため、attn_pool自体をバイパスする設計）。
# mean+max版（v001-v003）の「標準化後の平均特徴＝ゼロベクトル」フォールバックと同じ考え方
FALLBACK_STUDY_FEATURE = torch.zeros(1, int(ckpt["study_feature_dim"]), device=device)


def predict_fallback() -> np.ndarray:
    with torch.no_grad():
        logits = model.head(FALLBACK_STUDY_FEATURE)
    return torch.sigmoid(logits).cpu().numpy()[0]


# %%
def _read_dicom_slice(path: Path) -> tuple[int, np.ndarray]:
    ds = pydicom.dcmread(str(path))
    return int(ds.InstanceNumber), ds.pixel_array.astype(np.float32)


def load_series_volume(series_dir: Path) -> np.ndarray:
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"DICOMファイルが見つかりません: {series_dir}")
    slices = [_read_dicom_slice(f) for f in dcm_files]
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices], axis=0)


def predict_study(study_uid: str, series_df: pd.DataFrame) -> np.ndarray:
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    if len(study_series) == 0:
        raise ValueError(f"test_series.csv に {study_uid} のseriesがありません")

    series_list = []
    for _, row in study_series.iterrows():
        series_dir = COMP_DIR / "test_series" / study_uid / row["SeriesInstanceUID"]
        volume = load_series_volume(series_dir)
        processed = preprocess_series(volume)
        slice_features = dinov2_extractor.extract(processed, batch_size=BATCH_SIZE)  # (n_slices, embed_dim)
        if STANDARDIZE:
            slice_features = (slice_features - SCALER_MEAN) / SCALER_STD
        series_list.append({
            "slice_features": torch.from_numpy(slice_features.astype(np.float32)),
            "anatomical_plane": row["Anatomical_Plane"],
            "fluid_sensitive": int(row["Fluid_Sensitive"]),
            "fat_suppression": int(row["Fat_Suppression"]),
        })

    with torch.no_grad():
        logits = model.forward_study(series_list, device)
    return torch.sigmoid(logits).cpu().numpy()


# %%
test_df = pd.read_csv(COMP_DIR / "test.csv")
test_series_df = pd.read_csv(COMP_DIR / "test_series.csv")
logger.info("test件数: %d", len(test_df))

rows = []
failed_studies: dict[str, str] = {}
for i, study_uid in enumerate(test_df["StudyInstanceUID"], start=1):
    logger.info("[%d/%d] %s を処理中...", i, len(test_df), study_uid)
    try:
        probs = predict_study(study_uid, test_series_df)
    except Exception as e:  # noqa: BLE001 — 破損DICOM/欠損メタデータ等の外部データ起因の障害を1件のみ無効化するため広く捕捉
        logger.exception("[%d/%d] %s の推論に失敗。フォールバック予測（ゼロ特徴）で代替します", i, len(test_df), study_uid)
        failed_studies[study_uid] = f"{type(e).__name__}: {e}"
        probs = predict_fallback()

    row = {"StudyInstanceUID": study_uid}
    for j, label in enumerate(LABELS):
        row[label] = float(probs[j])
    rows.append(row)

if failed_studies:
    logger.warning("%d/%d件のstudyでフォールバック予測を使用しました: %s", len(failed_studies), len(test_df), failed_studies)
    with open("/kaggle/working/inference_failures.json", "w") as f:
        json.dump(failed_studies, f, indent=2, ensure_ascii=False)

submission = pd.DataFrame(rows, columns=["StudyInstanceUID", *LABELS])

# %%
# ── 提出前フォーマット検証（fail-fast）
assert len(submission) == len(test_df), f"行数不一致: submission={len(submission)}, test={len(test_df)}"
assert list(submission.columns) == ["StudyInstanceUID", *LABELS], f"列名/順序が不正: {list(submission.columns)}"
assert submission[LABELS].isna().sum().sum() == 0, "欠損値が含まれています"
assert (submission[LABELS].to_numpy() >= 0).all() and (submission[LABELS].to_numpy() <= 1).all(), "値域[0,1]を超えています"
assert set(submission["StudyInstanceUID"]) == set(test_df["StudyInstanceUID"]), "StudyInstanceUIDが一致しません"

submission.to_csv(OUTPUT_CSV, index=False)
logger.info("submission.csv を保存しました: %s (%d行)", OUTPUT_CSV, len(submission))
print(submission.head())
