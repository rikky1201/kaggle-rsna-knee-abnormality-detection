# %% [markdown]
# # RSNA Knee Abnormality Detection — 提出用推論Notebook（デュアルバックボーン列ルーティング版）
#
# Code Competition submission（`enable_internet: false` 必須）。
# 2026-09-16 のバックボーン比較検証（本番同等データ量 gold58+pseudo4,349でのpooled OOF）で、
# DINOv2-baseとRadImageNet-ResNet50はラベルごとに得意不得意が異なることが判明した:
# - Effusion(+0.111)・Synovitis(+0.125): RadImageNetの方が明確に良い
# - 残り10ラベル: DINOv2の方が良い（特にBaker's -0.230, ACL -0.186, MCL -0.229, Medial OA -0.123）
# そのため本Notebookは**列ごとに出力元バックボーンを切り替えるルーティング推論**を行う
# （ユーザー指示、2026-09-16）。両バックボーンとも gold全58件+pseudo全件で学習した
# フルデータモデル（`expA01_dinov2_baseline/train_full.py`）を使用。
#
# 依存データセット:
# - `rikky1201/rsna-knee-dinov2-baseline-assets`: preprocessing.py/pooling.py/
#   feature_extraction.py(DINOv2)/radimagenet_feature_extraction.py/model.py +
#   学習済み `full_model_dinov2.ckpt` / `full_model_radimagenet.ckpt`
# - `rikky1201/dinov2-base-weights-offline`: facebook/dinov2-base のオフライン重み
# - `rikky1201/radimagenet-resnet50-weights`: RadImageNet ResNet50 のオフライン重み
# - competition_sources: test.csv, test_series.csv, test_series/ のDICOM本体
#
# DICOM読み込み・前処理（`preprocess_series`まで）はバックボーン非依存のため、series単位で
# 一度だけ行い、両バックボーンのforwardに使い回す（DICOM I/Oが律速というこれまでの計測結果を
# 踏まえ、二重読み込みを避ける設計）。

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
# ── パス解決（マウント階層を仮定せず実ファイルを探索する。過去のトラブルシュートの教訓）
# 素朴な rglob だと競技データセット（80万ファイル超のDICOM）配下を毎回全走査してしまい
# 約14〜17分かかることを実測で確認した（浅い3階層のみチェックする簡易版でも、
# /kaggle/input/datasets/<user>/<slug>/<file> の4階層目に対応できず同様に遅かった）。
# os.walk を深さ制限付き・既知の巨大ディレクトリ（train_series/test_series, 80万ファイル規模の
# DICOM本体）を明示的に枝刈りしながら探索することで、マウント階層の変化に頑健かつ高速にする
def _find_by_marker(root: Path, marker_relpath: str, max_depth: int = 5) -> Path:
    import os

    skip_dirnames = {"train_series", "test_series"}  # マーカーファイルが入らないと分かっている巨大ツリー
    root_depth = len(root.parts)

    if root.exists():
        for dirpath, dirnames, _filenames in os.walk(root):
            dirpath_p = Path(dirpath)
            depth = len(dirpath_p.parts) - root_depth
            if depth >= max_depth:
                dirnames.clear()  # これ以上深く潜らない
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
ASSETS_DIR = _find_by_marker(INPUT_ROOT, "full_model_dinov2.ckpt")
DINOV2_WEIGHTS_DIR = _find_by_marker(INPUT_ROOT, "model.safetensors")
RADIMAGENET_WEIGHTS_DIR = _find_by_marker(INPUT_ROOT, "RadImageNet-ResNet50_notop.pth")
logger.info("competition dir: %s", COMP_DIR)
logger.info("assets dir: %s", ASSETS_DIR)
logger.info("dinov2 weights dir: %s", DINOV2_WEIGHTS_DIR)
logger.info("radimagenet weights dir: %s", RADIMAGENET_WEIGHTS_DIR)

sys.path.append(str(ASSETS_DIR))
from feature_extraction import Dinov2FeatureExtractor  # noqa: E402
from model import FeatureStandardizer, LinearProbeHead  # noqa: E402
from pooling import pool_series_to_study, pool_slices_to_series  # noqa: E402
from preprocessing import preprocess_series  # noqa: E402
from radimagenet_feature_extraction import RadImageNetFeatureExtractor  # noqa: E402

OUTPUT_CSV = Path("/kaggle/working/submission.csv")

# ── 列ごとのバックボーン割り当て（2026-09-16のバックボーン比較検証結果に基づく。
# 本番同等データ量(gold58+pseudo4,349)でのpooled OOF: Effusion +0.111, Synovitis +0.125で
# RadImageNetが明確に優位。残り10列はDINOv2が優位（詳細はdaily_reports/20260916.md参照）
RADIMAGENET_LABELS = {"Effusion", "Synovitis"}

# %%
# ── 学習済みチェックポイントのロード（DINOv2・RadImageNet両方）
# weights_only=False: 自作チェックポイント（numpy配列のscaler統計量を含む）のため明示的に許可する
device = torch.device("cuda")


def load_head(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    scaler = FeatureStandardizer()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.std_ = ckpt["scaler_std"]
    head = LinearProbeHead(ckpt["input_dim"], ckpt["num_classes"]).to(device)
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()
    return {"labels": ckpt["labels"], "scaler": scaler, "standardize": ckpt["standardize"],
            "head": head, "input_dim": ckpt["input_dim"]}


dinov2_ckpt = load_head(ASSETS_DIR / "full_model_dinov2.ckpt")
radimagenet_ckpt = load_head(ASSETS_DIR / "full_model_radimagenet.ckpt")
if dinov2_ckpt["labels"] != radimagenet_ckpt["labels"]:
    raise AssertionError(f"2モデルのラベル順が不一致: dinov2={dinov2_ckpt['labels']}, radimagenet={radimagenet_ckpt['labels']}")
LABELS = dinov2_ckpt["labels"]
logger.info("labels (%d): %s", len(LABELS), LABELS)
logger.info("RadImageNet採用列: %s / DINOv2採用列: %s", sorted(RADIMAGENET_LABELS), sorted(set(LABELS) - RADIMAGENET_LABELS))

dinov2_extractor = Dinov2FeatureExtractor(size="base", model_path=str(DINOV2_WEIGHTS_DIR), device="cuda")
logger.info("DINOv2-base（オフライン重み）ロード完了: embed_dim=%d", dinov2_extractor.embed_dim)
radimagenet_extractor = RadImageNetFeatureExtractor(
    weights_path=str(RADIMAGENET_WEIGHTS_DIR / "RadImageNet-ResNet50_notop.pth"), device="cuda"
)
logger.info("RadImageNet-ResNet50（オフライン重み）ロード完了: embed_dim=%d", radimagenet_extractor.embed_dim)

BATCH_SIZE = 32


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


def extract_study_features(study_uid: str, series_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """DICOM読み込み・前処理はバックボーン非依存のため1回だけ行い、両バックボーンの
    forwardに使い回す（DICOM I/Oが律速というこれまでの計測結果を踏まえた設計）。
    Returns: (dinov2_study_feature, radimagenet_study_feature)
    """
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    if len(study_series) == 0:
        raise ValueError(f"test_series.csv に {study_uid} のseriesがありません")

    dinov2_series_entries = []
    radimagenet_series_entries = []
    for _, row in study_series.iterrows():
        series_dir = COMP_DIR / "test_series" / study_uid / row["SeriesInstanceUID"]
        volume = load_series_volume(series_dir)
        processed = preprocess_series(volume)
        meta = {
            "anatomical_plane": row["Anatomical_Plane"],
            "fluid_sensitive": int(row["Fluid_Sensitive"]),
            "fat_suppression": int(row["Fat_Suppression"]),
        }
        dinov2_slice_features = dinov2_extractor.extract(processed, batch_size=BATCH_SIZE)
        dinov2_series_entries.append({"feature": pool_slices_to_series(dinov2_slice_features), **meta})
        radimagenet_slice_features = radimagenet_extractor.extract(processed, batch_size=BATCH_SIZE)
        radimagenet_series_entries.append({"feature": pool_slices_to_series(radimagenet_slice_features), **meta})

    return pool_series_to_study(dinov2_series_entries), pool_series_to_study(radimagenet_series_entries)


# %%
test_df = pd.read_csv(COMP_DIR / "test.csv")
test_series_df = pd.read_csv(COMP_DIR / "test_series.csv")
logger.info("test件数: %d", len(test_df))

# 標準化後の平均（=ゼロベクトル）。学習データで実際に2/4,407件のDICOM破損が見つかっている
# （expA01_dinov2_baseline/SESSION_NOTES.md §3.10）ため、非公開test setでも同種の破損・
# 欠損メタデータに当たる可能性がある。1studyの失敗でNotebook全体をクラッシュさせず、
# 該当studyのみ「標準化後の平均特徴」で代替してsubmissionの行数を維持する
DINOV2_FALLBACK_FEATURE = np.zeros(dinov2_ckpt["input_dim"], dtype=np.float32)
RADIMAGENET_FALLBACK_FEATURE = np.zeros(radimagenet_ckpt["input_dim"], dtype=np.float32)


def predict_probs(ckpt: dict, feat: np.ndarray) -> np.ndarray:
    if ckpt["standardize"]:
        feat = ckpt["scaler"].transform(feat[np.newaxis, :])[0]
    feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = ckpt["head"](feat_t)
    return torch.sigmoid(logits).cpu().numpy()[0]


rows = []
failed_studies: dict[str, str] = {}
for i, study_uid in enumerate(test_df["StudyInstanceUID"], start=1):
    logger.info("[%d/%d] %s を処理中...", i, len(test_df), study_uid)
    try:
        dinov2_feat, radimagenet_feat = extract_study_features(study_uid, test_series_df)
    except Exception as e:  # noqa: BLE001 — 破損DICOM/欠損メタデータ等の外部データ起因の障害を1件のみ無効化するため広く捕捉
        logger.exception("[%d/%d] %s の特徴抽出に失敗。フォールバック特徴（標準化後の平均）で代替します", i, len(test_df), study_uid)
        failed_studies[study_uid] = f"{type(e).__name__}: {e}"
        dinov2_feat, radimagenet_feat = DINOV2_FALLBACK_FEATURE, RADIMAGENET_FALLBACK_FEATURE

    dinov2_probs = predict_probs(dinov2_ckpt, dinov2_feat)
    radimagenet_probs = predict_probs(radimagenet_ckpt, radimagenet_feat)

    row = {"StudyInstanceUID": study_uid}
    for j, label in enumerate(LABELS):
        row[label] = float(radimagenet_probs[j] if label in RADIMAGENET_LABELS else dinov2_probs[j])
    rows.append(row)

if failed_studies:
    logger.warning("%d/%d件のstudyでフォールバック特徴を使用しました: %s", len(failed_studies), len(test_df), failed_studies)
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
