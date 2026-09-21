# %% [markdown]
# # RSNA Knee: 競技データのディレクトリ構造・DICOM読み込み確認（GPU不要, 探索専用）
#
# 画像本体（DICOM, 500GB規模）はローカルにダウンロードしない方針のため、実際の
# ディレクトリ構造・ファイル命名規則・pydicomでの読み込み結果を一度も確認していない。
# DINOv2特徴抽出パイプラインを組む前に、まずこの軽量Notebookで構造を把握する。

# %%
from pathlib import Path

input_root = Path("/kaggle/input")
print("=== /kaggle/input 直下 ===")
for p in sorted(input_root.iterdir()):
    print(" ", p)

# %%
# ── 階層を再帰的に浅く覗く（500GB規模なので深さ・件数を絞って安全に）
def list_tree(path: Path, depth: int = 0, max_depth: int = 5, max_entries: int = 15) -> None:
    if depth > max_depth:
        return
    try:
        entries = sorted(path.iterdir())[:max_entries]
    except (PermissionError, NotADirectoryError, OSError) as e:
        print("  " * depth + f"<読み取り不可: {e}>")
        return
    for e in entries:
        marker = "/" if e.is_dir() else ""
        print("  " * depth + e.name + marker)
        if e.is_dir():
            list_tree(e, depth + 1, max_depth, max_entries)


print("\n=== ディレクトリツリー（深さ5, 各階層最大15件） ===")
list_tree(input_root)

# %%
# ── train_series.csv から既知の StudyInstanceUID / SeriesInstanceUID で実ファイルを探す
KNOWN_STUDY_UID = "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260"
KNOWN_SERIES_UID = "1.2.826.0.1.3680043.8.498.12343110195036213483454091715412333772"

print(f"\n=== 既知の StudyInstanceUID を検索: {KNOWN_STUDY_UID} ===")
study_matches = list(input_root.rglob(f"*{KNOWN_STUDY_UID}*"))
print(f"マッチ数: {len(study_matches)}")
for m in study_matches[:20]:
    print(" ", m)

print(f"\n=== 既知の SeriesInstanceUID を検索: {KNOWN_SERIES_UID} ===")
series_matches = list(input_root.rglob(f"*{KNOWN_SERIES_UID}*"))
print(f"マッチ数: {len(series_matches)}")
for m in series_matches[:20]:
    print(" ", m)

# %%
# ── DICOMファイルが見つかれば pydicom で実際に読み込んでみる
dicom_candidates = [m for m in series_matches if m.is_file()]
if not dicom_candidates:
    # series配下のディレクトリの中身を見る
    series_dirs = [m for m in series_matches if m.is_dir()]
    for d in series_dirs:
        dicom_candidates.extend(sorted(d.iterdir())[:5])

print(f"\n=== DICOM候補ファイル ({len(dicom_candidates)}件) ===")
for f in dicom_candidates[:10]:
    print(" ", f, f.stat().st_size if f.is_file() else "-", "bytes")

if dicom_candidates:
    try:
        import pydicom
    except ImportError:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydicom"], check=True)
        import pydicom

    sample = dicom_candidates[0]
    print(f"\n=== pydicomで読み込み: {sample} ===")
    ds = pydicom.dcmread(str(sample))
    print("Modality:", getattr(ds, "Modality", None))
    print("Rows/Columns:", getattr(ds, "Rows", None), getattr(ds, "Columns", None))
    print("PixelSpacing:", getattr(ds, "PixelSpacing", None))
    print("BitsAllocated:", getattr(ds, "BitsAllocated", None))
    print("PhotometricInterpretation:", getattr(ds, "PhotometricInterpretation", None))
    print("SeriesInstanceUID:", getattr(ds, "SeriesInstanceUID", None))
    print("StudyInstanceUID:", getattr(ds, "StudyInstanceUID", None))
    print("InstanceNumber:", getattr(ds, "InstanceNumber", None))
    arr = ds.pixel_array
    print("pixel_array shape/dtype:", arr.shape, arr.dtype, "min/max:", arr.min(), arr.max())
else:
    print("DICOM候補が見つかりませんでした。ディレクトリツリーの出力を確認してパスを見直す必要があります。")

# %%
# ── 1 series 分の全スライス数を数える（想定シリーズの完全なディレクトリが分かれば）
series_dirs = [m for m in series_matches if m.is_dir()]
for d in series_dirs:
    files = sorted(d.iterdir())
    print(f"\n{d}: {len(files)} files")
    for f in files[:5]:
        print("  ", f.name)
