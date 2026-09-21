# %% [markdown]
# # report-labeler 本番適用: 4,349件の未ラベルレポートに疑似ラベルを生成
#
# `kaggle_qwen_benchmark/` で検証した Qwen2.5-14B-Instruct-AWQ + vLLM + guided decoding
# （58件検証: macro F1 = 0.693, exact_match = 8.6%, 1件あたり約2.90秒）と全く同じ構成を
# `data/unlabeled_4349.csv`（12ラベル全欠損の4,349件）に適用する。
#
# 推定所要時間: 2.9s × 4,349 ≈ 3.5時間 + モデルロード約200秒
# チャンク単位（CHUNK_SIZE件ごと）で `/kaggle/working/pseudo_labels.csv` に逐次追記する。
# 途中でエラー終了しても、`kaggle kernels output` でその時点までの結果を回収できるようにするため。

# %%
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "vllm"], check=True)

# %%
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPUが利用できません。kernel-metadata.json の enable_gpu / Accelerator設定を確認してください。")
logger.info("GPU: %s / capability: sm_%d%d", torch.cuda.get_device_name(0), *torch.cuda.get_device_capability(0))

# %%
# ── パス解決（Kaggle 環境を自動検出）。assets データセットを sys.path に追加して
#      prompt_template.py をそのまま import する（検証時と単一の真実源を共有）
def _find_assets_dir() -> Path:
    # マウントパスの階層（/kaggle/input/<slug>/ か /kaggle/input/datasets/<user>/<slug>/ か）に
    # 依存しないよう、ディレクトリ名を推測せず prompt_template.py を再帰的に探す
    input_root = Path("/kaggle/input")
    matches = list(input_root.rglob("prompt_template.py"))
    if matches:
        return matches[0].parent
    # 診断用: /kaggle/input 配下の実際の中身をエラーメッセージに含める
    if input_root.exists():
        listing = [str(p.relative_to(input_root)) for p in input_root.rglob("*")]
    else:
        listing = ["<'/kaggle/input' 自体が存在しません>"]
    raise FileNotFoundError(
        "report-labeler assets dataset (prompt_template.py) が /kaggle/input 配下に見つかりません。"
        " kernel-metadata.json の dataset_sources を確認してください。\n"
        "実際の /kaggle/input の中身:\n  " + "\n  ".join(listing)
    )


ASSETS_DIR = _find_assets_dir()
sys.path.append(str(ASSETS_DIR))

import prompt_template as pt  # noqa: E402

UNLABELED_CSV = ASSETS_DIR / "unlabeled_4349.csv"
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_CSV = OUTPUT_DIR / "pseudo_labels.csv"

logger.info("assets dir: %s", ASSETS_DIR)
logger.info("prompt_version: %s", pt.PROMPT_VERSION)

# %%
# ── 推論パラメータ（58件検証と完全に同じ構成。58件検証: macro F1=0.693, exact_match=8.6%）
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"
MAX_NEW_TOKENS = 256
GPU_MEMORY_UTILIZATION = 0.90
CHUNK_SIZE = 300  # この件数ごとに /kaggle/working/pseudo_labels.csv に追記する

# %%
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

logger.info("モデルをロードします（vLLM）: %s", MODEL_ID)
t_load_start = time.time()
llm = LLM(
    model=MODEL_ID,
    quantization="awq",
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_model_len=4096,
    dtype="float16",
)
logger.info("モデルロード完了: %.1fs", time.time() - t_load_start)

output_schema = pt.build_output_schema()
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_NEW_TOKENS,
    structured_outputs=StructuredOutputsParams(json=output_schema),
)

# %%
# ── 4,349件をチャンク単位でバッチ推論し、逐次 CSV に追記する
df = pd.read_csv(UNLABELED_CSV)
logger.info("対象件数: %d 件 / model=%s / prompt=%s / chunk_size=%d", len(df), MODEL_ID, pt.PROMPT_VERSION, CHUNK_SIZE)

if OUTPUT_CSV.exists():
    OUTPUT_CSV.unlink()

n_written = 0
n_parse_failed = 0
total_generated_tokens = 0
t_all_start = time.time()

for chunk_start in range(0, len(df), CHUNK_SIZE):
    chunk = df.iloc[chunk_start: chunk_start + CHUNK_SIZE]
    chunk_no = chunk_start // CHUNK_SIZE + 1
    n_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info("[chunk %d/%d] %d件を推論中... (累計 %d/%d)", chunk_no, n_chunks, len(chunk), chunk_start, len(df))

    conversations = [
        [
            {"role": "system", "content": pt.build_system_prompt()},
            {"role": "user", "content": pt.build_user_prompt(report)},
        ]
        for report in chunk["Report"]
    ]

    t_chunk_start = time.time()
    outputs = llm.chat(conversations, sampling_params=sampling_params)
    chunk_sec = time.time() - t_chunk_start

    rows = []
    for study_id, output in zip(chunk["StudyInstanceUID"], outputs):
        completion = output.outputs[0]
        total_generated_tokens += len(completion.token_ids)
        try:
            parsed = json.loads(completion.text)
            pt.validate_output(parsed)
        except (json.JSONDecodeError, ValueError) as e:
            # guided decodingによりほぼ発生しない想定だが、4,349件規模のバッチジョブで
            # 1件の異常のために全体を落とすのは非効率なため、ログに残して当該件のみスキップする
            # （fail-fastの原則は「握りつぶさない」ことであり、ここではログで可視化した上で
            # 個別にスキップし、最終サマリーで件数を報告する形にする）
            n_parse_failed += 1
            logger.warning("JSON解析/検証失敗（スキップ）: %s | %s | raw=%r", study_id, e, completion.text[:300])
            continue
        row = {"StudyInstanceUID": study_id}
        for label in pt.LABELS:
            row[label] = parsed[label]
        rows.append(row)

    chunk_df = pd.DataFrame(rows, columns=["StudyInstanceUID", *pt.LABELS])
    chunk_df.to_csv(OUTPUT_CSV, mode="a", header=(chunk_no == 1), index=False)
    n_written += len(rows)

    elapsed = time.time() - t_all_start
    rate = elapsed / (chunk_start + len(chunk))
    eta_sec = rate * (len(df) - (chunk_start + len(chunk)))
    logger.info(
        "[chunk %d/%d] 完了: %.1fs（1件 %.2fs）| 累計保存 %d件 | 失敗 %d件 | ETA %.1f分",
        chunk_no, n_chunks, chunk_sec, chunk_sec / len(chunk), n_written, n_parse_failed, eta_sec / 60,
    )

t_all_total = time.time() - t_all_start
logger.info(
    "全件完了: %d/%d 件保存（失敗 %d件）| 総推論時間 %.1f分（1件平均 %.2fs）| 平均生成トークン数 %.1f",
    n_written, len(df), n_parse_failed, t_all_total / 60, t_all_total / len(df), total_generated_tokens / len(df),
)

# %%
# ── サニティチェック: 58件gold labelの陽性率と大きく乖離していないか確認する
result_df = pd.read_csv(OUTPUT_CSV)
if len(result_df) != len(df) - n_parse_failed:
    raise AssertionError(f"保存件数が一致しません: expected={len(df) - n_parse_failed}, actual={len(result_df)}")

summary = {"n_total": len(df), "n_written": n_written, "n_parse_failed": n_parse_failed}
for label in pt.LABELS:
    summary[f"positive_rate_{label}"] = float(result_df[label].mean())

summary_path = OUTPUT_DIR / "pseudo_labels_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
logger.info("サマリーを保存しました: %s", summary_path)
print(json.dumps(summary, indent=2))
