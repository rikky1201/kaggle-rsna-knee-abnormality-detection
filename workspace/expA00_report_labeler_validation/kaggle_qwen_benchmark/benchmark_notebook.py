# %% [markdown]
# # report-labeler 候補モデル比較: Qwen2.5-14B-Instruct-AWQ (vLLM)
#
# v1（`benchmark_notebook_hf_transformers_archive.py`, 素の transformers.generate() +
# bitsandbytes nf4）は58件に約2時間かかった（1件平均 約124秒、JSON再試行は0回にも関わらず
# 40〜55秒 と 90〜190秒 の二峰性）。原因調査の結論:
# - モデル再ロードなし、GPUは Tesla T4 を正しく使用（CPU実行ではない）を確認済み
# - bitsandbytes nf4 の逆量子化カーネルは T4 上で最適化されておらず生成速度そのものが遅い
# - **出力フォーマットをAPI/エンジン側で強制していなかった**ため、モデルがJSON出力後も
#   停止条件を厳密に満たさず、ケースによって512トークン近くまで生成し続けていた
#   （二峰性の直接の原因と推測）
#
# → vLLM + AWQ量子化 + guided decoding（JSON Schema強制）に置き換えて再検証する。
# guided decodingで「有効なJSONが完成したら即停止」を保証し、AWQはTuring(T4)向けに
# 最適化されたカーネルを持つため、bitsandbytes nf4より高速な想定。
#
# 比較対象:
# - Claude Opus 5 (v2_fewshot): macro F1 = 0.733, 1件あたり数秒, コスト約$1.28
# - Qwen2.5-14B-Instruct (bitsandbytes nf4, transformers): macro F1 = 0.696, 1件平均約124秒, コスト$0

# %%
# vLLMをインストール。Tesla T4 (sm_75) は現行torchビルドの対応範囲内
# （sm_70以降）なので、P100の時のようなtorchピン止めは不要。
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

# ── 0. GPU確認（fail-fast）
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPUが利用できません。kernel-metadata.json の enable_gpu / Accelerator設定を確認してください。")
logger.info("GPU: %s / capability: sm_%d%d", torch.cuda.get_device_name(0), *torch.cuda.get_device_capability(0))
logger.info("GPU memory (allocated/reserved, ロード前): %.2f GB / %.2f GB",
            torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)

# %%
# ── 1. パス解決（Kaggle 環境を自動検出）。assets データセットを sys.path に追加して
#      prompt_template.py / metrics.py をそのまま import する（ローカルコードと単一の真実源を共有）
def _find_assets_dir() -> Path:
    candidates = sorted(Path("/kaggle/input").glob("*report-labeler*"))
    for p in candidates:
        if (p / "prompt_template.py").exists():
            return p
    raise FileNotFoundError(
        "report-labeler assets dataset (prompt_template.py) が /kaggle/input 配下に見つかりません。"
        " kernel-metadata.json の dataset_sources を確認してください。"
    )


ASSETS_DIR = _find_assets_dir()
sys.path.append(str(ASSETS_DIR))

import prompt_template as pt  # noqa: E402
from metrics import compute_metrics  # noqa: E402

VALIDATION_CSV = ASSETS_DIR / "validation_58.csv"
OUTPUT_DIR = Path("/kaggle/working")

logger.info("assets dir: %s", ASSETS_DIR)
logger.info("prompt_version: %s", pt.PROMPT_VERSION)

# %%
# ── 2. 推論パラメータ（ここだけで管理する）
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"  # AWQ 4bit 事前量子化済み（9.99GB, Turing世代のT4向け最適化カーネルあり）
MAX_NEW_TOKENS = 256  # guided decodingでJSON完成後は即停止するため、512は不要
GPU_MEMORY_UTILIZATION = 0.90

# %%
# ── 3. vLLM でモデルロード
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams  # 新API（旧 GuidedDecodingParams / guided_decoding は非推奨）

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
logger.info("GPU memory (allocated/reserved, ロード後): %.2f GB / %.2f GB",
            torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)

output_schema = pt.build_output_schema()
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_NEW_TOKENS,
    structured_outputs=StructuredOutputsParams(json=output_schema),
)

# %%
# ── 4. 58件をバッチ推論（vLLMのcontinuous batchingで全件を1回のllm.chat呼び出しに渡す）
val_df = pd.read_csv(VALIDATION_CSV)
logger.info("対象件数: %d 件 / model=%s / prompt=%s", len(val_df), MODEL_ID, pt.PROMPT_VERSION)

conversations = [
    [
        {"role": "system", "content": pt.build_system_prompt()},
        {"role": "user", "content": pt.build_user_prompt(report)},
    ]
    for report in val_df["Report"]
]

t_infer_start = time.time()
outputs = llm.chat(conversations, sampling_params=sampling_params)
t_infer_total = time.time() - t_infer_start
logger.info(
    "バッチ推論完了: 全%d件で %.1fs（1件平均 %.2fs）",
    len(outputs), t_infer_total, t_infer_total / len(outputs),
)

# %%
# ── 5. 出力パース + 検証（guided decodingにより有効なJSONのみ生成されるはずだが、
#      json.loadsでのパースとvalidate_output()は必ず経由する）
rows = []
total_generated_tokens = 0
for study_id, output in zip(val_df["StudyInstanceUID"], outputs):
    completion = output.outputs[0]
    n_tokens = len(completion.token_ids)
    total_generated_tokens += n_tokens

    parsed = json.loads(completion.text)
    pt.validate_output(parsed)

    row_result = {"StudyInstanceUID": study_id, "_n_tokens": n_tokens}
    for label in pt.LABELS:
        row_result[f"pred_{label}"] = parsed[label]
    rows.append(row_result)

logger.info(
    "生成トークン数: 平均 %.1f / 最大 %d / 最小 %d（旧transformers版の二峰性の原因診断用）",
    total_generated_tokens / len(rows),
    max(r["_n_tokens"] for r in rows),
    min(r["_n_tokens"] for r in rows),
)

pred_df = pd.DataFrame(rows)
true_df = val_df[["StudyInstanceUID", *pt.LABELS]].rename(columns={l: f"true_{l}" for l in pt.LABELS})
merged = true_df.merge(pred_df, on="StudyInstanceUID", how="inner")
if len(merged) != len(val_df):
    raise AssertionError(f"結合後の件数が一致しません: true={len(val_df)}, merged={len(merged)}")

merged_path = OUTPUT_DIR / "predictions_with_truth.csv"
merged.drop(columns=["_n_tokens"]).to_csv(merged_path, index=False)
logger.info("予測結果を保存しました: %s", merged_path)

# %%
# ── 6. 評価指標（Claude / 旧transformers版と同一の metrics.compute_metrics を使用）
metrics = compute_metrics(merged, pt.LABELS)
metrics["run_config"] = {
    "model": MODEL_ID,
    "engine": "vllm",
    "prompt_version": pt.PROMPT_VERSION,
    "quantization": "awq",
    "max_new_tokens": MAX_NEW_TOKENS,
    "n_studies": len(merged),
    "total_inference_sec": t_infer_total,
    "sec_per_item": t_infer_total / len(merged),
    "avg_generated_tokens": total_generated_tokens / len(merged),
    "model_load_sec": time.time() - t_load_start - t_infer_total,
}
metrics_path = OUTPUT_DIR / "metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
logger.info("評価指標を保存しました: %s", metrics_path)

logger.info("=== macro 指標 (Qwen2.5-14B-AWQ, vLLM) === %s", metrics["macro"])
logger.info("exact_match_rate: %.4f", metrics["exact_match_rate"])
logger.info("1件あたり推論時間: %.2fs（旧transformers+bnb版: 約124s）", t_infer_total / len(merged))
print("macro:", json.dumps(metrics["macro"], indent=2))
print("exact_match_rate:", metrics["exact_match_rate"])
print("sec_per_item:", t_infer_total / len(merged))
print("--- 比較対象 (Claude Opus 5, v2_fewshot): macro F1 = 0.733 ---")
print("--- 比較対象 (Qwen2.5-14B transformers+bnb nf4): macro F1 = 0.696, sec_per_item ≈ 124s ---")
