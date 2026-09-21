# %% [markdown]
# # report-labeler 候補モデル比較: Qwen2.5-14B-Instruct
#
# `workspace/expA00_report_labeler_validation/` の report-labeler 検証と同じ
# `prompt_template.py`（判定ルール v2_fewshot） / `metrics.py`（評価ロジック）を
# `rikky1201/rsna-knee-report-labeler-assets` データセット経由で import し、
# 同じ58件の gold label データセットで Qwen2.5-14B-Instruct を評価する。
#
# 比較対象: Claude Opus 5 (v2_fewshot) — macro F1 = 0.733, exact_match = 6.9%
# （`results/claude-opus-5/v2_fewshot_20260914_152840/metrics.json`）

# %%
# 事前インストール済みの torch を巻き込んでアップグレードしない（重要）。
# v3実行で「pip install -U transformers accelerate bitsandbytes」がtorchも
# 巻き込んでアップグレードし、Kaggle既定のGPU（Tesla P100, CUDA capability sm_60）を
# サポートしないビルドに変わってしまい、bitsandbytesのCUDAカーネル呼び出しで
# "Error named symbol not found" → Kernel死亡、という失敗を起こした。
# 現在の torch バージョンにピン止めしてから他パッケージだけ更新することで、
# pip resolver が torch を巻き込むのを防ぐ（互換性が無ければ pip 自体がエラーで
# fail-fast する。silent downgrade/upgrade より安全）。
import subprocess
import sys

import torch as _torch_before_upgrade

_torch_pin = f"torch=={_torch_before_upgrade.__version__}"
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes", _torch_pin],
    check=True,
)

# %%
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── 0. GPU互換性の早期チェック（fail-fast）。sm_XX が torch のビルド対応外だと
#      generate() 実行時まで気づかず "Kernel died" で無言落ちするため、ここで検出する
if torch.cuda.is_available():
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    device_arch = f"sm_{cap_major}{cap_minor}"
    supported_archs = torch.cuda.get_arch_list()
    logger.info("GPU: %s / capability: %s / torch対応arch: %s", torch.cuda.get_device_name(0), device_arch, supported_archs)
    if not any(device_arch in a for a in supported_archs):
        raise RuntimeError(
            f"GPU capability {device_arch} はこの torch ビルド（対応: {supported_archs}）でサポートされていません。"
            " Kaggle Notebook の Accelerator設定を GPU T4 x2 に変更するか、対応する torch バージョンを使ってください。"
        )
else:
    raise RuntimeError("CUDA GPUが利用できません。kernel-metadata.json の enable_gpu / Notebook の Accelerator設定を確認してください。")


# %%
# ── 1. パス解決（Kaggle 環境を自動検出）。assets データセットを sys.path に追加して
#      prompt_template.py / metrics.py をそのまま import する（ローカルコードと単一の真実源を共有）
#      ※ Dataset はサブフォルダを作らずファイルを直置きしている（kaggle CLI の --dir-mode
#      zip/tar でアーカイブ化される問題を避けるため。upload.sh 参照）
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
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MAX_NEW_TOKENS = 512
MAX_GEN_RETRIES = 3  # JSON解析/検証に失敗した場合のみ再生成（具体的な失敗種別 + ログ + 上限付き）

# %%
# ── 3. モデルロード（4bit量子化。単一GPUでも載る想定）
logger.info("モデルをロードします: %s", MODEL_ID)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
)
model.eval()
logger.info("モデルロード完了")


# %%
# ── 4. JSON抽出ユーティリティ
# Claude側は output_config(json_schema) で出力形状をAPI側に強制できるが、
# transformers の素の generate にはその機構がないため、抽出 + 検証 + リトライで対処する
# （prompt_template.validate_output は両スクリプトで共有）。
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


def label_one(report_text: str) -> dict:
    messages = [
        {"role": "system", "content": pt.build_system_prompt()},
        {"role": "user", "content": pt.build_user_prompt(report_text)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    last_exc: Exception | None = None
    for attempt in range(1, MAX_GEN_RETRIES + 1):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            parsed = extract_json(text)
            pt.validate_output(parsed)
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            last_exc = e
            logger.warning(
                "JSON解析/検証失敗 (attempt %d/%d): %s | raw=%r",
                attempt, MAX_GEN_RETRIES, e, text[:300],
            )
    raise RuntimeError(f"{MAX_GEN_RETRIES}回リトライしても有効なJSONが得られませんでした: {last_exc}")


# %%
# ── 5. 58件を評価
val_df = pd.read_csv(VALIDATION_CSV)
logger.info("対象件数: %d 件 / model=%s / prompt=%s", len(val_df), MODEL_ID, pt.PROMPT_VERSION)

rows = []
for i, row in enumerate(val_df.itertuples(index=False), start=1):
    logger.info("[%d/%d] %s を処理中...", i, len(val_df), row.StudyInstanceUID)
    t0 = time.time()
    parsed = label_one(row.Report)
    logger.info("  -> %.1fs", time.time() - t0)

    row_result = {"StudyInstanceUID": row.StudyInstanceUID}
    for label in pt.LABELS:
        row_result[f"pred_{label}"] = parsed[label]
    rows.append(row_result)

pred_df = pd.DataFrame(rows)
true_df = val_df[["StudyInstanceUID", *pt.LABELS]].rename(columns={l: f"true_{l}" for l in pt.LABELS})
merged = true_df.merge(pred_df, on="StudyInstanceUID", how="inner")
if len(merged) != len(val_df):
    raise AssertionError(f"結合後の件数が一致しません: true={len(val_df)}, merged={len(merged)}")

merged_path = OUTPUT_DIR / "predictions_with_truth.csv"
merged.to_csv(merged_path, index=False)
logger.info("予測結果を保存しました: %s", merged_path)

# %%
# ── 6. 評価指標（Claude側と同一の metrics.compute_metrics を使用）
metrics = compute_metrics(merged, pt.LABELS)
metrics["run_config"] = {
    "model": MODEL_ID,
    "prompt_version": pt.PROMPT_VERSION,
    "quantization": "4bit-nf4",
    "max_new_tokens": MAX_NEW_TOKENS,
    "n_studies": len(merged),
}
metrics_path = OUTPUT_DIR / "metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
logger.info("評価指標を保存しました: %s", metrics_path)

logger.info("=== macro 指標 (Qwen2.5-14B) === %s", metrics["macro"])
logger.info("exact_match_rate: %.4f", metrics["exact_match_rate"])
print("macro:", json.dumps(metrics["macro"], indent=2))
print("exact_match_rate:", metrics["exact_match_rate"])
print("--- 比較対象 (Claude Opus 5, v2_fewshot): macro F1 = 0.733, exact_match = 0.069 ---")
