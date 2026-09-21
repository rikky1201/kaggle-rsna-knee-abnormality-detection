"""
report-labeler の検証: Claude API に data/validation_58.csv の58件を投げ、
12ラベルの疑似ラベルを生成して gold label と比較・評価する。

推論パラメータはこのファイル冒頭の定数で管理する（ハードコーディング散在防止）。

Usage:
    bash run_llm_labeling.sh                 # 58件フル実行
    bash run_llm_labeling.sh --limit 3        # 動作確認用に3件だけ
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

from metrics import compute_metrics
from prompt_template import LABELS, PROMPT_VERSION, build_output_schema, build_request_kwargs, validate_output

# --- 推論パラメータ（ここだけで管理する） ---
MODEL_ID = "claude-opus-5"
MAX_TOKENS = 4096
MAX_API_RETRIES = 3  # RateLimitError / APIConnectionError のみリトライ対象
RETRY_BASE_WAIT_SEC = 5

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent
ENV_PATH = REPO_ROOT / ".env"
VALIDATION_CSV = EXP_DIR / "data" / "validation_58.csv"

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> None:
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_file = output_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


def build_client() -> anthropic.Anthropic:
    load_dotenv(ENV_PATH)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(f"ANTHROPIC_API_KEY が見つかりません（{ENV_PATH} を確認してください）")
    return anthropic.Anthropic(api_key=api_key)


def call_with_retry(client: anthropic.Anthropic, model: str, **kwargs) -> anthropic.types.Message:
    """RateLimitError / APIConnectionError / 5xx(APIStatusError) のみリトライ対象
    （具体的な例外型 + ログ + 上限付き）。それ以外（BadRequestError, AuthenticationError 等の
    4xx）は握りつぶさず即座に送出する。
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return client.messages.create(model=model, max_tokens=MAX_TOKENS, **kwargs)
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            last_exc = e
        except anthropic.APIStatusError as e:
            if e.status_code < 500:
                raise  # 4xx はリトライしても直らないので即座に送出
            last_exc = e
        wait = min(RETRY_BASE_WAIT_SEC * (2 ** (attempt - 1)), 60)
        logger.warning(
            "%s (attempt %d/%d) — %ds待ってリトライします: %s",
            type(last_exc).__name__, attempt, MAX_API_RETRIES, wait, last_exc,
        )
        time.sleep(wait)
    raise last_exc


def label_one(client: anthropic.Anthropic, model: str, report_text: str) -> dict:
    kwargs = build_request_kwargs(report_text)
    schema = build_output_schema()
    response = call_with_retry(
        client, model,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        **kwargs,
    )
    text_block = next(b for b in response.content if b.type == "text")
    parsed = json.loads(text_block.text)  # output_config が形を保証するが、パースは必ずjson.loadsで行う
    validate_output(parsed)
    return parsed, response.usage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に先頭N件だけ実行")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--sleep", type=float, default=0.0, help="各呼び出し間のスリープ秒数")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = EXP_DIR / "results" / args.model / f"{PROMPT_VERSION}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logger.info("検証用データセットを読み込みます: %s", VALIDATION_CSV)
    val_df = pd.read_csv(VALIDATION_CSV)
    if args.limit is not None:
        val_df = val_df.head(args.limit)
    logger.info("対象件数: %d 件 / model=%s / prompt=%s", len(val_df), args.model, PROMPT_VERSION)

    client = build_client()

    predictions_path = output_dir / "predictions.jsonl"
    rows = []
    total_input_tokens = 0
    total_output_tokens = 0

    with open(predictions_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(val_df.itertuples(index=False), start=1):
            study_id = row.StudyInstanceUID
            logger.info("[%d/%d] %s を処理中...", i, len(val_df), study_id)

            parsed, usage = label_one(client, args.model, row.Report)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens

            # 生のAPI出力をここに逐次書き出す（途中で例外が起きても、ここまでの結果は残る）
            record = {"StudyInstanceUID": study_id, "pred": parsed}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            row_result = {"StudyInstanceUID": study_id}
            for label in LABELS:
                row_result[f"pred_{label}"] = parsed[label]
            rows.append(row_result)

            if args.sleep > 0:
                time.sleep(args.sleep)

    pred_df = pd.DataFrame(rows)
    # true label を validation_58.csv から結合（列名の空白/アポストロフィ問題を避けるため位置ではなく列名で結合）
    true_df = val_df[["StudyInstanceUID", *LABELS]].rename(columns={l: f"true_{l}" for l in LABELS})
    merged = true_df.merge(pred_df, on="StudyInstanceUID", how="inner")
    if len(merged) != len(val_df):
        raise AssertionError(f"結合後の件数が一致しません: true={len(val_df)}, merged={len(merged)}")

    merged_path = output_dir / "predictions_with_truth.csv"
    merged.to_csv(merged_path, index=False)
    logger.info("予測結果を保存しました: %s", merged_path)

    metrics = compute_metrics(merged, LABELS)
    metrics["run_config"] = {
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "max_tokens": MAX_TOKENS,
        "n_studies": len(merged),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "run_id": run_id,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("評価指標を保存しました: %s", metrics_path)

    logger.info("=== macro 指標 === %s", metrics["macro"])
    logger.info("exact_match_rate: %.4f", metrics["exact_match_rate"])
    logger.info(
        "推定コスト: input=%d tok, output=%d tok",
        total_input_tokens, total_output_tokens,
    )


if __name__ == "__main__":
    main()
