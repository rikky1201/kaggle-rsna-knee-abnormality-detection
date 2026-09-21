"""
Report -> LLM -> 疑似ラベル抽出の共通プロンプトテンプレート。

train.csv の12ラベル列名と完全一致するキーで 0/1 の JSON を返させる。
このモジュールは候補LLMをまたいで共有する「単一の真実源」として使う
（プロンプト文言・ラベル列順序をここだけで管理し、他スクリプトはここから import する）。
"""

# プロンプト改訂のたびに上げる（結果比較時にどのプロンプト版か区別するため）
# v1: 初版（肯定/否定/曖昧=0 のシンプルな3分類ルールのみ）
# v2_fewshot: 58件の実行結果を踏まえ、intact/trace/probable(所見 vs 依頼理由)の3パターンをfew-shot例として追加
# v3_english: v2_fewshotのルール・few-shot例を完全に維持したまま英語に翻訳（レポート本文が英語のため、
#   指示言語をレポートと揃えることでQwen2.5-14B-AWQの指示追従が安定するか検証する目的。ユーザー指示、2026-09-19）
PROMPT_VERSION = "v3_english"

# train.csv の列順序と完全一致させること（この順序が崩れると下流の評価コードとズレる）
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

SYSTEM_PROMPT = """You are a medical text annotator that determines the presence or absence
of 12 specified findings from knee MRI radiology reports.
Strictly follow the classification rules below.

Classification rules:
- If the Findings / Impression section reports a concrete finding (with anatomical
  location, morphology, etc.), classify it as 1. Even if hedging language such as
  "probable", "suspected", or "likely" is attached, classify as 1 as long as a
  concrete finding is described (diagnostic confidence and presence of a finding
  are separate axes)
- Classify as 0 if the finding is explicitly negated (e.g. "intact", "not torn",
  "preserved") or not mentioned at all
- A differential diagnosis listing alone (e.g. "cannot exclude X", "differential
  includes X", with no concrete finding described) does NOT count as 1. Classify as 0
- Symptoms or reasons for referral stated in the Indication / Clinical History
  section at the top of the report should NOT be used as evidence unless confirmed
  in the Findings/Impression section. A suspected diagnosis mentioned only in the
  reason for referral does not count as 1
- A finding explicitly described as "trace" is considered clinically insignificant
  and classified as 0. "Small", "mild", "moderate", etc. are considered present
  findings and classified as 1
- Do not supplement with inference or general knowledge. Base judgments only on
  what is explicitly written in the report text

Examples of borderline cases:

Example 1 (explicit negation):
  Excerpt: "ACL is intact. PCL is preserved. The MCL is intact. Medial meniscus is not torn."
  Answer: ACL=0, MCL=0, Medial Meniscus=0
  Reason: "intact"/"not torn" are explicit negations. The mere mention of a
  structure name (ACL, MCL, etc.) in the text is not itself evidence

Example 2 (trace vs. small/mild distinction):
  Excerpt: "Small joint effusion. ... Trace Baker's cyst."
  Answer: Effusion=1, Baker's=0
  Reason: "small" counts as a present finding (1); "trace" is clinically
  negligible and counts as 0. The treatment differs by finding even within the
  same sentence

Example 3 (suspected finding in Findings vs. suspected diagnosis in
Indication/History):
  Excerpt A (Findings section): "Suspected small horizontal tear at the anterior
  horn of the lateral meniscus."
  Answer: Lateral Meniscus=1
  Reason: Despite the hedging word "suspected", a concrete tear finding with
  location and morphology is described in the Findings section, so it counts as 1
  Excerpt B (Clinical History section): "CLINICAL HISTORY: ... Suspected bone
  contusion/meniscal tear."
  Answer: Corresponding finding=0 (unless separately confirmed in
  Findings/Impression)
  Reason: This is the clinical suspicion at the time of the exam request (reason
  for referral), not a finding the radiologist confirmed on imaging

Output format:
- Output a single JSON object only, with an integer value of 0 or 1 for each of
  the 12 labels
- Key names must exactly match the following 12 (including case, apostrophes,
  and spaces):
  {labels}
"""

USER_PROMPT_TEMPLATE = """Below is a knee MRI radiology report. Following the classification rules,
determine the presence or absence of the 12 findings.

--- Report text ---
{report}
--- End of report text ---
"""


def build_system_prompt() -> str:
    labels_list = "\n".join(f"  - {label}" for label in LABELS)
    return SYSTEM_PROMPT.format(labels=labels_list)


def build_user_prompt(report_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(report=report_text)


def build_request_kwargs(report_text: str) -> dict:
    """Anthropic `client.messages.create(**kwargs, model=..., max_tokens=...)` にそのまま渡せる形。

    Anthropic API では system は messages 内の role ではなくトップレベル引数として渡す
    （OpenAI形式との違いに注意）。
    """
    return {
        "system": build_system_prompt(),
        "messages": [{"role": "user", "content": build_user_prompt(report_text)}],
    }


def build_output_schema() -> dict:
    """`output_config.format`（json_schema）に渡すスキーマ。

    型・必須キー・additionalProperties:false をAPI側で強制することで、
    「キー名が微妙にズレる」「JSON以外の文章が混ざる」を構造的に防ぐ。
    """
    return {
        "type": "object",
        "properties": {label: {"type": "integer", "enum": [0, 1]} for label in LABELS},
        "required": list(LABELS),
        "additionalProperties": False,
    }


def validate_output(parsed: dict) -> None:
    """LLMが返したJSON（dictにパース済み）がスキーマ通りか検証する。不正なら例外を投げる（握りつぶさない）。"""
    missing = [label for label in LABELS if label not in parsed]
    if missing:
        raise ValueError(f"出力JSONにキーが不足しています: {missing}")
    extra = [key for key in parsed if key not in LABELS]
    if extra:
        raise ValueError(f"出力JSONに未知のキーが含まれています: {extra}")
    invalid = {k: v for k, v in parsed.items() if v not in (0, 1)}
    if invalid:
        raise ValueError(f"0/1以外の値が含まれています: {invalid}")
