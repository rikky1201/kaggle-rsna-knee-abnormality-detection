# SESSION_NOTES — expA00_report_labeler_validation

## 1. セッション情報
- 日付: 2026-09-14
- 作業フォルダ: `workspace/expA00_report_labeler_validation`
- 目標: report-labeler（放射線読影レポート本文 → LLM → 12ラベルの疑似ラベル）の検証。
  1. 12ラベル全確定 58件の検証用データセット作成
  2. 共通プロンプトテンプレート（判定ルール固定）
  3. LLM出力フォーマットの固定（JSON, train.csv列名と完全一致）
  4. 命名規則に従った実験ディレクトリの作成
  5. （ユーザー承認後）Claude API (`claude-opus-5`) を実際に呼び出して58件で精度検証

## 2. 背景（なぜこの実験が必要か）
- `train.csv` は4,407 study全件にレポート本文があるが、12ラベルが埋まっているのはそのうち**58件のみ**（詳細: `daily_reports/20260914.md`, `survey/competition/overview.md`）
- 残り約4,349件のラベルなしレポートに疑似ラベルを付与しないと、通常の教師あり学習の学習データが極端に不足する
- そのため「レポートテキスト → LLM → 疑似ラベル」という report-labeler パイプラインが本コンペの事実上の前提タスクになっている
- 58件の gold label はこの report-labeler の**精度検証用の唯一の正解データ**であり、どのLLM・どのプロンプトを採用するかをこの58件でのスコア（今回のセッションでは未計算）に基づいて決める想定

## 3. 試したアプローチと結果

### 3.1 検証用データセットの抽出（`build_validation_set.py`）
- `datasets/rsna-knee-abnormality-detection/train.csv` を読み込み、12ラベル列（`prompt_template.LABELS`）が全て非欠損の行のみを抽出
- `len(fully_labeled) != 58` の場合は `AssertionError` で fail-fast（train.csv が更新された場合に気づかず古い前提で進めることを防ぐ）
- 結果: **58行、14列（StudyInstanceUID, Report + 12ラベル）を正しく抽出**。`data/validation_58.csv` に保存
- 実行コマンド: `bash run.sh`（`cd "$(dirname "$0")"` してから `python build_validation_set.py` を呼ぶ構成）

### 3.2 プロンプトテンプレート設計（`prompt_template.py`）
- `LABELS` 定数を単一の真実源（single source of truth）として定義し、`build_validation_set.py` からも import して列順序のズレを防ぐ
- 判定ルール（ユーザー指示を厳守）:
  - 明示的な肯定 → 1
  - 明示的な否定 / 言及なし → 0
  - 疑い・鑑別診断のみ・曖昧な記述 → **1にしない**（0扱い）
- 出力フォーマット: 12ラベル分の0/1のみを持つ単一JSONオブジェクト。キー名は `train.csv` の列名と完全一致（`Baker's` のアポストロフィ含む）
- `build_request_kwargs()` で Anthropic `client.messages.create(**kwargs, model=..., max_tokens=...)` にそのまま渡せる形（`system` はトップレベル引数、`messages` はuserのみ）を返す設計
- `build_output_schema()` で `output_config.format`（json_schema）用のスキーマを生成し、API側で出力形状を構造的に強制
- `validate_output()` でLLM出力JSONのキー過不足・値域（0/1以外）を検証する関数を用意（呼び出し側で握りつぶさずに例外を投げる設計。CLAUDE.mdのエラーハンドリング原則に準拠）

## 4. ファイル構成
```
workspace/expA00_report_labeler_validation/
├── SESSION_NOTES.md              # 本ファイル
├── run.sh                        # cd → build_validation_set.py 実行
├── run_llm_labeling.sh           # cd → run_llm_labeling.py 実行
├── build_validation_set.py       # train.csv → validation_58.csv (gold 58件) 抽出
├── build_unlabeled_set.py        # train.csv → unlabeled_4349.csv (未ラベル4349件) 抽出
├── build_full_labels.py          # gold + pseudo → train_full_labels_4407.csv 統合
├── run_llm_labeling.py           # Claude API 呼び出し + 評価指標計算（58件検証専用）
├── prompt_template.py            # 共通プロンプト・LABELS定義・出力スキーマ・出力検証関数
├── metrics.py                    # 評価指標計算（Claude/Qwen両スクリプトで共有）
├── requirements.txt              # pandas, scikit-learn, anthropic, python-dotenv
├── data/
│   ├── validation_58.csv         # gold label（58件）
│   ├── unlabeled_4349.csv        # 未ラベルレポート（4349件, report-labeler入力）
│   ├── pseudo_labels_4349.csv    # Qwen2.5-14B-AWQ+vLLMが生成した疑似ラベル（4349件）
│   └── train_full_labels_4407.csv # gold+pseudo統合（train.csv全件分, label_source列付き）
├── kaggle_qwen_benchmark/        # 58件でのモデル比較用Kaggle Notebook一式（検証専用）
├── kaggle_qwen_pseudolabel/      # 4349件本番適用用Kaggle Notebook一式
└── results/
    └── claude-opus-5/
        └── 20260914_150632/
            ├── predictions.jsonl            # 生API出力（逐次書き出し、監査証跡）
            ├── predictions_with_truth.csv   # gold labelと結合済みの予測結果
            ├── metrics.json                 # per-label + macro指標
            └── log_20260914_150632.txt      # 実行ログ（DEBUGレベル）
```
- 可視化結果: なし
- **最終成果物**: `data/train_full_labels_4407.csv`（次のfold設計・学習フェーズで使用する統合ラベルセット）

## 5. 重要な知見
- 58件の抽出は `dropna(subset=LABELS, how="any")` で機械的に行える（欠損値は空文字列としてpandasがNaN扱いしてくれる）
- **避けるべきアプローチ**: プロンプト内のラベルキー名をハードコードで複数箇所に書くと、列順序変更時に同期漏れが起きる → `prompt_template.LABELS` に一元化して import で共有する方式を採用（有効だったテクニック）
- **未実施・要注意**: LLMの実際の出力が本当に厳密JSONのみを返すかはモデル依存（前置き文や```json コードブロックを付けてくるモデルがある）。次段階でJSON抽出のロバスト化（正規表現でJSON部分だけ抜き出す等）が必要になる可能性が高い

### 3.3 Claude API 実装と実行（`run_llm_labeling.py`）— ユーザー承認後に実施
- Claude API キーは `.env`（リポジトリルート、`.gitignore` 済み）に `ANTHROPIC_API_KEY` として格納済み。`python-dotenv` で読み込み
- モデル: `claude-opus-5`（`claude-api` スキルのデフォルト方針に従う。ユーザーから他モデル指定なし）
- **`output_config.format`（json_schema）で出力形式をAPI側に強制**: `prompt_template.build_output_schema()` が12ラベルの `integer enum:[0,1]`, `required`全ラベル, `additionalProperties:false` を持つJSON Schemaを生成。プロンプト文言だけに頼らず構造的にキー欠落・型崩れ・JSON以外の文章混入を防ぐ設計にした（当初のプロンプトのみでの担保より頑健）
- **重要な設計修正**: 初期実装の `build_messages()` はOpenAI形式（`role:"system"`のメッセージ）を想定していたが、Anthropic APIでは `system` はトップレベル引数として別に渡す仕様のため誤り。`build_request_kwargs()` に置き換えて修正済み
- エラーハンドリング: `RateLimitError` / `APIConnectionError` のみ具体的な例外型でキャッチしログ+リトライ（上限3回、指数バックオフ）。それ以外（`BadRequestError`, JSON parse失敗, `validate_output()`失敗等）は握りつぶさず即座に送出（CLAUDE.mdのfail-fast原則に準拠）
- 結果は `results/{model}/{run_id}/` に集約: `predictions.jsonl`（逐次書き出し。生API出力の監査証跡）、`predictions_with_truth.csv`（gold labelと結合済み）、`metrics.json`（per-label + macro指標）、`log_*.txt`

### 3.4 実行結果（`results/claude-opus-5/20260914_150632/`）
- `--limit 2` でのドライラン → パイプライン正常動作を確認後、58件フル実行
- **トークン使用量**: input 106,236 tok, output 18,761 tok → **推定コスト約 $1.00**（input $0.53 + output $0.47、Opus 5料金 $5/$25 per 1M tok）
- **macro指標**（12ラベル平均）: accuracy 0.806, precision 0.701, recall 0.780, **F1 0.716**
- **exact_match_rate（12ラベル全一致）: 6.9%（58件中4件のみ）** — per-label指標はそこそこでも、study単位で完全一致する確率は低い（12個の独立した2値判定の掛け算効果）
- **ラベルごとの傾向**（`metrics.json` の `per_label` 参照）:
  - **過剰検出傾向**（recall高いがprecision低い = 疑わしきは1にしがち）: MCL（recall 1.0 / precision 0.5, 予測18件 vs 正解9件）, Effusion（recall 1.0 / precision 0.69, 予測51件 vs 正解35件）, ACL（recall 1.0 / precision 0.8, 予測30件 vs 正解24件）
  - **過小検出傾向**（precision高いがrecall低い = 見逃しが多い）: Synovitis（recall 0.48 / precision 0.76, 予測17件 vs 正解27件）, Medial Meniscus（recall 0.46 / precision 0.86, 予測14件 vs 正解26件）, Lateral Meniscus（recall 0.52 / precision 0.75, 予測16件 vs 正解23件）
  - 比較的バランス良い: Fracture（precision=recall=0.72）, Medial OA（0.69/0.73）

### 3.5 プロンプトv2（few-shot追加）と再実行（`prompt_template.PROMPT_VERSION = "v2_fewshot"`）
- ユーザー指示: 誤り事例の実データを確認してから、3パターンをfew-shot例としてプロンプトに埋め込む
  1. **intact→0**: `data/validation_58.csv` から `"intact"` を含む15件を検索。"ACL is intact"→ACL=0, "MCL is intact"→MCL=0 を確認し、明示的否定の例として採用
  2. **trace所見の扱い**: `"trace"` + `"Baker"` を検索。"Small joint effusion...Trace Baker's cyst" の実例で **Effusion=1, Baker's=0** を確認（"small"は所見あり、"trace"は臨床的に無視できる程度、という区別が実データで裏付けられた）
  3. **probableな半月板所見の扱い**: `"suspect"` 系ワードを検索した結果、単純な「疑い→0」では説明できない挙動を発見:
     - Findingsセクションの "Suspected small horizontal tear at the anterior horn of the lateral meniscus" → **Lateral Meniscus=1**（部位・形態を伴う具体的所見は、ヘッジ表現があっても1）
     - Clinical Historyセクションの "Suspected bone contusion/meniscal tear"（検査依頼理由）→ 対応する所見は0（読影所見で未確認のため）
     - **既存ルール「疑い・鑑別診断は1にしない」と矛盾する実例**だったため、ルール自体を「Findings内の具体的所見 vs Indication/History内の紹介理由」で切り分ける形に修正（単純な追加例ではなくルール本文も改訂）
- `PROMPT_VERSION` 定数を追加（`v1` → `v2_fewshot`）。結果フォルダ名にプロンプト版を含めて比較可能にした（`results/{model}/{prompt_version}_{run_id}/`）
- **v1 → v2 の比較（58件, claude-opus-5）**:

| 指標 | v1 | v2_fewshot | 差分 |
|------|-----|-----------|------|
| macro F1 | 0.716 | 0.733 | +0.017 |
| macro precision | 0.701 | 0.704 | +0.003 |
| macro recall | 0.780 | 0.807 | +0.027 |
| exact_match_rate | 6.9% | 6.9% | ±0（4/58のまま） |
| 推定コスト | $1.00 | $1.28 | +$0.28（few-shot分でinput token増） |

- **ラベル別の変化**（狙い通り改善したもの）:
  - **Lateral Meniscus: F1 0.615→0.769（+0.154）** — 「Findings内のsuspectedな断裂所見は1」の効果が明確に出た
  - **Baker's: F1 0.786→0.815（+0.03）** — trace所見の区別が効いた
  - Medial Meniscus: F1 0.6→0.636（+0.036、軽微改善）
- **変化なし・悪化したもの**:
  - Effusion, Synovitis: **完全に無変化**（予測件数も一致）。Synovitisの過小検出（recall 0.48）は今回のfew-shotでは対処できていない → 別パターンの誤りが支配的と推測
  - ACL: F1 0.889→0.842（悪化）。予測件数が30→33に増加し、over-predictionがむしろ悪化。"intact"例を入れたが別の原因（他の言及パターン）が残っている可能性
  - Lateral OA, Contusion: 軽微に悪化（F1 -0.02〜-0.03程度、n=58では誤差の範囲の可能性もある）

### 3.6 候補モデル比較: Qwen2.5-14B-Instruct（Kaggle GPU, 無料枠）
- ユーザーからのコスト意識の指摘（「Llamaなどのモデルを使えばコスト0でできますか？」）を受け、Qwen2.5-14B-Instruct を Kaggle Notebook の無料GPUで動かし、同じ58件でF1を比較するベンチマークを構築
- **インフラは元々保留していた「③ `src/` をプライベートKaggle Datasetとしてpushし、Notebook側でsys.path.appendでimport」の実地検証を兼ねる**
- 構成: `workspace/expA00_report_labeler_validation/kaggle_qwen_benchmark/`
  - `benchmark_notebook.py`（jupytext source）/ `.ipynb`（派生、gitignore）
  - `kernel-metadata.json`（GPU有効・internet有効・`dataset_sources`でassets datasetをattach）
  - `dataset-metadata.json`（assets dataset用）
  - `upload.sh`（assets push → jupytext変換 → kernel push を1コマンド化）
- **`prompt_template.py` / `metrics.py` / `validation_58.csv` を private Kaggle Dataset (`rikky1201/rsna-knee-report-labeler-assets`) として push**し、Notebook側で `sys.path.append` してそのまま import（Claude側スクリプトと完全に同一のプロンプト・評価ロジックを共有 = 単一の真実源）
- Qwen側はAnthropic APIのような `output_config.format` 相当の構造化出力保証機構がないため、JSON抽出＋`prompt_template.validate_output()`検証＋失敗時リトライ（最大3回、具体的な例外型 + ログ）で対処
- 4bit量子化（bitsandbytes nf4）でロードし、単一GPUでも載るようにした
- **ハマった点と対処**:
  1. `kaggle datasets create` はサブフォルダ（`src/`, `data/`）を `--dir-mode` 未指定だとスキップし「Please upload at least one file」で失敗。`--dir-mode zip/tar` は中身がアーカイブ化されNotebook側から読みにくくなる → **サブフォルダを作らずファイルを直置き**する構成に変更して解決
  2. kernel-metadata.json の `id` とタイトルのスラッグが一致していないと、Kaggle側がタイトル由来のスラッグでkernelを作成してしまい、以降の `kernels status`/`output` コマンドで指定するslugとズレる → 実際に解決されたslug（`qwen2-5-14b-report-labeler-benchmark`）に `id` を合わせて修正
  3. jupytextで生成した `.ipynb` に `kernelspec` メタデータが無いと、Kaggle側のpapermill実行が `ValueError: No kernel name found in notebook` で即座に落ちる → `jupytext --to ipynb --set-kernel python3` で明示的に付与して解決
- v3 push後、GPU上で実行中（Monitorツールでバックグラウンド監視）

### 3.7 Kaggle実行時のGPU互換性トラブル（v3〜v5）
- **v3の失敗**: 約200秒でNotebookが `nbclient.exceptions.DeadKernelError: Kernel died` で無言クラッシュ
  - ログを遡ると根本原因は2つ: (1) `Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation`（torchがsm_70以降のみ対応） (2) bitsandbytesのCUDAカーネル呼び出しで `Error named symbol not found at line 74 in file /src/csrc/ops.cu`
  - **原因の特定**: notebook冒頭の `%pip install -q -U transformers accelerate bitsandbytes` が `-U`（アップグレード）指定だったため、pipのdependency resolverが**torchも巻き込んでアップグレード**し、Kaggleのデフォルト割当GPU（Tesla P100 = Pascal世代 = sm_60）を切り捨てた新しいtorchビルドに変わってしまっていた（Kaggleプリインストールのtorchは元々P100と互換だった可能性が高い）
  - kaggle CLIのログ取得自体もハマった: Windows環境でPowerShell/bash問わず `'cp932' codec can't encode character` でクラッシュしログが0バイトで落ちる → `$env:PYTHONUTF8="1"`（Python 3.7+ UTF-8モード。PYTHONIOENCODINGでは直らなかった）で解決
- **`--accelerator` オプションの調査**: `kaggle kernels push --accelerator <値>` でGPUタイプ（T4x2等）を明示指定できるパラメータがCLIに存在するが、`kagglesdk` 側に列挙型が未実装で**クライアント側では値の妥当性を検証しない**（無効な値を渡しても素通りしてしまい、実際にGPUクォータを消費する実行が走ってしまう）。正しい値が分からない状態で推測を重ねるとクォータの無駄遣いになるため、**この方向は断念**
- **v5での修正方針**: GPUタイプの指定を諦め、**pipインストール時にtorchを現在のバージョンにピン止め**して巻き込みアップグレードを防止する方式に変更
  ```python
  torch_pin = f"torch=={_torch_before_upgrade.__version__}"
  subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes", torch_pin], check=True)
  ```
  - これにより、bitsandbytes/accelerateがtorchの新バージョンを要求する場合はpipのresolverエラーで**即座に失敗**する（fail-fast。3分待って無言でKernel diedするより遥かに良い）
  - さらに `torch.cuda.get_device_capability()` と `torch.cuda.get_arch_list()` を突き合わせる**早期互換性チェック**をモデルロード前に追加し、非対応GPUなら即座に明示的な `RuntimeError` で落とすようにした（診断可能なエラーメッセージに変換）
- **v5の結果**: 早期互換性チェックが正しく機能し、200秒の無言クラッシュ→**54秒で明示的な `RuntimeError` メッセージ**に改善。ただしログを見ると `torch対応arch: ['sm_70', ...]` は v3 と全く同じ → **仮説「pip -Uがtorchを巻き込みアップグレードした」は誤りだった**。実際は Kaggle のこのNotebook環境のベースイメージが**最初からP100(sm_60)非対応のtorchをプリインストールしている**ことが判明（torchピン止めは無意味だった。ただし早期チェック自体は有用なので残す）
- **根本対処**: `kaggle kernels push --accelerator <値>` でGPUタイプを明示指定できる。正しい値をWebSearchで特定: **`NvidiaTeslaT4`**（デフォルトGPU）。`NvidiaTeslaP100` は retired（廃止）扱いで、これがデフォルト割当になってしまうことがある。`upload.sh` に `--accelerator NvidiaTeslaT4` を追加
- v6 push後、T4 GPU上で再実行中

### 3.8 v6完了結果: Qwen2.5-14B-Instruct (transformers + bitsandbytes nf4, T4)
- **58件フル完走。所要時間 約2時間強**（モデルロード 約5分10秒 + 58件推論）
- **結果**: macro F1 = 0.696 / precision 0.656 / recall 0.778 / accuracy 0.787 / exact_match 6.9%（4/58, Claudeと同数）
- Claude Opus 5 (v2) との比較: F1 -0.037（そこまで大差ではない）。ラベル別ではMedial MeniscusはQwenが上回り（F1 0.636→0.8）、PF OA/Contusion/FractureはClaudeが明確に上回る
- **速度面の問題**: 1件平均 約124秒（想定10〜20分の10倍以上）。ユーザー指摘を受けて原因調査を実施

### 3.9 速度低下の原因調査（ユーザー指示の4点を確認）
ログ（`qwen2-5-14b-report-labeler-benchmark.log`）を解析:
1. **モデル再ロードの有無**: 無し。「モデルロード完了」ログは1回のみ、ループ外でロードしていることを確認 → 問題なし
2. **4bit量子化が実際に適用されているか**: ログに明示的なGPUメモリ使用量の記録が無く直接確認はできなかったが、T4(16GB)で14Bモデルがfp16(約28GB必要)ならOOMするはずのところ正常完走している → 間接的に4bit適用を推測（次回からは`torch.cuda.memory_allocated()`を明示ログする方針に変更）
3. **max_new_tokens=512が過大か**: **JSON解析/検証のリトライは全58件中0回**であったにも関わらず、処理時間が **40〜55秒の集団** と **90〜190秒の集団** に二極化。リトライが原因ではなく、**出力フォーマットをエンジン側で強制していなかったため、一部のケースでモデルがJSON完成後も生成を続け512トークン近くまで生成していた**ことが濃厚な原因と推測（生成トークン数を直接ログしていなかったため確証はないが、状況証拠から高確度）
4. **GPUが実際に使われているか**: ログに `GPU: Tesla T4 / capability: sm_75` を明示的に確認 → CPU実行ではない
- ベースの40〜55秒自体もClaudeの数秒と比べて遅く、bitsandbytes nf4の逆量子化カーネルがT4(Turing)上で最適化されておらず生成速度そのものが遅いことも一因と判断

### 3.10 対策: vLLM + AWQ量子化 + guided decoding に置き換え
- ユーザー指示: 素朴なbatchingではなくvLLMへの置き換えを実施し、同じ58件で再ベンチマーク
- モデルを `Qwen/Qwen2.5-14B-Instruct-AWQ`（Qwen公式のAWQ 4bit事前量子化版, 9.99GB）に変更。AWQはTuring(T4)向けに最適化されたカーネルを持ち、bitsandbytes nf4より高速な想定
- **`SamplingParams(guided_decoding=GuidedDecodingParams(json=schema))`** でJSON Schemaを直接エンジンに強制させ、「有効なJSON完成で即座に生成停止」を保証（3.9で推測した二峰性の原因を構造的に解消）
- `llm.chat(conversations, sampling_params=...)` で58件を**1回のバッチ呼び出し**に投入（vLLMのcontinuous batchingでGPU利用効率を最大化。逐次ループだった旧版との最大の違い）
- 診断のため `torch.cuda.memory_allocated()/memory_reserved()` をモデルロード前後で明示ログ、生成トークン数の平均/最大/最小もログするよう改善
- T4(sm_75)はP100と異なり現行torchビルドの対応範囲内のため、torchピン止めは不要（vLLMのpip installが自然に解決）
- 旧HF transformers版は `benchmark_notebook_hf_transformers_archive.py` として保存（比較・再現性のため）
- v7は `GuidedDecodingParams` のimportエラーで即失敗（`vllm.sampling_params` に存在しない）。WebSearchで調査した結果、**vLLMのAPIが `guided_decoding`/`GuidedDecodingParams` から `structured_outputs`/`StructuredOutputsParams` に変更されていた**ことが判明。コードを新API（`SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))`）に修正し v8 push

### 3.11 v8完了結果: Qwen2.5-14B-Instruct-AWQ (vLLM + guided decoding) — 大幅改善
- **58件フル完走**。モデルロード 約201秒 + バッチ推論 168秒 = **合計約6分**（旧transformers版は2時間強）
- **1件あたり処理時間: 約2.90秒**（旧版 約124秒から**約43倍高速化**）
- **平均生成トークン数: 109トークン** → guided decoding導入により、旧版で見られた「JSON完成後も生成を続け512トークン近くまで暴走する」問題が解消されたことを裏付け（3.9節の推測が正しかったことの実証）
- **精度**: macro F1 = 0.693（旧transformers版 0.696とほぼ同等）/ precision 0.647 / recall 0.787 / **exact_match_rate = 8.6%（5/58, 旧2回はいずれも6.9%=4/58だったので微改善）**
- **全4,349件への外挿**: 2.9秒×4,349 ≈ **約3.5時間**（+モデルロード）。旧版の推定150時間から現実的な時間に短縮された

### 3.12 3モデル比較まとめ（58件, gold label）
| 実行 | モデル/エンジン | macro F1 | precision | recall | exact_match | 1件あたり時間 | コスト |
|------|-----------------|----------|-----------|--------|--------------|--------------|--------|
| v2_fewshot | Claude Opus 5 | **0.733** | **0.704** | 0.807 | 6.9% | 数秒 | 約$1.28 |
| v6 | Qwen2.5-14B (transformers+bnb nf4) | 0.696 | 0.656 | 0.778 | 6.9% | 約124s | $0 |
| v8 | Qwen2.5-14B-AWQ (vLLM+guided decoding) | 0.693 | 0.647 | **0.787** | **8.6%** | **2.90s** | $0 |

### 3.13 本番適用: 4,349件の未ラベルレポートへの疑似ラベル生成
- ユーザー指示: v8（vLLM+AWQ+guided decoding）の構成で4,349件全件に本番適用
- `build_unlabeled_set.py` を新規作成（`build_validation_set.py` の対）。train.csv から12ラベル全欠損の行を抽出し `data/unlabeled_4349.csv` を作成（件数不一致で `AssertionError` の fail-fast チェック付き。実際に4,349件で一致確認）
- `workspace/expA00_report_labeler_validation/kaggle_qwen_pseudolabel/` を新設。`kaggle_qwen_benchmark/` の検証構成（vLLM + Qwen2.5-14B-Instruct-AWQ + guided decoding, T4 GPU）をそのまま流用
- **assets Kaggle Dataset (`rikky1201/rsna-knee-report-labeler-assets`) に `unlabeled_4349.csv` を追加してバージョン更新**（dataset versionは全置き換えのため、`metrics.py`/`validation_58.csv` も含めて毎回揃えてアップロードする方式に）
- **チャンク処理（300件ずつ）で `/kaggle/working/pseudo_labels.csv` に逐次追記**する設計にした。理由: 4,349件（58件の約75倍）× 推定3.5〜4時間の長時間ジョブで、途中エラー時に全結果を失わないようにするため（CLAUDE.mdの「チェックポイント再開必須」の精神を、この推論ジョブにも適用）。ただし真の意味でのクロスラン再開（前回の途中結果を読み込んで続きから）までは実装せず、単一実行内での安全性確保に留めた（コスト対効果を踏まえた判断）
- JSON解析/検証失敗時は、guided decodingによりほぼ発生しない想定だが、4,349件規模で1件のために全体を止めるのは非効率なため、**ログに記録した上で当該件のみスキップ**する設計（fail-fastの原則は「握りつぶさない」ことであり、可視化した上での意図的なスキップは許容と判断）
- 完了後、58件gold labelの陽性率分布との比較でサニティチェックする予定（`pseudo_labels_summary.json` に各ラベルの陽性率を出力）
- **ハマった点**: 今回もkernel titleとidのスラッグ不一致（"Qwen2.5-14B report pseudolabel generation" → 実際のslugは `qwen2-5-14b-report-pseudolabel-generation`、id指定は `qwen2-5-14b-report-pseudolabel` で不一致）。kernel-metadata.jsonのidを実際のslugに合わせて修正
- **v1は即座にエラー**: `FileNotFoundError: report-labeler assets dataset (prompt_template.py) が /kaggle/input 配下に見つかりません`。原因は dataset version 更新直後（"processing"中）に kernel push してしまい、Notebook実行時点でまだ新しいファイルが `/kaggle/input/` にマウントされていなかったこと（`tools/kaggle_code_competition_submission.md` に記載済みの既知の罠「Dataset processing 待ち」を実地で踏んだ）
- `upload.sh` に **dataset status が "ready" になるまで待機するポーリングループ（10秒間隔、最大30回=5分）** を追加してから v2 を再push
- **v2も同じ `FileNotFoundError` で失敗**（dataset status "ready" 確認済みでも再発）。`kaggle datasets files` でデータセットの中身自体は正しくアップロードされていることを確認、`kaggle kernels pull --metadata` でサーバー側の kernel-metadata.json も `dataset_sources` が正しく設定されていることを確認 → **データセット側・kernel設定側のどちらも問題なし**と判明
- `_find_assets_dir()` に診断用の `/kaggle/input` ディレクトリリスティングをエラーメッセージに追加してv3再push
- **v3のエラーメッセージで判明した真の原因**: `/kaggle/input` の中身が `datasets: ['rikky1201']` の1エントリのみ！ **従来想定していた `/kaggle/input/<dataset-slug>/` ではなく、`/kaggle/input/datasets/<username>/<dataset-slug>/` という新しいネスト構造**でマウントされていた（`kaggle_qwen_benchmark`側のv6/v8実行時とはマウント方式が異なっていた、もしくはKaggle環境側の仕様変更の可能性）
- **対処**: ディレクトリ名のパターンマッチ（`glob("*report-labeler*")`）をやめ、**`input_root.rglob("prompt_template.py")` で実ファイルを再帰的に探す方式**に変更。マウント階層がどう変わっても頑健になるよう設計を修正
- **v4完了**: 4,349件**全件成功（失敗0件）**。総推論時間202.2分（約3.4時間）、1件平均2.79秒（58件検証時の2.90秒とほぼ一致、スケールしても速度劣化なし）、平均生成トークン数109（58件検証と同水準）

### 3.14 gold labelとの陽性率比較 — 無視できない乖離を発見
`pseudo_labels_summary.json` の陽性率を58件gold labelの陽性率（3.5節参照）と比較:

| ラベル | 58件gold | 4,349件疑似ラベル | 差 |
|--------|---------|-------------------|-----|
| ACL | 41.4% | 27.5% | -13.9pt |
| MCL | 15.5% | 18.1% | +2.6pt |
| Medial Meniscus | 44.8% | 56.1% | +11.3pt |
| Lateral Meniscus | 39.7% | 28.3% | -11.4pt |
| Medial OA | 25.9% | 38.6% | +12.7pt |
| Lateral OA | 19.0% | 28.5% | +9.5pt |
| PF OA | 36.2% | 25.7% | -10.5pt |
| Effusion | 60.3% | 56.8% | -3.5pt |
| **Synovitis** | 46.6% | **18.3%** | **-28.3pt** |
| Baker's | 20.7% | 24.3% | +3.6pt |
| Contusion | 32.8% | 23.6% | -9.2pt |
| **Fracture** | 31.0% | **5.3%** | **-25.7pt** |

- 特にFracture（-25.7pt）とSynovitis（-28.3pt）の落差が大きい
- 検証セッション開始時点（daily_reports/20260914.md）から指摘していた懸念「58件の gold label がランダムサンプルでなく所見が豊富な症例に意図的に偏っている可能性」が現実化した可能性が高いと判断（専門家ラベリング対象として重大所見を含む症例が意図的に多く選ばれ、母集団全体の有病率とは乖離している、という解釈）
- モデルの系統的な過検出/過小検出の可能性も完全には排除できないが、**ユーザー判断によりこのまま採用**し、疑似ラベル精度の追加検証（ランダムサンプル目視確認等）は行わずに次工程へ進む

### 3.15 gold + pseudo 統合ラベルセットの作成
- `build_full_labels.py` を新規作成。`validation_58.csv`（gold, 58件）+ `pseudo_labels_4349.csv`（pseudo, 4,349件）を統合し、`data/train_full_labels_4407.csv`（train.csv全件分, 4,407行）を生成
- **`label_source` 列（"gold"/"pseudo"）を保持**。fold設計・学習時にソース別の重み付けや検証を行う可能性があるため
- StudyInstanceUIDの重複チェック・件数一致チェック（4407件）を fail-fast で実施済み

## 6. 次のステップ（優先度付き）
このセッションで report-labeler の検証〜本番適用（`data/train_full_labels_4407.csv` 完成）まで完了。次は別exp（fold設計・画像学習パイプライン）に引き継ぐ:
1. **[高]** fold設計（保留していたタスクに復帰）。`data/train_full_labels_4407.csv` の `label_source` 列（gold/pseudo）をfold間の分布バランスやCV信頼性評価にどう活かすか検討。MultilabelStratifiedKFoldの適用を検討
2. **[中]** 疑似ラベルの精度限界を踏まえた学習戦略の検討（pseudo-labelノイズを前提にした損失関数・sample weight・gold58件を検証専用に固定する案 等）
3. **[低]** 誤り事例の目視確認・プロンプトのさらなる改善（Synovitis等）— 優先度を下げて保留（ユーザー判断によりこのまま採用のため）
4. **[低]** Kaggle `/rules` ページの一次情報確認（チーム人数、外部データ可否等）— 未着手のまま持ち越し

## 7. 性能変化の記録
| 実行 | prompt | モデル/エンジン | 件数 | macro F1 | macro Precision | macro Recall | exact_match | 1件あたり時間 | 推定コスト |
|------|--------|-----------------|------|----------|-----------------|--------------|--------------|--------------|-----------|
| 20260914_150632 | v1 | claude-opus-5 | 58 | 0.716 | 0.701 | 0.780 | 6.9% | 数秒 | ~$1.00 |
| v2_fewshot_20260914_152840 | v2_fewshot | claude-opus-5 | 58 | 0.733 | 0.704 | 0.807 | 6.9% | 数秒 | ~$1.28 |
| v6 (kaggle) | v2_fewshot | Qwen2.5-14B (transformers+bnb nf4) | 58 | 0.696 | 0.656 | 0.778 | 6.9% | ~124s | $0 |
| v8 (kaggle) | v2_fewshot | Qwen2.5-14B-AWQ (vLLM+guided decoding) | 58 | 0.693 | 0.647 | 0.787 | 8.6% | ~2.90s | $0 |

## 8. コマンド履歴
```bash
mkdir -p workspace/expA00_report_labeler_validation/data
# prompt_template.py, build_validation_set.py, run.sh を作成
chmod +x workspace/expA00_report_labeler_validation/run.sh
cd workspace/expA00_report_labeler_validation
bash run.sh
# => data/validation_58.csv (58行) 生成を確認

# Claude API 実装・実行（ユーザー承認後）
python3 -m pip install anthropic python-dotenv  # bash側python3に追加インストール
bash run_llm_labeling.sh --limit 2   # ドライラン
bash run_llm_labeling.sh             # 58件フル実行
# => results/claude-opus-5/20260914_150632/{predictions_with_truth.csv, metrics.json, predictions.jsonl, log_*.txt}
```
