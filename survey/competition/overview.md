# コンペ概要 — RSNA Knee Abnormality Detection

最終更新: 2026-09-14

## 1. プラットフォーム
- Kaggle（**Code Competition**: Kaggle Notebook 提出、実行時間 9時間以内、インターネットアクセス不可）
- 主催: RSNA（Radiological Society of North America）+ Kaggle, Inc.

## 2. タスク定義
- **入力**: 膝MRI検査（DICOM画像、複数シーケンス/複数スライス）。学習データには加えて**放射線読影レポート本文（フリーテキスト、多言語）**が付属するが、**推論時（test）にはレポートは提供されない**（`test.csv` は `StudyInstanceUID` の1カラムのみ、実データで確認済み）
- **出力**: 12種類の二値所見（binary findings）それぞれの確率（0-1）。**正式名称・列順序を実データで確認済み**:
  1. ACL（前十字靭帯）
  2. MCL（内側側副靭帯）
  3. Medial Meniscus（内側半月板）
  4. Lateral Meniscus（外側半月板）
  5. Medial OA（内側区画 変形性関節症）
  6. Lateral OA（外側区画 変形性関節症）
  7. PF OA（膝蓋大腿区画 変形性関節症）
  8. Effusion（関節液貯留）
  9. Synovitis（滑膜炎）
  10. Baker's（ベーカー嚢腫）
  11. Contusion（挫傷）
  12. Fracture（骨折）
- **クラス数**: 12（マルチラベル二値分類。1検査につき12個の確率を出力）
- **評価単位**: 検査（study = `StudyInstanceUID`）単位
- **⚠️ 重要な構造上の特徴（実データで確認済み）**: `train.csv` は **4,407 study 全件にレポート本文があるが、12ラベルが埋まっているのはそのうち58件のみ**（残り4,349件は全ラベル空欄）。これは通常の教師あり学習ではなく、**実質的に「58件の gold label + 4,349件のレポート本文（ラベルなし）」という強い弱教師あり学習（weakly-supervised）構造**。レポートテキストから疑似ラベルを生成する report-labeler の構築が事実上の前提タスクになっている（コミュニティ実装でも同様の指摘あり）
- **タスクの特徴**: 画像 + レポートテキストの両方を学習に使う初のRSNAチャレンジ。推論時は画像のみ（レポート提供なし）なので、**学習時にレポートで得た知識を画像モデルへ転移させる設計が必須**

## 3. データ所在
- URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
- **確定値（軽量CSV実物で確認済み）**:
  - `train.csv`: 4,407 rows（= 4,407 study、重複なし）。カラム: `StudyInstanceUID, Report, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`
  - `train_series.csv`: 24,371 rows（4,407 study 全件に対応、1 studyあたり series 数は 3〜14 の分布、最頻値5）。カラム: `StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane`（Sagittal/Coronal/Axial の3方向）
  - `test.csv`: 4 rows（公開されているのはダミー/サンプルで、実際のtestはKaggle実行時に差し替わるcode competition方式と推測）。カラムは `StudyInstanceUID` のみ
  - `test_series.csv`: 16 rows（test.csv の4studyに対応、train_seriesと同カラム構成）
  - `sample_submission.csv`: 4 rows、全ラベル0.5固定
  - 画像本体（DICOM, ~500GB）は**方針によりダウンロードしない**。Kaggle Notebook上でコンペデータセットをアタッチして読み込む
- フォーマット: DICOM（画像）+ 上記メタデータCSV + フリーテキストレポート（train.csvのReport列、全4,407件に存在）
- ライセンス: 未確認
- ローカル配置先: `datasets/rsna-knee-abnormality-detection/`（軽量CSV5点のみ配置済み。画像は非配置）

## 4. 評価指標
- 名称: **macro-averaged AUC-ROC**
- 定義: 12個の binary finding ラベルそれぞれで AUC-ROC を計算し、**均等重み（macro average）**で平均
- 実装上の注意:
  - sklearn の `roc_auc_score(y_true, y_score, average='macro', multi_class=...)` を使う場合、マルチラベル（12個の独立した二値問題）として **ラベルごとに `roc_auc_score` を計算して単純平均**する実装が素直（`average='macro'` を直接マルチラベル行列に渡す挙動を要検証）
  - 陽性例が極端に少ないラベル（レアな所見）がある場合、fold内でAUC計算が不安定になる可能性 → 層別fold設計時に要考慮

## 5. 提出形式
- 種別: **(b) Kaggle Code Competition**
- 提出方法: Kaggle Notebook を Save & Run All → "Submit to Competition"（**最終提出は手動**、テンプレのルール通り）
- 実行制約: **9時間以内**、**インターネットアクセス不可**（`enable_internet: false`。pretrained weight は事前に Dataset としてアップロードしておく必要あり）
- 出力: `/kaggle/working/submission.csv`（12ラベル分のカラムを正しい順序で出力する必要あり）
- 詳細フローは `tools/kaggle_code_competition_submission.md` を参照

## 6. タイムライン
- 開始: 不明（要確認。2026年8月頃 RSNA アナウンス）
- **Entry Deadline（参加登録締切）**: 2026-10-15
- **最終締切（Final Submission Deadline）**: 2026-10-22（残り**約38日**、本日2026-09-14時点）
- 提出回数制限: 未確認

## 7. ルール
- チーム人数: 未確認（要 `/rules` ページ直接確認 — WebFetch/WebSearch では詳細が取得できなかった）
- 外部データ: 未確認
- 事前学習モデル: 使用実績あり（コミュニティのベースラインで DINOv2 等の pretrained model 使用が確認できる）が、**公式ルールでの明記は未確認**
- 商用ライセンス制約: 未確認
- **重要な注意点**: レポート本文（フリーテキスト、患者由来データ）を**外部の hosted LLM API に送信することは Rule 4.b（Data Security）に抵触する可能性がある**、とコミュニティで指摘あり。レポートテキストを扱う場合はローカル処理に留め、外部API送信は避けるべき（要 `/rules` ページで一次情報を確認してから対応方針を確定する）

## 不明点・確認待ち
- [ ] `/rules` ページの一次情報（チーム人数上限、外部データ可否、事前学習モデル可否、商用ライセンス制約、データセキュリティ条項の正確な文言）— WebFetch がSPAのため取得できず、要ログイン後の手動確認 or 再度WebFetch/スクレイピング
- [x] 12ラベルの正式名称・提出CSVでの列順序 → **実データで確認済み**（上記タスク定義参照）
- [ ] データライセンス（データ配布ページで要確認）
- [ ] コンペ開始日
- [ ] 提出回数制限（1日あたり）
- [x] test.csv の正確なカラム構成 → **実データで確認済み**（`StudyInstanceUID` 1カラムのみ。公開分は4行のダミーで、実評価は非公開test setに対しNotebook実行と推測）
- [ ] Validation phase / Test phase の分離有無（Code Competition では2段階採点のことが多い。要確認）
- [ ] **58件の gold label がどう選ばれたか**（ランダムサンプルか、特定施設 か）。fold設計・検証戦略に直結するため要調査（train.csv内の分布確認は次アクションで実施）

## 次のアクション（提案）
- ユーザーに Kaggle へのログインを依頼し、`/data` と `/rules` ページの内容を直接共有してもらう（もしくはブラウザで確認したスクリーンショットを提供してもらう）のが最速
- 堅実案: DICOM読み込み → 2.5D/3Dモデル（`reference/` のPyTorch Lightning + timm + smp テンプレをベースに、マルチラベル12出力ヘッドに変更）でまず baseline を作る
- 爆発案: レポートテキストを**学習時のみ**補助タスク（テキストからラベル抽出の弱教師あり学習 or マルチモーダル事前学習の補助損失）として活用し、画像単体モデルより強いラベル/特徴表現を得る（推論時はテキスト無しでも画像エンコーダに知識が転移している状態を狙う = 一種の知識蒸留）
