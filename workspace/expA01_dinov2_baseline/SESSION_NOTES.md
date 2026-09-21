# SESSION_NOTES — expA01_dinov2_baseline

## 1. セッション情報
- 日付: 2026-09-15
- 作業フォルダ: `workspace/expA01_dinov2_baseline`
- 目標: `KAGGLE_DIRECTION.md`「モデル設計方針（2026-09確定）」に基づく画像モデリングパイプラインの基盤実装。
  今回のスコープ（ユーザー指示により損失関数・fold学習ループは含めない）:
  1. report-labelerの全件適用（済み→今回はPPV/NPV算出）
  2. 前処理パイプライン（percentile clip正規化 + スライス間引き）
  3. Pooling（slice→series, series→study）
  4. DINOv2凍結特徴抽出 + 50〜100件での時間計測（全件実行はユーザー指示待ち）
  5. 特徴キャッシュのKaggle Dataset化（構成のみ、今回は本実行しない）

## 2. 背景
- `expA00_report_labeler_validation` で report-labeler（Qwen2.5-14B-Instruct-AWQ + vLLM + guided decoding）による疑似ラベル生成が完了（4,349件全件、`data/train_full_labels_4407.csv`）
- ユーザーが `KAGGLE_DIRECTION.md` に画像モデリングのモデル設計方針を追記。report-labelerタスクから画像分類パイプライン構築へのフェーズ移行のため、新しいexp番号（expA01）を切った

## 3. 試したアプローチと結果

### 3.1 report-labelerの信頼性テーブル（PPV/NPV）算出
- 既に4,349件全件への疑似ラベル適用は完了済みだったため、KAGGLE_DIRECTION.mdの指示通り「既に適用済みなら〜PPV・NPVを算出」の分岐で対応
- `expA00_report_labeler_validation/compute_label_reliability.py` を新規作成。58件gold labelに対するQwen2.5-14B-AWQ+vLLM+guided decoding（v2_fewshotプロンプト）の予測から、ラベルごとのPPV/NPVを算出
- 結果を `expA00_report_labeler_validation/data/pseudo_label_reliability.csv` に保存
- ファイル命名をKAGGLE_DIRECTION.md指定に合わせ、`data/pseudo_labels_qwen25_v2_guided.csv`（`pseudo_labels_4349.csv` と同内容）も作成
- **PPV/NPV結果**:

| label | PPV | NPV |
|-------|-----|-----|
| ACL | 0.733 | 0.929 |
| MCL | 0.444 | 0.975 |
| Medial Meniscus | 0.686 | 0.913 |
| Lateral Meniscus | 0.588 | 0.875 |
| Medial OA | 0.706 | 0.927 |
| Lateral OA | 0.429 | 0.946 |
| PF OA | 0.714 | 0.750 |
| Effusion | 0.708 | 0.900 |
| Synovitis | 0.778 | 0.675 |
| Baker's | 0.688 | 0.976 |
| Contusion | 0.560 | 0.848 |
| Fracture | 0.733 | 0.837 |

- 傾向はexpA00での58件検証結果（precision/recallの過剰・過小検出パターン）と整合的。Lateral OA・MCLはPPVが低め（過剰検出＝疑似ラベル1の信頼度が低い）、SynovitisはNPVが低め（過小検出＝疑似ラベル0の信頼度が低い）

### 3.2 前処理パイプライン（`src/preprocessing.py`）
- `percentile_clip_normalize(volume, low=0.5, high=99.5)`: study(volume)ごとにpercentile clip → [0,1]スケーリング
- `subsample_slices(volume, max_slices=60)`: 60枚超のseriesのみ均等間隔で間引き（`np.linspace`で両端を含む等間隔サンプリング）
- `preprocess_series()`: 正規化→間引きの順で実行するパイプライン関数（percentileは間引き前の全スライスから計算する方が統計的に安定するため、この順序にした）
- 定数画像（percentile範囲が0になる）は `ValueError` で fail-fast

### 3.3 Pooling（`src/pooling.py`）
- `pool_slices_to_series()`: (n_slices, feat_dim) → (2*feat_dim,) の mean+max concat
- `pool_series_to_study()`: Anatomical_Plane(3)×Fluid_Sensitive(2)の6スロット+other 1スロット=計7スロットに集約
- **設計判断（仕様に明記が無かった点、要レビュー）**:
  - 同一スロットに複数seriesが属する場合は特徴ベクトルを単純平均
  - Fat_Suppressionスカラーも同様に平均（0〜1の連続値になりうる）
  - 空スロットは特徴・Fat_Suppression共にゼロ
  - 未知のAnatomical_Plane値は "other" スロットに割り当て

### 3.4 ユニットテスト（`tests/test_preprocessing_pooling.py`）
- 合成データ（乱数配列）で13件のテストを作成、**全件PASS**
- 実DICOMは扱わないため、形状・値域・境界条件（定数画像・空配列・次元不一致等）のみ検証

### 3.5 データ構造探索（`kaggle_explore_data/`）— 完了
- DINOv2特徴抽出の前に、コンペの実DICOMディレクトリ構造を一度も確認していなかったため、GPU不要の軽量探索Notebook（`enable_gpu: false`）を作成して実行
- **判明した構造**:
  - 競技データのマウント先: `/kaggle/input/competitions/<comp-slug>/`（従来の `/kaggle/input/<comp-slug>/` ではなく、report-labeler側で発見したdatasetの新マウント規約 `/kaggle/input/datasets/<user>/<slug>/` と同様、競技データも `competitions/` という中間階層が挟まる新方式だった）
  - DICOMパス: `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`
  - **ファイル名はSOPInstanceUID（順不同）→ InstanceNumberでソートが必須**
  - DICOM属性: Modality=MR, Rows/Columns=512×512, PixelSpacing≈0.33mm, BitsAllocated=16, PhotometricInterpretation=MONOCHROME2, pixel_array dtype=uint16
- この情報を元に、パス解決は**ディレクトリ名を推測せず実ファイル（`train_series.csv`, `preprocessing.py`）を`rglob`で探す**方式に統一（report-labeler側の教訓を先取りして反映）

### 3.6 DINOv2特徴抽出コード + タイミング計測（`kaggle_dinov2_extract/`）— 実行中
- `src/feature_extraction.py`: `Dinov2FeatureExtractor` クラス。HuggingFace `transformers`（`AutoImageProcessor` + `AutoModel`）経由で `facebook/dinov2-small` / `facebook/dinov2-base` をロードし、全パラメータ凍結。画像バッチ→CLSトークン特徴を返す
  - 入力は `preprocessing.py` で [0,1] 正規化済みのグレースケール画像のため、processorには `do_rescale=False` を指定（二重リスケール防止）
- `kaggle_dinov2_extract/extract_features_notebook.py`: 実DICOM読み込み（InstanceNumberソート）→ `preprocess_series()` → `Dinov2FeatureExtractor.extract()` → `pool_slices_to_series()` → `pool_series_to_study()` の一連のパイプラインを実装
- **small・base の両サイズで100件（固定シード42）を計測**し、1件あたり時間・GPU使用量・全4,407件への外挿時間を比較してから、ユーザーにサイズ選定を委ねる設計にした
- assets（`preprocessing.py`/`pooling.py`/`feature_extraction.py`）を `rikky1201/rsna-knee-dinov2-baseline-assets` としてprivate Kaggle Dataset化（report-labelerとは別データセット。src/を分離して管理）
- **push後、一発で成功**（今回はマウントパス探索を事前に済ませていたため、report-labeler側のような複数回のトラブルシュートは不要だった）

### 3.7 タイミング計測結果（100件, 固定シード42）

| モデル | embed_dim | study特徴次元 | 平均 sec/件 | 中央値 | 最小-最大 | GPU使用量(allocated) | 全4,407件推定 |
|--------|-----------|---------------|------------|--------|-----------|----------------------|---------------|
| DINOv2-small | 384 | 5,383 | 4.58s | 3.34s | 1.63-13.74s | 0.10GB | 約5.60時間 |
| DINOv2-base | 768 | 10,759 | 4.87s | 4.20s | 2.33-10.88s | 0.36GB | 約5.97時間 |

- **small/baseの速度差はわずか6%**（DINOv2のforward計算自体ではなく、DICOM I/O（1studyあたり平均5series×数十枚のDICOMファイル読み込み）がボトルネックになっていると判断）
- GPU使用量はどちらも1.3GB未満とT4(16GB)に対して余裕が大きい
- mean > median の乖離（例: small 4.58s vs 3.34s）は、series数が多いstudy（train_series.csvで最大14series/study）が計算時間を押し上げているためと推測
- **推奨: DINOv2-base**。速度コストがほぼ同じ（+6%）にも関わらず表現力が高い（768次元 vs 384次元）ため、KAGGLE_DIRECTION.mdの「Efficiencyトラック考慮でlarge/giantは避ける」の範囲内でbaseを選ぶのが合理的と判断
- 全4,407件の推定所要時間（約6時間）はKaggle週間GPU予算内で実行可能

## 4. ファイル構成
```
workspace/expA01_dinov2_baseline/
├── SESSION_NOTES.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # percentile clip正規化 + スライス間引き
│   └── pooling.py             # slice→series→study pooling
├── tests/
│   └── test_preprocessing_pooling.py  # 合成データでのユニットテスト（13件PASS）
└── kaggle_explore_data/       # 実DICOM構造の探索用Kaggle Notebook（GPU不要）
    ├── explore_notebook.py
    ├── kernel-metadata.json
```
- `workspace/expA00_report_labeler_validation/compute_label_reliability.py` と `data/pseudo_label_reliability.csv`、`data/pseudo_labels_qwen25_v2_guided.csv`、`data/qwen_v2guided_validation_predictions.csv` も本セッションで追加（report-labeler側の成果物）

## 5. 重要な知見
- **Kaggleのマウントパス規約が変化している**（report-labeler側でも同様の発見）。`/kaggle/input/<slug>/` ではなく `/kaggle/input/competitions/<slug>/` や `/kaggle/input/datasets/<user>/<slug>/` という中間階層が入る。ディレクトリ名を仮定せず、目印となる実ファイル（`train_series.csv`, `preprocessing.py` 等）を `rglob` で探す設計に統一したことで、今回のKaggle実行は一発成功した（教訓を先取りして反映できた）
- **DICOMファイル名はSOPInstanceUID（順不同）**。スライスの空間的順序を保つには `InstanceNumber` でのソートが必須（気づかずに実装すると、スライス順がランダムになりpercentile/poolingの結果は数値的には壊れないが、mean/maxプーリングを使う設計なので実害は限定的ではあるものの、将来3D的な処理をする場合は致命的なバグになりうる）
- **DINOv2 small/baseの速度差はごくわずか（6%）**。ボトルネックはDICOM I/Oであり、GPU計算（DINOv2 forward）ではない。したがって「小さいモデルを選んで高速化する」という判断はほぼ意味がなく、表現力で選ぶべき → baseを推奨

### 3.8 DICOM読み込み並列化の検証 — 結果不一致のため不採用
- ユーザー指摘（「DataLoaderのnum_workersは？」）を受け、`load_series_volume()`が完全に逐次実装（並列化なし）だったことが判明
- `ThreadPoolExecutor(max_workers=8)` で1 series内のDICOM読み込みを並列化し、同じ100件・同じシードで再計測（v2）
- **結果が不一致**: small 4.58s→3.38s（26%改善）、base 4.87s→**5.35s（10%悪化）**
- Kaggle側のネットワークストレージI/O性能の実行間variance、またはスレッド数がストレージ帯域の頭打ちを超えている可能性が高く、この1回の計測だけでは並列化の効果を確定的に判断できないと判断
- **ユーザー判断: v1（逐次読み込み）のシンプルな実装のまま全件実行**することに決定

### 3.9 全4,407件への本番実行（`kaggle_dinov2_extract_full/`）
- `kaggle_dinov2_extract/`（ベンチマーク専用）とは別に、本番実行専用フォルダを新設（report-labelerで確立したbenchmark/production分離パターンを踏襲）
- 構成: DINOv2-base固定、逐次DICOM読み込み（v1方式）、**200件ごとのチャンク単位で `/kaggle/working/features_chunk_XXXX.npz` に保存**（npzは追記不可のためチャンクごとに別ファイル。1チャンクの失敗が既存チャンクを破壊しない設計）
- per-study処理失敗時は `Exception` を広く捕捉してログ＋`extraction_manifest.json`の`failed_uids`に記録した上でスキップ（4,407件規模のバッチジョブで1件のために6時間の計算を失うのを避けるため。pseudo-label生成時と同じ方針）
- push後、T4 GPU上で実行 → **完了（約10時間）**

### 3.10 全件実行の結果
- **実際の所要時間: 約10時間**（事前計測ベースの推定約6時間を大幅に超過。`mean_sec_per_study` = 7.55s、100件ベンチマーク時の4.87sより約55%遅い。原因は未調査だが、Kaggle側I/O性能のvariance、または大規模実行特有の負荷パターンの違いによるものと推測— 3.8節で観測した並列化の効果不一致と同根の「Kaggle共有インフラのI/O揺らぎ」が濃厚）
- **成功 4,405件 / 失敗 2件**（4,407件中）。失敗2件は `pydicom` のエラー「The number of bytes of pixel data is less than expected」= **実際にDICOMファイルが破損/不整合**（コード側の不具合ではない）。`extraction_manifest.json` の `failed_uids` に記録済み
- チャンク単位（200件, 計23チャンク）で正常に `/kaggle/working/features_chunk_XXXX.npz` に保存。チェックポイント設計が機能し、10時間の長時間ジョブでも欠損なく回収できた
- `merge_feature_chunks.py` で23チャンクを統合し `data/dinov2_base_features_4405.npz`（study_uids: (4405,), features: (4405, 10759), float32）を作成。件数・次元・重複無しのassertを通過
- **タスク5（特徴キャッシュのKaggle Dataset化）完了**: `rikky1201/rsna-knee-dinov2-base-features` としてprivate push（116MB）。今後の学習Notebookは `dataset_sources` に追加するだけで、DINOv2特徴の再抽出なしに学習へ進める

### 3.11 損失関数・fold設計・分類ヘッド学習ループの実装
- ユーザー指示: gold/pseudo結合時に破損DICOM2件（3.10節、いずれもpseudo側）を除外。実装は `src/data_prep.load_training_data()` で labels(4407) と features(4405) を inner join することで自動的に対応（join後に gold件数が58のまま変化しないことを assert で確認）

**fold設計**:
- `workspace/fold/generate_folds_multilabel.py` を新規作成（`iterstrat.MultilabelStratifiedKFold` を使う専用スクリプト。既存の `generate_folds.py` は単一ラベルのstratifyにしか対応していないため、README.mdの「専用スクリプトを同階層に追加する」規約に従って分離）
- gold 58件のみを対象に4分割（`workspace/fold/v1_multilabel_gold58/folds.csv`）。各fold 13〜17件、陽性0件のラベルは無し
- pseudo（4,347件）はfold番号なし（`fold=-1`）とし、**全foldで常にtrain側に含める**設計（`src/data_prep.split_fold()`: val = 対象foldのgoldのみ、train = 残りのgold + pseudo全件）

**損失関数**（`src/loss.py`）:
- `SoftTargetBCELoss`: `F.binary_cross_entropy_with_logits(reduction='none', pos_weight=π)` でラベル軸(12)の要素別損失を得てから `sum(dim=1)` → `mean()`（KAGGLE_DIRECTION.mdの `L=Σ_j ℓ_j` をバッチ平均する設計。デフォルトの `reduction='mean'` は全要素(batch×12)で割ってしまい定義とズレるため意図的に reduction='none' から手動集計）
- ソフトターゲット（`src/data_prep.py`）: pseudo行は `pred==1 → PPV_j`, `pred==0 → 1-NPV_j` にベクトル化変換。gold行は実測値をそのまま使用
- pos_weight: gold 58件の分布から算出。MCL(5.444)が最大（最も不均衡）、Effusion(0.657)が最小（陽性多数派）

**分類ヘッド**（`src/model.py`）:
- `LinearProbeHead`: `nn.Linear(10759, 12)` のみ（KAGGLE_DIRECTION.mdの「全パラメータ凍結・分類ヘッドのみ学習」方針通り、DINOv2側は既にキャッシュ済み特徴として固定済み）
- `FeatureStandardizer`: 各foldの **train側（gold train + pseudo全件）の統計量のみ**で標準化し、held-out gold検証セットにも同じ統計量を適用（CLAUDE.mdの前処理原則「trainの統計量で正規化、testの情報を使わない」を厳守）

**学習ループ**（`train.py`）: CLAUDE.mdの学習コードの鉄則に準拠
- config.yaml駆動（OmegaConf dotlist上書き対応）、実験フォルダ基準の絶対パスで `results/{experiment_name}/foldN/` に出力
- チェックポイント再開対応（`last.ckpt`/`best.ckpt`、`trainer.ckpt_path` で再開可能）
- AMP対応（`torch.amp.autocast`/`GradScaler`。ローカル実行環境がCPUのみだったため実質無効化されるが、GPU環境では有効になる設計）
- `logging` モジュールでコンソール(INFO)+ファイル(DEBUG)出力、`training_log.json`（学習曲線の生データ）+ `training_curve.png`（loss/AUC推移の可視化）を保存
- 早期終了（val macro AUC が20epoch改善しなければ停止）

**動作確認（fold0のみ先行実行）**:
- train_loss: 11.20 → 7.82（単調に近い減少を確認）
- val_macro_auc: epoch11で best 0.7589（AUC計算が正常に機能することを確認）
- 出力ファイル（ckpt, config.yamlコピー, training_curve.png, training_log.json, ログ）が全て正しく生成されることを確認

**全4fold実行結果**:

| fold | train件数 | val件数(gold) | best val macro AUC |
|------|-----------|----------------|---------------------|
| 0 | 4,392 | 13 | 0.7589 |
| 1 | ~4,390 | 17 | 0.7593 |
| 2 | ~4,393 | 15 | 0.7319 |
| 3 | ~4,391 | 13 | 0.7501 |
| **平均** | | | **0.7501** |

ラベル別AUC（4fold平均、各foldのbestエポック時点）: Medial OA(0.942)・Baker's(0.841)・Medial Meniscus(0.803)が高精度、**Synovitis(0.564)がほぼランダムに近く最弱**（report-labelerの検証時点からSynovitisはrecall/NPVが低く課題として認識していたパターンと一致）。Contusion(0.643)・Effusion(0.693)も相対的に弱い

### 3.11.1 gold OOF全体のpooled macro AUC（2026-09-16追記、`compute_oof_auc.py`）
- 上記「平均 val macro AUC = 0.7501」は各foldのbest epoch macro AUCを4個単純平均した値（macro-of-macros）。各fold 13〜17件と少数で分散が大きく、bestエポック選択自体がそのfoldのval AUCを見て行われるため軽い楽観バイアスがある
- 各foldのbest.ckptで該当val_idxを推論し、**58件全体をpoolしてラベルごとにAUC計算→macro平均**した「真のOOF AUC」を別途算出: **0.7375**（fold単純平均より-0.0126）
- ラベル別OOF AUC: ACL 0.717 / MCL 0.796 / Medial Meniscus 0.764 / Lateral Meniscus 0.707 / Medial OA 0.915 / Lateral OA 0.774 / PF OA 0.741 / Effusion 0.636 / **Synovitis 0.576（最弱）** / Baker's 0.875 / Contusion 0.655 / Fracture 0.694
- Public LB(0.688)との比較は、fold単純平均(0.7501)ではなくこのpooled OOF(0.7375)を基準にする方が妥当（差-0.0495）

### 3.12 初回submission（Kaggle Code Competition）パイプライン構築
- ユーザー要望: 一度Publicに提出したい。本コンペはCode Competition（`enable_internet:false`必須）のため、提出用Notebookは実際のtest DICOM画像に対してDICOM読み込み→前処理→DINOv2特徴抽出→分類ヘッド推論を通しで行う必要がある

**提出用「フルデータモデル」の学習**（`train_full.py`, 新規）:
- 4fold CVはあくまで検証目的。提出には gold全58件+pseudo全件（検証データを割かない）で学習し直したモデルを使う設計にした
- 固定epoch数 = 4fold CVのbest_epoch中央値（0,5,5,11 → 5）を採用（held-outが無いためearly stopping不可）
- 標準化統計量（train全件のmean/std）もチェックポイントに含めて保存（推論時に同じ変換を再現するため必須）
- `results/expA01_linear_probe/full/full_model.ckpt` を生成（train_loss 11.08→9.28で単調減少）

**オフライン推論のための準備**:
- Code Competitionは`enable_internet:false`必須。DINOv2の重みをHuggingFace Hubからオンライン取得できないため、**ローカルで`facebook/dinov2-base`をダウンロードし`save_pretrained()`で自己完結ディレクトリに保存**、`rikky1201/dinov2-base-weights-offline`としてprivate Kaggle Dataset化（330MB）
- `src/feature_extraction.py`に`model_path`パラメータを追加し、ローカルパスからのオフラインロードに対応
- `src/model.py`（LinearProbeHead, FeatureStandardizer）と`full_model.ckpt`を既存の`rsna-knee-dinov2-baseline-assets`データセットに追加してバージョン更新

**提出用Notebook**（`kaggle_submission/notebook/inference_notebook.py`）:
- `competition_sources`でtest.csv/test_series.csv/test_series/のDICOM本体にアクセス、`dataset_sources`でassets+DINOv2オフライン重みをattach
- パス解決は今までの教訓通り`rglob`で実ファイルを再帰探索（マウント階層を仮定しない）
- 出力前に行数・列名列順・値域[0,1]・欠損なし・StudyInstanceUID一致をassertでfail-fast検証

**ハマった点**: v1は`torch.load(ckpt)`が`UnpicklingError`で失敗。**PyTorch 2.6以降 `torch.load` のデフォルトが `weights_only=True` に変更**されており、チェックポイント内のnumpy配列（scaler統計量）がデフォルトの許可リストに無いため読み込み拒否された。自作の信頼できるチェックポイントのため `weights_only=False` を明示して解決（`train.py`のckpt再開処理にも同じ修正を適用済み）

**結果**: v2で一発成功。`submission.csv`がsample_submission.csvと列名・列順序完全一致、行数一致（4行=ヘッダー+3件のダミーtest）、値は0.15〜0.82の多様な確率（0.5固定でないことから過学習していないことを示唆）。`submission-validator` agentで独立検証中

### 3.13 submission-validator による検証 + パス解決の高速化
- agent検証結果: **Failなし**。kernel-metadata.json（enable_internet:false等）、self-contained設計、submission.csvフォーマット（列名・行数・値域・UID一致）、fail-fast assert、jupytext同期、gitignore対応すべてPass
- **Warning指摘（重要）**: パス解決（`_find_by_marker`のrglob）が競技データセット（80万ファイル超のDICOM）配下を毎回全走査し**約14分**かかっていた。実際の非公開test setが大きい場合、9時間の実行制限を圧迫するリスク
- 対処: 浅い階層チェック（3階層まで）を試したが `/kaggle/input/datasets/<user>/<slug>/<file>` は4階層のため不十分（v3はむしろ悪化、1039秒）。**`os.walk`で深さ制限(5)+ 既知の巨大ディレクトリ（train_series/test_series）を明示的に枝刈り**する実装に変更（v4）
- **v4結果: パス解決が23.5秒未満に短縮**（v1比 約97%減）。全体の実行時間も約2分（3件）に短縮。フォーマット再検証も全項目Pass
- `submit/SUBMISSIONS.md` にv001として記録（Public/Private LBは提出後に記入）

### 3.14 バックボーン比較検証: DINOv2 vs RadImageNet ResNet50（2026-09-16）
- n=58限定（pseudo不使用）とn=4,405（本番同等、gold58+pseudo4,349）の両方でOOF比較を実施
- **n=58限定**（`compare_backbones_gold_only.py`, LogisticRegression）: MACRO DINOv2 0.591 vs RadImageNet 0.585（ほぼ同水準）。PCA次元スイープ（`compare_radimagenet_pca_sweep.py`）でも次元数を揃えてもギャップは解消せず
- **本番同等**（`compute_oof_auc_radimagenet.py`, train.py本番手順）: MACRO DINOv2 0.7375 vs RadImageNet 0.6725（-0.065）。ただし**Synovitis(+0.125)・Effusion(+0.111)は明確に改善**、**ACL・MCL・Medial OA・Baker's(-0.230)は悪化**というラベル依存の結果
- RadImageNet特徴: `data/radimagenet_resnet50_features_4405.npz`（4405×28679, `kaggle_radimagenet_full_extract/`で全件抽出済み）。学習済みckpt: `results/expA01_linear_probe_radimagenet/foldN/best.ckpt`
- **判断**: DINOv2ベースライン維持、全面切り替えは見送り。Synovitis/Effusion特化のラベル別アンサンブル候補として終盤フェーズ用に記録（詳細は`daily_reports/20260916.md`）

### 3.15 列ごとのバックボーンルーティング推論を提出Notebookに実装（2026-09-16）
- ユーザー指示: Effusion・Synovitis列のみRadImageNet出力、残り10列はDINOv2出力を採用
- RadImageNet版フルデータモデル追加（`train_full.py`の`FIXED_EPOCHS`をconfig駆動`trainer.full_fixed_epochs`に変更、RadImageNetはCV best_epoch中央値21を使用）
- **事故と復旧**: `kaggle datasets version -p <folder>`はフォルダ内容で完全上書きするため、以前のRadImageNet gold58比較用pushで共有assetsデータセットから`model.py`/`full_model.ckpt`が消失していた。`kaggle_submission/kaggle_dataset_assets/`に全ファイルを統合し復旧。DINOv2/RadImageNetのfullモデルは`full_model_dinov2.ckpt`/`full_model_radimagenet.ckpt`と改名して共存
- `inference_notebook.py`: DICOM読み込み・前処理を1回だけ行い両バックボーンのforwardに使い回す設計。kernel v8で公開ダミー3件検証: 非routing10列は旧v6と完全一致、Effusion/Synovitisのみ値変化、エラーなし
- 詳細は`daily_reports/20260916.md`参照

### 3.16 gold OOF評価にbootstrap 95%信頼区間を追加（2026-09-16、ユーザー方針転換の契機）
- `src/metrics.py`新規: `bootstrap_label_aucs`（study単位重複ありリサンプリング）/`summarize_ci`/`bootstrap_paired_delta`（対応ありbootstrap、2モデル比較の有意性判定用）。テスト6件PASS
- `compute_oof_auc.py`/`compute_oof_auc_radimagenet.py`をOOF予測保存+CI自動出力に対応。`bootstrap_compare.py`新規（保存済みOOFから2モデル間の有意差検証）
- **判明**: RadImageNetルーティングで見えたEffusion(+0.109)・Synovitis(+0.123)の改善は、対応ありbootstrapの95%CIが共に0をまたぎ**統計的に非有意**。v003でLBが変化しなかったのはこれで定量的に説明がつく（Baker's悪化とMACRO悪化のみ有意）
- **運用変更**: 今後の比較検証は点推定だけでなくこのCIを見てから提出判断する（ユーザー方針）。詳細は`daily_reports/20260916.md`参照

## 6. 次のステップ（優先度付き）
1. **[高]**（ユーザー操作待ち）Kaggle UIでNotebookを開き、Save & Run All → Submit to Competition（手動）
2. **[中]** 提出後、Public LBスコアを `submit/SUBMISSIONS.md` に記入し、CV(0.7501)との相関を確認
3. **[中]** Synovitis・Contusion・Effusionの弱さへの対処検討（report-labeler側のプロンプト改善 or 損失関数の重み調整 or fine-tuningへの切り替え検討）
4. **[低]** DICOM破損2件の詳細確認（余力があれば）
5. **[低]** DINOv2の入力解像度の検証（現状processorデフォルト、おそらく224x224）

## 7. 性能変化の記録
| 実行 | 内容 | 結果 |
|------|------|------|
| - | ユニットテスト（前処理・pooling） | 13/13 PASS |
| DINOv2-small タイミング計測(v1逐次) | 100件, T4 GPU | 平均4.58s/件, 全件推定5.60時間 |
| DINOv2-base タイミング計測(v1逐次) | 100件, T4 GPU | 平均4.87s/件, 全件推定5.97時間 |
| DINOv2-small タイミング計測(v2並列io=8) | 100件, T4 GPU | 平均3.38s/件（26%改善） |
| DINOv2-base タイミング計測(v2並列io=8) | 100件, T4 GPU | 平均5.35s/件（10%悪化） |
| **DINOv2-base 全件本番実行(v1逐次)** | **4,407件, T4 GPU** | **成功4,405件/失敗2件, 平均7.55s/件, 総時間約10時間** |
| **linear probe 4fold CV** | **gold 58件, ローカルCPU** | **平均 val macro AUC = 0.7501**（fold: 0.759/0.759/0.732/0.750） |
| 提出Notebook パス解決(v1: rglob全探索) | test 3件, T4 GPU | 約14分 |
| 提出Notebook パス解決(v4: os.walk深さ制限) | test 3件, T4 GPU | **23.5秒未満**（約97%短縮） |

## 8. コマンド履歴
```bash
cd workspace/expA00_report_labeler_validation
python3 compute_label_reliability.py
# => data/pseudo_label_reliability.csv 生成

cd ../expA01_dinov2_baseline
python3 tests/test_preprocessing_pooling.py
# => 13 passed, 0 failed

cd kaggle_explore_data
python3 -m jupytext --to ipynb --set-kernel python3 explore_notebook.py
python3 -m kaggle kernels push -p .
# => データ構造判明: /kaggle/input/competitions/<slug>/train_series/<StudyUID>/<SeriesUID>/<SOPUID>.dcm

cd ../kaggle_dinov2_extract
PYTHONUTF8=1 bash upload.sh
# => 一発成功（v1逐次）。timing_benchmark.json 取得: small 4.58s/件, base 4.87s/件

# v2: DICOM並列読み込み(ThreadPoolExecutor)で再計測 → small改善/base悪化で不採用
PYTHONUTF8=1 python3 -m jupytext --to ipynb --set-kernel python3 extract_features_notebook.py
PYTHONUTF8=1 python3 -m kaggle kernels push -p . --accelerator NvidiaTeslaT4

# 全件本番実行（v1逐次, DINOv2-base固定）
cd ../kaggle_dinov2_extract_full
PYTHONUTF8=1 bash upload.sh
# => 約10時間後 COMPLETE。features_chunk_0001~0023.npz + extraction_manifest.json 取得

cd ..
python3 merge_feature_chunks.py
# => data/dinov2_base_features_4405.npz (4405, 10759) 生成

cd kaggle_features_cache
PYTHONUTF8=1 bash upload.sh
# => rikky1201/rsna-knee-dinov2-base-features としてprivate Kaggle Dataset化完了

# 損失関数・fold設計・学習ループ
cd ..
python3 prepare_gold_labels_for_fold.py
# => data/gold_58_labels.csv 生成

cd ../fold
python3 generate_folds_multilabel.py --input ../expA01_dinov2_baseline/data/gold_58_labels.csv \
    --version v1_multilabel_gold58 --id-col StudyInstanceUID \
    --target-cols "ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture" \
    --n-splits 4
# => workspace/fold/v1_multilabel_gold58/folds.csv 生成

cd ../expA01_dinov2_baseline
PYTHONUTF8=1 bash run.sh data.fold=0 trainer.max_epochs=100
# => fold0のみ先行実行。train_loss単調減少・val_macro_auc計算を確認（best 0.7589）

for fold in 1 2 3; do PYTHONUTF8=1 bash run.sh data.fold=$fold trainer.max_epochs=100; done
# => 全4fold完了。平均 val macro AUC = 0.7501
```
