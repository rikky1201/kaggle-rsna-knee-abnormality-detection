# SESSION_NOTES — expA02_resnet34_finetune

## 1. セッション情報
- 日付: 2026-09-20
- 作業フォルダ: `workspace/expA02_resnet34_finetune`
- 背景: expA01（DINOv2凍結+linear probe, LB 0.688）に対し、DINOv2 attention pooling版は
  CVで有意差なしと判明（`daily_reports/20260919.md`）。次の一手としてバックボーンの
  fine-tuning（凍結解除）をユーザー指示で検討開始
- 新しいexp番号にした理由: 「特徴を1回抽出してキャッシュ→軽い分類ヘッドのみ学習」から
  「エポックごとにDICOM読み込み→backbone forward→backward」への根本的なアーキテクチャ変更
  （CLAUDE.mdの「大きく方針が変わる時だけ新しいexp番号にする」に該当）

## 2. 経緯（fine-tuning候補選定）
expA01配下で5候補を100件ベンチマーク（`kaggle_dinov2_finetune_benchmark/`等、詳細は
`daily_reports/20260919.md`参照）:

| 候補 | trainable params | OOM率 | ピークメモリ | 30h予算epoch/fold(4fold) |
|---|---|---|---|---|
| DINOv2全層 | 86.6M | 16% | 98.98% | 1.33 |
| DINOv2後半4層 | 28.4M | 0% | 55.06% | 1.52 |
| RadImageNet R50全層 | 23.5M | 0% | 57.12% | 1.39 |
| ResNet18(ImageNet)全層 | 11.2M | 0% | 18.41% | 1.66 |
| **ResNet34(ImageNet)全層** | 21.3M | 0% | 28.06% | **1.83（最良）** |

RadImageNetは公式にResNet50/DenseNet121/InceptionResNetV2/InceptionV3のみ配布
（ResNet18/34の重みは存在しない。WebSearchで確認、`daily_reports/20260919.md`参照）。
**ユーザー判断: ResNet34（ImageNet事前学習, 全層fine-tuning）を採用**。

## 3. 学習ループ設計判断

### fold戦略: 単一train/val splitに集中（ユーザー判断）
- 4fold CVのままだと1.83epoch/foldしか回らずfine-tuningが収束前に終わるリスクが高いため、
  既存の`workspace/fold/v1_multilabel_gold58/folds.csv`のfold=0（gold 13件）をvalに固定し、
  残りgold 45件 + pseudo 4,349件 = 4,394件をtrainに使う単一splitに変更
- 30h予算 ÷ 1epoch時間(ResNet34ベンチマーク実績 4.10h/epoch, train=4390件換算) ≈ **7.3epoch**
- 代償: 4fold CVのbootstrap信頼区間比較のような厳密なOOF評価はできない（val 13件のみの
  単一評価になる）。fine-tuning自体の効果検証は、この単一val split上でのAUC推移で行う

### Kaggleセッション制約への対応: 1セッション=1epoch, kernel自己参照で再開
- Kaggle Notebookの1セッション実行時間上限（実質約9時間）に対し、1epoch ≈ 4.1時間のため
  安全マージンを見て**1 push = 1epoch**とする（複数epochを1セッションに詰め込まない）
- 再開の仕組み: `kernel-metadata.json`の`kernel_sources`に**自分自身**を指定し、前回pushの
  出力（`last.ckpt`）を次回pushで`/kaggle/input/`から読み込んで再開する（Kaggle Notebookの
  自己参照chaining。初回pushのみ`kernel_sources: []`、2回目以降は自身のslugを追加）
- 目標epoch数7に対し、7回程度手動でpush→完了確認→再pushを繰り返す運用になる

### その他の設計（pooling/損失関数/前処理はexpA01のパイプラインを踏襲）
- pooling: slice→series mean+max concat → series→study 7スロット集約（attention pooling
  ではなくexpA01のmean+max版を踏襲。バックボーンのfine-tuning効果のみを見るため、pooling
  方式は変数を増やさない）
- 損失関数: `SoftTargetBCELoss`（pos_weight/PPV・NPVソフトターゲット）はexpA01と同一
- slice間引き: `FINETUNE_MAX_SLICES_PER_SERIES=16`（ベンチマークと同一条件を維持。ResNet34は
  メモリに大きな余裕（28%使用）があるため増やす余地はあるが、まずはベンチマーク済みの
  条件で確実に動かすことを優先し、増量は効果検証後の追加検討とする）
- AMP(autocast fp16) + `gc.collect()`（OOM時の参照サイクル解放。全層DINOv2fine-tuning
  ベンチマークで判明したリーク対策と同じ）+ gradient accumulation(ACCUM_STUDIES=8)

## 4. ファイル構成
```
workspace/expA02_resnet34_finetune/
├── SESSION_NOTES.md
└── kaggle_train/
    ├── train_notebook.py       # Kaggle実行用の学習notebook（自己完結、1push=1epoch）
    ├── kernel-metadata.json
    └── upload.sh
```
- labels/reliability/fold CSVは`rikky1201/rsna-knee-dinov2-baseline-assets`データセットに
  追加（`train_full_labels_4407.csv`, `pseudo_label_reliability.csv`。foldは既存の
  `folds_gold58.csv`を流用）
- DICOM前処理は既存の`preprocessing.py`（同データセット内、既存）をそのまま再利用

## 5. 進捗（2026-09-20〜21）

- [x] labels/reliability CSVをKaggle assetsデータセットに追加
- [x] スモークテストで2件のバグを発見・修正（loss計算のdim不一致、resume chainingでのbest.ckpt消失）→詳細は`daily_reports/20260919.md`
- [x] resume機構（kernel自己参照chaining）を検証済み
- [x] epoch0〜2実行完了。**週間GPU予算切れで一時停止中**（累計15.88h/30h消費）

### epoch別結果
| epoch | val macro AUC | train_loss | 所要時間 |
|---|---|---|---|
| 0 | 0.5428 | 11.63 | 4.78h |
| 1 | **0.5609（best）** | 10.42 | 5.79h |
| 2 | 0.4685（悪化） | 10.28 | 5.31h |

- train_lossは単調減少だが、val macro AUC(n=13の小標本、CIが広い)は0.543→0.561→0.469と一貫した改善トレンドが見えていない。小標本ノイズの可能性が高いが要注視
- 現在のbest.ckptはepoch1時点の重み

## 6. RunPod移行 + epoch3〜6実行（2026-09-24〜25）

Kaggle週間GPU予算切れを機にRunPodへ移行（詳細設計・ハマりどころは`daily_reports/20260924.md`参照）。
生DICOM転送は1.36TBで非現実的と判明したため、前処理済み画像キャッシュ（224x224 uint8, slice cap16,
`kaggle_preprocess_cache/`で生成）を8個のKaggle datasetに分割保存し、RunPod pod（RTX4090）にダウンロードして
学習する方式に変更。`runpod_train/train_runpod.py`（DICOM I/O無し、複数epoch連続実行）を新規実装。

**結果**:
| epoch | 環境 | val macro AUC | train_loss |
|---|---|---|---|
| 0 | Kaggle T4 | 0.5428 | 11.63 |
| 1 | Kaggle T4 | 0.5609 | 10.42 |
| 2 | Kaggle T4 | 0.4685 | 10.28 |
| 3 | RunPod 4090 | 0.5954 | 10.09 |
| 4 | RunPod 4090 | 0.5872 | 9.80 |
| 5 | RunPod 4090 | 0.6035 | 9.55 |
| **6** | RunPod 4090 | **0.6137（best）** | **9.34** |

- 1epoch実測: 約5.3分（キャッシュ版, RTX4090）。Kaggle DICOM版(4.8〜5.8h)から桁違いに高速化
- epoch3以降はtrain_loss単調減少 + val macro AUC上昇トレンドが確認できた（epoch0-2の乱高下から改善）
- 4epoch(3〜6)の実コスト: $0.30、所要23分
- チェックポイントは`rikky1201/resnet34-finetune-runpod-ckpt`（Kaggle dataset）に永続化済み

## 7. 次のステップ
- [ ] （ユーザー判断待ち）コストが極めて低いため、さらにepochを重ねて収束を見るか判断
- [ ] baseline（CV 0.7375, LB 0.688）に到達/超過したらLB提出可否を判断
- [ ] 再開する場合: `resnet34-finetune-runpod-ckpt`データセットのlast.ckptを起点に、同じ`runpod_train/`一式でpodを再起動するだけでよい（新しい前処理キャッシュ生成は不要、既に完成済み）
