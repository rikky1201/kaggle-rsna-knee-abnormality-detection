# My Memo

このファイルは自分のメモを残すためのものです。

## メモ


### RSNA Knee Discussion — Efficiency LB上位者からの知見

**情報源**：Efficiency LB上位者による、1ヶ月にわたる複数のDiscussionコメント（[Best single-model score](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)、[9h runtime question](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740474)、[Maybe worth checking the Efficiency LB?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736678)）。誇張の動機が薄く、本人が繰り返し自分の限界を認めている点で信頼度が高い情報源と判断。

## 得られた知見

**①report-labelerの改善効果は限定的**
複数の上位者が「独自ラベルと公開データセットのラベルで、精度に大きな差を感じない」と証言。ただし1人は「独自ラベルを公開データセットと組み合わせたらLBが+0.015上がった」とも報告しており、効果はゼロではないが小さい。プロンプトの細かい調整に大きな伸びしろを期待すべきではない。

**②スコアを大きく動かした要因はエンコーダの追加事前学習と長時間学習**
0.936到達者が要因として挙げたのはこの2点。凍結運用ではなく、エンコーダ自体をこのデータに適応させる工程（fine-tuning）が主要な伸びしろだった。

**③CV/LB相関は「変更を1つずつ」でしか保てない**
同じ0.936到達者が「エンコーダのアーキテクチャ変更なども含め、一度に複数の変更を積み重ねたら、CVとLBの相関が見えなくなった」と明言。1回の検証で1つだけ変える、という慎重な進め方の必要性を裏付ける一次証言。

**④アンサンブル・複雑化は精度を上げても信頼性を上げるとは限らない**
5foldアンサンブルで0.947まで上げた人、単一モデルで0.938・0.949を出した人、いずれも「private LBで保つか自信がない」と発言。過学習リスクは複雑さと共に上がる。

**⑤バックボーンは地味な選択肢で十分戦える**
上位者の多くがEfficientNet・ResNet（"smaller resnet"）のみに時間を投資しており、DINOv2/v3のような大規模自己教師ありモデルを必須とはしていない。

**⑥Efficiency（実行時間）の実態**
DICOM読み込みが提出時間の60〜70%を占める、という証言複数。5〜6分が実用的な着地点で、そこから更に削る価値は薄いとの声。学習時のI/Oボトルネックが推論時にも同様に効くことの裏付け。

**⑦gold58とLBの関係の実例**
「single fold, single modelでLB 0.949、gold58で0.930」という組が公開されている。gold58を信頼できる評価基準として使う設計自体は妥当という傍証。

## 自分のパイプラインへの示唆
- report-labelerへの深追いは区切りをつけ、③の教訓（1変更ずつ）を守りながら②（fine-tuning）に軸足を移す
- バックボーンをDINOv2以外に広げる必要は今のところ薄い（⑤より）
- Efficiencyの最適化は後回しでよい（⑥より、5〜6分が目安）

---

**タグ案**：`#ml-engineering` `#kaggle-rsna-knee`
**ファイル名案**：`kaggle-rsna-knee-discussion-insights.md`
<!-- ここに自由にメモを追記してください -->

