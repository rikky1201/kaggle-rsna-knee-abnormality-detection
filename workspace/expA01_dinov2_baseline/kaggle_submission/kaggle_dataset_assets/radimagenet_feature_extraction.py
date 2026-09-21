"""
RadImageNet ResNet50（凍結）による特徴抽出。`feature_extraction.py`（DINOv2版）と対になる
バックボーン比較用の実装。インターフェースは `Dinov2FeatureExtractor.extract()` と揃えてあり、
`pooling.py`（slice→series→study）をそのまま流用できる。1ファイルに自己完結させているのは、
Kaggle Dataset化時にフラットにコピーして `sys.path.append` するデプロイ方式（既存の
feature_extraction.py/pooling.py/preprocessing.pyと同じ）と相性を合わせるため（相対import不可）。

アーキテクチャ出典: https://github.com/Warvito/radimagenet-models
（official RadImageNet の Keras ResNet50 実装 https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py
をPyTorchに移植した非公式リポジトリ）。state_dict は同リポジトリの `radimagenet_resnet50()` が
Google Driveから取得するものと同一ファイル `RadImageNet-ResNet50_notop.pth`。
torchvisionのresnet50とは重み初期化・bias付きconv・BatchNorm epsが異なるため、
ロード済みweightに strict=True で一致する独自アーキテクチャをそのまま再現している。

設計判断（公式リポジトリに明記が無く、`acl/acl_train.py`のACL検出リファレンス実装から逆算した点）:
- 入力サイズ: 224x224（DINOv2版と条件を揃えるため。ACLスクリプトのデフォルトは256だが
  ResNet50はglobal poolingのため入力サイズに強く依存しない。本比較ではバックボーン差のみを
  見たいため、前処理サイズは統一する）
- 前処理: 公式ACLスクリプトの `ImageDataGenerator(rescale=1./255,
  preprocessing_function=preprocess_input)`（`preprocess_input`はKeras
  `imagenet_utils.preprocess_input`のデフォルトmode='caffe'）という組み合わせをそのまま再現する。
  つまり [0,1]にスケーリング済みの入力に対して、caffe modeの平均減算
  （RGB→BGR + ImageNet平均[103.939,116.779,123.68]を減算、スケーリングなし）をそのまま適用する
  （0-255への逆スケールはしない。公式スクリプトの実際の挙動に合わせた意図的な選択）
- 特徴ベクトル: layer4出力(B,2048,7,7)をglobal average poolingして2048次元にする
  （Kerasの`ResNet50(pooling='avg')`と同等。DINOv2のCLSトークンに相当する「1枚の画像→1本のベクトル」表現）
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RADIMAGENET_EMBED_DIM = 2048
INPUT_SIZE = 224
_CAFFE_BGR_MEAN = (103.939, 116.779, 123.68)  # Keras imagenet_utils.preprocess_input(mode='caffe')


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class _Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        width = int(planes)
        self.conv1 = _conv1x1(inplanes, width, stride=stride)
        self.bn1 = nn.BatchNorm2d(width, eps=1.001e-5, momentum=0.99)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(width, eps=1.001e-5, momentum=0.99)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=1.001e-5, momentum=0.99)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _RadImageNetResNet50(nn.Module):
    """`include_top=False` 相当（分類ヘッド無し）。forwardの出力は (B, 2048, 7, 7) の特徴マップ
    （global average poolingは呼び出し側で行う。Keras ResNet50(pooling='avg')の`avg`は含まない）。
    """

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1.001e-5, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * _Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * _Bottleneck.expansion, eps=1.001e-5, momentum=0.99),
            )
        layers = [_Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def _caffe_preprocess(images_01: Tensor) -> Tensor:
    mean = torch.tensor(_CAFFE_BGR_MEAN, dtype=images_01.dtype, device=images_01.device).view(1, 3, 1, 1)
    return images_01 - mean


class RadImageNetFeatureExtractor:
    """凍結RadImageNet ResNet50で画像バッチ → global average pool特徴ベクトルを抽出するラッパー。"""

    def __init__(self, weights_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _RadImageNetResNet50()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = RADIMAGENET_EMBED_DIM

    @torch.no_grad()
    def extract(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """images: (N, H, W) の float32 配列（グレースケール, 値域 [0, 1] 想定）。

        Returns:
            (N, embed_dim) の特徴ベクトル（global average pool後）
        """
        if images.ndim != 3:
            raise ValueError(f"images は (N, H, W) の3次元配列である必要があります: shape={images.shape}")

        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            t = torch.from_numpy(batch).float().unsqueeze(1).to(self.device)  # (B,1,H,W)
            t = F.interpolate(t, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
            t = t.repeat(1, 3, 1, 1)  # グレースケール→3チャンネル複製
            t = _caffe_preprocess(t)
            feat_map = self.model(t)  # (B, 2048, 7, 7)
            pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # (B, 2048)
            features.append(pooled.cpu().numpy())

        return np.concatenate(features, axis=0).astype(np.float32)
