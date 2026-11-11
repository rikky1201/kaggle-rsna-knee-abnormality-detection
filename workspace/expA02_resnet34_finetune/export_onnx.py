"""
expA02 (ResNet34 fine-tuning, v005) のモデル構造を図示用に ONNX へ書き出す。

学習コード (runpod_train/train_runpod.py の forward_study) は series を Python の dict で
7スロットに振り分けるため、そのままでは ONNX にできない。ここでは図示用に
「1スロット = 1 series、スライス数 S 固定」の静的な形に組み直している。
同じスロットに複数 series がある場合の平均は省略している（図示用の簡略化）。

入力:
  slices          (7, S, 1, 224, 224)  0-1 に正規化済みのスライス画像
  slot_mask       (7,)                 そのスロットに series があれば 1、なければ 0
  fat_suppression (7,)                 スロットごとの Fat_Suppression フラグ
出力:
  logits          (12,)

使い方: python export_onnx.py  →  onnx/expA02_resnet34_v005.onnx を Netron 等で開く
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models import resnet34

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
CKPT_PATH = EXP_DIR / "final_ckpt" / "best.ckpt"
OUT_PATH = EXP_DIR / "onnx" / "expA02_resnet34_v005.onnx"

N_SLOTS = 7
N_SLICES = 16  # 学習時の slice cap (FINETUNE_MAX_SLICES_PER_SERIES) と同じ
INPUT_SIZE = 224
EMBED_DIM = 512
NUM_CLASSES = 12
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class StudyModelForDiagram(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet34(weights=None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(N_SLOTS * (2 * EMBED_DIM + 1), NUM_CLASSES)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, slices: Tensor, slot_mask: Tensor, fat_suppression: Tensor) -> Tensor:
        n_slots, n_slices = slices.shape[0], slices.shape[1]
        x = slices.reshape(n_slots * n_slices, 1, INPUT_SIZE, INPUT_SIZE)
        x = x.repeat(1, 3, 1, 1)                       # 1ch → 3ch
        x = (x - self.mean) / self.std                 # ImageNet 正規化
        feats = self.backbone(x)                       # (7*S, 512)
        feats = feats.reshape(n_slots, n_slices, EMBED_DIM)
        series_feat = torch.cat([feats.mean(dim=1), feats.amax(dim=1)], dim=-1)  # slice→series: mean+max (7, 1024)
        series_feat = series_feat * slot_mask.unsqueeze(-1)                      # 空スロットはゼロ
        slot_vec = torch.cat([series_feat, fat_suppression.unsqueeze(-1)], dim=-1)  # (7, 1025)
        study_feat = slot_vec.reshape(1, -1)                                     # series→study: 7スロット連結 (1, 7175)
        return self.head(study_feat).squeeze(0)                                  # (12,)


def main() -> None:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = StudyModelForDiagram()
    model.backbone.load_state_dict(ckpt["backbone_state_dict"])
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval()
    logger.info("ckpt 読み込み: epoch=%d, best_macro_auc=%.4f", ckpt["epoch"], ckpt["best_macro_auc"])

    dummy = (
        torch.rand(N_SLOTS, N_SLICES, 1, INPUT_SIZE, INPUT_SIZE),
        torch.ones(N_SLOTS),
        torch.zeros(N_SLOTS),
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(OUT_PATH),
        input_names=["slices", "slot_mask", "fat_suppression"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    logger.info("書き出し完了: %s (%.1f MB)", OUT_PATH, OUT_PATH.stat().st_size / 1e6)

    # 書き出したモデルと PyTorch の出力が一致するか確認
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(OUT_PATH))
    sess = ort.InferenceSession(str(OUT_PATH))
    onnx_out = sess.run(None, {k: v.numpy() for k, v in zip(["slices", "slot_mask", "fat_suppression"], dummy)})[0]
    with torch.no_grad():
        torch_out = model(*dummy).numpy()
    max_diff = float(abs(onnx_out - torch_out).max())
    logger.info("PyTorch と ONNX の最大差: %.2e", max_diff)
    assert max_diff < 1e-3, f"出力が一致しません: max_diff={max_diff}"


if __name__ == "__main__":
    main()
