from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torchvision.models import resnet50


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DECODER_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "ml_decoder"
)

ML_DECODER_FILE = (
    ML_DECODER_ROOT
    / "ml_decoder.py"
)

if not ML_DECODER_FILE.is_file():
    raise FileNotFoundError(
        "找不到官方 ML-Decoder 源码：\n"
        f"{ML_DECODER_FILE}"
    )

# 这里只调用官方实现，ML-Decoder 不属于本项目创新。
if str(ML_DECODER_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_DECODER_ROOT))

from ml_decoder import MLDecoder  # noqa: E402


def _build_torchvision_resnet50(
    pretrained: bool,
) -> nn.Module:
    """
    同时兼容新旧 torchvision 的 ResNet-50 权重接口。
    """

    if not pretrained:
        try:
            return resnet50(weights=None)
        except TypeError:
            return resnet50(pretrained=False)

    try:
        from torchvision.models import ResNet50_Weights

        return resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )
    except (ImportError, AttributeError, TypeError):
        return resnet50(pretrained=True)


class CIDERBase(nn.Module):
    """
    纯净多标签 baseline：

        Image
          -> ImageNet ResNet-50 spatial feature map
          -> official ML-Decoder
          -> multi-label logits

    该模型不包含任何：

    - 长尾重加权；
    - 重采样；
    - 伪相关建模；
    - 事实或反事实干预；
    - logit 校准创新模块。
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        num_groups: Optional[int] = None,
        decoder_embedding: int = 768,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                f"num_classes 必须为正数，收到 {num_classes}"
            )

        if decoder_embedding <= 0:
            raise ValueError(
                "decoder_embedding 必须为正数，"
                f"收到 {decoder_embedding}"
            )

        base_model = _build_torchvision_resnet50(
            pretrained=pretrained
        )

        # 保留 ResNet 的空间特征，不使用 avgpool 和 fc。
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.num_classes = int(num_classes)
        self.num_features = 2048
        self.decoder_embedding = int(decoder_embedding)

        # 对 VOC 20 类和 COCO 80 类，都使用一类一个 query。
        # 这是基础分类头配置，不属于核心创新。
        resolved_groups = (
            self.num_classes
            if num_groups is None or int(num_groups) <= 0
            else int(num_groups)
        )

        if resolved_groups > self.num_classes:
            resolved_groups = self.num_classes

        self.num_groups = resolved_groups

        self.ml_decoder = MLDecoder(
            num_classes=self.num_classes,
            num_of_groups=self.num_groups,
            decoder_embedding=self.decoder_embedding,
            initial_num_features=self.num_features,
            zsl=0,
        )

    def forward_features(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        返回 ResNet-50 最后一个空间特征图。

        输入：
            [B, 3, H, W]

        输出：
            [B, 2048, H/32, W/32]
        """

        if images.ndim != 4:
            raise ValueError(
                "输入图像应为四维张量 [B,C,H,W]，"
                f"实际 shape={tuple(images.shape)}"
            )

        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward_head(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """
        将空间特征输入官方 ML-Decoder。
        """

        logits = self.ml_decoder(feature_map)

        expected_shape = (
            feature_map.shape[0],
            self.num_classes,
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "ML-Decoder 输出形状错误："
                f"expected={expected_shape}, "
                f"actual={tuple(logits.shape)}"
            )

        return logits

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        feature_map = self.forward_features(images)
        logits = self.forward_head(feature_map)
        return logits

    def forward_outputs(self, images: Tensor) -> dict[str, Tensor]:
        """Return the three tensors shared by LCI, CED, and later modules."""
        feature_map = self.forward_features(images)
        logits = self.forward_head(feature_map)
        class_attention = getattr(
            self.ml_decoder.decoder.layers[-1], "last_attn_weights", None
        )
        if class_attention is None:
            raise RuntimeError("ML-Decoder did not expose class attention")
        height, width = feature_map.shape[-2:]
        expected = (images.shape[0], self.num_classes, height * width)
        if tuple(class_attention.shape) != expected:
            raise RuntimeError(
                "class attention has an invalid shape: "
                f"expected={expected}, actual={tuple(class_attention.shape)}"
            )
        return {
            "feature_map": feature_map,
            "logits": logits,
            "class_attention": class_attention.reshape(
                images.shape[0], self.num_classes, height, width
            ),
        }

    def parameter_summary(self) -> dict:
        total = sum(
            parameter.numel()
            for parameter in self.parameters()
        )

        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        backbone = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("ml_decoder.")
        )

        decoder = sum(
            parameter.numel()
            for parameter in self.ml_decoder.parameters()
        )

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "backbone_parameters": backbone,
            "ml_decoder_parameters": decoder,
        }


def build_cider_base(
    num_classes: int,
    pretrained: bool = True,
    num_groups: Optional[int] = None,
    decoder_embedding: int = 768,
) -> CIDERBase:
    return CIDERBase(
        num_classes=num_classes,
        pretrained=pretrained,
        num_groups=num_groups,
        decoder_embedding=decoder_embedding,
    )
