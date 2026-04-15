import torch
import torch.nn as nn
import timm


class MobileViTModel(nn.Module):
    TORCHVISION_VITS = {
        "vit_b_16": 768, "vit_b_32": 768,
        "vit_l_16": 1024, "vit_l_32": 1024,
        "vit_h_14": 1280,
    }
    TIMM_VIT_FALLBACKS = {
        "vit_b_16": "vit_base_patch16_224",
        "vit_b_32": "vit_base_patch32_224",
        "vit_l_16": "vit_large_patch16_224",
        "vit_l_32": "vit_large_patch32_224",
        "vit_h_14": "vit_huge_patch14_224.in21k_ft_in1k",
    }

    def __init__(
        self,
        num_classes: int,
        model_name: str = "mobilevit_xxs",
        pretrained: bool = True,
        temporal_pool: str = "avg",
        dropout: float = 0.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.temporal_pool = temporal_pool
        self._use_torchvision = False

        backbone, feat_dim = self._build_backbone(model_name, pretrained)

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.temporal_conv = None
        self.temporal_lstm = None

        if temporal_pool == "conv1d":
            self.temporal_conv = nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1)
        elif temporal_pool == "lstm":
            lstm_hidden = feat_dim // 2
            self.temporal_lstm = nn.LSTM(
                input_size=feat_dim,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            # Bidirectional doubles the output size, so it maps back to feat_dim
        self.classifier = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _build_backbone(self, model_name: str, pretrained: bool):
        if model_name in self.TORCHVISION_VITS:
            backbone = self._try_torchvision_vit(model_name, pretrained)
            if backbone is not None:
                self._use_torchvision = True
                return backbone, self.TORCHVISION_VITS[model_name]
            # Fallback to timm
            timm_name = self.TIMM_VIT_FALLBACKS[model_name]
            backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
            return backbone, self.TORCHVISION_VITS[model_name]

        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = getattr(backbone, "num_features", 256)
        return backbone, feat_dim

    def _try_torchvision_vit(self, model_name: str, pretrained: bool):
        try:
            from torchvision.models.vision_transformer import (
                vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14,
                ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights,
                ViT_L_32_Weights, ViT_H_14_Weights,
            )
            constructors = {
                "vit_b_16": (vit_b_16, ViT_B_16_Weights),
                "vit_b_32": (vit_b_32, ViT_B_32_Weights),
                "vit_l_16": (vit_l_16, ViT_L_16_Weights),
                "vit_l_32": (vit_l_32, ViT_L_32_Weights),
                "vit_h_14": (vit_h_14, ViT_H_14_Weights),
            }
            ctor, weights_cls = constructors[model_name]
            weights = weights_cls.DEFAULT if pretrained else None
            backbone = ctor(weights=weights)
            backbone.heads.head = nn.Identity()
            return backbone
        except (ImportError, ModuleNotFoundError):
            return None

    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_torchvision:
            feats = self.backbone(x)
        elif hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)

        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if isinstance(feats, dict):
            feats = next(iter(feats.values()))

        if feats.dim() == 2:
            pass  # Already (batch, feat_dim) — no pooling needed
        elif feats.dim() == 3:
            feats = feats[:, 0]
        elif feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])
        else:
            raise ValueError(
                f"Unexpected feature tensor dimensions: {feats.dim()}D "
                f"(shape {feats.shape}). Expected 2D, 3D, or 4D."
            )

        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self._extract_frame_features(x)
        feat_dim = feats.shape[1]
        feats = feats.view(b, t, feat_dim)

        if self.temporal_pool == "lstm" and self.temporal_lstm is not None:
            lstm_out, _ = self.temporal_lstm(feats)  # (b, t, 2*hidden)
            pooled = lstm_out.mean(dim=1)
        elif self.temporal_pool == "avg":
            pooled = feats.mean(dim=1)
        elif self.temporal_pool == "max":
            pooled, _ = feats.max(dim=1)
        elif self.temporal_pool == "conv1d" and self.temporal_conv is not None:
            pooled = self.temporal_conv(feats.transpose(1, 2)).mean(dim=2)
        else:
            pooled = feats.mean(dim=1)

        pooled = self.dropout(pooled)
        return self.classifier(pooled)
