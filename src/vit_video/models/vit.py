import torch
import torch.nn as nn
import timm

class MobileViTModel(nn.Module):
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

        torchvision_vit_models = {
            "vit_b_16": ("vit_b_16", 768),
            "vit_b_32": ("vit_b_32", 768),
            "vit_l_16": ("vit_l_16", 1024),
            "vit_l_32": ("vit_l_32", 1024),
            "vit_h_14": ("vit_h_14", 1280),
        }
        
        # Map our names to timm names when torchvision ViT is unavailable
        timm_vit_models = {
            "vit_b_16": "vit_base_patch16_224",
            "vit_b_32": "vit_base_patch32_224",
            "vit_l_16": "vit_large_patch16_224",
            "vit_l_32": "vit_large_patch32_224",
            "vit_h_14": "vit_huge_patch14_224.in21k_ft_in1k",
        }

        if model_name in torchvision_vit_models:
            backbone = None
            try:
                from torchvision.models.vision_transformer import (
                    vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14,
                    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights,
                    ViT_L_32_Weights, ViT_H_14_Weights,
                )
                weight_classes = {
                    "vit_b_16": ViT_B_16_Weights,
                    "vit_b_32": ViT_B_32_Weights,
                    "vit_l_16": ViT_L_16_Weights,
                    "vit_l_32": ViT_L_32_Weights,
                    "vit_h_14": ViT_H_14_Weights,
                }
                vit_constructors = {
                    "vit_b_16": vit_b_16,
                    "vit_b_32": vit_b_32,
                    "vit_l_16": vit_l_16,
                    "vit_l_32": vit_l_32,
                    "vit_h_14": vit_h_14,
                }
                constructor = vit_constructors[model_name]
                weights_cls = weight_classes[model_name]
                weights = weights_cls.DEFAULT if pretrained else None
                backbone = constructor(weights=weights)
            except (ImportError, ModuleNotFoundError):
                pass

            if backbone is not None:
                feat_dim = torchvision_vit_models[model_name][1]
                backbone.heads.head = nn.Identity()
                self._use_torchvision = True
            else:
                if timm is None:
                    raise RuntimeError("timm is required for ViT backbones.")
                timm_name = timm_vit_models.get(model_name, "vit_base_patch16_224")
                backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
                feat_dim = torchvision_vit_models[model_name][1]
                self._use_torchvision = False
        else:
            if timm is None:
                raise RuntimeError("timm is required for MobileViTModel.")
            try:
                backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
            except Exception:
                backbone = timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=0, global_pool="avg")

            feat_dim = getattr(backbone, "num_features", None)
            if feat_dim is None:
                feat_dim = 256

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.temporal_conv = None
        if temporal_pool == "conv1d":
            self.temporal_conv = nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.classifier = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

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
            for v in feats.values():
                feats = v
                break
        
        if feats.dim() == 3:
            feats = feats[:, 0]
        elif feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])

        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self._extract_frame_features(x)
        feat_dim = feats.shape[1]
        feats = feats.view(b, t, feat_dim)

        if self.temporal_pool == "avg":
            pooled = feats.mean(dim=1)
        elif self.temporal_pool == "max":
            pooled, _ = feats.max(dim=1)
        elif self.temporal_pool == "conv1d" and self.temporal_conv is not None:
            temporal = feats.transpose(1, 2)
            temporal = self.temporal_conv(temporal)
            pooled = temporal.mean(dim=2)
        else:
            pooled = feats.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
