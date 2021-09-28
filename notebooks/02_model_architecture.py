"""
================================================================================
SOTA PNEUMONIA DETECTION SYSTEM - MODULE 2: MODEL ARCHITECTURE
================================================================================
This module implements the detection model architecture:
- EfficientNetV2 backbone (from timm)
- Feature Pyramid Network (FPN) for multi-scale features
- Faster R-CNN detection head

Why EfficientNetV2 over ResNet?
- Better accuracy-efficiency tradeoff
- Progressive learning and adaptive regularization
- Fused MBConv blocks for faster training
- SOTA ImageNet performance with fewer parameters
================================================================================
"""

# ==============================================================================
# CELL 1: IMPORTS
# ==============================================================================

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Torchvision detection components
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


# ==============================================================================
# CELL 2: EFFICIENTNETV2 BACKBONE WRAPPER
# ==============================================================================

class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2 backbone wrapper for object detection.

    Uses timm's `features_only=True` mode to extract intermediate feature maps
    at multiple scales, which are essential for FPN.

    EfficientNetV2-S Architecture Output Channels:
    - Stage 0: 24 channels  (1/2 resolution)
    - Stage 1: 48 channels  (1/4 resolution)
    - Stage 2: 64 channels  (1/8 resolution)
    - Stage 3: 128 channels (1/16 resolution)
    - Stage 4: 160 channels (1/32 resolution)
    - Stage 5: 256 channels (1/32 resolution)

    We use stages 2-5 for FPN (1/8 to 1/32 resolution) as is standard practice.
    Earlier stages have too high resolution and would be memory-intensive.
    """

    def __init__(
        self,
        model_name: str = 'tf_efficientnetv2_s',
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (2, 3, 4, 5),
        freeze_bn: bool = False
    ):
        """
        Initialize EfficientNetV2 backbone.

        Args:
            model_name: timm model name (tf_efficientnetv2_s, tf_efficientnetv2_m, etc.)
            pretrained: Use ImageNet pretrained weights
            out_indices: Which feature stages to output (0-indexed)
            freeze_bn: Freeze BatchNorm layers (useful for small batch sizes)
        """
        super().__init__()

        # Create timm model with feature extraction mode
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices
        )

        # Get output channel dimensions for each stage
        # This is critical for connecting to FPN
        self.out_channels_list = self.body.feature_info.channels()
        self.out_indices = out_indices

        print(f"EfficientNetV2 Backbone initialized:")
        print(f"  Model: {model_name}")
        print(f"  Output stages: {out_indices}")
        print(f"  Output channels: {self.out_channels_list}")

        # Optionally freeze BatchNorm
        # Recommended for batch_size < 4 to prevent BN stats corruption
        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        """Freeze all BatchNorm layers."""
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            OrderedDict mapping feature names to tensors
            Keys are '0', '1', '2', '3' corresponding to out_indices
        """
        # Get features from all specified stages
        features = self.body(x)

        # Convert to OrderedDict with string keys (required by FPN)
        out = OrderedDict()
        for idx, feat in enumerate(features):
            out[str(idx)] = feat

        return out

    @property
    def out_channels(self) -> int:
        """Return the number of output channels after FPN (unified)."""
        # FPN unifies all channels to the same dimension
        # We'll set this in the full model
        return 256


# ==============================================================================
# CELL 3: CUSTOM FPN WITH EFFICIENTNETV2
# ==============================================================================

class EfficientNetV2WithFPN(nn.Module):
    """
    EfficientNetV2 backbone combined with Feature Pyramid Network.

    WHY FPN?
    ========
    1. Multi-scale detection: Pneumonia opacities vary greatly in size
    2. Combines low-level (edges, textures) and high-level (semantic) features
    3. Standard in all modern object detectors (Faster R-CNN, RetinaNet, YOLO)

    FPN Architecture:
    - Takes multi-scale features from backbone (C2, C3, C4, C5)
    - Creates pyramid features (P2, P3, P4, P5) with same channel dimension
    - Adds top-down pathway with lateral connections
    - Optionally adds P6 via max pooling for very large objects
    """

    def __init__(
        self,
        backbone_name: str = 'tf_efficientnetv2_s',
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        freeze_bn: bool = False,
        extra_blocks: bool = True
    ):
        """
        Initialize backbone + FPN.

        Args:
            backbone_name: timm model name
            pretrained: Use pretrained weights
            fpn_out_channels: Number of channels in FPN output (256 is standard)
            freeze_bn: Freeze BatchNorm layers
            extra_blocks: Add extra max-pool level (P6)
        """
        super().__init__()

        # Create backbone
        self.backbone = EfficientNetV2Backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            out_indices=(2, 3, 4, 5),  # Use stages with 1/8 to 1/32 resolution
            freeze_bn=freeze_bn
        )

        # Get input channels for FPN from backbone
        in_channels_list = self.backbone.out_channels_list

        # Create FPN
        # extra_blocks adds P6 level for detecting large objects
        extra = LastLevelMaxPool() if extra_blocks else None

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
            extra_blocks=extra
        )

        # Store output channels for detection head
        self.out_channels = fpn_out_channels

        print(f"\nFPN initialized:")
        print(f"  Input channels: {in_channels_list}")
        print(f"  Output channels: {fpn_out_channels}")
        print(f"  Extra P6 level: {extra_blocks}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through backbone and FPN.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Dictionary of FPN features {'0': P2, '1': P3, '2': P4, '3': P5, 'pool': P6}
        """
        # Extract multi-scale features from backbone
        backbone_features = self.backbone(x)

        # Pass through FPN
        fpn_features = self.fpn(backbone_features)

        return fpn_features


# ==============================================================================
# CELL 4: ANCHOR GENERATOR FOR CHEST X-RAYS
# ==============================================================================

def create_anchor_generator() -> AnchorGenerator:
    """
    Create anchor generator optimized for pneumonia detection.

    WHY CUSTOM ANCHORS?
    ===================
    Pneumonia opacities in chest X-rays have specific characteristics:
    1. They tend to be roughly square or slightly rectangular
    2. They vary significantly in size (small infiltrates to large consolidations)
    3. They can appear anywhere in the lung fields

    We customize:
    - Sizes: Cover range from small (32px) to very large (512px)
    - Aspect ratios: Mostly square-ish (0.5, 1.0, 2.0)

    Each FPN level handles different object sizes:
    - P2 (1/8): Small opacities (32px)
    - P3 (1/16): Medium-small (64px)
    - P4 (1/32): Medium (128px)
    - P5 (1/64): Medium-large (256px)
    - P6 (1/128): Large (512px)
    """
    # Anchor sizes for each FPN level
    # Each tuple corresponds to one FPN level
    anchor_sizes = (
        (32,),    # P2 - small opacities
        (64,),    # P3
        (128,),   # P4 - medium
        (256,),   # P5
        (512,),   # P6 - large consolidations
    )

    # Aspect ratios (same for all levels)
    # Pneumonia opacities are usually roughly square
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    print(f"\nAnchor Generator:")
    print(f"  Sizes per level: {anchor_sizes}")
    print(f"  Aspect ratios: {aspect_ratios[0]}")
    print(f"  Anchors per location: {len(aspect_ratios[0])}")

    return anchor_generator


# ==============================================================================
# CELL 5: ROI POOLER CONFIGURATION
# ==============================================================================

def get_roi_pooler_config() -> Dict:
    """
    Configuration for RoI (Region of Interest) pooling.

    RoI Align is used instead of RoI Pool for better accuracy.
    It uses bilinear interpolation to avoid quantization artifacts.
    """
    return {
        'featmap_names': ['0', '1', '2', '3'],  # FPN levels to use
        'output_size': 7,  # Output spatial size (7x7 is standard)
        'sampling_ratio': 2  # Sampling points for RoI Align
    }


# ==============================================================================
# CELL 6: MAIN PNEUMONIA DETECTOR CLASS
# ==============================================================================

class PneumoniaDetector(nn.Module):
    """
    Complete Pneumonia Detection Model.

    Architecture:
    1. EfficientNetV2-S backbone (timm)
    2. Feature Pyramid Network
    3. Region Proposal Network (RPN)
    4. RoI Align + Detection Head (Faster R-CNN)

    This is a two-stage detector:
    - Stage 1 (RPN): Proposes candidate regions
    - Stage 2 (Detection): Classifies and refines boxes

    Configuration Notes:
    - num_classes=2: Background (0) + Pneumonia (1)
    - Image mean/std: ImageNet normalization (handled in dataset)
    - NMS threshold: 0.5 for RPN, 0.3 for detection (will use WBF post-hoc)
    """

    def __init__(
        self,
        backbone_name: str = 'tf_efficientnetv2_s',
        num_classes: int = 2,  # Background + Pneumonia
        pretrained_backbone: bool = True,
        # RPN settings
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        # Detection settings
        box_score_thresh: float = 0.05,  # Low threshold - filter with WBF later
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        # Training settings
        freeze_bn: bool = True,  # Freeze BN for small batch sizes
        trainable_backbone_layers: int = 5,  # Fine-tune all layers
    ):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone_name

        print("=" * 60)
        print("INITIALIZING PNEUMONIA DETECTOR")
        print("=" * 60)

        # 1. Create backbone with FPN
        backbone_fpn = EfficientNetV2WithFPN(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            fpn_out_channels=256,
            freeze_bn=freeze_bn,
            extra_blocks=True  # Add P6 level
        )

        # 2. Create anchor generator
        anchor_generator = create_anchor_generator()

        # 3. Create RPN head
        # RPN predicts objectness and box deltas for each anchor
        rpn_head = RPNHead(
            in_channels=backbone_fpn.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0]
        )

        # 4. Create complete Faster R-CNN model
        self.model = FasterRCNN(
            backbone=backbone_fpn,
            num_classes=num_classes,
            # RPN parameters
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            rpn_nms_thresh=rpn_nms_thresh,
            rpn_fg_iou_thresh=rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=rpn_bg_iou_thresh,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            # Box parameters
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
            box_fg_iou_thresh=box_fg_iou_thresh,
            box_bg_iou_thresh=box_bg_iou_thresh,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,
        )

        print(f"\nFaster R-CNN initialized:")
        print(f"  Num classes: {num_classes}")
        print(f"  RPN NMS thresh: {rpn_nms_thresh}")
        print(f"  Box score thresh: {box_score_thresh}")
        print(f"  Box NMS thresh: {box_nms_thresh}")
        print("=" * 60)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        In training mode:
            Returns loss dictionary {'loss_classifier', 'loss_box_reg',
                                     'loss_objectness', 'loss_rpn_box_reg'}

        In inference mode:
            Returns list of detection dictionaries per image
            [{'boxes': Tensor, 'labels': Tensor, 'scores': Tensor}, ...]

        Args:
            images: List of tensors, each (3, H, W)
            targets: List of target dicts (only needed for training)

        Returns:
            Losses (training) or detections (inference)
        """
        return self.model(images, targets)

    def get_trainable_parameters(
        self,
        lr_backbone: float = 1e-5,
        lr_fpn: float = 1e-4,
        lr_head: float = 1e-3
    ) -> List[Dict]:
        """
        Get parameter groups with different learning rates.

        WHY DIFFERENT LEARNING RATES?
        =============================
        - Backbone: Pretrained, needs small LR to preserve learned features
        - FPN: New layers, can use moderate LR
        - Detection head: New layers, highest LR for fast convergence

        This is called "discriminative learning rates" and is critical
        for fine-tuning pretrained models.
        """
        # Separate parameters by component
        backbone_params = []
        fpn_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'backbone.backbone' in name:
                backbone_params.append(param)
            elif 'backbone.fpn' in name:
                fpn_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'},
            {'params': fpn_params, 'lr': lr_fpn, 'name': 'fpn'},
            {'params': head_params, 'lr': lr_head, 'name': 'head'},
        ]

        print(f"\nParameter groups:")
        print(f"  Backbone: {len(backbone_params)} params, lr={lr_backbone}")
        print(f"  FPN: {len(fpn_params)} params, lr={lr_fpn}")
        print(f"  Head: {len(head_params)} params, lr={lr_head}")

        return param_groups

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze the backbone for transfer learning."""
        for param in self.model.backbone.backbone.parameters():
            param.requires_grad = not freeze
        print(f"Backbone {'frozen' if freeze else 'unfrozen'}")

    def freeze_bn(self):
        """Set all BatchNorm layers to eval mode."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()


# ==============================================================================
# CELL 7: MODEL VARIANTS
# ==============================================================================

def create_efficientnetv2_s_fasterrcnn(
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> PneumoniaDetector:
    """Create Faster R-CNN with EfficientNetV2-S backbone."""
    return PneumoniaDetector(
        backbone_name='tf_efficientnetv2_s',
        num_classes=num_classes,
        pretrained_backbone=pretrained,
        **kwargs
    )


def create_efficientnetv2_m_fasterrcnn(
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> PneumoniaDetector:
    """Create Faster R-CNN with EfficientNetV2-M backbone (larger)."""
    return PneumoniaDetector(
        backbone_name='tf_efficientnetv2_m',
        num_classes=num_classes,
        pretrained_backbone=pretrained,
        **kwargs
    )


def create_efficientnetv2_b0_fasterrcnn(
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> PneumoniaDetector:
    """Create Faster R-CNN with EfficientNetV2-B0 backbone (smaller, faster)."""
    return PneumoniaDetector(
        backbone_name='tf_efficientnetv2_b0',
        num_classes=num_classes,
        pretrained_backbone=pretrained,
        **kwargs
    )


# ==============================================================================
# CELL 8: MODEL TESTING
# ==============================================================================

def test_model():
    """Test the model with dummy inputs."""
    print("\n" + "=" * 60)
    print("TESTING MODEL ARCHITECTURE")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create model
    model = create_efficientnetv2_s_fasterrcnn(
        num_classes=2,
        pretrained=True
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass (training mode)
    print("\n" + "-" * 40)
    print("Testing training forward pass...")
    model.train()

    # Create dummy batch
    batch_size = 2
    images = [torch.randn(3, 1024, 1024).to(device) for _ in range(batch_size)]
    targets = [
        {
            'boxes': torch.tensor([[100, 100, 300, 300], [400, 400, 600, 600]]).float().to(device),
            'labels': torch.tensor([1, 1]).long().to(device),
        }
        for _ in range(batch_size)
    ]

    # Forward pass
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        losses = model(images, targets)

    print(f"\nTraining losses:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.4f}")

    total_loss = sum(losses.values())
    print(f"  Total: {total_loss.item():.4f}")

    # Test forward pass (inference mode)
    print("\n" + "-" * 40)
    print("Testing inference forward pass...")
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            detections = model(images)

    print(f"\nInference results (per image):")
    for i, det in enumerate(detections):
        print(f"  Image {i}:")
        print(f"    Boxes: {det['boxes'].shape}")
        print(f"    Labels: {det['labels'].shape}")
        print(f"    Scores: {det['scores'].shape}")
        if len(det['scores']) > 0:
            print(f"    Top score: {det['scores'][0].item():.4f}")

    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETE")
    print("=" * 60)

    return model


def print_model_summary(model: PneumoniaDetector):
    """Print a summary of model components."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    print("\n1. BACKBONE (EfficientNetV2)")
    backbone = model.model.backbone.backbone.body
    for i, (name, module) in enumerate(backbone.named_children()):
        if i < 3:  # Just show first few
            print(f"   {name}: {type(module).__name__}")
    print("   ...")

    print("\n2. FPN (Feature Pyramid Network)")
    fpn = model.model.backbone.fpn
    print(f"   Inner blocks: {len(fpn.inner_blocks)}")
    print(f"   Layer blocks: {len(fpn.layer_blocks)}")

    print("\n3. RPN (Region Proposal Network)")
    rpn = model.model.rpn
    print(f"   Anchor generator sizes: {rpn.anchor_generator.sizes}")

    print("\n4. ROI HEADS")
    roi_heads = model.model.roi_heads
    print(f"   Box predictor: {type(roi_heads.box_predictor).__name__}")

    print("=" * 60)


# ==============================================================================
# CELL 9: MAIN
# ==============================================================================

if __name__ == "__main__":
    model = test_model()
    print_model_summary(model)

    # Get parameter groups for optimizer
    param_groups = model.get_trainable_parameters(
        lr_backbone=1e-5,
        lr_fpn=1e-4,
        lr_head=1e-3
    )
