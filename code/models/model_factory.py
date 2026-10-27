"""
================================================================================
MODEL FACTORY
================================================================================
DESCRIPTION:
    Builds a model from `config['model']`, dispatching on `model.type`:
      - 'unet'               -> unet.UNetHeatmapModel (from-scratch baseline)
      - 'pretrained_encoder' -> pretrained_encoder_unet.PretrainedEncoderUNetHeatmapModel
                                (any timm features_only backbone - covers both
                                the ImageNet-pretrained-CNN and pretrained-HRNet arms)
      - 'dinov2'             -> dinov2_head.DINOv2HeatmapModel (frozen backbone)

    All three share the same (image_tensor) -> (N, num_landmarks, H, W)
    interface (H, W depend on the model - see each config's heatmap_size),
    so code/training/train_backbone_variant.py doesn't need to know which
    one it's training.

    Each arm's import is deferred into its own branch below rather than done
    at module level: 'pretrained_encoder' needs timm and 'dinov2' needs
    transformers, and neither should be required just to build a plain
    'unet' model - e.g. a notebook/environment with only torch installed can
    still run the from-scratch baseline (see notebooks/inference_visualize.ipynb).
================================================================================
"""


def build_model(model_config, num_landmarks):
    model_type = model_config.get('type', 'unet')

    if model_type == 'unet':
        from unet import UNetHeatmapModel
        return UNetHeatmapModel(
            num_landmarks=num_landmarks,
            base_channels=model_config.get('base_channels', 32),
        )

    if model_type == 'pretrained_encoder':
        from pretrained_encoder_unet import PretrainedEncoderUNetHeatmapModel
        return PretrainedEncoderUNetHeatmapModel(
            num_landmarks=num_landmarks,
            encoder_name=model_config.get('encoder_name', 'resnet34'),
            pretrained=model_config.get('pretrained', True),
            freeze_encoder=model_config.get('freeze_encoder', False),
        )

    if model_type == 'dinov2':
        from dinov2_head import DINOv2HeatmapModel
        return DINOv2HeatmapModel(
            num_landmarks=num_landmarks,
            model_name=model_config.get('model_name', 'facebook/dinov2-small'),
            decoder_channels=model_config.get('decoder_channels', 256),
            # freeze_backbone defaults True = the Run 6 frozen arm; the Run 9
            # fine-tuned arm sets it False and pairs it with layer-wise LR decay
            # (training.optimizer.layerwise_decay - see optim_factory.py).
            freeze_backbone=model_config.get('freeze_backbone', True),
            feature_layers=model_config.get('feature_layers'),
            decoder_extra_blocks=model_config.get('decoder_extra_blocks', 0),
        )

    raise ValueError(f"Unknown model.type: '{model_type}' (expected 'unet', 'pretrained_encoder', or 'dinov2')")
