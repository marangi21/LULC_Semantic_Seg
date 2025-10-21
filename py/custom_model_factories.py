"""
Seguendo la documentazione per aggiungere modelli custom a Terratorch:
https://ibm.github.io/terratorch/models/#adding-a-new-model
"""
from terratorch.models.model import ModelFactory, Model
import torch.nn as nn
from custom_models import DINOv3EncoderDeeplabV3PlusDecoder
from terratorch.models.model import register_factory
from custom_model_wrappers import DINOv3ModelWrapper

@register_factory("dinov3_model_factory")
class DINOv3ModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        in_channels: int,
        num_classes: int,
        **kwargs,
    ) -> Model:
        if task != "segmentation":
            raise ValueError(f"DINOv3ModelFactory supporta solo 'segmentation', non '{task}'.")
        dinov3_model = DINOv3EncoderDeeplabV3PlusDecoder(
            num_classes=num_classes,
            in_channels=in_channels
        )
        return DINOv3ModelWrapper(dinov3_model)