"""
Seguendo la documentazione per aggiungere modelli custom a Terratorch:
https://ibm.github.io/terratorch/models/#adding-a-new-model
"""
from terratorch.models.model import ModelFactory, Model
from terratorch.registry import MODEL_FACTORY_REGISTRY
from custom_model_wrappers import DINOv3ModelWrapper

@MODEL_FACTORY_REGISTRY.register
class DINOv3ModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        num_classes: int,
        backbone_kwargs: dict,
        decoder_kwargs: dict,
        head_kwargs: dict,
        **kwargs,
    ) -> Model: 
        if task != "segmentation":
            raise ValueError(f"DINOv3ModelFactory supporta solo 'segmentation', non '{task}'.")
        #---------------------------------------------------------------------------------------
        #-------------------------- DINOv3 + Deeplabv3+ ----------------------------------------
        #---------------------------------------------------------------------------------------
        if decoder_kwargs.get("name")=="deeplabv3plus":
            from custom_models import DINOv3EncoderDeeplabV3PlusDecoder
            dinov3_model = DINOv3EncoderDeeplabV3PlusDecoder(
                num_classes=num_classes,
                in_channels=backbone_kwargs.get("in_channels", 4),
                backbone_kwargs=backbone_kwargs,
                decoder_kwargs=decoder_kwargs,
                head_kwargs=head_kwargs
            )
        #---------------------------------------------------------------------------------------
        #-------------------------- DINOv3 + Mask2Former ---------------------------------------
        #---------------------------------------------------------------------------------------
        elif decoder_kwargs.get("name")=="mask2former":
            from custom_models import DINOv3EncoderMask2FormerDecoder
            dinov3_model = DINOv3EncoderMask2FormerDecoder(
                num_classes=num_classes,
                in_channels=backbone_kwargs.get("in_channels", 4),
                backbone_kwargs=backbone_kwargs,
                decoder_kwargs=decoder_kwargs,
                head_kwargs=head_kwargs
            )
        else:
            raise ValueError(f"Decoder '{decoder_kwargs.get("name")}' non supportato in DINOv3ModelFactory.")
        
        return DINOv3ModelWrapper(dinov3_model)