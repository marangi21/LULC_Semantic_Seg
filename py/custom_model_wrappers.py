"""
Seguendo la documentazione per aggiungere modelli custom a Terratorch:
https://ibm.github.io/terratorch/models/#adding-a-new-model
"""
import torch.nn as nn
from terratorch.models.model import Model, ModelOutput
from custom_models import DINOv3BaseModel, DINOv3EncoderDeeplabV3PlusDecoder, DINOv3EncoderMask2FormerDecoder

class DINOv3ModelWrapper(Model, nn.Module):
    """Wrapper per i modelli DINOv3 in modo che siano compatibili con Terratorch.
    Espone gli attributi 'encoder', 'decoder' e 'head' per la gestione
    da parte di TerratorchTask (es. learning rate differenziali)"""
    def __init__(self, model: DINOv3BaseModel):
        super().__init__()
        
        # Espongo i componenti come attributi del wrapper, in modo tale che il TerratorchTask possa trovarli
        # necessario per l'implementazione con learning rate differenziali
        self.model = model
        self.encoder = self.model.backbone

        if isinstance(self.model, DINOv3EncoderDeeplabV3PlusDecoder):
            print("Configuring wrapper for DINOv3EncoderDeeplabV3PlusDecoder")
            self.decoder = self.model.deeplab_decoder
            self.head = self.model.segmentation_head
            # non c'è un neck separato, quindi non lo espongo
        elif isinstance(self.model, DINOv3EncoderMask2FormerDecoder):
            print("Configuring wrapper for DINOv3EncoderMask2FormerDecoder")
            self.decoder = self.model.pixel_decoder     # pixel decoder
            self.head = self.model.transformer_decoder  # transformer decoder
            # non c'è un neck separato, quindi non lo espongo
        else:
            raise TypeError(f"Model type non supportato nel DINOv3ModelWrapper: {type(model).__name__}")
    
    def forward(self, x):
        return ModelOutput(self.model(x))

    def freeze_encoder(self):
        """Congela i pesi del backbone DINOv3."""
        print("Freezing DINOv3 backbone weights (encoder)")
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        """Congela i pesi del decoder (es. FPN, PixelDecoder)."""
        print("Freezing decoder weights")
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_head(self):
        """Congela i pesi della testa di segmentazione."""
        print("Freezing head weights")
        for param in self.head.parameters():
            param.requires_grad = False

    