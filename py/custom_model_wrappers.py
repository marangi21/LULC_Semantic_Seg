"""
Seguendo la documentazione per aggiungere modelli custom a Terratorch:
https://ibm.github.io/terratorch/models/#adding-a-new-model
"""
import torch.nn as nn
from terratorch.models.model import Model, ModelOutput
from custom_models import DINOv3EncoderDeeplabV3PlusDecoder
# ToDo: implementare model wrapper per DINOv3 + DeeplabV3Plus

class DINOv3ModelWrapper(Model, nn.Module):
    def __init__(self, dinov3_model: DINOv3EncoderDeeplabV3PlusDecoder):
        super().__init__()
        
        # Espongo i componenti come attributi del wrapper, in modo tale che il TerratorchTask possa trovarli
        # necessario per l'implementazione con learning rate differenziali
        self.model = dinov3_model
        self.encoder = self.model.backbone
        self.decoder = self.model.deeplab_decoder
        self.head = self.model.segmentation_head
        # non c'è un neck separato, quindi non lo espongo
    
    def forward(self, x):
        return ModelOutput(self.model(x))

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    