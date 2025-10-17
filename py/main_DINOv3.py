from transformers import pipeline
from transformers import AutoImageProcessor, AutoModel
from datamodule import WUSUSegmentationDataModule
from custom_tasks import DiffLRSemanticSegmentationTask
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping
import warnings
import logging
from rasterio.errors import NotGeoreferencedWarning
from pathlib import Path
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning) # ignoro i warning di rasterio sulle immagini non georeferenziate
logging.getLogger("tensorboardX").setLevel(logging.WARNING) # imposto il livello di logging di tensorboard per non mostrare nulla al di sotto di un warning (tipo i messaggi INFO)

pretrained_model_name = "facebook/dinov3-vit7b16-pretrain-sat493m"
#processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name,
    device_map="auto" # usa tutte le GPU disponibili
)

print(pretrained_model_name)