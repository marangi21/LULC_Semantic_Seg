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
import torch
from models import DINOv3EncoderDeeplabV3PlusDecoder
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning) # ignoro i warning di rasterio sulle immagini non georeferenziate
logging.getLogger("tensorboardX").setLevel(logging.WARNING) # imposto il livello di logging di tensorboard per non mostrare nulla al di sotto di un warning (tipo i messaggi INFO)

def main():
    seed_everything(42, workers=True) # per riproducibilità
    REPO_ROOT = Path(__file__).parent.parent
    DATA_ROOT = REPO_ROOT / "data" / "WUSU_preprocessed"
    CLASS_MAPPING_PATH = REPO_ROOT / "data" / "OpenWUSU512" / "class_mapping.json"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Definisco data module: gestisce tutta la logica di caricamento dei set di dati e istanziazione dei dataloader
    datamodule = WUSUSegmentationDataModule(
        data_root=DATA_ROOT,
        class_mapping_path=CLASS_MAPPING_PATH,
        batch_size=1,
        num_workers=4
    )
    num_classes = len(datamodule.class_names)

    model = DINOv3EncoderDeeplabV3PlusDecoder(
        num_classes=num_classes,
        in_channels=4
    )
    model.to(device)

    input = torch.randn([4, 4, 512, 512]).to(device)
    output = model(input)
    print("Output shape:", output.shape)  # dovrebbe essere [8, num_classes, 512, 512]
    
    print()




if __name__ == "__main__":
    main()