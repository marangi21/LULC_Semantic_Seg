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
from custom_model_factories import DINOv3ModelFactory
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning) # ignoro i warning di rasterio sulle immagini non georeferenziate
logging.getLogger("tensorboardX").setLevel(logging.WARNING) # imposto il livello di logging di tensorboard per non mostrare nulla al di sotto di un warning (tipo i messaggi INFO)

ENCODER_LR = 1e-5
DECODER_LR = HEAD_LR = 1e-3 # using decoder lr for neck too
WEIGHT_DECAY = 1e-3
IN_CHANNELS=4
MERGE_CLASSES = True

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
        in_channels=IN_CHANNELS,
        batch_size=2,
        num_workers=4,
        merge_classes=MERGE_CLASSES
    )
    num_classes = len(datamodule.class_names)

    model_args = {
        "num_classes": num_classes,
        "backbone_kwargs": {
            "in_channels": IN_CHANNELS,
            "skip_feature_block_index": 9 # indice del blocco transformer da cui estrarre la feature map per la skip connection
        },
        "decoder_kwargs": {
            "aspp_dropout": 0.25 # dropout del modulo ASPP del decoder DeeplabV3+
        },
        "head_kwargs": {
            "dropout": 0.5
        } 
    }

    # parametri custom per l'ottimizzatore con lr differenziati
    custom_opt_params = {
        "encoder_lr": ENCODER_LR,
        "decoder_lr": DECODER_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY
        }

    # Inizializzo il task di segmentazione
    task = DiffLRSemanticSegmentationTask(
        model_args=model_args,
        model_factory="DINOv3ModelFactory",
        loss='ce',
        lr=1e-5,
        optimizer='AdamW',
        optimizer_hparams=custom_opt_params,
        scheduler='ReduceLROnPlateau',
        scheduler_hparams={
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-8
        },
        plot_on_val=5,
        class_names=datamodule.class_names,
        ignore_index=None,
        freeze_backbone = False,                     # congelamento del backbone
        freeze_decoder = False,                     # non congelo il decoder
        freeze_head = False,                        # non congelo la testa
    )
    encoder_name=task.model.encoder.__class__.__name__
    decoder_name=task.model.decoder.__class__.__name__
    is_bb_frozen=next(task.model.encoder.parameters()).requires_grad==False
    logger = pl_loggers.TensorBoardLogger(
        save_dir=REPO_ROOT / "lightning_logs",
        name=f" \
        {encoder_name} \
        _{decoder_name}\
        {"_frozenBB" if is_bb_frozen else ""}\
        {"_RGB" if IN_CHANNELS==3 else ""}\
        {"_mergedClasses" if MERGE_CLASSES else ""}"
    )

    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=500,
        logger=logger,
        #overfit_batches=1,
        log_every_n_steps=1,
        precision='16-mixed',
        accumulate_grad_batches=8,
        callbacks=[
            EarlyStopping(
                monitor='val/loss',
                mode='min',
                patience=10,
                verbose=True
            )
        ]
    )

    trainer.fit(task, datamodule)

if __name__ == "__main__":
    main()