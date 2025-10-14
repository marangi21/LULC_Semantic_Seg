from datamodule import WUSUSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping
import warnings
import logging
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning) # ignoro i warning di rasterio sulle immagini non georeferenziate
logging.getLogger("tendorboardX").setLevel(logging.WARNING) # imposto il livello di logging di tensorboard per non mostrare nulla al di sotto di un warning (tipo i messaggi INFO)

def main():
    seed_everything(42, workers=True) # per riproducibilità
    DATA_ROOT = "/shared/marangi/projects/EVOCITY/building_extraction/data/WUSU_preprocessed"
    CLASS_MAPPING_PATH = "/shared/marangi/projects/EVOCITY/building_extraction/data/OpenWUSU512/class_mapping.json"
    
    # Definisco data module: gestisce tutta la logica di caricamento dei set di dati e istanziazione dei dataloader
    datamodule = WUSUSegmentationDataModule(
        data_root=DATA_ROOT,
        class_mapping_path=CLASS_MAPPING_PATH,
        batch_size=8,
        num_workers=4
    )
    num_classes = len(datamodule.class_names)

    # Definisco il modello con un dizionario le cui chiavi sono gli argomenti di EncoderDecoderFactory.build_model
    model_args = {
        "backbone": "swin_base_patch4_window7_224",
        "decoder": "FPN",
        "num_classes": num_classes,
        "necks": [
            {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
            {"name": "ReshapeTokensToImage"},
            {"name": "LearnedInterpolateToPyramidal"}
        ],
        "backbone_kwargs": {
            "pretrained": True,
            "in_channels": 4,
            "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
            "num_frames": 1
        },
        "decoder_kwargs": {
            "channels": [256, 128, 64, 32]
        },
        "head_kwargs": {
            #"out_channels": num_classes
            "dropout": 0.5
        }
    }

    # Inizializzo il task di segmentazione semantica
    task = SemanticSegmentationTask(
        model_args=model_args,                      # Argomenti per la ModelFactory
        model_factory="EncoderDecoderFactory",      # ModelFactory class che "assembla" backbone, neck, decoder, head
        loss='ce',                                  # cross-entropy loss
        lr=1e-4,                                    # learning rate
        optimizer="AdamW",                          # ottimizzatore AdamW
        optimizer_hparams={"weight_decay": 0.001},   # dizionario di iperparametri per opt
        scheduler="ReduceLROnPlateau",           # lr scheduler 
        scheduler_hparams={
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-8
        },
        freeze_backbone = True,                     # congelamento del backbone
        freeze_decoder = False,                     # non congelo il decoder
        freeze_head = False,                        # non congelo la testa
        plot_on_val=5,                             # ogni quante epoche plottare visualizzazioni di validation
        class_names=datamodule.class_names,         # nomi delle classi per le visualizzazioni
        ignore_index=None                              # indice della classe da ignorare
    )

    # Definisco un logger per dare un nome all'esperimento
    logger = pl_loggers.TensorBoardLogger(
        save_dir="./lightning_logs",
        name=f"{model_args['backbone']}_{model_args['decoder']}_finetune_wusu_first_exp"
    )

    # Definisco il trainer
    trainer = pl.Trainer(
        accelerator="auto",         # seleziona automaticamente tutte le GPU disponibili
        max_epochs=500,             # numero di epoche
        logger=logger,               # logger definito sopra
        #overfit_batches=None,           # per debug: usa solo un batch (sia train che val) per overfitting
        log_every_n_steps=10,         # logga scalar metrics ogni n batch di training (default 50, mettere 1 per test overfit batches)
        precision="16-mixed",        # usa precisione mista (float16 e float32) per risparmiare memoria GPU
        accumulate_grad_batches=1,  # numero di batch da accumulare prima di un passo di ottimizzazione:
                                    # batch_size=2 e accumulate_grad_batches=4 => effettivo batch size 8,
                                    # utile se batch_size è limitato dalla memoria GPU
        callbacks=[EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=10,
            verbose=True)],          # early stopping sulla val_loss
    )
    # Addestramento
    trainer.fit(task, datamodule)
    # Valutazione sul test set
    #trainer.test(task, datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()