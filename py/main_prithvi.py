from datamodule import WUSUSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
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

# UPerNet: https://huggingface.co/docs/transformers/en/model_doc/upernet

ENCODER_LR = 1e-5
DECODER_LR = HEAD_LR = 1e-3 # using decoder lr for neck too
WEIGHT_DECAY = 1e-3

def main():
    seed_everything(42, workers=True) # per riproducibilità
    REPO_ROOT = Path(__file__).parent.parent
    DATA_ROOT = REPO_ROOT / "data" / "WUSU_preprocessed"
    CLASS_MAPPING_PATH = REPO_ROOT / "data" / "OpenWUSU512" / "class_mapping.json"
    
    # Definisco data module: gestisce tutta la logica di caricamento dei set di dati e istanziazione dei dataloader
    datamodule = WUSUSegmentationDataModule(
        data_root=DATA_ROOT,
        class_mapping_path=CLASS_MAPPING_PATH,
        batch_size=1,
        num_workers=4
    )
    num_classes = len(datamodule.class_names)

    # Definisco il modello con un dizionario le cui chiavi sono gli argomenti di EncoderDecoderFactory.build_model
    model_args = {
        "backbone": "prithvi_eo_v2_300",
        "decoder": "UperNetDecoder", 
        "num_classes": num_classes,
        #"necks": [
        # these are to transform prithvi tokens to image-like tensors, to make prithvi backbone compatible 
        # with convolutional decoders that expect a piramyd of feature maps
        #    {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        #    {"name": "ReshapeTokensToImage"},
        #    {"name": "LearnedInterpolateToPyramidal"}
        #],
        "backbone_kwargs": {
            "pretrained": True,
            "in_channels": 4,
            "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
            "num_frames": 1
        },
        "decoder_kwargs": {
            # dimensioni delle feature map prodotte dal neck per i convolutional decoders.
            # devono matchare numero e dimensioni attese dalle skip connection del decoder
            # 256 per il token in output dal 5° transformer block della backbone,
            # 128 per il token in output dall'11° transformer block della backbone ecc...
            #"channels": [256, 128, 64, 32]
        },
        "head_kwargs": {
            #"out_channels": num_classes
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
    
    # Inizializzo il task di segmentazione semantica
    task = DiffLRSemanticSegmentationTask(
        model_args=model_args,                      # Argomenti per la ModelFactory
        model_factory="EncoderDecoderFactory",      # ModelFactory class che "assembla" backbone, neck, decoder, head
        loss='ce',                                  # cross-entropy loss
        lr=1e-5,                                    # learning rate globale (utilizzato per i gruppi di parametri per 
                                                    #           i quali non viene specificato un lr)
        optimizer="AdamW",                          # ottimizzatore AdamW
        optimizer_hparams=custom_opt_params,        # dizionario di iperparametri per opt
        scheduler="ReduceLROnPlateau",              # lr scheduler 
        scheduler_hparams={
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-8
        },
        freeze_backbone = False,                     # congelamento del backbone
        freeze_decoder = False,                     # non congelo il decoder
        freeze_head = False,                        # non congelo la testa
        plot_on_val=5,                             # ogni quante epoche plottare visualizzazioni di validation
        class_names=datamodule.class_names,         # nomi delle classi per le visualizzazioni
        ignore_index=None                              # indice della classe da ignorare
    )

    # Definisco un logger per dare un nome all'esperimento
    logger = pl_loggers.TensorBoardLogger(
        save_dir= REPO_ROOT / "lightning_logs",
        name=f"{model_args['backbone']}_{model_args['decoder']}_wmeans_diffLRs"
    )

    # Definisco il trainer
    trainer = pl.Trainer(
        accelerator="auto",         # seleziona automaticamente tutte le GPU disponibili
        max_epochs=500,             # numero di epoche
        logger=logger,               # logger definito sopra
        #overfit_batches=None,           # per debug: usa solo un batch (sia train che val) per overfitting
        log_every_n_steps=10,         # logga scalar metrics ogni n batch di training (default 50, mettere 1 per test overfit batches)
        precision="16-mixed",        # usa precisione mista (float16 e float32) per risparmiare memoria GPU
        accumulate_grad_batches=8,  # numero di batch da accumulare prima di un passo di ottimizzazione:
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