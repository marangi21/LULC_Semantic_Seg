import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from dataset import BuildingDataset
from trainer import Trainer
import wandb
import os

DEBUG = True
DEBUG_SAMPLE_SIZE = 10

TRAIN_IMG_PATH = '/shared/marangi/projects/EVOCITY/building_extraction/data/AerialImageDataset/train/images'
TRAIN_LABELS_PATH = '/shared/marangi/projects/EVOCITY/building_extraction/data/AerialImageDataset/train/gt'
VAL_IMG_PATH = '/shared/marangi/projects/EVOCITY/building_extraction/data/AerialImageDataset/val/images'
VAL_LABELS_PATH = '/shared/marangi/projects/EVOCITY/building_extraction/data/AerialImageDataset/val/gt'

processor = SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
train_dataset = BuildingDataset(img_path=TRAIN_IMG_PATH, labels_path=TRAIN_LABELS_PATH, processor=processor, transform=None)
val_dataset = BuildingDataset(img_path=VAL_IMG_PATH, labels_path=VAL_LABELS_PATH, processor=processor, transform=None)
batch_size = 24

if DEBUG:
    train_dataset = torch.utils.data.Subset(train_dataset, range(DEBUG_SAMPLE_SIZE))
    val_dataset = torch.utils.data.Subset(val_dataset, range(DEBUG_SAMPLE_SIZE//5))
    batch_size = 2

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_labels = 2
id2label = {0:"background", 1:"building"}
label2id = {v:k for k,v in id2label.items()}

model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name_or_path='nvidia/segformer-b0-finetuned-ade-512-512',
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
    )

# Freeze encoder weights per il fine tuning.
# Decoder-head e classifier weights sono scongelati e trainati.
# total_weights: 3714658
# trainable_weights: 395266
# trainable_percentage: 10.64%
for param in model.segformer.encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0001)

config = {
    # Dataset info
    'dataset': 'INRIA_AerialImageDataset',
    'train_size': len(train_dataset),
    'val_size': len(val_dataset),
    'batch_size': batch_size,
    'num_classes': num_labels,
    'id2label': id2label,

    # Model info
    'model_name': model.__class__.__name__,
    'processor': processor.__class__.__name__,
    'optimizer': optimizer.__class__.__name__,
    'epochs': 5,

    # Early stopping info
    'es_patience': 10,
    'es_min_delta': 0,

    # LR Scheduler info
    'lr_scheduler' : 'ReduceLROnPlateau',
    'lr_factor': 0.5,
    'lr_patience': 5,
    'min_lr': 1e-7,

    # Hardware info
    'device': device
}

with wandb.init(project='building_extraction',
                config=config,
                tags=['building_segmentation', 'segformer', 'INRIA_Aerial_Dataset'],
                notes='Building extraction using SegFormer on INRIA Aerial Dataset'
                ) as run:
    # Log modello (architettura, parametri, gradienti)
    wandb.watch(model, log_freq=100)

    trainer = Trainer(model=model,
                    optimizer=optimizer, 
                    train_dataloader=train_dataloader, 
                    val_dataloader=val_dataloader, 
                    device=device, 
                    es_patience=run.config['es_patience'],
                    es_min_delta=run.config['es_min_delta'],
                    lr_factor=run.config['lr_factor'],
                    lr_patience=run.config['lr_patience'],
                    min_lr=run.config['min_lr'],
                    verbose=True
                    )
    
    trainer.train(num_epochs=run.config['epochs'])

    # ToDo: check model versioning
    model_path = os.path.join(os.path.join(os.path.join(os.getcwd(), '..'), 'experiments'), f'segformer_be_{run.id}')
    trainer.model.save_pretrained(model_path)
    artifact = wandb.Artifact(
        name=f"segformer_be_{run.id}",
        type="model",
        description=f"Trained SegFormer model for building extraction run {run.id}"
    )
    artifact.add_dir(model_path)
    run.log_artifact(artifact)

    # Log final summary
    run.summary.update({
        'final_train_loss': trainer.train_losses[-1],
        'final_val_loss': trainer.val_losses[-1],
        'best_val_loss': trainer.early_stopper.best_loss,
        'best_epoch': trainer.early_stopper.best_epoch,
        'total_epochs_trained': len(trainer.train_losses),
        'early_stopped': trainer.early_stopper.early_stop
    })

print()

