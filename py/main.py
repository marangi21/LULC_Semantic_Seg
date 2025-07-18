import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from dataset import BuildingDataset
from trainer import Trainer

TRAIN_IMG_PATH = '/shared/marangi/projects/EVOCITY/building_extraction/data/AerialImageDataset/train/images'
TRAIN_LABELS_PATH = '/shared/marangi/projects/EVOCITY/building_extraction/data/AerialImageDataset/train/gt'
VAL_IMG_PATH = '/shared/marangi/projects/EVOCITY/building_extraction/data/AerialImageDataset/val/images'
VAL_LABELS_PATH = '/shared/marangi/projects/EVOCITY/building_extraction/data/AerialImageDataset/val/gt'

processor = SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
train_dataset = BuildingDataset(img_path=TRAIN_IMG_PATH, labels_path=TRAIN_LABELS_PATH, processor=processor, transform=None)
val_dataset = BuildingDataset(img_path=VAL_IMG_PATH, labels_path=VAL_LABELS_PATH, processor=processor, transform=None)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

test = train_dataset.__getitem__(0)[1]

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

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0001)
trainer = Trainer(model, optimizer, train_dataloader, val_dataloader, device)
trainer.train(num_epochs=5)

print()

