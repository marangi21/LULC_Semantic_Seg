from torch.utils.data import Dataset
import os
import rasterio as rio
import torch
import numpy as np
from torchvision.transforms import v2

#IMG_PATH = '/shared/marangi/projects/EVOCITY/building_extraction/data/AerialImageDataset/train/images'
#LABELS_PATH = '/shared/marangi/projects/EVOCITY/building_extraction/data/AerialImageDataset/train/gt'

class BuildingDataset(Dataset):
    def __init__(self, img_path, labels_path, processor, transform=None):
        """
        Args:
            img_path (str): percorso alle immagini
            labels_path (str): percorso alle annotazioni
            transform (callable, optional): opzionale, insieme delle trasformazioni da applicare alle immagini
        """
        self.img_path = img_path
        self.images = os.listdir(self.img_path)
        self.labels_path = labels_path
        self.labels = os.listdir(self.labels_path)
        self.processor=processor
        self.transform=transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        try:
            with rio.open(os.path.join(self.img_path, image)) as r:
                img = r.read()
            with rio.open(os.path.join(self.labels_path, label)) as r:
                gt = r.read()
        except Exception as e:
            print(f"Errore nell'apertura dell'immagine o della label: {e}")
            raise
        
        if img.shape[1:] != gt.shape[1:]:
            raise ValueError(f"Shape non congruenti: img {img.shape} vs gt {gt.shape}")

        img = torch.from_numpy(img.astype(np.uint8))
        mask = torch.from_numpy(np.where(gt == 255, 1, 0).astype(np.uint8)) # convertito in 0,1 da 0,255

        processed = self.processor(
            images=img,
            segmentation_maps=mask,
            return_tensors='pt'
        )
        pixel_values = processed['pixel_values'].squeeze(0)
        labels = processed['labels'].squeeze(0)
        labels = torch.where(labels == 255, torch.tensor(1), torch.tensor(0))
        labels = labels.long()

        return pixel_values, labels
        