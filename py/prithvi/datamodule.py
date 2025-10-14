import lightning.pytorch as pl
from torch.utils.data import DataLoader
from glob import glob
from dataset import SSDataset
import json
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
Valori di media e deviazione standard per il dataset PRITHVI
6 bande: BLUE, GREEN, RED, NIR_NARROW, SWIR1, SWIR2
Fonte: https://github.com/IBM/terratorch/blob/main/examples/tutorials/PrithviEOv2/prithvi_v2_eo_300_tl_unet_burnscars.ipynb

means=[
      0.0333497067415863, BLUE
      0.0570118552053618, GREEN
      0.0588974813200132, RED
      0.2323245113436119, NIR_NARROW
      0.1972854853760658, SWIR_1
      0.1194491422518656, SWIR_2
    ],
    stds=[
      0.0226913556882377, BLUE
      0.0268075602230702, GREEN
      0.0400410984436278, RED
      0.0779173242367269, NIR_NARROW
      0.0870873883814014, SWIR_1
      0.0724197947743781, SWIR_2
    ]

    WUSU Dataset ha le immagini prese da Gaofen-2 che ha solo 4 bande nell'ordine: 
    1. BLUE
    2. GREEN
    3. RED
    4. NIR
"""
PRITHVI_MEANS = [0.0333497067415863, 0.0570118552053618, 0.0588974813200132, 0.2323245113436119]
PRITHVI_STDS = [0.0226913556882377, 0.0268075602230702, 0.0400410984436278, 0.0779173242367269]
WUSU_MEANS = [60.862431586572065, 60.14200775711625, 59.58274396944217, 69.85295084927102] 
WUSU_STDS = [33.92976461834848, 34.12665488243311, 35.27960972101711, 41.15969251882479] 
MEANS = WUSU_MEANS
STDS = WUSU_STDS

class WUSUSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_root, class_mapping_path, batch_size=8, num_workers=4):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = None

        # Remapping degli indici delle classi. WUSU Dataset non ha la classe 5 e questo crea problemi a cuda
        # Mi costruisco un tensore di conversione in modo tale che prendendo in input una maschera
        # mi restituisca la stessa maschera con gli indici delle classi modificati
        with open(class_mapping_path) as f:
            self.original_class_mapping = json.load(f)
        
        # Crea la lista ordinata dei nomi delle classi (per i plot di terratorch)
        # Ordina gli item del dizionario in base al valore numerico
        sorted_items = sorted(self.original_class_mapping.items(), key=lambda item: item[1])
        # Estrai i nomi delle classi nell'ordine corretto
        self.class_names = [item[0] for item in sorted_items]

        # Creo il tensore che mappa il vecchio indice al nuovo, usando 255 come valore temporaneo
        # Questo tensore funge da "dizionario" per la conversione (con advanced indexing di pytorch)
        sorted_original_values = [item[1] for item in sorted_items]
        self.remapping_tensor = torch.full((max(sorted_original_values) + 1,), 255, dtype=torch.long)
        for new_idx, old_idx in enumerate(sorted_original_values):
            self.remapping_tensor[old_idx] = new_idx
    
    def remap_mask(self, mask):
        """
        Applica la mappatura sfruttando indicizzazione avanzata di torch con il tensore di mapping 
        che ho creato, alla maschera fornita in input
        """
        return self.remapping_tensor[mask]
    
    def setup(self, stage=None):
        train_path = f"{self.data_root}/train"
        val_path = f"{self.data_root}/val"
        test_path = f"{self.data_root}/test"
        train_image_paths = sorted(glob(f"{train_path}/**/imgs/*.tif", recursive=True))
        train_mask_paths = sorted(glob(f"{train_path}/**/class/*.tif", recursive=True))
        val_image_paths = sorted(glob(f"{val_path}/**/imgs/*.tif", recursive=True))
        val_mask_paths = sorted(glob(f"{val_path}/**/class/*.tif", recursive=True))
        test_image_paths = sorted(glob(f"{test_path}/**/imgs/*.tif", recursive=True))
        test_mask_paths = sorted(glob(f"{test_path}/**/class/*.tif", recursive=True))
        if stage == 'stats':
            self.train_dataset = SSDataset(train_image_paths,
                                        train_mask_paths,
                                        transform=self.get_transform(stage='stats'),
                                        remap_function=self.remap_mask,
                                        class_mapping=self.original_class_mapping)
        else:
            self.train_dataset = SSDataset(train_image_paths,
                                        train_mask_paths,
                                        transform=self.get_transform(stage='train'),
                                        remap_function=self.remap_mask,
                                        class_mapping=self.original_class_mapping)
            self.val_dataset = SSDataset(val_image_paths,
                                        val_mask_paths,
                                        transform=self.get_transform(stage='val'),
                                        remap_function=self.remap_mask,
                                        class_mapping=self.original_class_mapping) 
            self.test_dataset = SSDataset(test_image_paths,
                                        test_mask_paths,
                                        transform=self.get_transform(stage='test'),
                                        remap_function=self.remap_mask,
                                        class_mapping=self.original_class_mapping)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_transform(self, stage: str | None = None) -> A.Compose:
        """
        Restituisce la pipeline di trasformazioni di Albumentations.
        Metodo overridato per poter utilizzare le statistiche di normalizzazione di Prithvi
        """
        if stage == 'train' or stage =='fit':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=MEANS, std=STDS),
                ToTensorV2()
            ])
        elif stage == 'stats':
            transform = A.Compose([
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Normalize(mean=MEANS, std=STDS),
                ToTensorV2()
            ])
        return transform