import lightning.pytorch as pl
from torch.utils.data import DataLoader
from glob import glob
from dataset import SSDataset
import json
import torch

class WUSUSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_root, class_mapping_path, batch_size=8, num_workers=4):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        # ToDo: definisco qui le trasformazioni
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
        self.train_dataset = SSDataset(train_image_paths, train_mask_paths, remap_function=self.remap_mask)
        self.val_dataset = SSDataset(val_image_paths, val_mask_paths, remap_function=self.remap_mask) 
        self.test_dataset = SSDataset(test_image_paths, test_mask_paths, remap_function=self.remap_mask)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)