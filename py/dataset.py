import rasterio as rio
import torch
import numpy as np
import matplotlib.pyplot as plt
from legend import SegmentationLegend

class SSDataset:
    def __init__(self, image_paths, mask_paths, transform=None, remap_function=None, class_mapping=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.remap_function = remap_function
        self.class_mapping = class_mapping
        self.legend = SegmentationLegend(class_mapping) if class_mapping else None # Crea un'istanza della legenda se viene fornito il mapping

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        try:
            with rio.open(image_path) as img_file:
                image = torch.from_numpy(img_file.read().astype(np.float32))
            with rio.open(mask_path) as mask_file:
                mask = torch.from_numpy(mask_file.read(1).astype('int64'))  # Assuming mask is single channel
        except Exception as e:
            print(f"Errore nell'apertura dell'immagine o della mask: {e}")
            raise
        if image.shape[1:] != mask.shape:
            raise ValueError(f"Shape non congruenti: img {image.shape} vs mask {mask.shape}")

        if self.remap_function:
            mask = self.remap_function(mask)

        if self.transform:
            augmented = self.transform(image=image.permute(1, 2, 0).numpy(), mask=mask.numpy()) # Albumentations vuole (H, W, C), restituisce autonomamente un tensore pytorch [C,H,W]
            image = augmented['image']
            mask = augmented['mask']

        return { 
            "image": image,
            "mask": mask,
            "filename": image_path
        }
    
    def plot(self, sample):
        """
        Visualizza un campione del dataset (immagine, maschera e prediction).
        `sample` è il dizionario restituito da __getitem__.
        Metodo utilizzato internamente da torchgeo (alla quale si appoggia terratorch) per visualizzare i batch
        di validazione ogni plot_on_val epoche.
        Il metodo è chiameto dal validatio_step del task di terratorch, che aggiunge autonomamente 
        la predizione al dizionario del campione.
        Utilizza la SegmentationLegend definita nel costruttore.
        """
        if not self.legend:
            raise RuntimeError("Impossibile plottare: class_mapping non fornito al Dataset.")
        
        image = sample['image']
        mask = sample['mask']

        # Le immagini satellitari spesso non sono in un range visualizzabile [0,1] o [0,255].
        # Per una visualizzazione semplice, prendiamo le prime 3 bande (assumendo siano RGB-like)
        # e le normalizziamo in modo aggressivo per renderle visibili.
        # Questo è solo per il plotting, non influisce sul training.
        rgb_image = torch.permute(image[:3,:,:], (1, 2, 0))  # Cambia da (C, H, W) a (H, W, C)
        rgb_image_np = rgb_image.numpy()
        rgb_min = rgb_image_np.min(axis=(0, 1), keepdims=True)
        rgb_max = rgb_image_np.max(axis=(0, 1), keepdims=True)
        rgb_image_np = (rgb_image_np - rgb_min) / (rgb_max - rgb_min + 1e-6)
        rgb_image = torch.from_numpy(np.clip(rgb_image_np, 0, 1))

        # Controlla se il sample contiene la predizione e imposta di conseguenza il numero di subplot
        has_prediction = 'prediction' in sample
        num_plots = 3 if has_prediction else 2
        fig, axs = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        
        if num_plots==1: # Rendi 'axs' sempre un array per consistenza (questa cosa non l'ho capita but here we are folks)
            axs = [axs]

        # Plot dell'immagine
        axs[0].imshow(rgb_image)
        axs[0].set_title("Image")
        axs[0].axis("off")
        # Plot dalla maschera (ground truth)
        axs[1].imshow(mask, cmap=self.legend.cmap, norm=self.legend.norm)
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")
        # Plot dalla predizione (se presente)
        if has_prediction:
            prediction = sample['prediction'].numpy()
            axs[2].imshow(prediction, cmap=self.legend.cmap, norm=self.legend.norm)
            axs[2].set_title("Prediction Mask")
            axs[2].axis("off")
        
        # Aggiunge la legenda
        self.legend.plot()
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        return fig