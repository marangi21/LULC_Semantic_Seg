import rasterio as rio
import torch
import numpy as np
import matplotlib.pyplot as plt

"""
Valori di media e deviazione standard per il dataset PRITHVI
6 bande: BLUE, GREEN, RED, NIR_NARROW, SWIR1, SWIR2
Fonte: https://github.com/IBM/terratorch/blob/main/examples/tutorials/PrithviEOv2/prithvi_v2_eo_300_tl_unet_burnscars.ipynb

means=[
      0.0333497067415863,
      0.0570118552053618,
      0.0588974813200132,
      0.2323245113436119,
      0.1972854853760658,
      0.1194491422518656,
    ],
    stds=[
      0.0226913556882377,
      0.0268075602230702,
      0.0400410984436278,
      0.0779173242367269,
      0.0870873883814014,
      0.0724197947743781,
    ]
"""

class SSDataset:
    def __init__(self, image_paths, mask_paths, transform=None, remap_function=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.remap_function = remap_function

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
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return { 
            "image": image,
            "mask": mask,
            "filename": image_path
        }
    
    def plot(self, sample):
        # ToDo: migliorare il plotting come ho fatto in tests.ipynb
        """
        Visualizza un campione del dataset (immagine e maschera).
        `sample` è il dizionario restituito da __getitem__.
        Metodo utilizzato internamente da torchgeo (alla quale si appoggia terratorch) per visualizzare i batch
        di validazione ogni plot_on_val epoche.
        """
        image = sample['image']
        mask = sample['mask']
        filename = sample['filename']

        # Le immagini satellitari spesso non sono in un range visualizzabile [0,1] o [0,255].
        # Per una visualizzazione semplice, prendiamo le prime 3 bande (assumendo siano RGB-like)
        # e le normalizziamo in modo aggressivo per renderle visibili.
        # Questo è solo per il plotting, non influisce sul training.
        rgb = image[:3, :, :].movedim([0], [2])  # Cambia da (C, H, W) a (H, W, C)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        # Creiamo una figura con due subplot: immagine e maschera
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        axs[0].imshow(rgb)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="viridis") # Usiamo una colormap per vedere le diverse classi
        axs[1].set_title("Mask")
        axs[1].axis("off")

        return fig