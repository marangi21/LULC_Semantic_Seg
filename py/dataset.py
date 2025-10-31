import rasterio as rio
import torch
import numpy as np
import matplotlib.pyplot as plt
from legend import SegmentationLegend

class SSDataset:
    def __init__(
            self,
            image_paths, 
            mask_paths, 
            in_channels=4, 
            transform=None, 
            remap_function=None, 
            class_mapping=None,
            source_gsd: float = 1.0, # espressa in metri
            target_gsd: float = 1.0,  # espressa in metri
            mask_mode = 'nearest' # algoritmo di resampling delle maschere
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.in_channels = in_channels
        self.transform = transform
        self.remap_function = remap_function
        self.class_mapping = class_mapping
        self.legend = SegmentationLegend(class_mapping) if class_mapping else None # Crea un'istanza della legenda se viene fornito il mapping
        self.source_gsd = source_gsd
        self.target_gsd = target_gsd
        self.resample = (source_gsd != target_gsd)
        self.mask_mode = mask_mode

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
        if image.shape[0] not in [3, 4]:
            raise ValueError(f"Numero di canali non supportato: {image.shape[0]}. Supportati solo 3 o 4 canali.")
        if image.shape[1:] != mask.shape:
            raise ValueError(f"Shape non congruenti: img {image.shape} vs mask {mask.shape}")
        # Adatta il numero di canali dell'immagine per esperimenti con immagini RGB
        if image.shape[0] == 4 and  self.in_channels == 3:
            image = image[:3, :, :]  # Prendi solo le prime 3 bande

        if self.remap_function:
            mask = self.remap_function(mask)

        if self.transform:
            augmented = self.transform(image=image.permute(1, 2, 0).numpy(), mask=mask.numpy()) # Albumentations vuole (H, W, C), restituisce autonomamente un tensore pytorch [C,H,W]
            image = augmented['image']
            mask = augmented['mask']

        if self.resample:
            image, mask = self.change_gsd(image, mask, self.mask_mode)

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

    def change_gsd(self, image, mask, mask_mode='nearest'):
        scale_factor = self.source_gsd / self.target_gsd
        original_shape = image.shape[-2:]
        new_shape = [int(dim*scale_factor) for dim in original_shape]
        # Downsampling con bilineare per l'immagine, nearest per la maschera
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=new_shape,
            mode='bilinear',
            antialias=True
        ).squeeze(0)
        if mask_mode=='mode':
            mask = self.downsample_mask_mode(mask, scale_factor)
        elif mask_mode=='nearest':
            mask = torch.nn.functional.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0),
                size=new_shape,
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
        else:
            raise ValueError(f"Metodo di resampling della maschera non riconosciuto: {mask_mode}")
        return image, mask

    def downsample_mask_mode(self, mask, scale_factor):
        from scipy.stats import mode
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        
        H, W = mask_np.shape
        if scale_factor > 1:
            return mask # downsampling non necessario, sarebbe upsampling.
        
        # calcolo dimensione finestra e dimensioni dell'imagine downsampled
        window_size = int(round(1/scale_factor))
        new_H = int(np.floor(H / window_size))
        new_W = int(np.floor(W / window_size))
        downsampled_mask = np.zeros((new_H, new_W), dtype=mask_np.dtype)

        # sliding window con calcolo della moda
        for i in range(new_H):
            for j in range(new_W):
                y0 = i * window_size
                y1 = y0 + window_size
                x0 = j * window_size
                x1 = x0 + window_size
                window = mask_np[y0:y1, x0:x1]

                # assegna la classe più comune nella finestra al pixel downsampled
                mode_value = mode(window, axis=None, keepdims=False)
                downsampled_mask[i,j] = mode_value[0]

        if isinstance(mask, torch.Tensor):
            return torch.from_numpy(downsampled_mask).to(mask.device)
        else:
            return downsampled_mask
                                    