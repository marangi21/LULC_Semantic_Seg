from torchvision.transforms import v2
import torchvision.transforms.functional as tvf
import torch.nn.functional as F
import random
import torch

class SegRandomCrop:
    def __init__(self, crop_size, p=0.5):
        self.crop_size = crop_size
        self.p = p
        pass

    def __call__(self, img, mask):

        # applico solo con probabilità p
        if random.random() > self.p:
            return img, mask
        # Assumo img.shape=[C,H,W], mask.shape=[H,W]
        # Assumo crop quadrato
        # boundary coords per il punto selezionato a random da cui far partire il crop
        max_x = img.shape[1] - self.crop_size
        max_y = img.shape[2] - self.crop_size
        # seleziono punto random
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        # croppo
        img = img[:, x:x+self.crop_size, y:y+self.crop_size]
        mask = mask[x:x+self.crop_size, y:y+self.crop_size]
        return img, mask
    
class SegRandomRotation:
    def __init__(self, degrees, bg_class=0, p=0.5):
        # serve per generare i due valoti tra cui scegliere l'angolo di rotazione casualmente
        # se ne passo uno fa +/-degrees, se no tra i due passati (angolo1, angolo2)
        # degrees è sempre tupla con due elementi quindi
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.bg_class = bg_class
        self.p = p
        pass
    
    def __call__(self, img, mask):
        # Assumo img.shape=[C,H,W], mask.shape=[H,W]
        # applico solo con probabilità p
        if random.random() > self.p:
            return img, mask
        # seleziona un angolo random dall'intervallo
        angle = random.uniform(self.degrees[0], self.degrees[1])
        original_size = img.shape[-2:] # (H, W)
        # Applica la rotazione con padding
        img_rotated = tvf.rotate(img, angle, expand=True, fill=0)
        mask_rotated = tvf.rotate(mask.unsqueeze(0), angle, expand=True, fill=self.bg_class)
        # Resize alla dimensione originale
        img_final = F.interpolate(img_rotated.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
        mask_final = F.interpolate(mask_rotated.unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest').squeeze(0).squeeze(0).long()
        return img_final, mask_final
    
class SegColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img, mask):
        if random.random() > self.p:
            return img, mask
        img_tensor = torch.from_numpy(img)
        #mask_tensor = torch.from_numpy(mask)
        color_transform = v2.Compose([
            v2.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        ])
        img_tensor = color_transform(img_tensor)
        img = img_tensor.numpy()
        #mask = mask_tensor.numpy()
        return img, mask
    
class SegRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() > self.p:
            return img, mask
        img_flipped = tvf.vflip(img)
        mask_flipped = tvf.vflip(mask.unsqueeze(0)).squeeze(0) # vflip vuole [C,H,W]
        return img_flipped, mask_flipped
    
class SegRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() > self.p:
            return img, mask
        img_flipped = tvf.hflip(img)
        mask_flipped = tvf.hflip(mask.unsqueeze(0)).squeeze(0) # hflip vuole [C,H,W]
        return img_flipped, mask_flipped