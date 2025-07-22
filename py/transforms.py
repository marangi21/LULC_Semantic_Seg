from torchvision.transforms import v2
import torchvision.transforms.functional as tvf
import torch.nn.functional as F
import random
import torch

class SegRandomSizeCrop:
    def __init__(self, min_size, max_size, p=0.5):
        self.min_size = min_size
        self.max_size = max_size
        self.p = p
        pass

    def __call__(self, img, mask):

        # applico solo con probabilità p
        if random.random() > self.p:
            return img, mask
        # Assumo img.shape=[C,H,W], mask.shape=[H,W]
        # Assumo crop quadrato, dimensione randomica per ogni chiamata
        crop_size = random.randint(self.min_size, self.max_size)
        # boundary coords per il punto selezionato a random da cui far partire il crop
        max_x = img.shape[1] - crop_size
        max_y = img.shape[2] - crop_size
        # seleziono punto random
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        # croppo
        img_cropped = img[:, x:x+crop_size, y:y+crop_size]
        mask_cropped = mask[x:x+crop_size, y:y+crop_size]
        original_size = (img.shape[1], img.shape[2])
        img_final = F.interpolate(img_cropped.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
        mask_final = F.interpolate(mask_cropped.unsqueeze(0).unsqueeze(0).float(), size=original_size, mode='nearest').squeeze(0).squeeze(0).long()
        return img_final, mask_final

    def __repr__(self):
        return f"{self.__class__.__name__}(min_size={self.min_size}, max_size={self.max_size}, p={self.p})"
    
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

    def __repr__(self):
        return f"{self.__class__.__name__}(degrees={self.degrees}, bg_class={self.bg_class}, p={self.p})"
    
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

    def __repr__(self):
        return f"{self.__class__.__name__}(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue}, p={self.p})"
    
class SegRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() > self.p:
            return img, mask
        img_flipped = tvf.vflip(img)
        mask_flipped = tvf.vflip(mask.unsqueeze(0)).squeeze(0) # vflip vuole [C,H,W]
        return img_flipped, mask_flipped

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
    
class SegRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() > self.p:
            return img, mask
        img_flipped = tvf.hflip(img)
        mask_flipped = tvf.hflip(mask.unsqueeze(0)).squeeze(0) # hflip vuole [C,H,W]
        return img_flipped, mask_flipped
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

class Compose:
        """
        Unisce più trasformazioni
        """

        def __init__(self, transforms: list):
            self.transforms = transforms

        def __call__(self, input, target):
            for tr in self.transforms:
                input, target = tr(input, target)
            return input, target

        def __repr__(self):
            format_string = self.__class__.__name__ + '('
            for t in self.transforms:
                format_string += '\n'
                format_string += f'    {t}'
            format_string += '\n)'
            return format_string
        
def get_training_transforms():
    return Compose([
        SegRandomSizeCrop(min_size=384, max_size=460, p=0.5),
        SegRandomHorizontalFlip(p=0.5),
        SegRandomVerticalFlip(p=0.5),
        SegColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        SegRandomRotation(degrees=180, bg_class=0, p=0.5)
    ])