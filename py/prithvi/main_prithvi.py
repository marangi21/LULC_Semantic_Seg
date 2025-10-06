import terratorch as tt
from terratorch.datamodules import GenericNonGeoClassificationDataModule
import rasterio as rio

TRAIN_PATH = "/shared/marangi/projects/EVOCITY/building_extraction/data/OpenWUSU512/train"
TEST_PATH = "/shared/marangi/projects/EVOCITY/building_extraction/data/OpenWUSU512/test"

with rio.open(f"{TRAIN_PATH}/JA/imgs/JA15_0.tif") as img:
    with rio.open(f"{TRAIN_PATH}/JA/class/JA15_0.tif") as mask:
        print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")