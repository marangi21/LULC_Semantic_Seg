# LULC Semantic Segmentation with Geospatial Foundation Models

## Project Overview

This readme documents my hypotesis-driven series of experiments focused on **Land Use/Land Cover (LULC) Semantic Segmentation and Semantic Change Detection** using state-of-the-art foundation models for Earth Observation (like Prithvi-EO-2.0, Clay and DINOv3-sat). The goal of the project is to develop robust computer vision models in order to automate change detection tasks in urban environments like detection of new buildings, excavation sites or roads.
The experiments are conducted in a python 3.12 environment using the Terratorch framework. 
## Dataset: WUSU (Wuhan-Hongshan-Jiang'an)

The project utilizes the WUSU dataset, a tri-temporal, high-resolution dataset for semantic segmentation and change detection. The dataset is composed by Gaofen-2 512x512 image patches with 4 spectral bands: Blue, Green, Red and Near-InfraRed (NIR). Annotations for semantic segmentation include the following 11 classes: Road, Low building, High building, Arable land, Woodland, Grassland, River, Lake, Structure, Excavation, Bare surface. Annotations for pixel-wise Binary Change Detection ("Changed" or "Unchanged") and Semantic Change Detection ("From Woodland to Excavation", "From Arable land to Building"...) are also provided.

**Key Statistics**
- **Areas of Interest (AoIs)**: 2 (HS and JA)
- **Time Instances**: 3 (2015, 2016, 2018)
- **Patches per AoI**: 288 (HS) + 270 (JA) = 558
- **Total Images (Train/Test)**: 1,674 / 1,674
- **Total Segmentation Masks (Train/Test)**: 1,674 / 1,674
- **Total Change Masks (Train/Test)**: 1,116 / 1,116

I chose this dataset mainly for two reasons:
- Images have a Ground Sampling Distance of 1 meter, which allows to detect changes even in small buildings and roads.
- Aannotations for all 3 tasks (SS, BCD and SCD) are provided, allowing for a fair comparison among these approaches to find out which one is the best for my use cases.

![Dataset Structure](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/Untitled.png)

## Experiments Roadmap

I will start by framing the problem as a Semantic Segmentation task, adopting a two-stage approach:
- Stage 1: The primary objective is to first build and refine a state-of-the-art semantic segmentation model. This model's sole responsibility is to generate a highly accurate Land Use/Land Cover (LULC) map from a single satellite image at a specific temporal instant ($T_1$ or $T_2$).
- Stage 2: Once the segmentation model is trained, it is applied independently to images from two different times ($T_1$ and $T_2$). A deterministic post-processing pipeline then will take these two resulting LULC maps as input and compares them pixel-by-pixel to generate the final **semantic change mask**. Given the 11 distinct LULC classes, this translates to a fine-grained change detection task with **111 possible output classes** ($11 \times 10$ possible "from-to" transitions + 1 "Unchanged" class).

To systematically find the optimal solution, the research is structured into three distinct phases, ordered by expected impact.
### Phase 1: Domain Adaptation
This phase focuses on addressing the domain shift between the data used for pre-training the foundation model and the target dataset. The domain shift arises from several key differences:
- **Sensor Mismatch**: Geospatial foundation models are usually pre-trained on large-scale datasets, usualy Harmonized Landsat Sentinel data, while the target dataset is composed of imagery from the Gaofen-2 satellite.
- **Spatial and Spectral Discrepancy**: The pre-training data (e.g., 10-30m GSD for HLS) and the target data (~1m GSD for Gaofen-2) have vastly different spatial resolutions. Furthermore, their spectral bands and sensitivities differ. The model must learn to adapt its feature extractors from a coarse to a fine-grained view of the world.
- **Geographical differences**: the WUSU dataset images are taken in urban environments from two districts of wuhan, which can have their paricular architechtural, geometrical and vegetation features
To build a solid foundation and mitigate these challenge, i will implement two strategies:
- **Image Normalization**: Analyze the impact of using dataset's specific per-channel statistics (mean/std) for data normalization and compare them to using the pre-training statistics of the foundation model.
- **Backbone Unfreezing with Differential Learning Rates**: Fine-tuning the entire model to allow the backbone to adapt to the new data domain, with a lower learning rate to avoid catastrophic forgetting.
### Phase 2: Architecture Optimization
This phase focuses on finding the ideal model architecture by systematically experimenting with different combinations of backbones (e.g., Prithvi, DINOv3...) and decoders (e.g., UNet, FPN, DeepLabV3+, Transformer-based decoders...). Once promising architectures are identified, perform hyperparameter tuning to maximize their performance and find the best model.
### Phase 3: Other Techniques
This phase involves more speculative experiments to push performance boundaries, like:
- Calulate spectral indices (e.g., NDVI, NDBI) to add as input channels in order to provide the model with more explicit domain-specific features.
- Investigate the impact of specialized loss functions like Focal Loss (for class imbalance) or Dice Loss (to directly optimize IoU).
- **Staged Fine-Tuning**: Gradually adapt the model by fine-tuning on progressively higher-resolution images (e.g., 10m -> 5m -> 1m GSD) to experiment with a smoother domain adaptation.
## Experiment Log

This section details the objective, hypothesis, methodology, and conclusions for each experiment conducted.
### Experiment #1: Pipeline Validation
- **Objective**: Validate the deep learning pipeline and create a codebase for future experiments.
- **Methodology**: Used a Prithvi-EO-2.0 backbone and a U-Net decoder. A neck module takes output tokens from the 5th, 11th, 17th, and 23rd transformer blocks and maps them to 2D feature maps (with 256, 128, 64, and 32 channels, respectively) to supply the decoder's skip connections. Used the backbone's pre-training statistics to normalize the data.
- **Results**: The model learned successfully, but overfitting was observed. The training process appeared unstable, likely a side effect of using the original pre-training statistics to normalize the Gaofen-2 data. Visual inspection showed the model often confused spectrally similar classes (e.g., High_Building vs. Low_Building, Structure vs Excavation).
- **Conclusion**: The pipeline is functioning and validated. The next step is to address the unstable training and overfitting.
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/exp%231.png)
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/exp%231_2.png)
  

### Experiment #2: Architectural Modularity Test
- **Objective**: Test the modularity of the `terratorch` pipeline by swapping the backbone and decoder. Verify if training instability persists with a larger backbone and a different decoder.
- **Methodology**: Replaced the Prithvi-EO-2.0-300M with the larger Prithvi-EO-2.0-600M backbone and the U-Net decoder with an FPN decoder.
- **Results**: The framework handled the architectural changes seamlessly. Performance was similar to the first experiment, and the training process remained unstable.

| train/loss | train/acc | train/pixel_acc | train/mIoU | train/F1 | val/loss | val/acc | val/pixel_acc | val/mIoU | val/F1 |
| ---------- | --------- | --------------- | ---------- | -------- | -------- | ------- | ------------- | -------- | ------ |
| 0.448      | 0.7543    | 0.8218          | 0.6679     | 0.77     | 0.5286   | 1.109   | 0.6717        | 0.3918   | 0.537  |

- **Conclusion**: The pipeline is robust and flexible. The core performance issue is not solely due to the choice of backbone or decoder but could lie in the data processing strategy.
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/exp%232_1.png)
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/exp%232_2.png)

### Experiment #3: Correcting for Domain Shift via Normalization
- **Hypothesis**: The unstable training is caused by a severe domain shift between Prithvi's pre-training data (HLS) and the WUSU dataset (Gaofen-2). Normalizing WUSU with its own statistics should stabilize training and improve results.
- **Methodology**: Calculated per-channel mean and standard deviation for the WUSU dataset's training images. Replicated the Experiment #2 setup but normalized the data with the new calculated statistics. Confirmed a discrepancy in stats, for RGB channels they differ by an order of magnitude:
	- **Prithvi Stats**: MEANS = \[0.0333497067415863, 0.0570118552053618, 0.0588974813200132, 0.2323245113436119], STDS = \[0.0226913556882377, 0.0268075602230702, 0.0400410984436278, 0.0779173242367269]
	- **WUSU Stats**: MEANS = \[0.23867621592812355, 0.2358510244232708, 0.23365783291727554, 0.27393315260030393], STDS = \[0.13305789483810862, 0.1338300122317386, 0.1383514053044646, 0.1614105588074333]
- **Results**:
    - The training process stabilized significantly, which was especially evident in the training loss behavior.
    - The new normalization led to demonstrably better results on the validation set and improved visual predictions, particularly for classes like Road.
- **Conclusion**: **Using dataset-specific normalization statistics is critical.** Foundation models are powerful enough to handle domain shift, but they must be provided with correctly processed data. This was a major breakthrough.

Up to this point, experiments were conducted with a completely frozen backbone. Unfreezing the backbone and training its layers with a small learning rate could help the model adapt to the new domain and obtain better representations in the latent space. This could improve the segmentation of small objects and roads, as the model seems to currently detect them but has trouble segmenting their shapes correctly.
### Experiment #4: Differential Learning Rates
- **Hypothesis**: Unfreezing the backbone and fine-tuning it with a smaller learning rate will allow for better model adaptation to the new domain.
- **Methodology**: Implemented differential learning rates: $1e-5$ for the backbone and $1e-3$ for the decoder, neck, and head.
- **Results**: Significant performance improvement; this configuration achieved the best results so far. This confirms the hypothesis that a carefully tuned differential learning rate is effective for domain adaptation. The model still shows signs of overfitting, which can be addressed later with regularization and more training data.

| train/loss | train/acc | train/pixel_acc | train/mIoU | train/F1 | val/loss | val/acc | val/pixel_acc | val/mIoU | val/F1 |
| ---------- | --------- | --------------- | ---------- | -------- | -------- | ------- | ------------- | -------- | ------ |
| 0.2681     | 0.8258    | 0.8508          | 0.7433     | 0.8446   | 1.031    | 0.5645  | 0.6792        | 0.4129   | 0.5609 |

- **Conclusion**: This strategy will be used for all future experiments. The focus now shifts to optimizing the model architecture.

![training losses](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/image16.png)
![validation losses](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/image17.png)
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/fpn_difflr.png)
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/fpn2.png)

### Experiment #5: Evaluating atrous convolutions
- **Hypothesis**: A DeepLabV3+ decoder, leveraging atrous convolutions, should handle multi-scale objects and complex boundaries more effectively than an FPN decoder.
- **Methodology**: Replicated Experiment #4, replacing the FPN decoder with DeepLabV3+.
- **Results**: Achieved the best validation loss so far (0.9295). While the training loss was slightly worse than FPN, the lower validation loss indicates better generalization.
    - Visual inspection confirmed that DeepLabV3+ produced clearer boundaries for small buildings and was the first model to correctly identify some of the smaller roads.
    - A key insight emerged: the model often correctly identified objects (like roads or woodland) that were missing or incorrectly labeled in the ground truth. This means the model's actual performance is probably higher than the metrics suggest, and Prithvi's feature extraction is exceptionally powerful.
 
| train/loss | train/acc | train/pixel_acc | train/mIoU | train/F1 | val/loss | val/acc | val/pixel_acc | val/mIoU | val/F1 |
| ---------- | --------- | --------------- | ---------- | -------- | -------- | ------- | ------------- | -------- | ------ |
| 0.2805     | 0.7986    | 0.8275          | 0.7114     | 0.8196   | 0.9295   | 0.5097  | 0.6812        | 0.38     | 0.5111 |

- **Conclusion**: DeepLabV3+ appears to be a better decoder for this task. The dilated convolutions proved effective. The results also highlight the limitations of the dataset's annotations.

![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/deeplab.png)
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/deeplab2.png)
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/image1c.png)
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/image1d.png)
### Experiment #6: DINOv3 Backbone
- **Objective**: Evaluate the performance of the DINOv2 backbone compared to Prithvi-EO-2.0.
- **Hypothesis**: `DINOv3` is pre-trained on very high-resolution (0.6m GSD) RGB imagery, and may offer superior spatial feature extraction, potentially improving small object shape detection. However, its lack of pre-training on multispectral data might degrade classification accuracy (as NIR data helps identify materials and vegetation). This presents an interesting trade-off between spatial and spectral adaptation to investigate.
- **Methodology**: Replicated Experiment #5, replacing the Prithvi-EO-2.0-600M backbone with a dinov3-vitl16-pretrain-sat493m backbone. Built a custom neck that takes the output of the 10th transformer block of the backbone, reshapes it into an image (with subsequent 1x1 conv) and feeds it as a skip connection for the deeplabv3+ decoder. The index of the transformer block for this process might be a tunable hyperparameter in future experiments. Also, excluded the firs 5 tokens from the backbone embeddings when generating the feature maps as they are 1 CLS token and 4 Register tokens, which are not spatially informative and made token length=1029 which is not a perfect square.
- **Results**: All evaluation metrics on the validation set improved compared to past experiments. From a visual inspection of some validation batch predictions, it looks like the model is more precise when detecting small buildings shapes and minor roads. After the first epoch the model started labeling lakes as "River". I don't think this is gonna be an issue for this use case because they could be grouped into the same superclass "Water" as the goal is to detect new buildings, roads and excavations.
| train/loss | train/acc | train/pixel_acc | train/mIoU | train/F1 | val/loss | val/acc | val/pixel_acc | val/mIoU | val/F1 |
| ---------- | --------- | --------------- | ---------- | -------- | -------- | ------- | ------------- | -------- | ------ |
| 0.3955     | 0.7999    | 0.8436          | 0.7177     | 0.8235   | 0.7985   | 0.5439  | 0.7155        | 0.4253     | 0.5553 |
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/dinov3_eval.png)
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/dinov3_eval2.png)
![](https://github.com/marangi21/LULC_Semantic_Seg/blob/main/images/dinov3_eval3.png)
- **Conclusion**: This architecture outperformed all the previous ones across every evaluation metric, proving that DINOv3-sat backbone is better for this task.

## Experiment #7: Evaluating a Mask2Former decoder
- **Objective:** evaluating the impact of a transformer-based decoder on performances, compared to previous conv-based decoders. 
- **Methodology:** \[In Progress]
- **Results:** \[In Progress]
| train/loss | train/acc | train/pixel_acc | train/mIoU | train/F1 | val/loss | val/acc | val/pixel_acc | val/mIoU | val/F1 |
| ---------- | --------- | --------------- | ---------- | -------- | -------- | ------- | ------------- | -------- | ------ |
|      |     |         |      |    |    |   |         |      |  |
