# LungGuardian

Pneumonia detection from chest X-rays using Faster R-CNN with EfficientNetV2 backbone. Built for the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).

<p align="center">
  <img src="assets/detection_demo.png" alt="Pneumonia Detection Demo" width="800"/>
</p>

## What it does

This is an **object detection** model, not a classifier. It finds and localizes pneumonia opacities in chest X-rays with bounding boxes.

<p align="center">
  <img src="assets/classification_vs_detection.png" alt="Classification vs Detection" width="700"/>
</p>

| Input | Output |
|:-----:|:------:|
| ![Input X-Ray](assets/sample_input.png) | ![Detection Output](assets/sample_output.png) |

## Tech stack

- **Backbone**: EfficientNetV2-S (via `timm`)
- **Detector**: Faster R-CNN with FPN
- **Loss**: Focal Loss (handles class imbalance)
- **Post-processing**: Weighted Box Fusion
- **Augmentation**: Albumentations + Mosaic
- **Splitting**: StratifiedGroupKFold (no patient leakage)

## Setup

```bash
git clone https://github.com/yourusername/LungGuardian.git
cd LungGuardian
pip install -r requirements.txt
```

Download the [RSNA dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) and extract to `data/rsna/`.

## Training

```bash
# Single fold
python notebooks/04_training_engine.py --fold 0 --epochs 15

# All folds
python notebooks/04_training_engine.py --all_folds
```

## Project structure

```
├── notebooks/
│   ├── 01_data_pipeline.py      # Dataset & augmentation
│   ├── 02_model_architecture.py # EfficientNetV2 + FPN + Faster R-CNN
│   ├── 03_losses_and_utilities.py # Focal loss, WBF, metrics
│   └── 04_training_engine.py    # Training loop
├── assets/                      # Demo images
├── requirements.txt
└── README.md
```

## Key configs

| Param | Value |
|-------|-------|
| Image size | 1024×1024 |
| Batch size | 4 |
| LR (backbone) | 1e-5 |
| LR (head) | 1e-4 |
| Scheduler | Cosine w/ warmup |
| Epochs | 15 |

## References

- [EfficientNetV2](https://arxiv.org/abs/2104.00298)
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [Focal Loss](https://arxiv.org/abs/1708.02002)
- [Weighted Box Fusion](https://arxiv.org/abs/1910.13302)

## License

MIT
