"""
================================================================================
SOTA PNEUMONIA DETECTION SYSTEM - MODULE 1: DATA PIPELINE
================================================================================
This module handles all data loading, preprocessing, and augmentation for the
RSNA Pneumonia Detection Challenge dataset.

Key Features:
- DICOM file reading and normalization
- Bounding box grouping (multiple boxes per patient)
- StratifiedGroupKFold splitting (patient-level grouping + class stratification)
- Strong albumentations augmentation pipeline
================================================================================
"""

# ==============================================================================
# CELL 1: IMPORTS AND SETUP
# ==============================================================================

import os
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold

# Albumentations for SOTA augmentation pipeline
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

# Configuration
class Config:
    """Central configuration for data pipeline."""
    DATA_DIR = "./data/rsna"
    TRAIN_DIR = os.path.join(DATA_DIR, "stage_2_train_images")
    TEST_DIR = os.path.join(DATA_DIR, "stage_2_test_images")
    TRAIN_LABELS = os.path.join(DATA_DIR, "stage_2_train_labels.csv")

    # Image settings
    IMG_SIZE = 1024  # RSNA images are 1024x1024
    NUM_WORKERS = 4

    # Cross-validation settings
    N_FOLDS = 5
    SEED = 42

    # Training settings
    BATCH_SIZE = 4  # Small batch due to large images


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(Config.SEED)


# ==============================================================================
# CELL 2: DICOM UTILITIES
# ==============================================================================

def read_dicom(path: str, normalize: bool = True) -> np.ndarray:
    """
    Read a DICOM file and convert to numpy array.

    RSNA Challenge uses 8-bit grayscale DICOM images (0-255).
    We convert to float32 and optionally normalize to [0, 1].

    Args:
        path: Path to .dcm file
        normalize: If True, normalize pixel values to [0, 1]

    Returns:
        Image as numpy array with shape (H, W) or (H, W, 3) for RGB
    """
    dcm = pydicom.dcmread(path)

    # Extract pixel array
    img = dcm.pixel_array.astype(np.float32)

    # Handle PhotometricInterpretation
    # MONOCHROME1: 0 = white, MONOCHROME2: 0 = black
    if hasattr(dcm, 'PhotometricInterpretation'):
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            # Invert so that 0 = black (standard)
            img = img.max() - img

    # Normalize to [0, 1] range
    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Convert grayscale to 3-channel (required for pretrained backbones)
    # This is critical: EfficientNetV2 expects RGB input
    img = np.stack([img, img, img], axis=-1)

    return img


def resize_image_and_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    target_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image and scale bounding boxes accordingly.

    Args:
        image: Input image (H, W, C)
        boxes: Bounding boxes in [x_min, y_min, x_max, y_max] format
        target_size: Target size (square)

    Returns:
        Resized image and scaled boxes
    """
    h, w = image.shape[:2]

    # Resize image
    image_resized = cv2.resize(image, (target_size, target_size))

    if len(boxes) > 0:
        # Scale boxes
        scale_x = target_size / w
        scale_y = target_size / h

        boxes = boxes.copy().astype(np.float32)
        boxes[:, [0, 2]] *= scale_x  # x coordinates
        boxes[:, [1, 3]] *= scale_y  # y coordinates

    return image_resized, boxes


# ==============================================================================
# CELL 3: BOUNDING BOX GROUPING
# ==============================================================================

def group_boxes_by_patient(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Group bounding boxes by patientId.

    The RSNA dataset has multiple rows per patient (one per bounding box).
    This function groups them so each patient has:
    - A list of bounding boxes (if pneumonia positive)
    - A target label (1 = pneumonia, 0 = no pneumonia)

    CSV Format:
        patientId, x, y, width, height, Target
        - Target=1: Pneumonia present (has bounding box)
        - Target=0: No pneumonia (x, y, width, height are NaN)

    Args:
        df: Raw training labels DataFrame

    Returns:
        Dictionary mapping patientId to {'boxes': [...], 'target': int}
    """
    grouped = defaultdict(lambda: {'boxes': [], 'target': 0})

    for _, row in df.iterrows():
        patient_id = row['patientId']
        target = int(row['Target'])

        grouped[patient_id]['target'] = target

        # Only add boxes for positive cases (Target=1)
        if target == 1 and not pd.isna(row['x']):
            # Convert from [x, y, width, height] to [x_min, y_min, x_max, y_max]
            x_min = float(row['x'])
            y_min = float(row['y'])
            x_max = x_min + float(row['width'])
            y_max = y_min + float(row['height'])

            grouped[patient_id]['boxes'].append([x_min, y_min, x_max, y_max])

    # Convert box lists to numpy arrays
    for patient_id in grouped:
        boxes = grouped[patient_id]['boxes']
        if len(boxes) > 0:
            grouped[patient_id]['boxes'] = np.array(boxes, dtype=np.float32)
        else:
            # Empty array with correct shape for no boxes
            grouped[patient_id]['boxes'] = np.zeros((0, 4), dtype=np.float32)

    return dict(grouped)


def create_patient_df(grouped_data: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a patient-level DataFrame from grouped data.

    This is used for stratified splitting - we need one row per patient
    with their target label for stratification.

    Args:
        grouped_data: Output from group_boxes_by_patient()

    Returns:
        DataFrame with columns: patientId, target, num_boxes
    """
    records = []
    for patient_id, data in grouped_data.items():
        records.append({
            'patientId': patient_id,
            'target': data['target'],
            'num_boxes': len(data['boxes'])
        })

    return pd.DataFrame(records)


# ==============================================================================
# CELL 4: STRATIFIED GROUP K-FOLD SPLITTING
# ==============================================================================

def create_folds(
    patient_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create StratifiedGroupKFold splits.

    WHY STRATIFIED GROUP K-FOLD?
    ============================
    1. GROUP: Same patientId must NEVER appear in both train and validation.
       This prevents data leakage - if the same patient is in both sets,
       the model can "memorize" patient-specific features rather than
       learning generalizable pneumonia patterns.

    2. STRATIFIED: Maintain the ratio of pneumonia-positive to negative cases
       across all folds. The dataset is imbalanced (~26% positive), so random
       splitting could create folds with very different class distributions.

    Args:
        patient_df: DataFrame with patientId and target columns
        n_folds: Number of folds
        seed: Random seed for reproducibility

    Returns:
        DataFrame with added 'fold' column
    """
    patient_df = patient_df.copy()
    patient_df['fold'] = -1

    # StratifiedGroupKFold ensures:
    # - Groups (patientId) are not split across train/val
    # - Class distribution (target) is preserved in each fold
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    X = patient_df['patientId'].values
    y = patient_df['target'].values
    groups = patient_df['patientId'].values  # Each patient is their own group

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        patient_df.loc[val_idx, 'fold'] = fold_idx

    return patient_df


def get_fold_data(
    patient_df: pd.DataFrame,
    grouped_data: Dict[str, Dict],
    fold: int
) -> Tuple[List[str], List[str]]:
    """
    Get train and validation patient IDs for a specific fold.

    Args:
        patient_df: DataFrame with fold assignments
        grouped_data: Grouped bounding box data
        fold: Fold number to use as validation

    Returns:
        Tuple of (train_patient_ids, val_patient_ids)
    """
    train_patients = patient_df[patient_df['fold'] != fold]['patientId'].tolist()
    val_patients = patient_df[patient_df['fold'] == fold]['patientId'].tolist()

    return train_patients, val_patients


# ==============================================================================
# CELL 5: AUGMENTATION PIPELINES
# ==============================================================================

def get_train_transforms(img_size: int = 1024) -> A.Compose:
    """
    Strong training augmentation pipeline using Albumentations.

    WHY THESE AUGMENTATIONS?
    ========================
    1. HorizontalFlip: Chest X-rays can be flipped (though heart position changes)
    2. ShiftScaleRotate: Simulates patient positioning variations
    3. RandomBrightnessContrast: Handles different X-ray machine settings
    4. GaussNoise: Simulates image noise from different machines
    5. CLAHE: Enhances local contrast (common in medical imaging)
    6. RandomGamma: Handles exposure variations
    7. CoarseDropout: Regularization, forces model to use all features

    bbox_params ensures boxes are transformed along with the image
    and invalid boxes (out of frame, too small) are filtered out.
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.7
        ),

        # Intensity augmentations (critical for medical imaging)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.7),

        # Local contrast enhancement (SOTA for chest X-rays)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

        # Noise augmentation
        A.GaussNoise(var_limit=(10, 50), p=0.3),

        # Regularization via dropout
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size // 20,
            max_width=img_size // 20,
            min_holes=1,
            fill_value=0,
            p=0.3
        ),

        # Normalize to ImageNet stats (required for pretrained EfficientNetV2)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=1.0  # Our images are already [0, 1]
        ),

        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',  # [x_min, y_min, x_max, y_max]
        label_fields=['labels'],
        min_area=100,  # Filter tiny boxes
        min_visibility=0.3  # Keep boxes at least 30% visible after augmentation
    ))


def get_valid_transforms(img_size: int = 1024) -> A.Compose:
    """
    Validation/Test transforms - only normalization, no augmentation.

    We keep validation deterministic for reliable metric comparison.
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=1.0
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_area=0,
        min_visibility=0.0
    ))


# ==============================================================================
# CELL 6: MOSAIC AUGMENTATION (SOTA)
# ==============================================================================

class MosaicAugmentation:
    """
    Mosaic Augmentation: Stitches 4 images together.

    WHY MOSAIC?
    ===========
    1. Forces model to detect objects at various scales in one image
    2. Increases effective batch diversity (4 images combined)
    3. Provides natural regularization
    4. Originally from YOLOv4, now standard in SOTA detectors

    The 4 images are placed in a 2x2 grid with a random center point,
    creating varied object scales and positions.
    """

    def __init__(
        self,
        dataset: 'RSNADataset',
        img_size: int = 1024,
        p: float = 0.5
    ):
        self.dataset = dataset
        self.img_size = img_size
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mosaic augmentation with probability p.

        Args:
            image: Primary image
            boxes: Primary bounding boxes
            labels: Primary labels
            index: Index of primary image in dataset

        Returns:
            Mosaic image, combined boxes, combined labels
        """
        if np.random.random() > self.p:
            return image, boxes, labels

        # Get 3 additional random images
        indices = [index]
        while len(indices) < 4:
            idx = np.random.randint(0, len(self.dataset))
            if idx != index:
                indices.append(idx)

        # Random center point for mosaic
        cx = np.random.randint(self.img_size // 4, 3 * self.img_size // 4)
        cy = np.random.randint(self.img_size // 4, 3 * self.img_size // 4)

        # Create mosaic canvas
        mosaic_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        mosaic_boxes = []
        mosaic_labels = []

        # Positions: top-left, top-right, bottom-left, bottom-right
        positions = [
            (0, 0, cx, cy),           # top-left
            (cx, 0, self.img_size, cy),    # top-right
            (0, cy, cx, self.img_size),    # bottom-left
            (cx, cy, self.img_size, self.img_size)  # bottom-right
        ]

        for i, idx in enumerate(indices):
            # Get image and boxes (without augmentation)
            if i == 0:
                img, bxs, lbls = image, boxes, labels
            else:
                img, bxs, lbls = self.dataset.get_raw_item(idx)

            x1, y1, x2, y2 = positions[i]
            tile_w, tile_h = x2 - x1, y2 - y1

            # Resize image to fit tile
            img_resized = cv2.resize(img, (tile_w, tile_h))
            mosaic_img[y1:y2, x1:x2] = img_resized

            # Scale and shift boxes
            if len(bxs) > 0:
                scale_x = tile_w / self.img_size
                scale_y = tile_h / self.img_size

                bxs_scaled = bxs.copy()
                bxs_scaled[:, [0, 2]] = bxs_scaled[:, [0, 2]] * scale_x + x1
                bxs_scaled[:, [1, 3]] = bxs_scaled[:, [1, 3]] * scale_y + y1

                # Clip to mosaic bounds
                bxs_scaled[:, [0, 2]] = np.clip(bxs_scaled[:, [0, 2]], 0, self.img_size)
                bxs_scaled[:, [1, 3]] = np.clip(bxs_scaled[:, [1, 3]], 0, self.img_size)

                # Filter invalid boxes (too small after clipping)
                valid = (bxs_scaled[:, 2] - bxs_scaled[:, 0] > 10) & \
                        (bxs_scaled[:, 3] - bxs_scaled[:, 1] > 10)

                mosaic_boxes.append(bxs_scaled[valid])
                mosaic_labels.append(lbls[valid])

        # Combine all boxes and labels
        if mosaic_boxes:
            mosaic_boxes = np.concatenate(mosaic_boxes, axis=0)
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
        else:
            mosaic_boxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.array([], dtype=np.int64)

        return mosaic_img, mosaic_boxes, mosaic_labels


# ==============================================================================
# CELL 7: CUSTOM DATASET CLASS
# ==============================================================================

class RSNADataset(Dataset):
    """
    PyTorch Dataset for RSNA Pneumonia Detection Challenge.

    Handles:
    - DICOM file reading and conversion to RGB tensors
    - Multiple bounding boxes per image
    - Integration with albumentations for bbox-aware augmentation
    - Optional mosaic augmentation

    Target Format (for Faster R-CNN):
    {
        'boxes': FloatTensor[N, 4] - [x_min, y_min, x_max, y_max]
        'labels': Int64Tensor[N] - class labels (1 for pneumonia)
        'image_id': Int64Tensor[1] - unique image identifier
        'area': FloatTensor[N] - box areas
        'iscrowd': Int64Tensor[N] - 0 for all (no crowd annotations)
    }
    """

    def __init__(
        self,
        patient_ids: List[str],
        grouped_data: Dict[str, Dict],
        image_dir: str,
        transforms: Optional[A.Compose] = None,
        img_size: int = 1024,
        use_mosaic: bool = False,
        mosaic_prob: float = 0.5
    ):
        """
        Initialize the dataset.

        Args:
            patient_ids: List of patient IDs to include
            grouped_data: Dictionary from group_boxes_by_patient()
            image_dir: Directory containing DICOM files
            transforms: Albumentations transform pipeline
            img_size: Target image size
            use_mosaic: Whether to use mosaic augmentation
            mosaic_prob: Probability of applying mosaic
        """
        self.patient_ids = patient_ids
        self.grouped_data = grouped_data
        self.image_dir = image_dir
        self.transforms = transforms
        self.img_size = img_size
        self.use_mosaic = use_mosaic

        # Initialize mosaic augmentation if enabled
        self.mosaic = None
        if use_mosaic:
            self.mosaic = MosaicAugmentation(
                dataset=self,
                img_size=img_size,
                p=mosaic_prob
            )

    def __len__(self) -> int:
        return len(self.patient_ids)

    def get_raw_item(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get raw image and boxes without augmentation.
        Used by mosaic augmentation to fetch additional images.

        Returns:
            image: numpy array (H, W, 3)
            boxes: numpy array (N, 4)
            labels: numpy array (N,)
        """
        patient_id = self.patient_ids[idx]
        data = self.grouped_data[patient_id]

        # Load DICOM image
        dcm_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        image = read_dicom(dcm_path, normalize=True)

        # Get boxes and create labels
        boxes = data['boxes'].copy()

        # Resize if needed
        if image.shape[0] != self.img_size:
            image, boxes = resize_image_and_boxes(image, boxes, self.img_size)

        # Labels: all boxes are class 1 (pneumonia)
        labels = np.ones(len(boxes), dtype=np.int64)

        return image, boxes, labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            image: Tensor of shape (3, H, W)
            target: Dictionary with 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        """
        patient_id = self.patient_ids[idx]

        # Get raw image and boxes
        image, boxes, labels = self.get_raw_item(idx)

        # Apply mosaic augmentation if enabled
        if self.mosaic is not None and self.training_mode:
            image, boxes, labels = self.mosaic(image, boxes, labels, idx)

        # Apply albumentations transforms
        if self.transforms is not None:
            # Albumentations expects list of labels
            if len(boxes) > 0:
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes.tolist(),
                    labels=labels.tolist()
                )
                image = transformed['image']
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['labels'], dtype=np.int64)
            else:
                # No boxes - just transform the image
                transformed = self.transforms(
                    image=image,
                    bboxes=[],
                    labels=[]
                )
                image = transformed['image']
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.array([], dtype=np.int64)

        # Handle empty boxes (negative samples)
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
            area = np.array([], dtype=np.float32)
        else:
            # Calculate box areas
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
            'patient_id': patient_id  # For reference
        }

        return image, target

    @property
    def training_mode(self) -> bool:
        """Check if in training mode (for mosaic)."""
        return self.transforms is not None and hasattr(self.transforms, 'transforms')


# ==============================================================================
# CELL 8: COLLATE FUNCTION
# ==============================================================================

def collate_fn(batch: List[Tuple]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Custom collate function for object detection.

    Faster R-CNN expects:
    - List of images (not stacked tensor, as sizes may vary)
    - List of target dictionaries

    This is different from classification where we stack into a single tensor.
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets


# ==============================================================================
# CELL 9: DATA LOADING UTILITIES
# ==============================================================================

def prepare_data(config: Config = Config()) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Load and prepare all training data.

    Returns:
        patient_df: DataFrame with patient info and fold assignments
        grouped_data: Dictionary with grouped bounding boxes
    """
    print("Loading training labels...")
    df = pd.read_csv(config.TRAIN_LABELS)
    print(f"  Total rows: {len(df)}")

    print("\nGrouping bounding boxes by patient...")
    grouped_data = group_boxes_by_patient(df)
    print(f"  Unique patients: {len(grouped_data)}")

    print("\nCreating patient-level DataFrame...")
    patient_df = create_patient_df(grouped_data)

    # Class distribution
    pos_count = patient_df['target'].sum()
    neg_count = len(patient_df) - pos_count
    print(f"  Positive (pneumonia): {pos_count} ({100*pos_count/len(patient_df):.1f}%)")
    print(f"  Negative (no pneumonia): {neg_count} ({100*neg_count/len(patient_df):.1f}%)")

    print(f"\nCreating {config.N_FOLDS}-fold splits...")
    patient_df = create_folds(patient_df, n_folds=config.N_FOLDS, seed=config.SEED)

    # Verify fold distributions
    print("\nFold class distributions:")
    for fold in range(config.N_FOLDS):
        fold_df = patient_df[patient_df['fold'] == fold]
        fold_pos = fold_df['target'].sum()
        print(f"  Fold {fold}: {len(fold_df)} samples, "
              f"{fold_pos} positive ({100*fold_pos/len(fold_df):.1f}%)")

    return patient_df, grouped_data


def create_dataloaders(
    patient_df: pd.DataFrame,
    grouped_data: Dict[str, Dict],
    fold: int,
    config: Config = Config()
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for a specific fold.

    Args:
        patient_df: DataFrame with fold assignments
        grouped_data: Grouped bounding box data
        fold: Fold number to use as validation
        config: Configuration object

    Returns:
        train_loader, val_loader
    """
    # Get patient IDs for this fold
    train_patients, val_patients = get_fold_data(patient_df, grouped_data, fold)

    print(f"\nFold {fold}:")
    print(f"  Training patients: {len(train_patients)}")
    print(f"  Validation patients: {len(val_patients)}")

    # Create datasets
    train_dataset = RSNADataset(
        patient_ids=train_patients,
        grouped_data=grouped_data,
        image_dir=config.TRAIN_DIR,
        transforms=get_train_transforms(config.IMG_SIZE),
        img_size=config.IMG_SIZE,
        use_mosaic=True,  # Enable mosaic for training
        mosaic_prob=0.3   # 30% chance of mosaic
    )

    val_dataset = RSNADataset(
        patient_ids=val_patients,
        grouped_data=grouped_data,
        image_dir=config.TRAIN_DIR,
        transforms=get_valid_transforms(config.IMG_SIZE),
        img_size=config.IMG_SIZE,
        use_mosaic=False  # No mosaic for validation
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


# ==============================================================================
# CELL 10: TESTING & VISUALIZATION
# ==============================================================================

def visualize_sample(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    denormalize: bool = True
) -> np.ndarray:
    """
    Visualize a sample with bounding boxes.

    Args:
        image: Tensor of shape (3, H, W)
        target: Target dictionary with 'boxes'
        denormalize: Whether to reverse ImageNet normalization

    Returns:
        Image with drawn bounding boxes as numpy array
    """
    # Convert to numpy
    img = image.permute(1, 2, 0).numpy().copy()

    # Denormalize
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean

    # Clip to valid range and convert to uint8
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # Draw boxes
    boxes = target['boxes'].numpy()
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img


def test_data_pipeline():
    """
    Test the complete data pipeline.

    This function validates that:
    1. DICOM files are read correctly
    2. Bounding boxes are grouped properly
    3. Augmentations work with bboxes
    4. DataLoader produces valid batches
    """
    print("=" * 60)
    print("TESTING DATA PIPELINE")
    print("=" * 60)

    config = Config()

    # Check if data exists
    if not os.path.exists(config.TRAIN_LABELS):
        print(f"\nERROR: Data not found at {config.DATA_DIR}")
        print("Please download the RSNA Pneumonia Detection Challenge dataset")
        print("and place it in the ./data/rsna/ directory.")
        return None

    # Prepare data
    patient_df, grouped_data = prepare_data(config)

    # Create dataloaders for fold 0
    train_loader, val_loader = create_dataloaders(
        patient_df, grouped_data, fold=0, config=config
    )

    # Test a single batch
    print("\n" + "=" * 60)
    print("TESTING BATCH LOADING")
    print("=" * 60)

    images, targets = next(iter(train_loader))

    print(f"\nBatch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Image dtype: {images[0].dtype}")
    print(f"Image range: [{images[0].min():.3f}, {images[0].max():.3f}]")

    for i, target in enumerate(targets):
        print(f"\nTarget {i}:")
        print(f"  Patient ID: {target['patient_id']}")
        print(f"  Boxes shape: {target['boxes'].shape}")
        print(f"  Labels: {target['labels']}")
        if len(target['boxes']) > 0:
            print(f"  Box example: {target['boxes'][0]}")

    print("\n" + "=" * 60)
    print("DATA PIPELINE TEST COMPLETE")
    print("=" * 60)

    return train_loader, val_loader


# ==============================================================================
# CELL 11: MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Run the test
    result = test_data_pipeline()

    if result is not None:
        train_loader, val_loader = result
        print(f"\n✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches")
        print("\nData pipeline is ready for training!")
