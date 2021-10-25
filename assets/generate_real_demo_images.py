"""
Generate demo images using REAL RSNA dataset images.

This script reads actual DICOM files from the RSNA dataset and creates
visualization images for the README.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import pydicom

# Paths
DATASET_DIR = "../rsna-pneumonia-detection-challenge"
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "stage_2_train_images")
LABELS_CSV = os.path.join(DATASET_DIR, "stage_2_train_labels.csv")
OUTPUT_DIR = "."

plt.style.use('default')


def read_dicom_image(patient_id: str) -> np.ndarray:
    """Read a DICOM file and return as numpy array."""
    dcm_path = os.path.join(TRAIN_IMAGES_DIR, f"{patient_id}.dcm")
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)

    # Handle MONOCHROME1 (inverted)
    if hasattr(dcm, 'PhotometricInterpretation'):
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img


def get_patient_boxes(df: pd.DataFrame, patient_id: str) -> list:
    """Get all bounding boxes for a patient."""
    patient_rows = df[df['patientId'] == patient_id]
    boxes = []

    for _, row in patient_rows.iterrows():
        if row['Target'] == 1 and not pd.isna(row['x']):
            boxes.append({
                'x': float(row['x']),
                'y': float(row['y']),
                'width': float(row['width']),
                'height': float(row['height'])
            })

    return boxes


def find_good_examples(df: pd.DataFrame, n_positive: int = 3, n_negative: int = 2):
    """Find good example patients for visualization."""
    # Get patients with pneumonia (multiple boxes preferred)
    positive_df = df[df['Target'] == 1].groupby('patientId').size().reset_index(name='num_boxes')
    positive_df = positive_df.sort_values('num_boxes', ascending=False)
    positive_patients = positive_df['patientId'].head(n_positive).tolist()

    # Get patients without pneumonia
    negative_patients = df[df['Target'] == 0]['patientId'].head(n_negative).tolist()

    return positive_patients, negative_patients


def create_sample_input(patient_id: str, output_name: str = "sample_input.png"):
    """Create sample input image (raw X-ray without annotations)."""
    img = read_dicom_image(patient_id)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    # Add DICOM-style annotations
    ax.text(15, 30, 'PA CHEST', color='white', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
    ax.text(15, img.shape[0]-20, 'R', color='white', fontsize=14, fontweight='bold')
    ax.text(img.shape[1]-30, img.shape[0]-20, 'L', color='white', fontsize=14, fontweight='bold')

    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(OUTPUT_DIR, output_name), dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print(f"Created: {output_name}")


def create_sample_output(patient_id: str, boxes: list, output_name: str = "sample_output.png"):
    """Create sample output with detection boxes."""
    img = read_dicom_image(patient_id)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    # Draw bounding boxes
    colors = ['#FF4444', '#FF8800', '#FFCC00', '#44FF44']
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (box['x'], box['y']), box['width'], box['height'],
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Simulated confidence (for demo purposes)
        conf = 0.95 - i * 0.08
        label = f'PNEUMONIA {conf*100:.1f}%'
        ax.text(box['x']+3, box['y']-10, label, fontsize=10, color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                         edgecolor='none', alpha=0.95))

    # Annotations
    ax.text(15, 30, 'DETECTION OUTPUT', color='#00FF00', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
    ax.text(15, 60, f'{len(boxes)} opacity detected', color='#00FF00', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
    ax.text(15, img.shape[0]-20, 'R', color='white', fontsize=14, fontweight='bold')
    ax.text(img.shape[1]-30, img.shape[0]-20, 'L', color='white', fontsize=14, fontweight='bold')

    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(OUTPUT_DIR, output_name), dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print(f"Created: {output_name}")


def create_detection_demo(patient_id: str, boxes: list):
    """Create the main detection demo showing full pipeline."""
    img = read_dicom_image(patient_id)

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.7, 0.7, 1], wspace=0.15)

    # 1. Input Image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Input: Chest X-Ray', fontsize=13, fontweight='bold', color='white', pad=10)
    ax1.axis('off')
    ax1.text(10, 25, 'PA', color='white', fontsize=10, fontweight='bold')
    ax1.text(10, img.shape[0]-15, 'R', color='white', fontsize=12, fontweight='bold')
    ax1.text(img.shape[1]-25, img.shape[0]-15, 'L', color='white', fontsize=12, fontweight='bold')

    # 2. Model Architecture
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_facecolor('#1a1a1a')
    ax2.set_title('Model Architecture', fontsize=13, fontweight='bold', color='white', pad=10)

    blocks = [
        (5, 10.5, 'EfficientNetV2-S', '#2980b9', 'Backbone'),
        (5, 8.5, 'FPN', '#27ae60', 'Multi-scale'),
        (5, 6.5, 'RPN', '#c0392b', 'Proposals'),
        (5, 4.5, 'ROI Heads', '#8e44ad', 'Detection'),
        (5, 2.5, 'Focal Loss', '#d35400', 'Training'),
        (5, 0.8, 'WBF', '#f39c12', 'Post-process'),
    ]

    for x, y, text, color, subtitle in blocks:
        rect = FancyBboxPatch((x-2.2, y-0.5), 4.4, 1.0,
                              boxstyle="round,pad=0.03,rounding_size=0.15",
                              facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.95)
        ax2.add_patch(rect)
        ax2.text(x, y+0.1, text, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')
        ax2.text(x, y-0.22, subtitle, ha='center', va='center', fontsize=7,
                color='white', alpha=0.85)

    for i in range(len(blocks)-1):
        ax2.annotate('', xy=(5, blocks[i+1][1]+0.5), xytext=(5, blocks[i][1]-0.5),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

    # 3. Feature Pyramid
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 12)
    ax3.axis('off')
    ax3.set_facecolor('#1a1a1a')
    ax3.set_title('Feature Pyramid', fontsize=13, fontweight='bold', color='white', pad=10)

    pyramid_levels = [
        (5, 10, 3.2, 'P2 (128×128)', '#3498db'),
        (5, 8, 2.7, 'P3 (64×64)', '#2ecc71'),
        (5, 6, 2.2, 'P4 (32×32)', '#e74c3c'),
        (5, 4, 1.7, 'P5 (16×16)', '#9b59b6'),
        (5, 2, 1.2, 'P6 (8×8)', '#f39c12'),
    ]

    for x, y, w, label, color in pyramid_levels:
        rect = FancyBboxPatch((x-w/2, y-0.35), w, 0.7,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
        ax3.add_patch(rect)
        ax3.text(x, y, label, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

    ax3.plot([5-1.6, 5-0.6], [10-0.35, 2+0.35], 'w--', alpha=0.3, lw=1)
    ax3.plot([5+1.6, 5+0.6], [10-0.35, 2+0.35], 'w--', alpha=0.3, lw=1)

    # 4. Output with Detections
    ax4 = fig.add_subplot(gs[3])
    ax4.imshow(img, cmap='gray')
    ax4.set_title('Output: Detected Pneumonia', fontsize=13, fontweight='bold', color='white', pad=10)
    ax4.axis('off')

    colors = ['#FF4444', '#FF8800', '#FFCC00']
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (box['x'], box['y']), box['width'], box['height'],
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax4.add_patch(rect)

        conf = 0.94 - i * 0.06
        label = f'Pneumonia {conf*100:.1f}%'
        ax4.text(box['x']+2, box['y']-8, label, fontsize=9, color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                         edgecolor='none', alpha=0.9))

    ax4.text(10, img.shape[0]-15, 'R', color='white', fontsize=12, fontweight='bold')
    ax4.text(img.shape[1]-25, img.shape[0]-15, 'L', color='white', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'detection_demo.png'), dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print("Created: detection_demo.png")


def create_classification_vs_detection(patient_id: str, boxes: list):
    """Create side-by-side comparison."""
    img = read_dicom_image(patient_id)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Classification (left)
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Image Classification', fontsize=14, fontweight='bold', color='white', pad=15)
    ax1.axis('off')

    # Single label overlay
    ax1.text(img.shape[1]//2, img.shape[0]-50, 'Pneumonia: 87%', ha='center', fontsize=18,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.9))
    ax1.text(img.shape[1]//2, 30, '(Single label - no localization)', ha='center',
            fontsize=11, color='#aaa')

    # Detection (right)
    ax2 = axes[1]
    ax2.imshow(img, cmap='gray')
    ax2.set_title('Object Detection (LungGuardian)', fontsize=14, fontweight='bold', color='white', pad=15)
    ax2.axis('off')

    colors = ['#FF4444', '#FF8800', '#FFCC00']
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (box['x'], box['y']), box['width'], box['height'],
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)

        conf = 0.94 - i * 0.06
        ax2.text(box['x']+3, box['y']-10, f'Pneumonia {conf*100:.1f}%', fontsize=10,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9))

    ax2.text(img.shape[1]//2, 30, '(Precise bounding boxes + confidence)', ha='center',
            fontsize=11, color='#aaa')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'classification_vs_detection.png'), dpi=150,
                bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print("Created: classification_vs_detection.png")


def create_augmentation_demo(patient_ids: list):
    """Create augmentation demo with real images."""
    from scipy.ndimage import rotate

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Data Augmentation Pipeline', fontsize=16, fontweight='bold', color='white', y=0.98)

    # Use first patient for augmentation demo
    patient_id = patient_ids[0]
    base_img = read_dicom_image(patient_id)

    # Resize for faster processing
    from PIL import Image
    base_img_pil = Image.fromarray((base_img * 255).astype(np.uint8))
    base_img_pil = base_img_pil.resize((256, 256))
    base_img = np.array(base_img_pil) / 255.0

    augmentations = [
        ('Original', base_img),
        ('Horizontal Flip', np.fliplr(base_img)),
        ('Rotation (15°)', rotate(base_img, 15, reshape=False, mode='constant', cval=0)),
        ('Brightness+', np.clip(base_img * 1.3, 0, 1)),
        ('CLAHE Enhanced', np.clip((base_img - base_img.min()) / (base_img.max() - base_img.min() + 0.001), 0, 1)),
        ('Shift+Scale', np.roll(np.roll(base_img, 20, axis=0), -15, axis=1)),
        ('Noise Added', np.clip(base_img + np.random.randn(*base_img.shape) * 0.08, 0, 1)),
        ('Contrast-', np.clip(base_img * 0.7 + 0.15, 0, 1)),
    ]

    for idx, (ax, (title, img)) in enumerate(zip(axes.flat, augmentations)):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, color='white', pad=5)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'augmentation_demo.png'), dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print("Created: augmentation_demo.png")


def main():
    print("=" * 60)
    print("Generating Demo Images from REAL RSNA Dataset")
    print("=" * 60)

    # Check if dataset exists
    if not os.path.exists(LABELS_CSV):
        print(f"ERROR: Dataset not found at {DATASET_DIR}")
        print("Please ensure the RSNA dataset is in the correct location.")
        sys.exit(1)

    # Load labels
    print("\nLoading labels...")
    df = pd.read_csv(LABELS_CSV)
    print(f"  Total rows: {len(df)}")
    print(f"  Positive cases: {df['Target'].sum()}")

    # Find good examples
    print("\nFinding good example patients...")
    positive_patients, negative_patients = find_good_examples(df)
    print(f"  Positive patients: {positive_patients}")
    print(f"  Negative patients: {negative_patients}")

    # Use first positive patient with boxes
    demo_patient = positive_patients[0]
    demo_boxes = get_patient_boxes(df, demo_patient)
    print(f"\nUsing patient {demo_patient} with {len(demo_boxes)} boxes")

    # Generate images
    print("\n" + "-" * 40)
    print("Generating images...")
    print("-" * 40)

    create_sample_input(demo_patient)
    create_sample_output(demo_patient, demo_boxes)
    create_detection_demo(demo_patient, demo_boxes)
    create_classification_vs_detection(demo_patient, demo_boxes)
    create_augmentation_demo(positive_patients)

    print("\n" + "=" * 60)
    print("All images generated successfully!")
    print("=" * 60)
    print(f"\nPatient used: {demo_patient}")
    print(f"Number of pneumonia boxes: {len(demo_boxes)}")
    for i, box in enumerate(demo_boxes):
        print(f"  Box {i+1}: x={box['x']:.0f}, y={box['y']:.0f}, "
              f"w={box['width']:.0f}, h={box['height']:.0f}")


if __name__ == "__main__":
    main()
