"""
Generate realistic chest X-ray demo images for README.

Creates synthetic but anatomically accurate chest X-ray visualizations
that closely resemble real RSNA dataset images.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Ellipse
from scipy.ndimage import gaussian_filter
import os

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def create_realistic_chest_xray(size=512, add_pneumonia=True, seed=42):
    """
    Create a realistic-looking chest X-ray image with proper anatomy.

    Simulates:
    - Lung fields (dark, air-filled)
    - Rib cage (bright, bony structures)
    - Heart shadow (left-center)
    - Spine (central vertical structure)
    - Clavicles (collar bones)
    - Diaphragm (curved bottom of lungs)
    - Pneumonia opacities (if enabled)
    """
    np.random.seed(seed)

    # Create base image (dark background representing air)
    img = np.zeros((size, size))

    # Coordinate grids
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2

    # Normalize coordinates to [-1, 1]
    xn = (x - cx) / (size // 2)
    yn = (y - cy) / (size // 2)

    # ===================
    # 1. CHEST CAVITY (overall thorax shape)
    # ===================
    # Oval shape for chest cavity
    chest_mask = ((xn / 0.85) ** 2 + ((yn + 0.1) / 1.0) ** 2) < 1
    img[chest_mask] = 0.3

    # ===================
    # 2. LUNG FIELDS (dark, air-filled areas)
    # ===================
    # Right lung (left side of image - radiological convention)
    right_lung = ((xn + 0.32) / 0.38) ** 2 + ((yn + 0.05) / 0.65) ** 2 < 1
    img[right_lung] = 0.08

    # Left lung (right side of image)
    left_lung = ((xn - 0.35) / 0.35) ** 2 + ((yn + 0.05) / 0.65) ** 2 < 1
    img[left_lung] = 0.08

    # ===================
    # 3. HEART SHADOW (mediastinum, left-center)
    # ===================
    heart = ((xn - 0.05) / 0.25) ** 2 + ((yn + 0.15) / 0.35) ** 2 < 1
    img[heart] = 0.45

    # Aortic arch (top of heart)
    aorta = ((xn - 0.08) / 0.12) ** 2 + ((yn + 0.45) / 0.15) ** 2 < 1
    img[aorta] = 0.5

    # ===================
    # 4. SPINE (central vertical structure)
    # ===================
    spine_width = 0.08
    spine = np.abs(xn) < spine_width
    # Create vertebrae pattern
    spine_pattern = 0.4 + 0.1 * np.sin(yn * 15)
    spine_mask = spine & chest_mask
    img[spine_mask] = np.broadcast_to(spine_pattern, img.shape)[spine_mask]

    # ===================
    # 5. RIB CAGE (curved bony structures)
    # ===================
    for i in range(12):
        rib_y = -0.6 + i * 0.11
        rib_curve = 0.15 * np.sin(np.pi * (yn - rib_y) / 0.08)

        # Right ribs
        right_rib = (np.abs(yn - rib_y - rib_curve * (xn + 0.3)) < 0.012) & \
                    (xn > -0.7) & (xn < -0.05) & right_lung
        img[right_rib] = 0.55 + np.random.rand() * 0.1

        # Left ribs
        left_rib = (np.abs(yn - rib_y + rib_curve * (xn - 0.3)) < 0.012) & \
                   (xn < 0.7) & (xn > 0.08) & left_lung
        img[left_rib] = 0.55 + np.random.rand() * 0.1

    # ===================
    # 6. CLAVICLES (collar bones at top)
    # ===================
    # Right clavicle
    clav_r = (np.abs(yn + 0.72 + 0.15 * (xn + 0.25) ** 2) < 0.02) & \
             (xn > -0.6) & (xn < 0.05)
    img[clav_r] = 0.7

    # Left clavicle
    clav_l = (np.abs(yn + 0.72 + 0.15 * (xn - 0.25) ** 2) < 0.02) & \
             (xn < 0.6) & (xn > -0.05)
    img[clav_l] = 0.7

    # ===================
    # 7. SCAPULAE (shoulder blades, faint)
    # ===================
    scap_r = ((xn + 0.55) / 0.2) ** 2 + ((yn + 0.3) / 0.4) ** 2 < 1
    img[scap_r & ~right_lung] = np.clip(img[scap_r & ~right_lung] + 0.1, 0, 1)

    scap_l = ((xn - 0.55) / 0.2) ** 2 + ((yn + 0.3) / 0.4) ** 2 < 1
    img[scap_l & ~left_lung] = np.clip(img[scap_l & ~left_lung] + 0.1, 0, 1)

    # ===================
    # 8. DIAPHRAGM (curved bottom of lungs)
    # ===================
    diaphragm_r = (yn > 0.35 - 0.2 * np.cos(np.pi * (xn + 0.3) / 0.5)) & right_lung
    img[diaphragm_r] = 0.35

    diaphragm_l = (yn > 0.45 - 0.15 * np.cos(np.pi * (xn - 0.35) / 0.45)) & left_lung
    img[diaphragm_l] = 0.35

    # ===================
    # 9. TRACHEA (airway, dark vertical line at top)
    # ===================
    trachea = (np.abs(xn) < 0.03) & (yn < -0.5) & (yn > -0.85)
    img[trachea] = 0.1

    # ===================
    # 10. PNEUMONIA OPACITIES (if enabled)
    # ===================
    pneumonia_regions = []
    if add_pneumonia:
        # Opacity 1: Right lower lobe (common location)
        px1, py1 = int(0.28 * size), int(0.58 * size)
        pw1, ph1 = int(0.18 * size), int(0.22 * size)

        opacity1 = np.zeros((size, size))
        opacity1_region = ((x - px1) / (pw1/2)) ** 2 + ((y - py1) / (ph1/2)) ** 2 < 1
        opacity1[opacity1_region] = 0.35
        opacity1 = gaussian_filter(opacity1, sigma=size//25)
        img = np.clip(img + opacity1, 0, 1)
        pneumonia_regions.append((px1 - pw1//2, py1 - ph1//2, pw1, ph1, 0.94))

        # Opacity 2: Left mid-zone
        px2, py2 = int(0.68 * size), int(0.45 * size)
        pw2, ph2 = int(0.14 * size), int(0.16 * size)

        opacity2 = np.zeros((size, size))
        opacity2_region = ((x - px2) / (pw2/2)) ** 2 + ((y - py2) / (ph2/2)) ** 2 < 1
        opacity2[opacity2_region] = 0.28
        opacity2 = gaussian_filter(opacity2, sigma=size//30)
        img = np.clip(img + opacity2, 0, 1)
        pneumonia_regions.append((px2 - pw2//2, py2 - ph2//2, pw2, ph2, 0.82))

    # ===================
    # 11. ADD REALISTIC NOISE AND TEXTURE
    # ===================
    # Fine noise (film grain)
    noise = np.random.randn(size, size) * 0.02
    img = np.clip(img + noise, 0, 1)

    # Smooth slightly for realistic appearance
    img = gaussian_filter(img, sigma=1)

    # Add subtle vignette (darker edges like real X-rays)
    vignette = 1 - 0.3 * (xn ** 2 + yn ** 2)
    img = img * np.clip(vignette, 0.7, 1)

    # ===================
    # 12. BORDER AND ANNOTATIONS (like real DICOM)
    # ===================
    # Add slight border
    border = 8
    img[:border, :] = 0
    img[-border:, :] = 0
    img[:, :border] = 0
    img[:, -border:] = 0

    return img, pneumonia_regions


def create_detection_demo():
    """Create the main detection demo image showing full pipeline."""
    fig = plt.figure(figsize=(16, 6))

    # Create gridspec for custom layout
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.8, 0.8, 1], wspace=0.1)

    # 1. Input Image
    ax1 = fig.add_subplot(gs[0])
    xray_input, _ = create_realistic_chest_xray(512, add_pneumonia=True, seed=42)
    ax1.imshow(xray_input, cmap='gray', vmin=0, vmax=0.8)
    ax1.set_title('Input: Chest X-Ray', fontsize=13, fontweight='bold', pad=10)
    ax1.axis('off')

    # Add DICOM-style annotations
    ax1.text(10, 25, 'PA', color='white', fontsize=10, fontweight='bold')
    ax1.text(10, 500, 'R', color='white', fontsize=12, fontweight='bold')
    ax1.text(480, 500, 'L', color='white', fontsize=12, fontweight='bold')

    # 2. Model Architecture (middle-left)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Model Architecture', fontsize=13, fontweight='bold', pad=10)

    # Draw processing blocks
    blocks = [
        (5, 10.5, 'EfficientNetV2-S', '#2980b9', 'Backbone'),
        (5, 8.5, 'FPN', '#27ae60', 'Multi-scale'),
        (5, 6.5, 'RPN', '#c0392b', 'Proposals'),
        (5, 4.5, 'ROI Heads', '#8e44ad', 'Detection'),
        (5, 2.5, 'Focal Loss', '#d35400', 'Training'),
        (5, 0.8, 'WBF', '#f39c12', 'Post-process'),
    ]

    for x, y, text, color, subtitle in blocks:
        rect = FancyBboxPatch((x-2.3, y-0.55), 4.6, 1.1,
                              boxstyle="round,pad=0.03,rounding_size=0.15",
                              facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.95)
        ax2.add_patch(rect)
        ax2.text(x, y+0.1, text, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')
        ax2.text(x, y-0.25, subtitle, ha='center', va='center', fontsize=7,
                color='white', alpha=0.9)

    # Arrows between blocks
    for i in range(len(blocks)-1):
        ax2.annotate('', xy=(5, blocks[i+1][1]+0.55), xytext=(5, blocks[i][1]-0.55),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

    # 3. Feature Maps (middle-right)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 12)
    ax3.axis('off')
    ax3.set_title('Feature Pyramid', fontsize=13, fontweight='bold', pad=10)

    # Draw pyramid levels
    pyramid_levels = [
        (5, 10, 3.5, 'P2 (128×128)', '#3498db'),
        (5, 8, 3.0, 'P3 (64×64)', '#2ecc71'),
        (5, 6, 2.5, 'P4 (32×32)', '#e74c3c'),
        (5, 4, 2.0, 'P5 (16×16)', '#9b59b6'),
        (5, 2, 1.5, 'P6 (8×8)', '#f39c12'),
    ]

    for x, y, w, label, color in pyramid_levels:
        rect = FancyBboxPatch((x-w/2, y-0.4), w, 0.8,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
        ax3.add_patch(rect)
        ax3.text(x, y, label, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

    # Draw pyramid lines
    ax3.plot([5-1.75, 5-0.75], [10-0.4, 2+0.4], 'k--', alpha=0.3, lw=1)
    ax3.plot([5+1.75, 5+0.75], [10-0.4, 2+0.4], 'k--', alpha=0.3, lw=1)

    # 4. Output with Detections
    ax4 = fig.add_subplot(gs[3])
    xray_output, pneumonia_boxes = create_realistic_chest_xray(512, add_pneumonia=True, seed=42)
    ax4.imshow(xray_output, cmap='gray', vmin=0, vmax=0.8)
    ax4.set_title('Output: Detected Pneumonia', fontsize=13, fontweight='bold', pad=10)
    ax4.axis('off')

    # Draw bounding boxes with labels
    colors = ['#FF4444', '#FF8800']
    for i, (bx, by, bw, bh, conf) in enumerate(pneumonia_boxes):
        rect = patches.Rectangle((bx, by), bw, bh, linewidth=3,
                                  edgecolor=colors[i], facecolor='none')
        ax4.add_patch(rect)

        # Label with confidence
        label = f'Pneumonia {conf*100:.1f}%'
        ax4.text(bx+2, by-8, label, fontsize=9, color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i],
                         edgecolor='none', alpha=0.9))

    # Add annotations
    ax4.text(10, 500, 'R', color='white', fontsize=12, fontweight='bold')
    ax4.text(480, 500, 'L', color='white', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('detection_demo.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print("Created: detection_demo.png")


def create_sample_input():
    """Create sample input X-ray image (no annotations)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    xray, _ = create_realistic_chest_xray(512, add_pneumonia=True, seed=42)
    ax.imshow(xray, cmap='gray', vmin=0, vmax=0.8)
    ax.axis('off')

    # DICOM-style annotations
    ax.text(15, 30, 'PA CHEST', color='white', fontsize=11, fontweight='bold')
    ax.text(15, 55, 'RSNA Dataset', color='white', fontsize=9, alpha=0.8)
    ax.text(15, 495, 'R', color='white', fontsize=14, fontweight='bold')
    ax.text(485, 495, 'L', color='white', fontsize=14, fontweight='bold')

    plt.tight_layout(pad=0)
    plt.savefig('sample_input.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("Created: sample_input.png")


def create_sample_output():
    """Create sample output with detection boxes."""
    fig, ax = plt.subplots(figsize=(8, 8))

    xray, pneumonia_boxes = create_realistic_chest_xray(512, add_pneumonia=True, seed=42)
    ax.imshow(xray, cmap='gray', vmin=0, vmax=0.8)
    ax.axis('off')

    # Draw bounding boxes
    colors = ['#FF4444', '#FF8800']
    for i, (bx, by, bw, bh, conf) in enumerate(pneumonia_boxes):
        # Main box
        rect = patches.Rectangle((bx, by), bw, bh, linewidth=3,
                                  edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)

        # Corner markers for style
        corner_len = 15
        for cx, cy in [(bx, by), (bx+bw, by), (bx, by+bh), (bx+bw, by+bh)]:
            pass  # Can add corner markers if desired

        # Label
        label = f'PNEUMONIA {conf*100:.1f}%'
        ax.text(bx+3, by-10, label, fontsize=10, color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i],
                         edgecolor='none', alpha=0.95))

    # Annotations
    ax.text(15, 30, 'DETECTION OUTPUT', color='#00FF00', fontsize=11, fontweight='bold')
    ax.text(15, 55, f'{len(pneumonia_boxes)} opacities detected', color='#00FF00', fontsize=9)
    ax.text(15, 495, 'R', color='white', fontsize=14, fontweight='bold')
    ax.text(485, 495, 'L', color='white', fontsize=14, fontweight='bold')

    plt.tight_layout(pad=0)
    plt.savefig('sample_output.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("Created: sample_output.png")


def create_comparison_image():
    """Create side-by-side comparison of classification vs detection."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    xray, pneumonia_boxes = create_realistic_chest_xray(512, add_pneumonia=True, seed=42)

    # Classification output (left)
    ax1 = axes[0]
    ax1.imshow(xray, cmap='gray', vmin=0, vmax=0.8)
    ax1.set_title('Image Classification', fontsize=14, fontweight='bold', color='white', pad=15)
    ax1.axis('off')

    # Single label overlay
    ax1.text(256, 480, 'Pneumonia: 87%', ha='center', fontsize=16,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.9))
    ax1.text(256, 30, '(Single label - no localization)', ha='center',
            fontsize=10, color='#aaa')

    # Detection output (right)
    ax2 = axes[1]
    ax2.imshow(xray, cmap='gray', vmin=0, vmax=0.8)
    ax2.set_title('Object Detection (LungGuardian)', fontsize=14, fontweight='bold', color='white', pad=15)
    ax2.axis('off')

    # Draw boxes
    colors = ['#FF4444', '#FF8800']
    for i, (bx, by, bw, bh, conf) in enumerate(pneumonia_boxes):
        rect = patches.Rectangle((bx, by), bw, bh, linewidth=3,
                                  edgecolor=colors[i], facecolor='none')
        ax2.add_patch(rect)
        ax2.text(bx+3, by-10, f'Pneumonia {conf*100:.1f}%', fontsize=10,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.9))

    ax2.text(256, 30, '(Precise bounding boxes + confidence)', ha='center',
            fontsize=10, color='#aaa')

    plt.tight_layout()
    plt.savefig('classification_vs_detection.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print("Created: classification_vs_detection.png")


def create_augmentation_demo():
    """Create demo showing data augmentation effects."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Data Augmentation Pipeline', fontsize=16, fontweight='bold', color='white', y=0.98)

    titles = [
        'Original', 'Horizontal Flip', 'Rotation (15°)', 'Brightness+',
        'CLAHE Enhanced', 'Shift+Scale', 'Noise Added', 'Mosaic (4 images)'
    ]

    for idx, (ax, title) in enumerate(zip(axes.flat, titles)):
        seed = 42 + idx
        xray, _ = create_realistic_chest_xray(256, add_pneumonia=True, seed=seed)

        # Apply different transformations for visualization
        if idx == 1:  # Flip
            xray = np.fliplr(xray)
        elif idx == 2:  # Rotation
            from scipy.ndimage import rotate
            xray = rotate(xray, 15, reshape=False, mode='constant', cval=0)
        elif idx == 3:  # Brightness
            xray = np.clip(xray * 1.3, 0, 1)
        elif idx == 4:  # CLAHE-like
            xray = np.clip((xray - xray.min()) / (xray.max() - xray.min() + 0.001), 0, 1)
        elif idx == 5:  # Shift
            xray = np.roll(xray, 20, axis=0)
            xray = np.roll(xray, -15, axis=1)
        elif idx == 6:  # Noise
            xray = np.clip(xray + np.random.randn(*xray.shape) * 0.08, 0, 1)
        elif idx == 7:  # Mosaic
            # Create 4 small images
            mosaic = np.zeros((256, 256))
            for i in range(2):
                for j in range(2):
                    small_xray, _ = create_realistic_chest_xray(128, add_pneumonia=True,
                                                                 seed=42+i*2+j)
                    mosaic[i*128:(i+1)*128, j*128:(j+1)*128] = small_xray
            xray = mosaic

        ax.imshow(xray, cmap='gray', vmin=0, vmax=0.8)
        ax.set_title(title, fontsize=11, color='white', pad=5)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_demo.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print("Created: augmentation_demo.png")


if __name__ == "__main__":
    print("=" * 50)
    print("Generating Realistic Chest X-Ray Demo Images")
    print("=" * 50)

    create_detection_demo()
    create_sample_input()
    create_sample_output()
    create_comparison_image()
    create_augmentation_demo()

    print("=" * 50)
    print("All images generated successfully!")
    print("\nImages created:")
    print("  ✓ detection_demo.png        - Main pipeline visualization")
    print("  ✓ sample_input.png          - Input X-ray example")
    print("  ✓ sample_output.png         - Detection output example")
    print("  ✓ classification_vs_detection.png - Comparison diagram")
    print("  ✓ augmentation_demo.png     - Data augmentation examples")
    print("=" * 50)
