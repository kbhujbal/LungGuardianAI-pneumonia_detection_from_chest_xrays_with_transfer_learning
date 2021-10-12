"""
Generate demo images for README.

This script creates sample visualization images showing:
1. detection_demo.png - Main demo showing detection pipeline
2. sample_input.png - Example input X-ray
3. sample_output.png - Example output with bounding boxes

Run this script after training to generate actual detection visualizations,
or use the synthetic examples for README demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def create_detection_demo():
    """Create the main detection demo image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('LungGuardian: Pneumonia Detection Pipeline', fontsize=16, fontweight='bold')

    # 1. Input Image (simulated X-ray)
    ax1 = axes[0]
    # Create a synthetic chest X-ray-like image
    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    X, Y = np.meshgrid(x, y)

    # Chest cavity shape
    chest = np.exp(-(X**2 + Y**2) / 0.5) * 0.7
    # Add some lung-like structures
    left_lung = np.exp(-((X+0.3)**2 + Y**2) / 0.15) * 0.3
    right_lung = np.exp(-((X-0.3)**2 + Y**2) / 0.15) * 0.3
    # Combine
    xray = chest - left_lung - right_lung
    # Add noise
    xray += np.random.randn(256, 256) * 0.05
    # Add pneumonia-like opacity
    pneumonia1 = np.exp(-((X+0.25)**2 + (Y-0.1)**2) / 0.02) * 0.4
    pneumonia2 = np.exp(-((X-0.35)**2 + (Y+0.15)**2) / 0.015) * 0.3
    xray += pneumonia1 + pneumonia2

    ax1.imshow(xray, cmap='bone')
    ax1.set_title('Input: Chest X-Ray', fontsize=12, fontweight='bold')
    ax1.axis('off')
    ax1.text(128, 245, 'DICOM Image (1024×1024)', ha='center', fontsize=9, color='white')

    # 2. Model Processing
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('LungGuardian Model', fontsize=12, fontweight='bold')

    # Draw processing blocks
    blocks = [
        (5, 8.5, 'EfficientNetV2-S\nBackbone', '#3498db'),
        (5, 6.5, 'Feature Pyramid\nNetwork (FPN)', '#2ecc71'),
        (5, 4.5, 'Region Proposal\nNetwork (RPN)', '#e74c3c'),
        (5, 2.5, 'Faster R-CNN\nDetection Head', '#9b59b6'),
        (5, 0.8, 'Weighted Box\nFusion (WBF)', '#f39c12'),
    ]

    for x, y, text, color in blocks:
        rect = FancyBboxPatch((x-2, y-0.6), 4, 1.2,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax2.add_patch(rect)
        ax2.text(x, y, text, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

    # Arrows
    for i in range(len(blocks)-1):
        ax2.annotate('', xy=(5, blocks[i+1][1]+0.6), xytext=(5, blocks[i][1]-0.6),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # 3. Output with detections
    ax3 = axes[2]
    ax3.imshow(xray, cmap='bone')
    ax3.set_title('Output: Detected Pneumonia', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Draw bounding boxes
    # Box 1 (left lung opacity)
    rect1 = patches.Rectangle((70, 90), 70, 80, linewidth=3,
                               edgecolor='#e74c3c', facecolor='none')
    ax3.add_patch(rect1)
    ax3.text(72, 85, 'Pneumonia 94.2%', fontsize=8, color='#e74c3c',
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Box 2 (right lung opacity)
    rect2 = patches.Rectangle((155, 115), 55, 60, linewidth=3,
                               edgecolor='#e74c3c', facecolor='none')
    ax3.add_patch(rect2)
    ax3.text(157, 110, 'Pneumonia 78.6%', fontsize=8, color='#e74c3c',
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('detection_demo.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: detection_demo.png")


def create_sample_input():
    """Create sample input X-ray image."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create synthetic X-ray
    x = np.linspace(-1, 1, 512)
    y = np.linspace(-1, 1, 512)
    X, Y = np.meshgrid(x, y)

    chest = np.exp(-(X**2 + Y**2) / 0.5) * 0.7
    left_lung = np.exp(-((X+0.3)**2 + Y**2) / 0.15) * 0.3
    right_lung = np.exp(-((X-0.3)**2 + Y**2) / 0.15) * 0.3
    xray = chest - left_lung - right_lung
    xray += np.random.randn(512, 512) * 0.03

    # Add pneumonia opacities
    pneumonia1 = np.exp(-((X+0.25)**2 + (Y-0.1)**2) / 0.02) * 0.4
    pneumonia2 = np.exp(-((X-0.35)**2 + (Y+0.15)**2) / 0.015) * 0.3
    xray += pneumonia1 + pneumonia2

    ax.imshow(xray, cmap='bone')
    ax.axis('off')
    ax.set_title('Input: Chest X-Ray', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('sample_input.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: sample_input.png")


def create_sample_output():
    """Create sample output with detections."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Same X-ray as input
    x = np.linspace(-1, 1, 512)
    y = np.linspace(-1, 1, 512)
    X, Y = np.meshgrid(x, y)

    chest = np.exp(-(X**2 + Y**2) / 0.5) * 0.7
    left_lung = np.exp(-((X+0.3)**2 + Y**2) / 0.15) * 0.3
    right_lung = np.exp(-((X-0.3)**2 + Y**2) / 0.15) * 0.3
    xray = chest - left_lung - right_lung
    xray += np.random.randn(512, 512) * 0.03
    pneumonia1 = np.exp(-((X+0.25)**2 + (Y-0.1)**2) / 0.02) * 0.4
    pneumonia2 = np.exp(-((X-0.35)**2 + (Y+0.15)**2) / 0.015) * 0.3
    xray += pneumonia1 + pneumonia2

    ax.imshow(xray, cmap='bone')
    ax.axis('off')
    ax.set_title('Output: Pneumonia Detected', fontsize=14, fontweight='bold', pad=10)

    # Detection boxes
    rect1 = patches.Rectangle((140, 175), 140, 160, linewidth=4,
                               edgecolor='#FF4444', facecolor='none')
    ax.add_patch(rect1)
    ax.text(142, 165, 'PNEUMONIA 94.2%', fontsize=10, color='white',
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='#FF4444', edgecolor='none', alpha=0.9))

    rect2 = patches.Rectangle((305, 225), 110, 120, linewidth=4,
                               edgecolor='#FF8800', facecolor='none')
    ax.add_patch(rect2)
    ax.text(307, 215, 'PNEUMONIA 78.6%', fontsize=10, color='white',
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='#FF8800', edgecolor='none', alpha=0.9))

    plt.tight_layout()
    plt.savefig('sample_output.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: sample_output.png")


def create_architecture_diagram():
    """Create a detailed architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('LungGuardian Architecture', fontsize=18, fontweight='bold', pad=20)

    # Input
    input_box = FancyBboxPatch((0.5, 4), 2, 2,
                                boxstyle="round,pad=0.05", facecolor='#ecf0f1',
                                edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 5, 'Input\nX-Ray\n1024×1024', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Backbone
    backbone_box = FancyBboxPatch((3.5, 3.5), 2.5, 3,
                                   boxstyle="round,pad=0.05", facecolor='#3498db',
                                   edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(backbone_box)
    ax.text(4.75, 5, 'EfficientNetV2-S\n\nStage 2: 64ch\nStage 3: 128ch\nStage 4: 160ch\nStage 5: 256ch',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # FPN
    fpn_box = FancyBboxPatch((6.5, 3.5), 2, 3,
                              boxstyle="round,pad=0.05", facecolor='#2ecc71',
                              edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(fpn_box)
    ax.text(7.5, 5, 'FPN\n\nP2, P3, P4\nP5, P6\n\n256 channels',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # RPN
    rpn_box = FancyBboxPatch((9, 5.5), 2, 1.5,
                              boxstyle="round,pad=0.05", facecolor='#e74c3c',
                              edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(rpn_box)
    ax.text(10, 6.25, 'RPN\nProposals',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # ROI Head
    roi_box = FancyBboxPatch((9, 3), 2, 1.5,
                              boxstyle="round,pad=0.05", facecolor='#9b59b6',
                              edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(roi_box)
    ax.text(10, 3.75, 'ROI Head\nFocal Loss',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # WBF
    wbf_box = FancyBboxPatch((11.5, 4), 2, 2,
                              boxstyle="round,pad=0.05", facecolor='#f39c12',
                              edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(wbf_box)
    ax.text(12.5, 5, 'WBF\nBox Fusion',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='#2c3e50', lw=2)
    ax.annotate('', xy=(3.5, 5), xytext=(2.5, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(6.5, 5), xytext=(6, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(9, 6.25), xytext=(8.5, 5.5), arrowprops=arrow_style)
    ax.annotate('', xy=(9, 3.75), xytext=(8.5, 4.5), arrowprops=arrow_style)
    ax.annotate('', xy=(11.5, 5), xytext=(11, 5.5), arrowprops=arrow_style)
    ax.annotate('', xy=(11.5, 5), xytext=(11, 4), arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: architecture_diagram.png")


if __name__ == "__main__":
    print("Generating demo images for README...")
    print("-" * 40)

    create_detection_demo()
    create_sample_input()
    create_sample_output()
    create_architecture_diagram()

    print("-" * 40)
    print("All images generated successfully!")
    print("\nImages created in current directory:")
    print("  - detection_demo.png")
    print("  - sample_input.png")
    print("  - sample_output.png")
    print("  - architecture_diagram.png")
