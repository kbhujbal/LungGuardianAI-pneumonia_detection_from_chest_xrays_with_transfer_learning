"""
================================================================================
SOTA PNEUMONIA DETECTION SYSTEM - MODULE 4: TRAINING ENGINE
================================================================================
This module contains the complete training pipeline:
1. Training and validation loops
2. Mixed precision training (AMP)
3. Learning rate scheduling
4. Model checkpointing
5. Logging and monitoring
================================================================================
"""

# ==============================================================================
# CELL 1: IMPORTS
# ==============================================================================

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Import our modules
from notebooks.data_pipeline_01 import (
    Config, RSNADataset, prepare_data, create_dataloaders,
    get_train_transforms, get_valid_transforms, collate_fn,
    set_seed, get_fold_data
)
from notebooks.model_architecture_02 import (
    PneumoniaDetector, create_efficientnetv2_s_fasterrcnn
)
from notebooks.losses_and_utilities_03 import (
    FocalLoss, FocalLossWithLogits, focal_loss_for_rpn,
    apply_wbf, calculate_map, calculate_iou_matrix,
    EarlyStopping, AverageMeter
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CELL 2: TRAINING CONFIGURATION
# ==============================================================================

class TrainConfig(Config):
    """Extended configuration for training."""

    # Training hyperparameters
    EPOCHS = 15
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS

    # Learning rates (discriminative)
    LR_BACKBONE = 1e-5
    LR_FPN = 5e-5
    LR_HEAD = 1e-4

    # Optimizer settings
    WEIGHT_DECAY = 0.01
    BETAS = (0.9, 0.999)

    # Scheduler settings
    SCHEDULER = 'cosine_warmup'  # 'cosine_warmup', 'reduce_on_plateau', 'one_cycle'
    WARMUP_EPOCHS = 2
    MIN_LR = 1e-7

    # Mixed precision
    USE_AMP = True

    # Early stopping
    PATIENCE = 5
    MIN_DELTA = 0.001

    # Checkpointing
    SAVE_DIR = './checkpoints'
    SAVE_BEST_ONLY = True

    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    EVAL_INTERVAL = 1  # Evaluate every N epochs

    # WBF settings for validation
    WBF_IOU_THR = 0.5
    WBF_SKIP_THR = 0.001

    # Validation mAP thresholds
    MAP_IOU_THRESHOLDS = [0.4, 0.45, 0.5, 0.55, 0.6]


# ==============================================================================
# CELL 3: TRAINING UTILITIES
# ==============================================================================

def setup_directories(config: TrainConfig) -> Path:
    """Create necessary directories for checkpoints and logs."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(config.SAVE_DIR) / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)

    return run_dir


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict,
    path: Path,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }

    # Save regular checkpoint
    torch.save(checkpoint, path / f'checkpoint_epoch_{epoch}.pt')

    # Save best model separately
    if is_best:
        torch.save(checkpoint, path / 'best_model.pt')
        logger.info(f"Saved best model with mAP: {metrics.get('mAP', 0):.4f}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    path: Path
) -> Tuple[int, Dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint.get('metrics', {})


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ==============================================================================
# CELL 4: LEARNING RATE SCHEDULER FACTORY
# ==============================================================================

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler based on configuration.

    Options:
    1. CosineAnnealingWarmRestarts: Good for long training, handles restarts
    2. ReduceLROnPlateau: Adaptive, reduces LR when validation stalls
    3. OneCycleLR: SOTA scheduler, warm-up + annealing in one cycle

    For detection tasks, OneCycleLR or Cosine with warm-up work best.
    """
    total_steps = steps_per_epoch * config.EPOCHS
    warmup_steps = steps_per_epoch * config.WARMUP_EPOCHS

    if config.SCHEDULER == 'cosine_warmup':
        # Custom cosine with linear warmup
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return max(
                    config.MIN_LR / config.LR_HEAD,
                    0.5 * (1 + np.cos(np.pi * progress))
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.SCHEDULER == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize mAP
            factor=0.5,
            patience=2,
            min_lr=config.MIN_LR,
            verbose=True
        )

    elif config.SCHEDULER == 'one_cycle':
        # OneCycleLR: SOTA scheduler
        # Warm-up to max LR then anneal down
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[config.LR_BACKBONE * 10, config.LR_FPN * 10, config.LR_HEAD * 10],
            epochs=config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=10,  # Initial LR = max_lr / 10
            final_div_factor=100  # Final LR = max_lr / 100
        )

    else:
        raise ValueError(f"Unknown scheduler: {config.SCHEDULER}")

    return scheduler


# ==============================================================================
# CELL 5: TRAIN ONE EPOCH
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    scaler: Optional[GradScaler] = None
) -> Dict[str, float]:
    """
    Train for one epoch.

    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation for larger effective batch size
    - Gradient clipping for stability
    - Detailed loss logging

    Args:
        model: Detection model
        train_loader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Training device
        epoch: Current epoch number
        config: Training configuration
        scaler: GradScaler for AMP

    Returns:
        Dictionary of average losses for the epoch
    """
    model.train()

    # Loss meters
    loss_meters = {
        'loss_classifier': AverageMeter(),
        'loss_box_reg': AverageMeter(),
        'loss_objectness': AverageMeter(),
        'loss_rpn_box_reg': AverageMeter(),
        'total_loss': AverageMeter()
    }

    # Progress bar
    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f'Epoch {epoch}/{config.EPOCHS} [Train]'
    )

    optimizer.zero_grad()

    for batch_idx, (images, targets) in pbar:
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in targets]

        # Skip if no valid targets (all images have no boxes)
        valid_targets = [t for t in targets if len(t['boxes']) > 0]
        if len(valid_targets) == 0 and batch_idx % 5 != 0:
            # Allow some negative-only batches but not too many
            continue

        # Forward pass with mixed precision
        with autocast(enabled=config.USE_AMP):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Scale for gradient accumulation
            losses = losses / config.ACCUMULATION_STEPS

        # Backward pass
        if config.USE_AMP and scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        # Gradient accumulation
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
            if config.USE_AMP and scaler is not None:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            optimizer.zero_grad()

            # Step scheduler (if per-batch scheduler like OneCycleLR)
            if config.SCHEDULER in ['one_cycle', 'cosine_warmup']:
                scheduler.step()

        # Update loss meters
        batch_size = len(images)
        for key in loss_dict:
            if key in loss_meters:
                loss_meters[key].update(loss_dict[key].item(), batch_size)
        loss_meters['total_loss'].update(
            losses.item() * config.ACCUMULATION_STEPS, batch_size
        )

        # Update progress bar
        if batch_idx % config.LOG_INTERVAL == 0:
            pbar.set_postfix({
                'loss': f"{loss_meters['total_loss'].avg:.4f}",
                'cls': f"{loss_meters['loss_classifier'].avg:.4f}",
                'box': f"{loss_meters['loss_box_reg'].avg:.4f}",
                'lr': f"{get_lr(optimizer):.2e}"
            })

    return {k: v.avg for k, v in loss_meters.items()}


# ==============================================================================
# CELL 6: VALIDATION
# ==============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    config: TrainConfig
) -> Tuple[Dict[str, float], List[Dict], List[Dict]]:
    """
    Validate the model.

    Computes:
    1. Validation loss (in training mode to get losses)
    2. mAP metrics (in eval mode with WBF post-processing)

    Args:
        model: Detection model
        val_loader: Validation dataloader
        device: Device
        epoch: Current epoch
        config: Training configuration

    Returns:
        Tuple of (metrics_dict, all_predictions, all_ground_truths)
    """
    # First pass: compute validation loss (training mode)
    model.train()
    loss_meters = {
        'loss_classifier': AverageMeter(),
        'loss_box_reg': AverageMeter(),
        'loss_objectness': AverageMeter(),
        'loss_rpn_box_reg': AverageMeter(),
        'total_loss': AverageMeter()
    }

    for images, targets in tqdm(val_loader, desc=f'Epoch {epoch} [Val Loss]'):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in targets]

        # Skip empty batches for loss computation
        valid_targets = [t for t in targets if len(t['boxes']) > 0]
        if len(valid_targets) == 0:
            continue

        with autocast(enabled=config.USE_AMP):
            loss_dict = model(images, targets)

        batch_size = len(images)
        for key in loss_dict:
            if key in loss_meters:
                loss_meters[key].update(loss_dict[key].item(), batch_size)
        loss_meters['total_loss'].update(
            sum(loss_dict.values()).item(), batch_size
        )

    # Second pass: compute mAP (evaluation mode)
    model.eval()
    all_predictions = []
    all_ground_truths = []

    for images, targets in tqdm(val_loader, desc=f'Epoch {epoch} [Val mAP]'):
        images = [img.to(device) for img in images]

        with autocast(enabled=config.USE_AMP):
            predictions = model(images)

        # Apply WBF to predictions
        predictions_wbf = apply_wbf(
            predictions,
            image_size=config.IMG_SIZE,
            iou_thr=config.WBF_IOU_THR,
            skip_box_thr=config.WBF_SKIP_THR
        )

        # Collect predictions and ground truths
        for pred, target in zip(predictions_wbf, targets):
            all_predictions.append({
                'boxes': pred['boxes'],
                'scores': pred['scores'],
                'labels': pred['labels']
            })
            all_ground_truths.append({
                'boxes': target['boxes'],
                'labels': target['labels']
            })

    # Calculate mAP
    map_results = calculate_map(
        all_predictions,
        all_ground_truths,
        iou_thresholds=config.MAP_IOU_THRESHOLDS
    )

    # Combine all metrics
    metrics = {k: v.avg for k, v in loss_meters.items()}
    metrics.update(map_results)

    return metrics, all_predictions, all_ground_truths


# ==============================================================================
# CELL 7: MAIN TRAINING LOOP
# ==============================================================================

def train(
    fold: int = 0,
    config: TrainConfig = TrainConfig(),
    resume_path: Optional[str] = None
) -> Dict:
    """
    Main training function.

    Implements complete training pipeline with:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Detailed logging

    Args:
        fold: Which fold to use for validation
        config: Training configuration
        resume_path: Path to checkpoint to resume from

    Returns:
        Dictionary with training history
    """
    # Setup
    set_seed(config.SEED)
    run_dir = setup_directories(config)

    # Save config
    with open(run_dir / 'config.json', 'w') as f:
        json.dump({k: v for k, v in vars(config).items()
                  if not k.startswith('_')}, f, indent=2, default=str)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ====================
    # DATA
    # ====================
    logger.info("Preparing data...")
    patient_df, grouped_data = prepare_data(config)

    train_loader, val_loader = create_dataloaders(
        patient_df, grouped_data, fold=fold, config=config
    )

    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # ====================
    # MODEL
    # ====================
    logger.info("Creating model...")
    model = create_efficientnetv2_s_fasterrcnn(
        num_classes=2,
        pretrained=True
    )
    model = model.to(device)

    # Freeze BatchNorm for small batch sizes
    model.freeze_bn()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ====================
    # OPTIMIZER
    # ====================
    logger.info("Setting up optimizer...")

    # Discriminative learning rates
    param_groups = model.get_trainable_parameters(
        lr_backbone=config.LR_BACKBONE,
        lr_fpn=config.LR_FPN,
        lr_head=config.LR_HEAD
    )

    optimizer = AdamW(
        param_groups,
        weight_decay=config.WEIGHT_DECAY,
        betas=config.BETAS
    )

    # ====================
    # SCHEDULER
    # ====================
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    logger.info(f"Using scheduler: {config.SCHEDULER}")

    # ====================
    # MIXED PRECISION
    # ====================
    scaler = GradScaler() if config.USE_AMP else None
    logger.info(f"Mixed precision training: {config.USE_AMP}")

    # ====================
    # RESUME
    # ====================
    start_epoch = 0
    if resume_path:
        logger.info(f"Resuming from {resume_path}")
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, Path(resume_path))
        start_epoch += 1

    # ====================
    # TRAINING LOOP
    # ====================
    early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)
    best_map = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_map': [],
        'learning_rates': []
    }

    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start = time.time()

        # Training
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch + 1,
            config=config,
            scaler=scaler
        )

        # Validation
        if (epoch + 1) % config.EVAL_INTERVAL == 0:
            val_metrics, _, _ = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch + 1,
                config=config
            )
        else:
            val_metrics = {'total_loss': 0, 'mAP': 0}

        # Update scheduler (for ReduceLROnPlateau)
        if config.SCHEDULER == 'reduce_on_plateau':
            scheduler.step(val_metrics['mAP'])

        # Logging
        epoch_time = time.time() - epoch_start
        current_lr = get_lr(optimizer)

        logger.info(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} | "
            f"Val Loss: {val_metrics['total_loss']:.4f} | "
            f"Val mAP: {val_metrics['mAP']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Detailed metrics
        logger.info(
            f"  Classifier: {train_metrics['loss_classifier']:.4f} | "
            f"Box Reg: {train_metrics['loss_box_reg']:.4f} | "
            f"Objectness: {train_metrics['loss_objectness']:.4f} | "
            f"RPN Box: {train_metrics['loss_rpn_box_reg']:.4f}"
        )

        for iou_thr in config.MAP_IOU_THRESHOLDS:
            key = f'mAP@{iou_thr}'
            if key in val_metrics:
                logger.info(f"  {key}: {val_metrics[key]:.4f}")

        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['val_map'].append(val_metrics['mAP'])
        history['learning_rates'].append(current_lr)

        # Checkpointing
        is_best = val_metrics['mAP'] > best_map
        if is_best:
            best_map = val_metrics['mAP']

        if is_best or not config.SAVE_BEST_ONLY:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                path=run_dir / 'checkpoints',
                is_best=is_best
            )

        # Early stopping
        if early_stopping(val_metrics['mAP']):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # ====================
    # FINISH
    # ====================
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best mAP: {best_map:.4f}")
    logger.info(f"Checkpoints saved to: {run_dir}")
    logger.info("=" * 60)

    # Save history
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return {
        'best_map': best_map,
        'history': history,
        'run_dir': str(run_dir)
    }


# ==============================================================================
# CELL 8: MULTI-FOLD TRAINING (OPTIONAL)
# ==============================================================================

def train_all_folds(config: TrainConfig = TrainConfig()) -> Dict:
    """
    Train on all folds for K-Fold cross-validation.

    This is useful for:
    1. Robust performance estimation
    2. Creating ensemble of models
    3. Competition submissions (average or ensemble predictions)

    Returns:
        Dictionary with results for each fold
    """
    all_results = {}

    for fold in range(config.N_FOLDS):
        logger.info("=" * 60)
        logger.info(f"TRAINING FOLD {fold + 1}/{config.N_FOLDS}")
        logger.info("=" * 60)

        fold_results = train(fold=fold, config=config)
        all_results[f'fold_{fold}'] = fold_results

    # Summary
    best_maps = [all_results[f'fold_{i}']['best_map'] for i in range(config.N_FOLDS)]
    mean_map = np.mean(best_maps)
    std_map = np.std(best_maps)

    logger.info("\n" + "=" * 60)
    logger.info("K-FOLD CROSS-VALIDATION RESULTS")
    logger.info("=" * 60)
    for fold in range(config.N_FOLDS):
        logger.info(f"Fold {fold + 1}: mAP = {best_maps[fold]:.4f}")
    logger.info(f"Mean mAP: {mean_map:.4f} ± {std_map:.4f}")

    all_results['summary'] = {
        'mean_map': mean_map,
        'std_map': std_map,
        'per_fold_map': best_maps
    }

    return all_results


# ==============================================================================
# CELL 9: QUICK TRAINING TEST
# ==============================================================================

def test_training_loop():
    """
    Quick test of the training loop with minimal data.

    Use this to verify everything works before full training.
    """
    logger.info("=" * 60)
    logger.info("TESTING TRAINING LOOP (Quick Test)")
    logger.info("=" * 60)

    # Minimal config for testing
    config = TrainConfig()
    config.EPOCHS = 1
    config.BATCH_SIZE = 2
    config.ACCUMULATION_STEPS = 1
    config.LOG_INTERVAL = 1
    config.USE_AMP = torch.cuda.is_available()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Check if data exists
    if not os.path.exists(config.TRAIN_LABELS):
        logger.warning(f"Data not found at {config.DATA_DIR}")
        logger.info("Running with synthetic data for testing...")

        # Create synthetic data test
        return test_with_synthetic_data(config, device)

    # Run actual training test
    try:
        results = train(fold=0, config=config)
        logger.info("Training loop test PASSED")
        return results
    except Exception as e:
        logger.error(f"Training loop test FAILED: {e}")
        raise


def test_with_synthetic_data(config: TrainConfig, device: torch.device):
    """Test training with synthetic data when real data isn't available."""
    logger.info("Creating synthetic test data...")

    # Create model
    model = create_efficientnetv2_s_fasterrcnn(num_classes=2, pretrained=False)
    model = model.to(device)

    # Synthetic batch
    images = [torch.randn(3, 512, 512).to(device) for _ in range(2)]
    targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200]]).float().to(device),
            'labels': torch.tensor([1]).long().to(device),
        }
        for _ in range(2)
    ]

    # Test forward pass (training)
    model.train()
    loss_dict = model(images, targets)
    total_loss = sum(loss_dict.values())

    logger.info(f"Forward pass: OK (loss = {total_loss.item():.4f})")

    # Test backward pass
    total_loss.backward()
    logger.info("Backward pass: OK")

    # Test inference
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    logger.info(f"Inference: OK (detected {len(predictions[0]['boxes'])} boxes)")

    # Test WBF
    predictions_wbf = apply_wbf(predictions, image_size=512)
    logger.info(f"WBF: OK (fused to {len(predictions_wbf[0]['boxes'])} boxes)")

    logger.info("\nSynthetic data test PASSED")
    return {'status': 'synthetic_test_passed'}


# ==============================================================================
# CELL 10: MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Pneumonia Detection Model')
    parser.add_argument('--fold', type=int, default=0, help='Fold to train on')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--all_folds', action='store_true', help='Train all folds')

    args = parser.parse_args()

    # Update config with args
    config = TrainConfig()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR_HEAD = args.lr

    if args.test:
        # Quick test
        test_training_loop()
    elif args.all_folds:
        # Train all folds
        results = train_all_folds(config)
    else:
        # Train single fold
        results = train(fold=args.fold, config=config, resume_path=args.resume)
        print(f"\nTraining complete! Best mAP: {results['best_map']:.4f}")
