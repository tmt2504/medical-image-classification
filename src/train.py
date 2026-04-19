"""
Training script with MLflow experiment tracking and early stopping.

Supports training three model architectures:
- Transfer Learning (ResNet50, EfficientNet-B0)
- Vision Transformer (ViT)
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np

# Set MLflow tracking URI (default to local mlruns if not set)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

from src.data_pipeline import create_data_loaders, NUM_CLASSES
from src.models.transfer_learning import ResNet50Model, EfficientNetModel
from src.models.vit_model import ViTModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def get_model(model_name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Factory function to create model by name."""
    models_map = {
        "resnet50": lambda: ResNet50Model(num_classes=num_classes),
        "efficientnet": lambda: EfficientNetModel(num_classes=num_classes),
        "vit": lambda: ViTModel(num_classes=num_classes),
    }

    if model_name not in models_map:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(models_map.keys())}"
        )

    return models_map[model_name]()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return {"loss": epoch_loss, "accuracy": epoch_acc}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    metrics = {
        "loss": epoch_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f2": fbeta_score(
            all_labels, all_preds, beta=2, average="macro", zero_division=0
        ),
    }

    return metrics


def train_model(
    model_name: str,
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    num_workers: int = 4,
    experiment_name: str = "skin-cancer-classification",
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Full training pipeline with MLflow tracking.

    Args:
        model_name: One of 'resnet50', 'efficientnet', 'vit'
        data_dir: Path to ISIC dataset
        output_dir: Directory to save trained models
        epochs: Maximum number of training epochs
        batch_size: Batch size for data loaders
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        patience: Early stopping patience
        num_workers: Number of data loader workers
        experiment_name: MLflow experiment name

    Returns:
        Trained model and best validation metrics
    """
    # Use MPS (Apple Silicon) if available, then CUDA, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Training {model_name} on {device} with lr={learning_rate}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Data loaders
    train_loader, val_loader, _, class_weights = create_data_loaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Model
    model = get_model(model_name)
    model = model.to(device)

    # Loss with class weights + label smoothing for better generalization
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Optimizer — AdamW for better weight decay handling
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Cosine annealing scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # MLflow tracking
    mlflow.set_experiment(experiment_name)

    best_val_metrics = None
    best_val_loss = float("inf")

    with mlflow.start_run(run_name=model_name):
        # Log hyperparameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "patience": patience,
                "optimizer": "Adam",
                "loss": "CrossEntropyLoss (weighted)",
                "device": str(device),
                "num_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
        )

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_metrics = validate(model, val_loader, criterion, device)

            # Step scheduler (CosineAnnealingWarmRestarts uses epoch count)
            scheduler.step(epoch)

            epoch_time = time.time() - start_time

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "val_f2": val_metrics["f2"],
                    "epoch_time": epoch_time,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            logger.info(
                f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val Recall: {val_metrics['recall']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}, "
                f"Val F2: {val_metrics['f2']:.4f}"
            )

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_metrics = val_metrics
                model_path = os.path.join(output_dir, f"{model_name}_best.pth")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "num_classes": NUM_CLASSES,
                    },
                    model_path,
                )
                logger.info(f"Saved best model to {model_path}")

            # Early stopping check
            if early_stopping(val_metrics["loss"]):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Log best metrics
        mlflow.log_metrics({f"best_{k}": v for k, v in best_val_metrics.items()})

        # Log model artifact
        model_path = os.path.join(output_dir, f"{model_name}_best.pth")
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)

        # Log model with MLflow
        mlflow.pytorch.log_model(model, f"{model_name}_model")

    logger.info(f"Training complete. Best validation metrics: {best_val_metrics}")
    return model, best_val_metrics


def train_all_models(
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = None,
) -> Dict[str, Dict[str, float]]:
    """Train all models and return their metrics for comparison."""
    model_names = ["resnet50", "efficientnet", "vit"]
    # Per-model learning rates for full fine-tuning
    lr_map = {"resnet50": 1e-4, "efficientnet": 1e-4, "vit": 5e-5}
    results = {}

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model: {model_name}")
        logger.info(f"{'='*60}\n")

        lr = (
            learning_rate if learning_rate is not None else lr_map.get(model_name, 1e-4)
        )

        _, metrics = train_model(
            model_name=model_name,
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
        )
        results[model_name] = metrics

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train skin cancer classification models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["resnet50", "efficientnet", "vit", "all"],
        help="Model to train",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Path to ISIC dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=7, help="Early stopping patience"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Data loader workers"
    )

    args = parser.parse_args()

    if args.model == "all":
        results = train_all_models(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        print("\n=== Training Results ===")
        for name, metrics in results.items():
            print(f"\n{name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
    else:
        train_model(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            num_workers=args.num_workers,
        )
