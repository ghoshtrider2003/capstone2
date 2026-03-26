"""Training script for HMSAN plant leaf disease detection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from dataset.data_module import DataConfig, create_dataloaders
from models.hmsan import HMSAN
from utils.metrics import MetricTracker, format_metrics
from utils.train_utils import EarlyStopping, plot_confusion_matrix, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HMSAN for plant leaf disease detection")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to ImageFolder dataset")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"])
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained backbone weights")
    parser.add_argument("--use-domain-adaptation", action="store_true")
    parser.add_argument("--domain-loss-weight", type=float, default=0.1)
    parser.add_argument("--num-domains", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    return parser.parse_args()


def build_optimizer(model: nn.Module, args: argparse.Namespace):
    if args.optimizer == "adam":
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args: argparse.Namespace):
    if args.scheduler == "step":
        return StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return CosineAnnealingLR(optimizer, T_max=args.epochs)


def run_epoch(
    model: HMSAN,
    dataloader,
    criterion: nn.Module,
    device: str,
    optimizer=None,
    domain_criterion: nn.Module | None = None,
    domain_loss_weight: float = 0.1,
) -> Tuple[float, dict, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    tracker = MetricTracker()

    all_targets = []
    all_preds = []

    with torch.set_grad_enabled(is_train):
        for images, targets in tqdm(dataloader, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            logits = outputs["logits"]
            loss = criterion(logits, targets)

            if domain_criterion is not None and "domain_logits" in outputs:
                # Placeholder single-domain labels (replace with domain metadata in real setups).
                domain_targets = torch.zeros(targets.size(0), dtype=torch.long, device=device)
                domain_loss = domain_criterion(outputs["domain_logits"], domain_targets)
                loss = loss + domain_loss_weight * domain_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            tracker.update(targets.detach().cpu(), preds.detach().cpu())

            all_targets.extend(targets.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    epoch_loss = total_loss / len(dataloader.dataset)
    metrics = tracker.compute()
    return epoch_loss, metrics, np.array(all_targets), np.array(all_preds)


def main() -> None:
    args = parse_args()
    device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, test_loader, class_names = create_dataloaders(data_config)

    model = HMSAN(
        num_classes=len(class_names),
        backbone=args.backbone,
        pretrained=args.pretrained,
        use_domain_adaptation=args.use_domain_adaptation,
        num_domains=args.num_domains,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss() if args.use_domain_adaptation else None

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    early_stopping = EarlyStopping(patience=args.early_stop_patience, mode="max")

    best_f1 = -1.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss, train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            domain_criterion=domain_criterion,
            domain_loss_weight=args.domain_loss_weight,
        )

        val_loss, val_metrics, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            domain_criterion=domain_criterion,
            domain_loss_weight=args.domain_loss_weight,
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | {format_metrics(train_metrics)}")
        print(f"Val   Loss: {val_loss:.4f} | {format_metrics(val_metrics)}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "class_names": class_names,
                    "args": vars(args),
                },
                str(best_path),
            )
            print(f"Saved new best model to {best_path}")

        if early_stopping.step(val_metrics["f1"]):
            print("Early stopping triggered.")
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics, y_true, y_pred = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
    )
    print(f"\nTest Loss: {test_loss:.4f} | {format_metrics(test_metrics)}")

    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=str(cm_path))
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
