"""
Complete training, validation, and held-out test script for BraTS Binary U-Net.

Designed for controlled experiments (Intro to ML in Biomedicine — Project 04):
  RQ1 (Modalities, fixed BCE+Dice):
    python3 train.py --modalities flair       --loss bce_dice --run-name flair_bce_dice
    python3 train.py --modalities t1gd        --loss bce_dice --run-name t1gd_bce_dice
    python3 train.py --modalities flair t1gd  --loss bce_dice --run-name flair_t1gd_bce_dice

  RQ2 (Loss Functions, fixed best modality setup):
    python3 train.py --modalities flair t1gd  --loss bce      --run-name flair_t1gd_bce
    python3 train.py --modalities flair t1gd  --loss dice     --run-name flair_t1gd_dice
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BraTSBinaryDataset, get_patient_splits, parse_modalities
from losses import get_loss_fn
from unet import UNet


def get_device() -> torch.device:
    """Select the fastest available accelerator (Apple Silicon MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Fix RNG seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_patients(
    model: nn.Module,
    dataset: BraTSBinaryDataset,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model at the 3D patient level across all axial slices of each patient.

    Stacking each patient's predicted 2D slices into a 3D volume avoids empty-slice
    inflation and gives clinically meaningful Dice, IoU, Precision, Recall,
    small-vs-large tumor performance, and per-subregion recall diagnostics.
    """
    model.eval()
    eps = 1e-6

    total_loss = 0.0
    total_batches = 0
    per_patient: list[dict] = []

    for case_id in dataset.case_ids:
        imgs_vol, raw_lbl_vol = dataset.get_patient_volume(case_id)  # (S, C, H, W), (S, H, W)
        bin_lbl_vol = (raw_lbl_vol > 0).astype(np.uint8)

        pred_chunks: list[np.ndarray] = []
        n_slices = imgs_vol.shape[0]
        for start in range(0, n_slices, batch_size):
            img_b = imgs_vol[start : start + batch_size].to(device)
            lbl_b = torch.from_numpy(bin_lbl_vol[start : start + batch_size]).unsqueeze(1).float().to(device)

            logits = model(img_b)
            total_loss += criterion(logits, lbl_b).item()
            total_batches += 1

            preds_b = (torch.sigmoid(logits) >= threshold).squeeze(1).cpu().numpy().astype(np.uint8)
            pred_chunks.append(preds_b)

        pred_vol = np.concatenate(pred_chunks, axis=0)  # (S, H, W)

        tp = float(((pred_vol == 1) & (bin_lbl_vol == 1)).sum())
        fp = float(((pred_vol == 1) & (bin_lbl_vol == 0)).sum())
        fn = float(((pred_vol == 0) & (bin_lbl_vol == 1)).sum())
        tumor_voxels = int(bin_lbl_vol.sum())

        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)

        # Sub-region recall: what fraction of each original BraTS class was captured?
        sub_recalls: dict[str, float | None] = {}
        for cls_id, cls_name in [(1, "edema"), (2, "non_enhancing"), (3, "enhancing")]:
            cls_mask = raw_lbl_vol == cls_id
            cls_total = int(cls_mask.sum())
            if cls_total > 0:
                sub_recalls[cls_name] = float((pred_vol[cls_mask] == 1).sum() / cls_total)
            else:
                sub_recalls[cls_name] = None

        per_patient.append(
            {
                "case_id": case_id,
                "tumor_voxels": tumor_voxels,
                "dice": round(dice, 5),
                "iou": round(iou, 5),
                "precision": round(precision, 5),
                "recall": round(recall, 5),
                "subregion_recall": sub_recalls,
            }
        )

    def _mean_std(values: list[float]) -> tuple[float, float]:
        arr = np.array(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    dice_mean, dice_std = _mean_std([p["dice"] for p in per_patient])
    iou_mean, iou_std = _mean_std([p["iou"] for p in per_patient])
    prec_mean, prec_std = _mean_std([p["precision"] for p in per_patient])
    rec_mean, rec_std = _mean_std([p["recall"] for p in per_patient])

    # Small vs. Large tumor breakdown (split at median tumor volume)
    median_voxels = float(np.median([p["tumor_voxels"] for p in per_patient]))
    small_dices = [p["dice"] for p in per_patient if p["tumor_voxels"] <= median_voxels]
    large_dices = [p["dice"] for p in per_patient if p["tumor_voxels"] > median_voxels]

    # Mean recall per BraTS sub-region across patients where that sub-region is present
    subregion_summary: dict[str, float] = {}
    for cls_name in ("edema", "non_enhancing", "enhancing"):
        vals = [p["subregion_recall"][cls_name] for p in per_patient if p["subregion_recall"][cls_name] is not None]
        if vals:
            subregion_summary[cls_name] = round(float(np.mean(vals)), 5)

    ranked = sorted(per_patient, key=lambda p: p["dice"])

    return {
        "loss": round(total_loss / max(total_batches, 1), 5),
        "dice_mean": round(dice_mean, 5),
        "dice_std": round(dice_std, 5),
        "iou_mean": round(iou_mean, 5),
        "iou_std": round(iou_std, 5),
        "precision_mean": round(prec_mean, 5),
        "precision_std": round(prec_std, 5),
        "recall_mean": round(rec_mean, 5),
        "recall_std": round(rec_std, 5),
        "small_tumor_dice": round(float(np.mean(small_dices)), 5) if small_dices else None,
        "large_tumor_dice": round(float(np.mean(large_dices)), 5) if large_dices else None,
        "median_tumor_voxels": median_voxels,
        "subregion_recall_mean": subregion_summary,
        "worst_3": ranked[:3],
        "best_3": ranked[-3:][::-1],
        "per_patient": per_patient,
    }


def save_training_curves(history: list[dict], out_path: Path, title: str) -> None:
    """Save a crisp 2-panel figure of Train/Val Loss and Val Dice/Recall for slides."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_dice = [h["val_dice"] for h in history]
    val_recall = [h["val_recall"] for h in history]
    val_prec = [h["val_precision"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(epochs, train_loss, marker="o", ms=4, label="Train Loss")
    ax1.plot(epochs, val_loss, marker="s", ms=4, label="Val Loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, val_dice, marker="o", ms=4, linewidth=2, label="Val 3D Dice")
    ax2.plot(epochs, val_recall, marker="^", ms=4, linestyle="--", label="Val Recall")
    ax2.plot(epochs, val_prec, marker="v", ms=4, linestyle=":", label="Val Precision")
    ax2.set_title("Validation Patient-Level Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train compact 2D U-Net on BraTS binary tumor masks.")
    parser.add_argument("--data-root", type=str, default="Task01_BrainTumour", help="Path to Task01_BrainTumour")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["flair", "t1gd"],
        help="1 or 2 MRI modalities: flair, t1w, t1gd (t1ce), t2w",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="bce_dice",
        choices=["bce", "dice", "bce_dice"],
        help="Loss function to optimize",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial AdamW learning rate")
    parser.add_argument("--image-size", type=int, default=128, help="Resized axial dimension (divisible by 16)")
    parser.add_argument("--base-features", type=int, default=16, help="Base U-Net channels (16=compact, 32=full)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Bottleneck Dropout2d probability")
    parser.add_argument("--num-cases", type=int, default=150, help="Total patients in fixed subset")
    parser.add_argument("--train-count", type=int, default=100, help="Training patients")
    parser.add_argument("--val-count", type=int, default=25, help="Validation patients")
    parser.add_argument("--test-count", type=int, default=25, help="Held-out test patients")
    parser.add_argument("--slice-start", type=int, default=30, help="First axial slice index")
    parser.add_argument("--slice-end", type=int, default=125, help="Last axial slice index (exclusive)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split & training")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary prediction")
    parser.add_argument("--run-name", type=str, default=None, help="Output folder name under runs/")
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentations")
    parser.add_argument("--no-cache", action="store_true", help="Read directly from .nii.gz instead of .npy cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    base_dir = Path(__file__).parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        if (base_dir / data_root).exists():
            data_root = base_dir / data_root
        elif (base_dir / "data" / data_root).exists():
            data_root = base_dir / "data" / data_root
        else:
            data_root = base_dir / data_root

    mod_indices, mod_names = parse_modalities(args.modalities)
    mod_slug = "_".join(m.lower() for m in mod_names)
    run_name = args.run_name or f"{mod_slug}__{args.loss}"
    run_dir = base_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    slice_range = (args.slice_start, args.slice_end)
    cache_dir = None if args.no_cache else base_dir / f".cache_{args.image_size}_z{args.slice_start}_{args.slice_end}"

    # 1. Fixed patient-level split (saved to JSON and shared across all runs)
    splits = get_patient_splits(
        root_dir=data_root,
        num_cases=args.num_cases,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )

    print(f"=== Run: {run_name} ===")
    print(f"Device: {device} | Modalities: {mod_names} | Loss: {args.loss} | Image size: {args.image_size}x{args.image_size}")
    print(f"Patient split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # 2. Datasets & DataLoader
    train_ds = BraTSBinaryDataset(
        root_dir=data_root,
        case_ids=splits["train"],
        modalities=args.modalities,
        slice_range=slice_range,
        image_size=args.image_size,
        cache_dir=cache_dir,
        augment=not args.no_augment,
    )
    val_ds = BraTSBinaryDataset(
        root_dir=data_root,
        case_ids=splits["val"],
        modalities=args.modalities,
        slice_range=slice_range,
        image_size=args.image_size,
        cache_dir=cache_dir,
        augment=False,
    )
    test_ds = BraTSBinaryDataset(
        root_dir=data_root,
        case_ids=splits["test"],
        modalities=args.modalities,
        slice_range=slice_range,
        image_size=args.image_size,
        cache_dir=cache_dir,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 3. Compact U-Net, Loss, Optimizer, LR Scheduler
    b = args.base_features
    features = (b, b * 2, b * 4, b * 8)
    model = UNet(
        in_channels=len(mod_indices),
        out_channels=1,
        features=features,
        dropout=args.dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = get_loss_fn(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = {
        "run_name": run_name,
        "device": str(device),
        "modalities": mod_names,
        "modality_indices": mod_indices,
        "loss": args.loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "image_size": args.image_size,
        "features": list(features),
        "dropout": args.dropout,
        "trainable_params": total_params,
        "slice_range": list(slice_range),
        "num_cases": args.num_cases,
        "split_counts": {"train": args.train_count, "val": args.val_count, "test": args.test_count},
        "seed": args.seed,
        "threshold": args.threshold,
        "augment": not args.no_augment,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Model: Compact UNet {features} ({total_params:,} params) | Train slices: {len(train_ds):,}")

    # 4. Epoch Training + Patient-Level Validation Loop
    best_val_dice = -1.0
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Patient-level 3D validation
        val_res = evaluate_patients(
            model, val_ds, criterion, device, batch_size=args.batch_size * 2, threshold=args.threshold
        )

        record = {
            "epoch": epoch,
            "lr": round(current_lr, 6),
            "train_loss": round(train_loss, 5),
            "val_loss": val_res["loss"],
            "val_dice": val_res["dice_mean"],
            "val_iou": val_res["iou_mean"],
            "val_precision": val_res["precision_mean"],
            "val_recall": val_res["recall_mean"],
        }
        history.append(record)

        marker = ""
        if val_res["dice_mean"] > best_val_dice:
            best_val_dice = val_res["dice_mean"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            marker = "  <-- best"

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_res['loss']:.4f} | "
            f"val_dice={val_res['dice_mean']:.4f} | val_prec={val_res['precision_mean']:.4f} | "
            f"val_rec={val_res['recall_mean']:.4f}{marker}"
        )

    elapsed = time.time() - start_time
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    save_training_curves(
        history,
        run_dir / "curves.png",
        title=f"{run_name} ({'+'.join(mod_names)} | {args.loss})",
    )

    # 5. Final Held-Out Test Evaluation (using best validation checkpoint)
    print("\nEvaluating best checkpoint on held-out TEST set...")
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    test_res = evaluate_patients(
        model, test_ds, criterion, device, batch_size=args.batch_size * 2, threshold=args.threshold
    )
    test_res["training_seconds"] = round(elapsed, 1)
    test_res["best_val_dice"] = round(best_val_dice, 5)

    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_res, f, indent=2)

    small_str = f"{test_res['small_tumor_dice']:.4f}" if test_res["small_tumor_dice"] is not None else "N/A"
    large_str = f"{test_res['large_tumor_dice']:.4f}" if test_res["large_tumor_dice"] is not None else "N/A"
    print(
        f"\nTEST SUMMARY ({run_name}) — {len(splits['test'])} patients ({elapsed:.1f}s total):\n"
        f"  3D Dice:      {test_res['dice_mean']:.4f} ± {test_res['dice_std']:.4f}\n"
        f"  3D IoU:       {test_res['iou_mean']:.4f} ± {test_res['iou_std']:.4f}\n"
        f"  Precision:    {test_res['precision_mean']:.4f} ± {test_res['precision_std']:.4f}\n"
        f"  Recall:       {test_res['recall_mean']:.4f} ± {test_res['recall_std']:.4f}\n"
        f"  Small Tumors: {small_str} | Large Tumors: {large_str}\n"
        f"  Sub-region Recall: {test_res['subregion_recall_mean']}\n"
        f"  Saved outputs -> {run_dir}"
    )


if __name__ == "__main__":
    main()
