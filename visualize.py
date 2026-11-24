"""
Visualization & Slide-Figure Utility for BraTS Binary U-Net Experiments.

Three subcommands:
  1. Explore a raw NIfTI patient (interactive slice slider, or --save PNG):
       python3 visualize.py explore --case BRATS_001
       python3 visualize.py explore --case BRATS_001 --save brats_001.png

  2. Visualize a trained run on held-out test patients (Best / Median / Worst):
       python3 visualize.py predict --run runs/flair_t1gd_bce_dice --save pred_grid.png

  3. Compare multiple experiment runs (prints benchmark table + side-by-side figure):
       python3 visualize.py compare \
         --runs runs/flair_bce_dice runs/t1gd_bce_dice runs/flair_t1gd_bce_dice \
         --save rq1_modalities.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from dataset import BraTSBinaryDataset, get_patient_splits
from train import get_device
from unet import UNet

# Multi-class BraTS colormap (0=transparent, 1=edema, 2=non-enhancing, 3=enhancing)
MULTICLASS_CMAP = ListedColormap(
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.20, 0.80, 0.35, 0.55),
        (0.95, 0.75, 0.10, 0.70),
        (0.95, 0.22, 0.22, 0.75),
    ]
)

# Error-map colormap (0=TN transparent, 1=TP green, 2=FP red, 3=FN yellow)
ERROR_CMAP = ListedColormap(
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.18, 0.82, 0.38, 0.65),  # TP
        (0.95, 0.22, 0.22, 0.75),  # FP
        (0.98, 0.80, 0.12, 0.80),  # FN
    ]
)

# Binary mask overlay colormap (0=transparent, 1=cyan)
BINARY_CMAP = ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.15, 0.78, 0.96, 0.55)])


def _resolve_data_root(data_root_arg: str) -> Path:
    """Resolve Task01_BrainTumour from either ./Task01_BrainTumour or ./data/Task01_BrainTumour."""
    base_dir = Path(__file__).parent
    p = Path(data_root_arg)
    if p.is_absolute():
        return p
    if (base_dir / p).exists():
        return base_dir / p
    if (base_dir / "data" / p).exists():
        return base_dir / "data" / p
    return base_dir / p


def _contrast_window(sl: np.ndarray) -> tuple[float, float]:
    """Compute robust 1st-99.5th percentile display limits over non-zero pixels."""
    nz = sl[sl != 0]
    if nz.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(nz, [1, 99.5])
    if vmax <= vmin:
        return float(nz.min()), float(nz.max()) + 1e-5
    return float(vmin), float(vmax)


def _prep_display(sl: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Contrast-stretch a 2D MRI slice while keeping empty background (0.0) black."""
    vmin, vmax = _contrast_window(sl)
    disp = np.where(sl == 0, vmin, sl)
    return disp, vmin, vmax


def _error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Encode binary prediction vs ground truth into {0:TN, 1:TP, 2:FP, 3:FN}."""
    emap = np.zeros_like(gt, dtype=np.uint8)
    emap[(pred == 1) & (gt == 1)] = 1  # True Positive
    emap[(pred == 1) & (gt == 0)] = 2  # False Positive
    emap[(pred == 0) & (gt == 1)] = 3  # False Negative
    return emap


def _load_run(run_dir: Path, data_root: Path, device: torch.device):
    """Reload model, config, test metrics, and test dataset for a completed run."""
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(run_dir / "test_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    splits = get_patient_splits(
        root_dir=data_root,
        num_cases=cfg["num_cases"],
        train_count=cfg["split_counts"]["train"],
        val_count=cfg["split_counts"]["val"],
        test_count=cfg["split_counts"]["test"],
        seed=cfg["seed"],
    )

    slice_range = tuple(cfg["slice_range"])
    cache_dir = Path(__file__).parent / f".cache_{cfg['image_size']}_z{slice_range[0]}_{slice_range[1]}"
    if not cache_dir.exists():
        cache_dir = None

    test_ds = BraTSBinaryDataset(
        root_dir=data_root,
        case_ids=splits["test"],
        modalities=cfg["modalities"],
        slice_range=slice_range,
        image_size=cfg["image_size"],
        cache_dir=cache_dir,
        augment=False,
    )

    model = UNet(
        in_channels=len(cfg["modalities"]),
        out_channels=1,
        features=tuple(cfg["features"]),
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    return model, cfg, metrics, test_ds


@torch.no_grad()
def _predict_patient(
    model: UNet, test_ds: BraTSBinaryDataset, case_id: str, device: torch.device, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(imgs_np, raw_lbl_np, pred_bin_np)` for all slices of `case_id`."""
    imgs_t, raw_lbl = test_ds.get_patient_volume(case_id)
    preds = []
    for start in range(0, imgs_t.shape[0], 32):
        logits = model(imgs_t[start : start + 32].to(device))
        p = (torch.sigmoid(logits) >= threshold).squeeze(1).cpu().numpy().astype(np.uint8)
        preds.append(p)
    return imgs_t.numpy(), raw_lbl, np.concatenate(preds, axis=0)


def cmd_explore(args: argparse.Namespace) -> None:
    """Visualize a raw `.nii.gz` patient (interactive slider or saved multi-panel PNG)."""
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    data_root = _resolve_data_root(args.data_root)

    img_path = data_root / "imagesTr" / f"{args.case}.nii.gz"
    lbl_path = data_root / "labelsTr" / f"{args.case}.nii.gz"
    img = np.asarray(nib.load(str(img_path)).dataobj, dtype=np.float32)
    lbl = np.asarray(nib.load(str(lbl_path)).dataobj, dtype=np.uint8)

    z0 = args.slice if args.slice is not None else int(np.argmax((lbl > 0).sum(axis=(0, 1))))
    modality_names = ["0: FLAIR", "1: T1w", "2: T1gd", "3: T2w"]

    if args.save:
        fig, axes = plt.subplots(1, 6, figsize=(17, 3.6), facecolor="#0f1117")
        for c in range(4):
            sl = np.rot90(img[:, :, z0, c])
            vmin, vmax = _contrast_window(sl)
            axes[c].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
            axes[c].set_title(modality_names[c], color="white", fontsize=10, fontweight="bold")
            axes[c].axis("off")

        flair_sl = np.rot90(img[:, :, z0, 0])
        vmin, vmax = _contrast_window(flair_sl)

        # Panel 5: Multi-class mask
        axes[4].imshow(flair_sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[4].imshow(np.rot90(lbl[:, :, z0]), cmap=MULTICLASS_CMAP, vmin=0, vmax=3, interpolation="nearest")
        axes[4].set_title(f"4-Class Mask (z={z0})", color="white", fontsize=10, fontweight="bold")
        axes[4].axis("off")

        # Panel 6: Binary whole-tumor target (Project 04)
        axes[5].imshow(flair_sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[5].imshow(np.rot90((lbl[:, :, z0] > 0).astype(np.uint8)), cmap=BINARY_CMAP, vmin=0, vmax=1, interpolation="nearest")
        axes[5].set_title("Binary Target (Slide 06)", color="#7dd3fc", fontsize=10, fontweight="bold")
        axes[5].axis("off")

        legend_handles = [
            Patch(facecolor=(0.20, 0.80, 0.35, 0.8), label="Edema (1)"),
            Patch(facecolor=(0.95, 0.75, 0.10, 0.9), label="Non-enhancing (2)"),
            Patch(facecolor=(0.95, 0.22, 0.22, 0.9), label="Enhancing (3)"),
            Patch(facecolor=(0.15, 0.78, 0.96, 0.8), label="Binary Tumor (1|2|3)"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            frameon=True,
            facecolor="#1c212b",
            edgecolor="#334155",
            labelcolor="white",
            fontsize=9,
        )
        fig.suptitle(f"{args.case} — MRI Modalities & Binary Whole-Tumor Target", color="white", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0.08, 1, 0.93])
        fig.savefig(args.save, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved exploration figure -> {args.save}")
        return

    # Interactive Matplotlib axial slice scrubber
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8))
    plt.subplots_adjust(bottom=0.2)

    handles = []
    for c in range(4):
        sl = np.rot90(img[:, :, z0, c])
        vmin, vmax = _contrast_window(sl)
        h = axes[c].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[c].set_title(modality_names[c])
        axes[c].axis("off")
        handles.append(h)

    flair_sl = np.rot90(img[:, :, z0, 0])
    vmin, vmax = _contrast_window(flair_sl)
    bg_h = axes[4].imshow(flair_sl, cmap="gray", vmin=vmin, vmax=vmax)
    mk_h = axes[4].imshow(np.rot90(lbl[:, :, z0]), cmap=MULTICLASS_CMAP, vmin=0, vmax=3, interpolation="nearest")
    axes[4].set_title(f"FLAIR + Mask (z={z0})")
    axes[4].axis("off")

    slider_ax = plt.axes([0.2, 0.06, 0.6, 0.05])
    z_slider = Slider(slider_ax, "Axial Slice z", 0, img.shape[2] - 1, valinit=z0, valstep=1)

    def _update(_val):
        z = int(z_slider.val)
        for c in range(4):
            handles[c].set_data(np.rot90(img[:, :, z, c]))
        bg_h.set_data(np.rot90(img[:, :, z, 0]))
        mk_h.set_data(np.rot90(lbl[:, :, z]))
        axes[4].set_title(f"FLAIR + Mask (z={z})")
        fig.canvas.draw_idle()

    z_slider.on_changed(_update)
    fig.suptitle(f"Interactive BraTS Explorer: {args.case}")
    plt.show()


def cmd_predict(args: argparse.Namespace) -> None:
    """Plot Best, Median, and Worst test patients with TP/FP/FN error maps."""
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = get_device()
    base_dir = Path(__file__).parent
    data_root = _resolve_data_root(args.data_root)
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = base_dir / run_dir

    model, cfg, metrics, test_ds = _load_run(run_dir, data_root, device)

    ranked = sorted(metrics["per_patient"], key=lambda p: p["dice"])
    selected = [
        ("Best", ranked[-1]),
        ("Median", ranked[len(ranked) // 2]),
        ("Worst", ranked[0]),
    ]

    fig, axes = plt.subplots(len(selected), 4, figsize=(12.5, 9.2), facecolor="#0f1117")
    col_titles = [
        f"Input ({cfg['modalities'][0]})",
        "Ground Truth (Binary)",
        "U-Net Prediction",
        "Error Map (TP / FP / FN)",
    ]

    for row, (tag, info) in enumerate(selected):
        cid = info["case_id"]
        imgs, raw_lbl, preds = _predict_patient(model, test_ds, cid, device, threshold=cfg["threshold"])
        gt_bin = (raw_lbl > 0).astype(np.uint8)

        # Pick slice with largest ground-truth tumor area
        best_z = int(np.argmax(gt_bin.sum(axis=(1, 2))))
        bg, vmin, vmax = _prep_display(np.rot90(imgs[best_z, 0]))
        gt_sl = np.rot90(gt_bin[best_z])
        pr_sl = np.rot90(preds[best_z])
        err_sl = _error_map(pr_sl, gt_sl)

        panels = [
            (bg, None, None, None),
            (bg, gt_sl, BINARY_CMAP, (0, 1)),
            (bg, pr_sl, BINARY_CMAP, (0, 1)),
            (bg, err_sl, ERROR_CMAP, (0, 3)),
        ]
        for col, (base_im, ov, cmap, clim) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(base_im, cmap="gray", vmin=vmin, vmax=vmax)
            if ov is not None:
                ax.imshow(ov, cmap=cmap, vmin=clim[0], vmax=clim[1], interpolation="nearest")
            if row == 0:
                ax.set_title(col_titles[col], color="white", fontsize=10, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(
                    f"{tag}: {cid}\n3D Dice={info['dice']:.3f}\nRec={info['recall']:.2f}",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
            ax.set_xticks([])
            ax.set_yticks([])

    legend_handles = [
        Patch(facecolor=(0.18, 0.82, 0.38, 0.85), label="True Positive (TP)"),
        Patch(facecolor=(0.95, 0.22, 0.22, 0.85), label="False Positive (FP)"),
        Patch(facecolor=(0.98, 0.80, 0.12, 0.90), label="False Negative (FN)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor="#1c212b",
        edgecolor="#334155",
        labelcolor="white",
        fontsize=9.5,
    )
    fig.suptitle(
        f"Test Set Qualitative Analysis — {cfg['run_name']} (Mean 3D Dice = {metrics['dice_mean']:.4f})",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved prediction grid -> {args.save}")
    else:
        plt.show()


def cmd_compare(args: argparse.Namespace) -> None:
    """Print a benchmark table across runs and plot side-by-side overlays on one test case."""
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = get_device()
    base_dir = Path(__file__).parent
    data_root = _resolve_data_root(args.data_root)

    loaded = []
    print("\n=== Experiment Comparison (Held-Out Test Set, Patient-Level 3D) ===")
    header = (
        f"{'Run':<24} | {'Modalities':<12} | {'Loss':<8} | "
        f"{'Dice':<13} | {'IoU':<6} | {'Prec':<6} | {'Rec':<6} | {'Small':<6} | {'EdemaR':<6} | {'EnhR':<6}"
    )
    print(header)
    print("-" * len(header))

    for r in args.runs:
        run_dir = Path(r)
        if not run_dir.is_absolute():
            run_dir = base_dir / run_dir
        model, cfg, m, test_ds = _load_run(run_dir, data_root, device)
        loaded.append((model, cfg, m, test_ds))
        sub = m.get("subregion_recall_mean", {})
        small_str = f"{m['small_tumor_dice']:.4f}" if m.get("small_tumor_dice") is not None else "  N/A "
        print(
            f"{cfg['run_name']:<24} | {'+'.join(cfg['modalities']):<12} | {cfg['loss']:<8} | "
            f"{m['dice_mean']:.4f}±{m['dice_std']:.3f} | {m['iou_mean']:.4f} | "
            f"{m['precision_mean']:.4f} | {m['recall_mean']:.4f} | "
            f"{small_str} | {sub.get('edema', 0):.4f} | {sub.get('enhancing', 0):.4f}"
        )

    # Pick common test patient for visual comparison (use --case or median case of first run)
    first_metrics = loaded[0][2]
    if args.case:
        case_id = args.case
    else:
        ranked = sorted(first_metrics["per_patient"], key=lambda p: p["dice"])
        case_id = ranked[len(ranked) // 2]["case_id"]

    # Load FLAIR background + multi-class GT for consistent visual backdrop
    first_cfg = loaded[0][1]
    slice_range = tuple(first_cfg["slice_range"])
    cache_dir = Path(__file__).parent / f".cache_{first_cfg['image_size']}_z{slice_range[0]}_{slice_range[1]}"
    ref_ds = BraTSBinaryDataset(
        root_dir=data_root,
        case_ids=[case_id],
        modalities=["flair"],
        slice_range=slice_range,
        image_size=first_cfg["image_size"],
        cache_dir=cache_dir if cache_dir.exists() else None,
        augment=False,
    )
    ref_imgs, ref_raw_lbl = ref_ds.get_patient_volume(case_id)
    gt_bin = (ref_raw_lbl > 0).astype(np.uint8)
    best_z = int(np.argmax(gt_bin.sum(axis=(1, 2))))
    flair_bg, vmin, vmax = _prep_display(np.rot90(ref_imgs[best_z, 0].numpy()))
    gt_sl = np.rot90(gt_bin[best_z])

    n_cols = 1 + len(loaded)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 4.0), facecolor="#0f1117")
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(flair_bg, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].imshow(np.rot90(ref_raw_lbl[best_z]), cmap=MULTICLASS_CMAP, vmin=0, vmax=3, interpolation="nearest")
    axes[0].set_title(f"Ground Truth ({case_id})\nGreen=Edema | Red=Enh", color="white", fontsize=9.5, fontweight="bold")
    axes[0].axis("off")

    for i, (model, cfg, m, test_ds) in enumerate(loaded, start=1):
        _, _, preds = _predict_patient(model, test_ds, case_id, device, threshold=cfg["threshold"])
        pr_sl = np.rot90(preds[best_z])
        err_sl = _error_map(pr_sl, gt_sl)
        case_dice = next(p["dice"] for p in m["per_patient"] if p["case_id"] == case_id)

        axes[i].imshow(flair_bg, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i].imshow(err_sl, cmap=ERROR_CMAP, vmin=0, vmax=3, interpolation="nearest")
        axes[i].set_title(
            f"{'+'.join(cfg['modalities'])} | {cfg['loss']}\nCase Dice={case_dice:.3f} (Test={m['dice_mean']:.3f})",
            color="white",
            fontsize=9.5,
            fontweight="bold",
        )
        axes[i].axis("off")

    legend_handles = [
        Patch(facecolor=(0.18, 0.82, 0.38, 0.85), label="TP"),
        Patch(facecolor=(0.95, 0.22, 0.22, 0.85), label="FP"),
        Patch(facecolor=(0.98, 0.80, 0.12, 0.90), label="FN"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor="#1c212b",
        edgecolor="#334155",
        labelcolor="white",
        fontsize=9,
    )
    fig.suptitle(f"Model Comparison on Held-Out Test Case {case_id}", color="white", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\nSaved comparison figure -> {args.save}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BraTS MRI scans and U-Net predictions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("explore", help="Explore a raw NIfTI MRI scan")
    p_exp.add_argument("--data-root", default="Task01_BrainTumour")
    p_exp.add_argument("--case", default="BRATS_001", help="Patient ID (e.g. BRATS_001)")
    p_exp.add_argument("--slice", type=int, default=None, help="Axial slice index (default: peak tumor)")
    p_exp.add_argument("--save", default=None, help="Save static PNG instead of opening interactive slider")

    p_pred = sub.add_parser("predict", help="Plot Best/Median/Worst test predictions for one run")
    p_pred.add_argument("--data-root", default="Task01_BrainTumour")
    p_pred.add_argument("--run", required=True, help="Path to run directory (e.g. runs/flair_t1gd_bce_dice)")
    p_pred.add_argument("--save", default=None, help="Save figure to PNG path")

    p_cmp = sub.add_parser("compare", help="Compare multiple completed runs side-by-side")
    p_cmp.add_argument("--data-root", default="Task01_BrainTumour")
    p_cmp.add_argument("--runs", nargs="+", required=True, help="List of run directories to compare")
    p_cmp.add_argument("--case", default=None, help="Optional specific test patient ID")
    p_cmp.add_argument("--save", default=None, help="Save comparison figure to PNG path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "explore":
        cmd_explore(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
