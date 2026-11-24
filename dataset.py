"""
BraTS (MSD Task01_BrainTumour) 2D Binary Segmentation Dataset & Caching Pipeline.

Design aligned with Intro to ML in Biomedicine — Project 04:
  - Selectable MRI modalities (1 or 2 channels, e.g. FLAIR, T1gd, or FLAIR + T1gd).
  - Binary whole-tumor mask: combines classes 1 (edema), 2 (non-enhancing), and
    3 (enhancing) into 1 (tumor) vs. 0 (background).
  - Fixed, reproducible patient-level train / val / test split (e.g. 100 / 25 / 25
    out of 150 cases) saved to JSON so all experiments share the exact same split.
  - One-time `.npy` preprocessing cache (volume-level brain-only Z-score + resize)
    with memory-mapped reads so multi-epoch training does not bottleneck on gzip.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

# Channel mapping from Task01_BrainTumour/dataset.json (plus common aliases)
MODALITY_MAP: dict[str, int] = {
    "flair": 0,
    "t1w": 1,
    "t1": 1,
    "t1gd": 2,
    "t1ce": 2,
    "t2w": 3,
    "t2": 3,
}

CANONICAL_MODALITY_NAMES: dict[int, str] = {
    0: "FLAIR",
    1: "T1w",
    2: "T1gd",
    3: "T2w",
}


def parse_modalities(modalities: Sequence[str]) -> tuple[list[int], list[str]]:
    """Convert modality names (e.g. ['flair', 't1gd']) to channel indices and labels."""
    if not modalities:
        raise ValueError("Specify at least one modality (e.g. 'flair', 't1gd').")
    indices: list[int] = []
    names: list[str] = []
    for mod in modalities:
        key = mod.strip().lower()
        if key not in MODALITY_MAP:
            valid = ", ".join(sorted(MODALITY_MAP.keys()))
            raise ValueError(f"Unknown modality '{mod}'. Valid options: {valid}")
        idx = MODALITY_MAP[key]
        if idx not in indices:
            indices.append(idx)
            names.append(CANONICAL_MODALITY_NAMES[idx])
    return indices, names


def normalize_volume_brain_only(vol_4d: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each MRI modality across non-zero brain voxels of a 3D volume.

    Doing normalization per 3D patient volume keeps intensity consistent across all
    axial slices of the same scan while leaving empty background strictly at 0.0.
    """
    out = np.zeros_like(vol_4d, dtype=np.float32)
    for c in range(vol_4d.shape[-1]):
        channel = vol_4d[..., c]
        brain_mask = channel > 0
        if np.any(brain_mask):
            vals = channel[brain_mask]
            mean = vals.mean()
            std = vals.std()
            out[..., c][brain_mask] = (vals - mean) / (std + 1e-8)
    return out


def _resize_slices(
    img_slices: np.ndarray, lbl_slices: np.ndarray, image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize extracted axial slices to (N, 4, image_size, image_size) and (N, image_size, image_size).
    Uses bilinear interpolation for MRI intensities and nearest-neighbor for integer masks.
    """
    if img_slices.shape[-2] == image_size and img_slices.shape[-1] == image_size:
        return img_slices, lbl_slices

    img_t = torch.from_numpy(img_slices)  # (N, 4, H, W)
    img_resized = F.interpolate(
        img_t, size=(image_size, image_size), mode="bilinear", align_corners=False
    ).numpy()

    lbl_t = torch.from_numpy(lbl_slices).unsqueeze(1).float()  # (N, 1, H, W)
    lbl_resized = (
        F.interpolate(lbl_t, size=(image_size, image_size), mode="nearest")
        .squeeze(1)
        .numpy()
        .astype(np.uint8)
    )
    return img_resized, lbl_resized


def get_patient_splits(
    root_dir: str | Path,
    num_cases: int = 150,
    train_count: int = 100,
    val_count: int = 25,
    test_count: int = 25,
    seed: int = 42,
    split_file: str | Path | None = None,
) -> dict[str, list[str]]:
    """
    Create or load a fixed, deterministic patient-level train / val / test split.

    Always splits by patient ID (never by slice) to prevent data leakage across splits.
    """
    if train_count + val_count + test_count != num_cases:
        raise ValueError(
            f"train_count ({train_count}) + val_count ({val_count}) + "
            f"test_count ({test_count}) must equal num_cases ({num_cases})."
        )

    root_dir = Path(root_dir)
    if split_file is None:
        split_file = Path(__file__).parent / f"splits_{num_cases}_seed{seed}.json"
    split_file = Path(split_file)

    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if (
            len(saved.get("train", [])) == train_count
            and len(saved.get("val", [])) == val_count
            and len(saved.get("test", [])) == test_count
        ):
            return {k: saved[k] for k in ("train", "val", "test")}

    dataset_json = root_dir / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"Could not find {dataset_json}.")

    with open(dataset_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Collect valid patient IDs in sorted order for OS-independent reproducibility
    all_cases: list[str] = []
    for item in meta["training"]:
        img_path = root_dir / item["image"].lstrip("./")
        lbl_path = root_dir / item["label"].lstrip("./")
        if img_path.exists() and lbl_path.exists():
            case_id = img_path.name.replace(".nii.gz", "")
            all_cases.append(case_id)
    all_cases.sort()

    if num_cases > len(all_cases):
        raise ValueError(f"Requested {num_cases} cases, but only {len(all_cases)} exist.")

    rng = random.Random(seed)
    rng.shuffle(all_cases)
    selected = all_cases[:num_cases]

    splits = {
        "train": sorted(selected[:train_count]),
        "val": sorted(selected[train_count : train_count + val_count]),
        "test": sorted(selected[train_count + val_count :]),
    }

    payload = {
        "seed": seed,
        "num_cases": num_cases,
        "counts": {"train": train_count, "val": val_count, "test": test_count},
        **splits,
    }
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return splits


def prepare_cache(
    root_dir: str | Path,
    case_ids: Sequence[str],
    cache_dir: str | Path,
    slice_range: tuple[int, int] = (30, 125),
    image_size: int = 128,
) -> Path:
    """
    Pre-extract normalized 2D slices for the selected patients into uncompressed `.npy` files.

    Stores all 4 modalities in float16 (`(N_slices, 4, H, W)`) and raw uint8 multi-class
    labels (`(N_slices, H, W)`), so any modality combination reuses the exact same cache
    and sub-region error analysis remains possible alongside binary training.
    """
    root_dir = Path(root_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    z0, z1 = slice_range

    missing = [
        cid
        for cid in case_ids
        if not (cache_dir / f"{cid}_img.npy").exists()
        or not (cache_dir / f"{cid}_lbl.npy").exists()
    ]
    if not missing:
        return cache_dir

    print(
        f"Preparing preprocessed cache in {cache_dir} for {len(missing)} patients "
        f"(slices {z0}..{z1 - 1}, size {image_size}x{image_size})..."
    )
    for cid in tqdm(missing, desc="Caching NIfTI -> .npy"):
        img_path = root_dir / "imagesTr" / f"{cid}.nii.gz"
        lbl_path = root_dir / "labelsTr" / f"{cid}.nii.gz"

        vol = np.asarray(nib.load(str(img_path)).dataobj, dtype=np.float32)  # (240, 240, 155, 4)
        lbl = np.asarray(nib.load(str(lbl_path)).dataobj, dtype=np.uint8)    # (240, 240, 155)

        vol = normalize_volume_brain_only(vol)

        # Slice along axial axis and move to (N_slices, C, H, W)
        img_slices = np.transpose(vol[:, :, z0:z1, :], (2, 3, 0, 1)).copy()
        lbl_slices = np.transpose(lbl[:, :, z0:z1], (2, 0, 1)).copy()

        img_slices, lbl_slices = _resize_slices(img_slices, lbl_slices, image_size)

        np.save(cache_dir / f"{cid}_img.npy", img_slices.astype(np.float16))
        np.save(cache_dir / f"{cid}_lbl.npy", lbl_slices.astype(np.uint8))

    return cache_dir


class BraTSBinaryDataset(Dataset):
    """
    PyTorch Dataset yielding 2D axial MRI slices and binary whole-tumor masks.

    Each item is `(image, mask)`:
      - image: float32 tensor of shape `(C, H, W)` where `C = len(modalities)`
      - mask:  float32 tensor of shape `(1, H, W)` with values in `{0.0, 1.0}`
    """

    def __init__(
        self,
        root_dir: str | Path,
        case_ids: Sequence[str],
        modalities: Sequence[str] = ("flair", "t1gd"),
        slice_range: tuple[int, int] = (30, 125),
        image_size: int = 128,
        cache_dir: str | Path | None = None,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.case_ids = list(case_ids)
        self.modality_indices, self.modality_names = parse_modalities(modalities)
        self.slice_start, self.slice_end = slice_range
        self.slices_per_volume = self.slice_end - self.slice_start
        self.image_size = image_size
        self.augment = augment

        if cache_dir is not None:
            self.cache_dir = prepare_cache(
                self.root_dir,
                self.case_ids,
                cache_dir,
                slice_range=slice_range,
                image_size=image_size,
            )
        else:
            self.cache_dir = None

        # Open read-only memory-mapped handles per worker for sub-millisecond reads
        self._img_mmaps: dict[str, np.ndarray] = {}
        self._lbl_mmaps: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.case_ids) * self.slices_per_volume

    def _get_patient_arrays(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return `(N_slices, 4, H, W)` image array and `(N_slices, H, W)` uint8 label array."""
        if self.cache_dir is not None:
            if case_id not in self._img_mmaps:
                self._img_mmaps[case_id] = np.load(
                    self.cache_dir / f"{case_id}_img.npy", mmap_mode="r"
                )
                self._lbl_mmaps[case_id] = np.load(
                    self.cache_dir / f"{case_id}_lbl.npy", mmap_mode="r"
                )
            return self._img_mmaps[case_id], self._lbl_mmaps[case_id]

        # Direct NIfTI fallback (slower for multi-epoch training due to gzip)
        img_path = self.root_dir / "imagesTr" / f"{case_id}.nii.gz"
        lbl_path = self.root_dir / "labelsTr" / f"{case_id}.nii.gz"
        vol = normalize_volume_brain_only(
            np.asarray(nib.load(str(img_path)).dataobj, dtype=np.float32)
        )
        lbl = np.asarray(nib.load(str(lbl_path)).dataobj, dtype=np.uint8)
        z0, z1 = self.slice_start, self.slice_end
        img_slices = np.transpose(vol[:, :, z0:z1, :], (2, 3, 0, 1)).copy()
        lbl_slices = np.transpose(lbl[:, :, z0:z1], (2, 0, 1)).copy()
        return _resize_slices(img_slices, lbl_slices, self.image_size)

    def get_patient_volume(self, case_id: str) -> tuple[torch.Tensor, np.ndarray]:
        """
        Return all slices of a single patient for 3D patient-level evaluation/visualization:
          - images: float32 tensor `(N_slices, C, H, W)` for selected modalities
          - raw_labels: uint8 ndarray `(N_slices, H, W)` preserving classes 0..3
        """
        img_arr, lbl_arr = self._get_patient_arrays(case_id)
        selected = np.array(img_arr[:, self.modality_indices, :, :], dtype=np.float32)
        raw_lbl = np.array(lbl_arr, dtype=np.uint8)
        return torch.from_numpy(selected), raw_lbl

    @staticmethod
    def _augment_pair(
        img: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Light, anatomically valid 2D augmentations (flips + mild intensity jitter)."""
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])   # Horizontal flip (left-right brain symmetry)
            mask = torch.flip(mask, dims=[2])
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[1])   # Vertical flip
            mask = torch.flip(mask, dims=[1])
        if torch.rand(1).item() < 0.5:
            # Mild brightness/contrast jitter on non-zero brain pixels only
            scale = 1.0 + (torch.rand(img.shape[0], 1, 1) * 0.2 - 0.1)
            shift = torch.rand(img.shape[0], 1, 1) * 0.2 - 0.1
            brain = img != 0
            img = torch.where(brain, img * scale + shift, img)
        return img, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_idx = idx // self.slices_per_volume
        local_z = idx % self.slices_per_volume
        case_id = self.case_ids[case_idx]

        img_arr, lbl_arr = self._get_patient_arrays(case_id)

        # Select requested MRI modalities -> (C, H, W) float32
        img_slice = np.array(img_arr[local_z, self.modality_indices, :, :], dtype=np.float32)
        # Combine tumor classes (1: edema, 2: non-enhancing, 3: enhancing) into binary mask
        bin_mask = (lbl_arr[local_z] > 0).astype(np.float32)[None, :, :]  # (1, H, W)

        img_tensor = torch.from_numpy(img_slice)
        mask_tensor = torch.from_numpy(bin_mask)

        if self.augment:
            img_tensor, mask_tensor = self._augment_pair(img_tensor, mask_tensor)

        return img_tensor, mask_tensor
