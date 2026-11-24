# Brain Tumor Segmentation — Compact 2D U-Net (BraTS)

*Intro to ML in Biomedicine — Project 04*

Reproducible PyTorch pipeline for binary whole-tumor segmentation on MSD **`Task01_BrainTumour` (BraTS)** using a compact 2D U-Net (`1.94M` params, `features=(16, 32, 64, 128)`), fixed patient-level split (`100` train / `25` val / `25` test in `splits_150_seed42.json`), and `128×128` axial slices (`z = 30..124`, volume-level brain-only Z-score normalization).

---

## Research Questions

1. **RQ1 — Modalities:** How well does `FLAIR` or `T1gd` alone segment the combined binary tumor mask (edema + non-enhancing + enhancing) compared to `FLAIR + T1gd`?
2. **RQ2 — Loss Functions:** Under severe class imbalance (~98.7% background vs. ~1.3% tumor), how do **BCE**, **Soft Dice**, and **BCE + Soft Dice** affect 3D overlap, recall, and smaller tumors?

**Why `FLAIR` and `T1gd` (out of the 4 BraTS modalities: `FLAIR`, `T1w`, `T1gd`, `T2w`)?**
- **`FLAIR` over `T2w`:** Both highlight fluid and peritumoral edema, but `T2w` also makes normal cerebrospinal fluid (CSF) bright white, creating confounders near ventricles. `FLAIR` suppresses CSF to dark so the **whole-tumor/edema boundary** stands out cleanly.
- **`T1gd` over native `T1w`:** Native `T1w` mainly gives baseline anatomy where lesions can be subtle, whereas `T1gd` (post-gadolinium contrast) leaks across a disrupted blood–brain barrier to highlight the **active enhancing tumor rim** and dark necrotic core.
- **Most complementary pair (within the 1–2 modality project scope):** `FLAIR` captures the outer tumor/edema extent while `T1gd` pinpoints the active tumor core — and comparing `FLAIR` vs. `FLAIR + T1gd` directly tests whether gadolinium contrast meaningfully improves whole-tumor segmentation.

---

## Evaluation Metrics & Loss Functions

### Patient-Level 3D Metrics (Binary Prediction $P$ vs. Expert Mask $Y$)
All 95 axial slices per test patient are stacked into a 3D volume to count voxel-level **TP**, **FP**, and **FN** on the combined binary tumor mask (`classes 1+2+3 → 1`):

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \qquad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \qquad \text{IoU} = \frac{|P \cap Y|}{|P \cup Y|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}, \qquad \text{Dice} = \frac{2\,|P \cap Y|}{|P| + |Y|} = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}$$

### Loss Functions (`losses.py`)
Let $z_i \in \mathbb{R}$ be the raw U-Net logit at pixel $i$, $p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}} \in (0, 1)$ its predicted tumor probability, and $y_i \in \{0, 1\}$ the binary expert label.

1. **Binary Cross-Entropy (`bce` — `nn.BCEWithLogitsLoss`)**
   $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N} \Big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\Big]$$
   - *Why chosen:* Standard probabilistic baseline with smooth, well-calibrated pixel-wise gradients (computed from raw logits via log-sum-exp to avoid $\log 0$ underflow).
   - *Limitation:* Weights all pixels equally ($\frac{1}{N}$). With ~98.7% background pixels, BCE looks tiny (`< 0.01`) while under-segmenting smaller tumors (lower recall).

2. **Soft Dice Loss (`dice` — `BinarySoftDiceLoss`)**
   Literally the same $\text{Dice} = \frac{2|P \cap Y|}{|P| + |Y|}$ formula above, made differentiable by using continuous probabilities $p_i$ instead of hard `0/1` masks and subtracting from `1` so PyTorch minimizes it:
   $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}, \qquad \epsilon = 10^{-5}$$
   - *Why chosen:* Directly optimizes the Dice overlap metric and ignores true-negative background pixels ($y = 0, p \approx 0$ contribute $0$), preventing background from drowning out small tumors.
   - *Stability:* Sums over all pixels in the 32-slice mini-batch at once ("Batch Dice") so tumor-free slices don't cause noisy gradients, with $\epsilon = 10^{-5}$ preventing `0 / 0`.

3. **Combined BCE + Soft Dice (`bce_dice` — `BCEDiceLoss`)**
   $$\mathcal{L}_{\text{BCE+Dice}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$
   - *Why chosen:* Standard medical segmentation hybrid (e.g., nnU-Net) combining BCE's smooth pixel calibration with Dice's foreground overlap focus.

---

## Results (`n = 25` Held-Out Test Patients, Patient-Level 3D)

All runs share the same split, `15` epochs, `batch_size=32`, `AdamW (lr=1e-3)`, `CosineAnnealingLR`, and best-validation checkpoint selection.

| RQ | Modalities | Loss | Test 3D Dice | IoU | Precision | Recall | Small Tumors* | Edema Rec† | Enh Rec† |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RQ1** | `T1gd` | `bce_dice` | `0.6459 ± 0.216` | `0.5097` | `0.7515` | `0.6155` | `0.5567` | `51.5%` | `86.9%` |
| **RQ1** | `FLAIR` | `bce_dice` | `0.8048 ± 0.110` | `0.6866` | **`0.8706`** | `0.7772` | `0.7856` | `75.1%` | `85.7%` |
| **RQ1/2** | `FLAIR + T1gd` | `bce_dice` | `0.8330 ± 0.085` | `0.7223` | `0.8560` | `0.8312` | `0.7883` | `78.3%` | `92.8%` |
| **RQ2** | `FLAIR + T1gd` | `bce` | `0.8171 ± 0.100` | `0.7020` | `0.8571` | `0.8060` | `0.7677` | `75.9%` | `89.3%` |
| **RQ2** | `FLAIR + T1gd` | `dice` | **`0.8392 ± 0.079`** | **`0.7305`** | `0.8675` | **`0.8319`** | **`0.7969`** | **`78.7%`** | **`94.2%`** |

*\*Small Tumors = smaller half of test patients split at the median expert ground-truth tumor volume (`≤ 19,109` resized voxels, ≈ `67 cm³`, `n = 13`).*
*†Post-hoc diagnostic recall: fraction of expert edema (class 1) or enhancing tumor (class 3) voxels captured by the binary `0/1` prediction.*

- **RQ1:** `FLAIR` beats `T1gd` by +15.9 Dice pts because `T1gd` misses almost half of peritumoral edema (`51.5%` vs. `75.1%`), despite capturing the enhancing rim slightly better (`86.9%` vs. `85.7%`). Combining `FLAIR + T1gd` improves every metric (`0.833` Dice).
- **RQ2:** Both Dice-based losses (`Soft Dice`: `0.839`, `BCE + Dice`: `0.833`) outperform pure `BCE` (`0.817`), especially on smaller tumors (`0.797` / `0.788` vs. `0.768`) and recall. `BCE + Dice` peaked slightly higher on validation (`0.834` vs. `0.827`), while batch-level `Soft Dice` edged ahead on the test set.

---

## Quick Start

Extract `Task01_BrainTumour/` under `data/Task01_BrainTumour/` (or `./Task01_BrainTumour/` — both are auto-detected and ignored by `.gitignore`), then:

```bash
pip install -r requirements.txt
python3 train.py --modalities flair t1gd --loss bce_dice --epochs 15 --batch-size 32
python3 visualize.py explore --case BRATS_001
python3 visualize.py compare --runs runs/t1gd_bce_dice runs/flair_bce_dice runs/flair_t1gd_bce_dice --save compare.png
```
