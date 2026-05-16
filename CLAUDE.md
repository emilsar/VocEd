# VocEd — Project Context for Claude

## What This Project Is
Applied deep learning for cytology image segmentation, taught as a series of Jupyter notebooks.
**Goal:** Predict the nucleus-to-cytoplasm (N/C) ratio from urothelial cell images using CNNs.
Audience: LA Mission College VOC Ed students (introductory level).

Companion textbook: [cvmath.club](https://cvmath.club/)

## Notebooks (Labs)

| File | Topic |
|---|---|
| `01_exploratory_segmentation.ipynb` | Grayscale thresholding, Dice score |
| `02_bayesian_optimisation.ipynb` | Auto-search for best thresholds, train/test split |
| `03_image_processing.ipynb` | Denoising, morphological opening/closing |
| `03v2_targeted_artifact_removal.ipynb` | Nucleus blemish removal via connected-component size filtering |
| `04_pixel_classifiers.ipynb` | k-NN on RGB colour vectors |
| `05_convolutions.ipynb` | Manual convolution, minimal PyTorch CNN |
| `06_unet.ipynb` | Encoder-decoder U-Net with skip connections |
| `07_nc_ratio_pipeline.ipynb` | End-to-end clinical N/C ratio evaluation |

## Dataset

- `imagedata/X/` — 200 RGB images as `.npy` files, shape `(3, 256, 256)`, float32 `[0, 1]`
- `imagedata/y/` — 200 segmentation masks as `.npy` files, shape `(256, 256)`
  - Labels: `0` background · `1` cytoplasm · `2` nucleus
- Source: Cedars/Project3 on Google Drive (synced as `.npy` arrays)

## Notebook Coding Style

Notebooks are student-facing — keep code parsimonious and readable:
- Use literal values instead of computed ones (e.g. `4` not `len(sample_ids)`)
- Use inline built-in colormaps (e.g. `cmap='viridis'`) — no custom `ListedColormap` or colormap variables
- No legend machinery (`mpatches`, `fig.legend`, `legend_patches`) unless strictly necessary
- No `tight_layout(rect=[...])` hacks — use plain `plt.tight_layout()` or nothing
- No unnecessary `fontsize=` annotations
- Avoid extra imports when `plt` or `np` already covers the need

## Git Repository
- GitHub: `emilsar/VocEd` (public)
- Branch: `main`, remote: `origin/main`
- Students open notebooks directly in Colab via the README badge links
