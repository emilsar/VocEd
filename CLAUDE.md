# VocEd — Project Context for Claude

## What This Project Is
Applied deep learning for cytology image segmentation, taught as a series of Jupyter notebooks.
**Goal:** Predict the nucleus-to-cytoplasm (N/C) ratio from urothelial cell images using CNNs.
Audience: LA Mission College VOC Ed students (introductory level).

Companion textbook: [cvmath.club](https://cvmath.club/)

## Notebooks (Labs)

Current labs are in `Labs/`.  They contain fill-in-the-blank puzzles (`___`), each with a
textbook link (obooks.tech/imageprocessing) and click-to-open hints, followed by an
`assert` check that prints ✅.

| File | Topic |
|---|---|
| `Labs/Lab1_exploratory_analysis.ipynb` | Purely exploratory: shapes, channels, transpose, overlays, class counts, true N/C ratio, BT.601 grayscale, per-class brightness histogram. Stops before thresholding. |
| `Labs/Lab2_thresholding_denoising_evaluation.ipynb` | Two-threshold segmenter; accuracy trap, precision/recall, Dice, IoU (Appendix C); Gaussian/median denoising (Ch 2 §2.6); stratified train/test split and tuning `t_nucleus` on train only; N/C scatter on test. |

**Answer keys** are in `Labs/Solutions/`, which is listed in `.gitignore` so they never reach
the public repo.  They are generated, together with the student versions, from one script
per lab where answers are written as `«answer»` (student build → `___`).  The scripts are in
`Labs/Solutions/` too: `cd Labs/Solutions && python3 make_lab1.py ../Lab1_exploratory_analysis.ipynb Lab1_exploratory_analysis_solutions.ipynb`.  If you edit a lab
by hand, edit its answer key the same way.

The original series (01–07, 03v2), plus `Machine Learning Approaches.ipynb`,
`feature_engineering.ipynb` and `project_3_outline.ipynb`, was moved to `Labs/Old_Labs/` on 2026-09-30.  Book
Chapter 8 links to Old Labs 05 and 06 by path.

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
