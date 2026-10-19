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
| `Labs/Lab3_grid_search_bayesian_optimisation_pipeline.ipynb` | Train/validation/test (same 40 test images as Lab 2); 11×11 grid over both thresholds; Bayesian optimisation with `gp_minimize` (Appendix B); pipeline of image steps (blur, median, NLM, unsharp) and mask steps (opening/closing) as lists of functions; ablation with thresholds re-tuned per pipeline; 5-knob BO; final test once + N/C R². Light code puzzles, mostly 🤔 Think puzzles. Runs in ~3½ min locally. |
| `Labs/Lab4_targeted_cleanup_connected_components.ipynb` | Connected components (`skimage.measure.label`, `connectivity=2`; areas via `np.bincount`, not `remove_small_objects`, whose `min_size` is deprecated in skimage 0.26 and whose `max_size` does not exist before it); `remove_specks`, `fill_holes`, ready-made `remove_debris`; how often each step's changes are right; ablation with thresholds re-tuned per pipeline (20 BO calls); 3-knob BO; final test once, with N/C within ±0.05. Runs in ~1 min locally. |

**Think puzzles** (from Lab 3 on): a markdown question headed `### 🤔 Think n — …`, followed by
an answer cell made with `nb.answer(...)` in `nbtools.py` — students see "✏️ Your answer", the
key shows "✏️ Model answer".  Model answers quote numbers from an executed run of the key; if
code changes, re-run the key with `nbclient` and re-check them.

**Lab 3 results to remember** (key run, `random_state=0`): tuning both thresholds moves
`t_background` from 0.85 to 1.00 (background is saturated white) and training Dice 0.734 → 0.781;
every pipeline step changes validation Dice by < 0.004 and unsharp masking lowers it by ~0.03;
5-knob BO picks σ ≈ 0.15 and radii 4/5, validation 0.812; test Dice: Lab 2 0.719, two thresholds
0.749, five knobs 0.770 — N/C R² stays negative throughout.  NumPy trap: `gray_all` is float32,
so `gray < 0.84` and `gray < np.float64(0.84)` can disagree on pixels exactly at the threshold.

**Lab 4 results to remember** (key run): image 7 has 2 true nuclei but 47 predicted nucleus blobs, 39 under 100 px; true nuclei (train) median 2,049 px, 5th percentile 115.  `remove_specks` is right on 87% of the pixels it changes, `fill_holes` on 62% (training image 167: it fills 4,853 true-cytoplasm pixels inside a dark ring).  Validation: none 0.802, specks 0.803, fill 0.806, debris 0.802 (does nothing with `t_background = 1.00`), specks+fill 0.812 with `t_nucleus` re-tuned down to 0.353, Lab 3 open/close r=2 0.805.  3-knob BO picks min_size 197: train higher, validation lower (0.806).  Test: threshold only 0.749 / R² −0.757 / 22% within ±0.05; specks+fill 0.772 / −0.465 / 35%.  Lab 3's five-knob row, re-run: 0.770, R² −0.812.

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
