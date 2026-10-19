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
| `Labs/Lab5_pixels_as_vectors_knn_logistic_regression.ipynb` | Pixel × RGB table (transpose/reshape); class-balanced vs random training tables; k-NN with k chosen on 10 validation images; logistic regression weights vs mean class colours; final test. ~11 min locally (k-NN prediction). |
| `Labs/Lab6_random_forest_mlp_colour_ceiling.ipynb` | Random forest with leaf sizes 1/20/100, importances; MLP weight count; colour ceiling (32 levels per channel, majority label per colour) framed as a pixel-*accuracy* ceiling; final test. ~3 min. |
| `Labs/Lab7_feature_engineering.ipynb` | Ten feature maps (R, G, B, blur2, blur8, median5, sobel, std5, gabor0, gabor90) as forest columns; ablation by group; importances; Lab 4 clean-up on top; final test. GLCM left out (a Think explains the cost). ~3½ min. |
| `Labs/Lab8_convolutions_minimal_cnn.ipynb` | Convolution by hand (checked against scipy), output sizes, MinimalCNN (5,187 weights), 25 epochs with validation Dice per epoch, learned kernels and maps, final test. ~4 min on CPU. |
| `Labs/Lab9_unet.ipynb` | SmallUNet (472,899 weights), CE + soft Dice, flip/rot90 augmentation, `train()` keeps the best validation epoch; runs: plain, + augmentation, no skips; saves `unet_weights.pt`; final test + N/C scatter. ~80 min on this 1-CPU box; minutes on a Colab GPU. |
| `Labs/Lab10_end_to_end_nc_ratio.ipynb` | Rebuilds Lab 4 (thresholds rounded to 0.353 / 1.00) and Lab 7's forest, loads the U-Net from `models/unet_lab9.pt` (a copy of Lab 9's key-run weights); R², bias, within ±0.05; three worst images; flagging at N/C ≥ 0.5 with sensitivity/specificity; whole-course table. ~35 s. |

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

**Labs 5–10 results to remember** (key runs; test = the same 40 images, as Dice / N/C R² / within ±0.05):
- Lab 5: k = 25 chosen on validation.  The balanced table calls more true cytoplasm nucleus (18.7% vs 11.7% random).  Test k-NN 0.763 / −0.341 / 30%, LR 0.752 / −0.955 / 30%.
- Lab 6: validation RF(100) 0.818, MLP 0.817; colour ceiling = pixel accuracy 0.898 (its Dice is *below* RF's, hence the accuracy framing).  Test RF 0.768 / −0.412 / 32%, MLP 0.768 / −0.467 / 30%.
- Lab 7: validation colour 0.818 → + neighbourhood 0.821 → + edges 0.822 → + texture 0.821 (texture slightly hurts).  Test 0.774 / −0.343 / 30%; + clean-up 0.771 / −0.296.
- Lab 8: validation Dice 0.755 at epoch 5, flat ~0.79 from epoch 17, 0.796 at 25 (0.802 at 40 in an experiment).  Test 0.755 / −0.714 / 32%.  5 of 16 conv1 maps are blank.
- Lab 9: best validation U-Net 0.825, + aug 0.827, no skips 0.825 — the skips and augmentation are within noise on these blocky labels; curves dip to ~0.69 on single epochs.  Test (+ aug) 0.784 / −1.120 / 25%: best Dice, worst N/C.
- Lab 10: bias Lab 4 +0.066, forest +0.027, U-Net +0.240.  Worst N/C for all three models: test images 145, 108, 154 (darkly stained cytoplasm called nucleus); image 40 has its nucleus missed by all three.  Cutoff 0.5: 16 should be flagged; Lab 4 / forest sensitivity 62%, specificity 75%; U-Net 69% / 67%.  No model reaches a positive N/C R².

**Answer keys** are in `Labs/Solutions/`, which is listed in `.gitignore` so they never reach
the public repo.  They are generated, together with the student versions, from one script
per lab where answers are written as `«answer»` (student build → `___`).  The scripts are in
`Labs/Solutions/` too: `cd Labs/Solutions && python3 make_lab1.py ../Lab1_exploratory_analysis.ipynb Lab1_exploratory_analysis_solutions.ipynb`.  If you edit a lab
by hand, edit its answer key the same way.

The original series (01–07, 03v2), plus `Machine Learning Approaches.ipynb`,
`feature_engineering.ipynb` and `project_3_outline.ipynb`, was moved to `Labs/Old_Labs/` on 2026-09-30.  Book
Chapter 8 links to Old Labs 05 and 06 by path (not yet repointed at Labs 8 and 9).

`models/unet_lab9.pt` is committed on purpose: Lab 10 loads it so it does not depend on a Lab 9
session.  If Lab 9's code changes, re-run its key and copy the new `unet_weights.pt` there.

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
