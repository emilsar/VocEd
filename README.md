# VocEd — Cytology Segmentation Course

Applied deep learning for cytology image segmentation.
**Goal:** Predict the nucleus-to-cytoplasm (N/C) ratio using convolutional neural networks.

## Labs

Labs live in `Labs/`.  Each one contains fill-in-the-blank puzzles that link to the matching
section of the textbook and come with hints.

| # | Notebook | Topic | Open in Colab |
|---|---|---|---|
| 1 | Exploratory Analysis | Loading the data, channels, displaying images and masks, class counts, N/C ratio, grayscale, brightness by class | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Lab1_exploratory_analysis.ipynb) |
| 2 | Thresholding, Denoising & Evaluation | Two-threshold segmenter, accuracy vs Dice/IoU, Gaussian & median filters, stratified train/test split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Lab2_thresholding_denoising_evaluation.ipynb) |
| 3 | Grid Search, Bayesian Optimisation & a Pipeline | Train/validation/test, grid search vs Bayesian optimisation, a switchable pipeline of blur, median, NLM, unsharp masking and opening/closing, ablation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Lab3_grid_search_bayesian_optimisation_pipeline.ipynb) |
| 4 | Targeted Cleanup & Connected Components | Counting and sizing blobs with `label` and `np.bincount`, removing nucleus specks, filling holes, ablation, final test of the thresholding approach | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Lab4_targeted_cleanup_connected_components.ipynb) |
| 5 | Pixels as Vectors: k-NN & Logistic Regression | A pixel as a point in colour space, class-balanced vs random training tables, choosing k on validation, reading logistic-regression weights | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Lab5_pixels_as_vectors_knn_logistic_regression.ipynb) |
| 6 | Random Forest, MLP & the Colour Ceiling | Leaf size and memorising, feature importances, counting MLP weights, measuring the best any colour-only rule can do | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Lab6_random_forest_mlp_colour_ceiling.ipynb) |
| 7 | Feature Engineering | Blur, median, Sobel, local standard deviation and Gabor feature maps as columns for a random forest, ablation by feature group | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Lab7_feature_engineering.ipynb) |
| 8 | Convolutions & a Minimal CNN (GPU) | Convolution by hand, a 5,187-weight CNN in PyTorch, the training loop, when training is finished, learned kernels, receptive field | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Lab8_convolutions_minimal_cnn.ipynb) |
| 9 | U-Net (GPU) | Encoder, decoder and skip connections, CE + soft Dice loss, data augmentation, early stopping on validation, skip-connection ablation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Lab9_unet.ipynb) |
| 10 | End to End: the N/C Ratio | Thresholds, forest and U-Net judged by the N/C ratio: R², bias, worst cases, flagging with sensitivity and specificity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Lab10_end_to_end_nc_ratio.ipynb) |

### Earlier labs

The original lab series is kept in `Labs/Old_Labs/`.

| # | Notebook | Topic | Open in Colab |
|---|---|---|---|
| 01 | Exploratory Segmentation | Grayscale thresholding, Dice score | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/01_exploratory_segmentation.ipynb) |
| 02 | Bayesian Optimisation | Auto-search for best thresholds, train/test split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/02_bayesian_optimisation.ipynb) |
| 03 | Image Processing | Denoising, morphological opening/closing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/03_image_processing.ipynb) |
| 03v2 | Targeted Artifact Removal | Nucleus blemish removal via connected-component size filtering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/03v2_targeted_artifact_removal.ipynb) |
| 04 | Pixel Classifiers | k-NN on RGB colour vectors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/04_pixel_classifiers.ipynb) |
| 05 | Convolutions & CNN | Manual convolution, minimal PyTorch CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/05_convolutions.ipynb) |
| 06 | U-Net | Encoder-decoder with skip connections | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/06_unet.ipynb) |
| 07 | N/C Ratio Pipeline | End-to-end clinical evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/07_nc_ratio_pipeline.ipynb) |
| — | Machine Learning Approaches | Logistic regression, random forest, MLP on pixels | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/Machine%20Learning%20Approaches.ipynb) |
| — | Feature Engineering | Handcrafted feature maps (Chapter 7) on real cell images | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/feature_engineering.ipynb) |
| — | Project 3 Outline | Extending the U-Net: five levels of student projects | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/VocEd/blob/main/Labs/Old_Labs/project_3_outline.ipynb) |

## Dataset

`imagedata/X/` — 200 RGB images, shape `(3, 256, 256)`, float32 `[0, 1]`
`imagedata/y/` — 200 segmentation masks, shape `(256, 256)`, labels: `0` background · `1` cytoplasm · `2` nucleus

## Textbook

Course textbook: [cvmath.club](https://cvmath.club/)
