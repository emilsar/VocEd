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
