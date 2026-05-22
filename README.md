# Redshift-Independent  Machine Learning for Celestial Classification

## Overview

Classifying celestial objects such as stars, galaxies, and quasars is a fundamental problem in astrophysics, playing a major role in understanding galaxy evolution and the large-scale structure of the Universe. Traditionally, accurate classification has depended heavily on spectroscopic redshift measurements. However, spectroscopic observations are expensive and time-consuming, meaning that only a tiny fraction of the billions of objects detected in modern sky surveys have confirmed redshift data.

This project introduces a redshift-independent machine learning framework that classifies celestial objects using only photometric and morphological information from the Sloan Digital Sky Survey (SDSS). The pipeline combines advanced feature engineering with powerful machine learning models, including XGBoost, Feedforward Neural Networks (FNNs), and a stacking ensemble architecture, to effectively distinguish stars, galaxies, and quasars even in the absence of spectroscopic data.

## Key Features

* **Redshift Independent Machine learning framework:**
  Simulates real-world survey conditions by removing spectroscopic redshift information and relying entirely on photometric features for classification.

* **Advanced Feature Engineering:**
  Uses higher-order color indices such as (u-r), (r-z), and (g-i) to capture broader spectral patterns, while also incorporating morphological features like `psf_model_diff` to differentiate compact and extended objects.

* **Multiple Machine Learning Models:**
  Implements and compares Random Forest, XGBoost, Feedforward Neural Networks, and a Stacking Ensemble framework.

* **Model Explainability with SHAP:**
  Applies SHAP analysis to interpret feature importance and understand how the models compensate for missing redshift information.

## Dataset

The models are trained and validated using data from **SDSS Data Release 17 (DR17)**.

* **Input Features:** Optical broad-band magnitudes ((u, g, r, i, z))
* **Target Classes:** Star, Galaxy, and Quasar (QSO)
* **Application Dataset:** A large unlabeled photometric dataset used to evaluate real-world deployment capability.

## Repository Structure

* `/data/` – SDSS datasets and application datasets *(excluded due to size)*
* `/models/` – Saved trained models and preprocessing objects
* `/notebooks/` – Jupyter notebooks for each experimental stage
* `/results/` – Evaluation reports, confusion matrices, and SHAP visualizations

## Results

The models achieved extremely high performance when spectroscopic redshift information was available, reaching accuracies of up to 99.90%. Even after removing redshift information, the models maintained strong performance and demonstrated impressive robustness.

### Redshift-Agnostic Accuracy Benchmarks

| Model                         | Basic Colors | Colors + Morphology | Colors + Higher-Order Features |
| ----------------------------- | ------------ | ------------------- | ------------------------------ |
| Random Forest                 | 92.92%       | 92.22%              | 92.88%                         |
| XGBoost                       | 93.27%       | 92.94%              | 93.37%                         |
| FNN (Keras)                   | 92.22%       | 91.41%              | 93.06%                         |
| Stacking Ensemble (XGB + FNN) | **93.41%**   | **92.83%**          | **93.48%**                     |

The inclusion of higher-order color indices significantly improved quasar detection performance, highlighting the importance of broader spectral information in photometric classification.

## SHAP Analysis

SHAP-based interpretability revealed how the models adapt when redshift information is unavailable:

1. With morphological features included, structural parameters such as `psf_model_diff_i` became dominant predictors for separating galaxies from point-like sources.
2. In purely photometric settings, the models relied heavily on engineered color indices such as (r-i), (g-r), and (r-z) to distinguish quasars from stars.

## Future Work

Potential future improvements include:

* Integrating mid-infrared data from ALLWISE to improve quasar-star separation
* Developing multimodal neural networks that combine photometric data with raw astronomical image cutouts
* Applying semi-supervised learning techniques to leverage unlabeled survey data


