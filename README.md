# Modeling bacterial single-cell expression using a conditional variational autoencoder

## Overview
This repository contains the implementation and initial evaluation of a conditional variational autoencoder (CVAE) trained to predict bacterial single-cell expression profiles in response to cefepime treatment.

## Model description
The model uses a CVAE architecture, with condition embeddings injected in the latent space after latent sampling.

[insert image]

Key design features:
- Condition information embedding injected in latent space, allowing simple inference
- Adversarial regressors allow encoder to learn condition-free, basal encodings of gene expression

At inference time:
1. Sample from isotropic Gaussian in latent space
2. Embed desired time and dose
3. Compose condition embeddings with latent sample
4. Feed composed latent through decoder to obtain gene expression vector


## Repository structure
```text
singlecell-CVAE/
├── notebooks/         
  ├── 01-train.ipynb   # Example training loop
  └── 02-eval.ipynb    # Inference and model evaluation
├── src/               
  ├── bulk_data.py     # Pipeline to load bulk RNA-seq data
  ├── eda.py           # Exploratory analysis
  ├── eval.py          # Model evaluation plots
  ├── metadata.py      # Metadata and filename parser
  ├── sc_data.py       # Pipeline to load single-cell data
  ├── train.py         # Training loop
  └── vae.py           # CVAE architecture
```
## Data
The model was trained on 80,000 *Streptococcus pneumoniae* scRNA-seq profiles colleted across varying timepoints and doses of cefepime. The EDA also includes a comparison between the single-cell and bulk transcriptional profiles collected from a separate experiment.

## Requirements and setup

## 1. Clone the repository 
```bash 
git clone https://github.com/eddykang06/singlecell-CVAE.git cd singlecell-CVAE
``` 
## 2. Create and activate the environment 
```bash 
conda create -n singlecell-cvae python -y conda activate singlecell-cvae
```
## 3. Install dependencies 
```bash
pip install -r requirements.txt
```
