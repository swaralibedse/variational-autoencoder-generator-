# variational-autoencoder-generator-
Variational Autoencoder (VAE) for molecular SMILES generation trained on the MOSES dataset — part of the MolVista AI project for AI-powered drug discovery.
# 🧬 MolVista AI – VAE Molecule Generator

This module implements a **Variational Autoencoder (VAE)** for generating molecular SMILES strings, trained on the **MOSES dataset**.
It is a key component of **MolVista AI – AI-Powered Molecular Docking & Drug Generation for Accelerated Drug Discovery**.

---

## 🌍 Overview

**MolVista AI** aims to accelerate early-stage drug discovery by combining:

* 🧠 **Deep Generative Models (VAE)** for molecule creation
* 🧩 **Graph Neural Networks (GNN)** for binding affinity prediction
* 🧫 **AI-assisted Docking and Visualization**

This **VAE Molecule Generator** learns molecular patterns from SMILES strings and generates new, chemically valid molecules.

---

## 🏗️ Project Architecture

Below is the conceptual flow of the **VAE Molecule Generator** module within MolVista AI:

```
                   ┌──────────────────────────────┐
                   │     MOSES Dataset (SMILES)   │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                       ┌────────────────────┐
                       │   tokenizer.py     │
                       │ (Encode/Decode)    │
                       └────────┬───────────┘
                                │
                                ▼
                        ┌──────────────┐
                        │   vae.py     │
                        │ (Encoder +   │
                        │  Decoder)    │
                        └──────┬───────┘
                               │
                               ▼
                     ┌─────────────────────┐
                     │   train_vae.py      │
                     │  (Training Loop)    │
                     └────────┬────────────┘
                              │
                 ┌────────────┼─────────────────────────────┐
                 │                                            │
                 ▼                                            ▼
       ┌──────────────────┐                       ┌──────────────────┐
       │ sample_vae.py    │                       │ evaluate.py      │
       │ Generate SMILES  │                       │ Uniqueness Check │
       └──────────────────┘                       └──────────────────┘

Output Files:
 - vae_model.pt
 - loss_curve.png
 - sample_output.txt
 - uniqueness.txt
```

---

## 🔧 Components

| File                  | Description                                       |
| --------------------- | ------------------------------------------------- |
| `tokenizer.py`        | Tokenizes and detokenizes SMILES strings          |
| `vae.py`              | PyTorch-based MoleculeVAE model (Encoder–Decoder) |
| `train_vae.py`        | Trains VAE model with checkpoint resumption       |
| `sample_vae.py`       | Samples and decodes new molecules                 |
| `evaluate.py`         | Computes uniqueness percentage                    |
| `models/vae_model.pt` | Saved trained model weights                       |
| `loss_curve.png`      | Reconstruction + KL Loss visualization            |

---

## 📁 Dataset

This module uses the **MOSES dataset**, a benchmark for generative molecule models.
Your dataset file should be named **`moses_train.csv`** and include a column named `SMILES`.

Example:

```csv
SMILES
CCO
C1=CC=CC=C1
CC(=O)OC1=CC=CC=C1C(=O)O
```

---

## 📈 Outputs

| Output              | Description                              |
| ------------------- | ---------------------------------------- |
| `vae_model.pt`      | Trained model weights (saved per epoch)  |
| `loss_curve.png`    | Plot of Reconstruction + KL Loss         |
| `sample_output.txt` | Input vs Decoded SMILES samples          |
| `uniqueness.txt`    | Percentage of unique generated molecules |

---

## ▶️ Training & Usage

To train, generate samples, and evaluate results, follow these commands:

```bash
# 1. Activate environment
conda activate molvista

# 2. Navigate to project
cd C:\molvista-ai

# 3. Set Python path
set PYTHONPATH=C:\molvista-ai

# 4. Train or resume VAE
python generator/scripts/train_vae.py
# → Saves model (vae_model.pt) and loss curve (loss_curve.png)

# 5. Sample new molecules
python generator/scripts/sample_vae.py
# → Outputs sample_output.txt (Input vs Decoded SMILES)

# 6. Evaluate uniqueness
python generator/scripts/evaluate.py
# → Outputs uniqueness.txt (Unique SMILES %)
```

---

## 🧠 Model Details

* **Framework:** PyTorch
* **Architecture:** Variational Autoencoder (Encoder + Decoder)
* **Input Representation:** SMILES strings
* **Latent Space:** Continuous vector space for molecule encoding
* **Loss Function:**

  * Reconstruction Loss
  * KL Divergence Loss
* **Optimization:** Adam optimizer

The VAE learns to encode molecular structure into a latent space and decode it back into valid SMILES — enabling *de novo* molecule generation.

---

## 🧩 Example Results

**Loss Curve:**
Displays reconstruction + KL divergence loss over epochs.
*File: `loss_curve.png`*

**Sample Output:**

```
Input SMILES  →  Decoded SMILES
CCO           →  CCO
C1=CC=CC=C1   →  C1=CC=CC=C1
CC(=O)OCC     →  CC(=O)OCC
```

**Uniqueness Report:**

```
Unique SMILES: 92.4%
```

---

## 🚀 Features

* ✅ Trainable VAE model for molecular SMILES
* ✅ Resume training from checkpoints
* ✅ Automatic generation & uniqueness evaluation
* ✅ Visualization of training progress
* ✅ Modular design (easy integration with future GNN modules)

---

## 🔬 Future Work

Upcoming modules in **MolVista AI** will include:

* **GNN-Based Binding Affinity Predictor**
  Predicts how strongly generated molecules bind to target proteins.

* **Conditional Molecule Generation**
  Control properties like logP or QED during generation.

* **Integration with Docking Pipelines**
  For end-to-end AI-driven drug design.

---

## 📜 Citation

If you use this work or codebase, please cite:

> Bedse, Swarali (2025). *MolVista AI – VAE Molecule Generator: A Variational Autoencoder for Molecular SMILES Generation.*

---

## 👩‍💻 Author

**Swarali Bedse**
Final Year B.Tech – Artificial Intelligence & Data Science
📫 [LinkedIn](https://www.linkedin.com) • [GitHub](https://github.com/shreeyan)

---

## 🖼️ Suggested Repository Banner

**Title:** *AI Generating Molecules*
Image Idea: A glowing neural network creating molecular structures — symbolizing AI chemistry.
Find one on [Unsplash](https://unsplash.com/s/photos/ai-molecule) or [Pexels](https://www.pexels.com/search/ai%20molecule/).

---

## 🏷️ Tags

`#DeepLearning` `#DrugDiscovery` `#GenerativeAI` `#PyTorch` `#MolecularDesign` `#AI` `#ComputationalChemistry` `#MachineLearning` `#VAE` `#Bioinformatics`
