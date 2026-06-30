# DMCL-MoRF

A Contrastive Learning–Based Dual-Modal Sequence–Structure Fusion Framework for MoRF Prediction

Identifying Molecular Recognition Features (MoRFs) within intrinsically disordered proteins is important for understanding protein interaction networks, but accurate residue-level MoRF prediction remains challenging because MoRFs are short, sparse, and closely related to disorder-to-order binding transitions. DMCL-MoRF is a dual-modal deep learning framework that integrates protein sequence representations and structure-derived topology for MoRF prediction.

DMCL-MoRF uses ProtT5 and ESM-2 embeddings to capture complementary residue-level sequence semantics. To adaptively integrate these two pretrained protein language model representations, the framework introduces a Discrepancy-driven Convolutional Gated Fusion (DCGF) module, which uses local representation discrepancies as gating cues to suppress redundant or conflicting information. In parallel, AlphaFold2-predicted structures are converted into residue-level geometric graphs, and an edge-aware graph neural network is used to encode spatial topology and distance-dependent residue interactions. Finally, fragment-level cross-modal contrastive learning is introduced to align sequence and structure representations in a shared latent space, improving multimodal consistency and robustness under class-imbalanced conditions.

<img width="554" height="373" alt="image" src="https://github.com/user-attachments/assets/a45ded2f-35e3-41c7-95a7-8d7a6062aecc" />

# System requirement

Python == 3.9.21
NumPy == 1.26.4
PyTorch == 2.6.0+cu124
PyTorch Scatter == 2.1.2+pt26cu124
PyTorch Cluster == 1.6.3+pt26cu124
PyTorch Geometric == 2.6.1
Biopython == 1.82
fair-esm == 2.0.0
sentencepiece == 0.2.0
transformers == 4.49.0

# Description

DMCL-MoRF is implemented in Python based on the PyTorch framework for residue-level prediction of Molecular Recognition Features (MoRFs). The model adopts a dual-branch sequence–structure architecture. The sequence branch extracts and fuses ProtT5 and ESM-2 embeddings using the Discrepancy-driven Convolutional Gated Fusion (DCGF) module. The structure branch constructs AlphaFold2-based residue graphs and uses an edge-aware graph neural network to capture spatial residue dependencies. A fragment-level cross-modal contrastive learning objective is further used to align sequence and structural representations, improving the robustness of MoRF prediction.

# Datasets

DMCL-MoRF provides several datasets in FASTA format for model training and evaluation, including:

 `Train.fasta`:
  The main training set used for model learning in DMCL-MoRF.

 `Test1.fasta`:
  The held-out test set constructed from the integrated benchmark dataset. It is used to evaluate the residue-level prediction performance of DMCL-MoRF under the main experimental setting.

 `Test2.fasta`:
  An independent test set reconstructed from the previously reported Test49 dataset according to the published UniProt accession numbers. It is used to further evaluate the generalization ability of DMCL-MoRF.

 `Training421.fasta`:
  The classical Training421 dataset, consisting of 421 protein sequences. It is one of the public benchmark datasets used in MoRF prediction studies.

 `Test 419.fasta`:
  The classical Test419 dataset, consisting of 419 protein sequences. It is used as one of the public source datasets for benchmark construction.

 `Test 45.fasta`:
  The Test45 dataset proposed in previous MoRF prediction studies. In this work, it is used as part of the main benchmark construction and also for supplementary low-homology validation.

 `test49.fasta`:
  The original or reconstructed Test49-related dataset used for independent evaluation.

All datasets are protein-sequence-level files used for training, testing, or supplementary validation in the MoRF prediction task.

# Feature extraction

The `Extract_features` directory contains scripts for generating sequence and structure features. Given the input dataset path, users can extract:

 ESM-2 residue-level embeddings
 ProtT5 residue-level embeddings
 AlphaFold2-derived structural features

These features are used as inputs for the sequence branch and structure branch of DMCL-MoRF.

# Model

The model-related code is stored in the `model` directory. It contains scripts corresponding to:

 the sequence representation branch,
 the structure representation branch,
 the DCGF-based fusion module,
 the final residue-level prediction module.

The trained models are saved in the `save_model` directory.

# Test

The testing script is located in `Test.py`. Users can run this script to evaluate the trained DMCL-MoRF model on Test1, Test2, or other prepared FASTA datasets.

# Citation

If you use DMCL-MoRF in your research, please cite our paper once it is available.
