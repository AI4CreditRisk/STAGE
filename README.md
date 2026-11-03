# STAGE: Structured Temporal and Graph Encoding Network for Credit Default Prediction

This repository provides the implementation of **STAGE (Structured Temporal and Graph Encoding Network)** for credit default prediction.

## Overview

STAGE is a deep learning framework designed to capture structured risk information and borrower relationships while addressing heterogeneous risk signals and class imbalance.

The framework consists of three main components:

- **Structured Risk Encoding (SRE):** Organizes risk attributes according to domain-defined orderings and captures borrower relationships through ordered feature encoding and multi-view graph attention.
- **Gated Adaptive Fusion (GAF):** Learns borrower-specific weights to adaptively combine ordered risk representations and graph representations.
- **Composite Loss Optimization (CLO):** Integrates class weighting, focal modulation, and label smoothing to improve learning under imbalanced class distributions.

## Datasets

The experiments are conducted on four credit datasets:

- LendingClub (LC)
- Give Me Some Credit (GMSC)
- Chinese Personal Loan (CPL)
- UCI Default of Credit Card Clients (UCI)

Dataset access and preprocessing instructions will be provided separately.

## Requirements

The implementation is based on Python.

Detailed package dependencies and environment specifications will be provided in `requirements.txt`.

## Usage

Training and evaluation instructions will be added after the final code structure and execution arguments are verified.

## Reproducibility

The repository is intended to support reproducibility of the main experiments, including:

- Performance comparisons with baseline models
- Component ablation studies
- Mechanism-oriented diagnostic experiments
- Sensitivity and computational efficiency analyses

## License

License information will be added before public release.
