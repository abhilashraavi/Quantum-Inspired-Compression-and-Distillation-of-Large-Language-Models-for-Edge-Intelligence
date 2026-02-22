Quantum-Inspired Optimisation for Transformer Pruning Under Edge Constraints
Overview

This repository contains the implementation for the paper:

“Quantum-Inspired Compression and Distillation of Large Language Models for Edge Intelligence”

The project explores a probabilistic, multi-objective optimisation framework for guiding pruning configurations in transformer-based language models under simulated edge-device constraints.

Unlike fixed magnitude-based pruning, this framework formulates pruning configuration selection as a constrained optimisation problem that jointly considers:

Predictive accuracy

Inference latency

Model size

The optimisation process is inspired by quantum-inspired q-bit representation and rotation-based probability updates.

Key Features

Quantum-inspired probabilistic component selection

Multi-objective fitness function (accuracy + latency + memory)

Rotation-based parameter update mechanism

Iterative global search of pruning configurations

CPU-based edge simulation

Confusion matrix generation

Latency comparison analysis

Model and Dataset

Model: distilbert-base-uncased-finetuned-sst-2-english

Dataset: SST-2 (GLUE benchmark)

Task: Binary sentiment classification

Evaluation: CPU-based inference latency simulation

Method Summary

Each linear module within the transformer architecture is assigned a probabilistic pruning intensity controlled by a quantum-inspired angle parameter (θ).

At each optimisation iteration:

A pruning configuration is sampled using sin²(θ) probability mapping.

The model is evaluated.

A multi-objective fitness score is computed.

Rotation-based updates adjust θ values.

The best configuration is retained.

The optimisation does not physically remove parameters but applies weight masking via L1 pruning.

Results (Current Implementation)
Model	Accuracy	Latency (ms)
Teacher (DistilBERT)	91.06%	178.64
L1 Pruned	90.48%	188.90
Quantum-Inspired	91.40%	189.14

The optimisation-guided configuration achieves competitive predictive performance under simulated edge conditions.

Installation
pip install -r requirements.txt
Run the Experiment
python main.py

Or execute in Google Colab.
