# Quantum-Inspired Compression and Distillation of Large Language Models for Edge Intelligence

## 📌 Overview

This repository contains the implementation and evaluation code for the research paper:

**"Quantum-Inspired Compression and Distillation of Large Language Models for Edge Intelligence"**

The project explores a probabilistic optimisation-guided structured pruning approach combined with knowledge distillation to enable efficient deployment of Large Language Models (LLMs) under simulated edge-device constraints.

---

## 🎯 Objective

Large Language Models (LLMs) achieve strong performance but are computationally expensive for edge deployment.

This work investigates whether a quantum-inspired probabilistic optimisation strategy can:

- Preserve classification accuracy
- Maintain competitive inference latency
- Enable structured compression under CPU-based edge simulation

---

## 🧠 Methodology

The framework consists of:

1. Pretrained Teacher Model (DistilBERT)
2. Structured L1 Pruning
3. Quantum-inspired probabilistic component selection
4. Knowledge distillation
5. CPU-based latency evaluation

Compression is formulated as a constrained optimisation problem balancing:

- Accuracy degradation
- Inference latency

---

## 📊 Experimental Setup

- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Dataset: SST-2 (GLUE Benchmark)
- Hardware: CPU-only (edge simulation)
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Per-sample inference latency

---

## 📈 Results (SST-2 Validation Set)

| Model | Accuracy (%) | Per-Sample Latency (ms) |
|--------|--------------|--------------------------|
| Teacher (DistilBERT) | 91.06 | 191.59 |
| L1 Pruned | 90.48 | 182.52 |
| Proposed Quantum-Inspired | 90.37 | 184.63 |

### Proposed Model Metrics

- Accuracy: 90.37%
- Precision: 90.36%
- Recall: 90.77%
- F1-Score: 90.56%
- Per-Sample Latency: 184.63 ms

---

## 🔍 Confusion Matrix (Proposed Model)

True Negative: 385  
False Positive: 43  
False Negative: 41  
True Positive: 403  

Balanced classification performance is maintained after compression.

---

## 🚀 How to Run (Google Colab Compatible)

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
