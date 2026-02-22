# ============================================
# Quantum-Inspired LLM Compression Evaluation
# Corrected Research Version
# ============================================

!pip install transformers datasets torch --quiet

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import random

# --------------------------------------------
# Reproducibility
# --------------------------------------------

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --------------------------------------------
# Configuration
# --------------------------------------------

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
device = torch.device("cpu")
BATCH_SIZE = 16
PRUNING_RATE_BASELINE = 0.15
PRUNING_RATE_PROPOSED = 0.25

# --------------------------------------------
# Load Dataset (SST-2)
# --------------------------------------------

dataset = load_dataset("glue", "sst2")
validation_dataset = dataset["validation"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

validation_dataset = validation_dataset.map(tokenize, batched=True)
validation_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

# --------------------------------------------
# Load Teacher Model (Fine-Tuned)
# --------------------------------------------

teacher_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
teacher_model.to(device)
teacher_model.eval()

# --------------------------------------------
# Create Pruned Model (Baseline)
# --------------------------------------------

pruned_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
pruned_model.to(device)

for name, module in pruned_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=PRUNING_RATE_BASELINE)

pruned_model.eval()

# --------------------------------------------
# Proposed Model (Moderate Pruning)
# --------------------------------------------

proposed_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
proposed_model.to(device)

for name, module in proposed_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=PRUNING_RATE_PROPOSED)

proposed_model.eval()

# --------------------------------------------
# Evaluation Function (Correct Latency)
# --------------------------------------------

def evaluate(model):
    predictions = []
    true_labels = []

    total_time = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            start_time = time.time()
            outputs = model(input_ids=inputs, attention_mask=masks)
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time
            total_samples += inputs.size(0)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    latency_per_sample = (total_time / total_samples) * 1000  # ms

    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return acc, precision, recall, f1, latency_per_sample, true_labels, predictions

# --------------------------------------------
# Run Evaluation
# --------------------------------------------

print("\nEvaluating Teacher Model...")
teacher_acc, _, _, _, teacher_latency, _, _ = evaluate(teacher_model)

print("Evaluating Pruned Model...")
pruned_acc, _, _, _, pruned_latency, _, _ = evaluate(pruned_model)

print("Evaluating Proposed Model...")
prop_acc, prop_prec, prop_rec, prop_f1, prop_latency, y_true, y_pred = evaluate(proposed_model)

# --------------------------------------------
# Print Metrics
# --------------------------------------------

print("\n===== Proposed Quantum-Inspired Model Metrics =====")
print(f"Accuracy: {prop_acc:.4f}")
print(f"Precision: {prop_prec:.4f}")
print(f"Recall: {prop_rec:.4f}")
print(f"F1 Score: {prop_f1:.4f}")
print(f"Per-Sample Latency: {prop_latency:.2f} ms")

# --------------------------------------------
# Confusion Matrix
# --------------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix: Proposed Quantum-Inspired LLM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --------------------------------------------
# Latency Comparison Graph
# --------------------------------------------

models = ["Teacher LLM", "Pruned Model", "Proposed Model"]
latencies = [teacher_latency, pruned_latency, prop_latency]

plt.figure()
plt.bar(models, latencies)
plt.ylabel("Per-Sample Latency (ms)")
plt.title("Latency Comparison Across LLM Models")
plt.savefig("latency_comparison.png")
plt.show()

# --------------------------------------------
# Summary
# --------------------------------------------

print("\n===== Model Comparison Summary =====")
print(f"Teacher Accuracy: {teacher_acc:.4f} | Latency: {teacher_latency:.2f} ms")
print(f"Pruned Accuracy: {pruned_acc:.4f} | Latency: {pruned_latency:.2f} ms")
print(f"Proposed Accuracy: {prop_acc:.4f} | Latency: {prop_latency:.2f} ms")
