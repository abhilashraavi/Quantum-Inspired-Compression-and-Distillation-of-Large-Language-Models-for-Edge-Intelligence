# ============================================
# Quantum-Inspired LLM Compression Framework
# Full Research-Aligned Implementation
# ============================================

!pip install transformers datasets torch --quiet

import torch
import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune

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

# Multi-objective weights (as in paper)
ALPHA = 0.7   # Accuracy priority
BETA = 0.2    # Latency weight
GAMMA = 0.1   # Memory weight

DELTA_THETA = 0.05
ITERATIONS = 5

# --------------------------------------------
# Load Dataset (SST-2)
# --------------------------------------------

dataset = load_dataset("glue", "sst2")
validation_dataset = dataset["validation"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["sentence"],
                     padding="max_length",
                     truncation=True,
                     max_length=128)

validation_dataset = validation_dataset.map(tokenize, batched=True)
validation_dataset.set_format(type="torch",
                              columns=["input_ids", "attention_mask", "label"])

loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

# --------------------------------------------
# Evaluation Function
# --------------------------------------------

def evaluate_full(model):
    model.eval()
    total_time = 0
    total_samples = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            start = time.time()
            outputs = model(input_ids=inputs, attention_mask=masks)
            end = time.time()

            total_time += (end - start)
            total_samples += inputs.size(0)

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    latency = (total_time / total_samples) * 1000
    acc = accuracy_score(true_labels, predictions)

    return acc, latency, true_labels, predictions

# --------------------------------------------
# Model Size Calculation
# --------------------------------------------

def model_size_mb(model):
    param_count = sum(p.numel() for p in model.parameters())
    return param_count / 1e6, (param_count * 4) / (1024**2)

# --------------------------------------------
# Get Linear Layers
# --------------------------------------------

def get_linear_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers.append(name)
    return layers

# --------------------------------------------
# Apply Pruning Configuration
# --------------------------------------------

def apply_pruning(model, config):
    for name, module in model.named_modules():
        if name in config:
            prune.l1_unstructured(module, name="weight", amount=config[name])

# --------------------------------------------
# Multi-objective Fitness Function
# --------------------------------------------

def compute_fitness(acc, latency, size_mb,
                    alpha=ALPHA, beta=BETA, gamma=GAMMA):

    acc_loss = 1 - acc
    latency_norm = latency / 200
    memory_norm = size_mb / 300

    return alpha * acc_loss + beta * latency_norm + gamma * memory_norm

# --------------------------------------------
# Quantum-Inspired Optimisation
# --------------------------------------------

def quantum_optimisation():

    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    base_model.to(device)

    layers = get_linear_layers(base_model)

    # Q-bit angle representation (θ)
    theta = {layer: math.pi/4 for layer in layers}

    best_config = None
    best_fitness = float("inf")

    for iteration in range(ITERATIONS):

        candidate_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        candidate_model.to(device)

        config = {}

        for layer in layers:
            prob = math.sin(theta[layer])**2
            if random.random() < prob:
                config[layer] = 0.25
            else:
                config[layer] = 0.10

        apply_pruning(candidate_model, config)

        acc, latency, _, _ = evaluate_full(candidate_model)
        params_m, size_mb = model_size_mb(candidate_model)

        fitness = compute_fitness(acc, latency, size_mb)

        print(f"\nIteration {iteration+1}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Latency: {latency:.2f} ms")
        print(f"Model Size: {size_mb:.2f} MB")
        print(f"Fitness: {fitness:.4f}")

        # Rotation Update Rule
        for layer in layers:
            if fitness < best_fitness:
                theta[layer] += DELTA_THETA
            else:
                theta[layer] -= DELTA_THETA

        if fitness < best_fitness:
            best_fitness = fitness
            best_config = config

    return best_config

# --------------------------------------------
# Run Optimisation
# --------------------------------------------

print("\nRunning Quantum-Inspired Optimisation...")
best_configuration = quantum_optimisation()

# --------------------------------------------
# Evaluate Teacher Model
# --------------------------------------------

teacher_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
teacher_model.to(device)
teacher_acc, teacher_latency, _, _ = evaluate_full(teacher_model)

# --------------------------------------------
# Baseline L1 Pruned Model
# --------------------------------------------

baseline_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
baseline_model.to(device)

for name, module in baseline_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.15)

baseline_acc, baseline_latency, _, _ = evaluate_full(baseline_model)

# --------------------------------------------
# Final Quantum-Inspired Model
# --------------------------------------------

final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
final_model.to(device)
apply_pruning(final_model, best_configuration)

final_acc, final_latency, y_true, y_pred = evaluate_full(final_model)
params_m, size_mb = model_size_mb(final_model)

print("\n===== Final Optimised Model =====")
print(f"Accuracy: {final_acc:.4f}")
print(f"Latency: {final_latency:.2f} ms")
print(f"Parameters: {params_m:.2f} Million")
print(f"Model Size: {size_mb:.2f} MB")

# --------------------------------------------
# Confusion Matrix
# --------------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix: Quantum-Inspired Model")
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

models = ["Teacher", "L1 Pruned", "Quantum-Inspired"]
latencies = [teacher_latency, baseline_latency, final_latency]
accuracies = [teacher_acc, baseline_acc, final_acc]

plt.figure()
plt.bar(models, latencies)
plt.ylabel("Per-Sample Latency (ms)")
plt.title("Latency Comparison Across Models")
plt.tight_layout()
plt.savefig("latency_comparison.png")
plt.show()

plt.figure()
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison Across Models")
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

print("\n===== Model Comparison Summary =====")
print(f"Teacher Accuracy: {teacher_acc:.4f} | Latency: {teacher_latency:.2f} ms")
print(f"L1 Pruned Accuracy: {baseline_acc:.4f} | Latency: {baseline_latency:.2f} ms")
print(f"Quantum-Inspired Accuracy: {final_acc:.4f} | Latency: {final_latency:.2f} ms")
