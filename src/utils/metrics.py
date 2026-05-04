"""
src/utils/metrics.py
All PlantNet-300K recommended metrics + validation framework
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# ──────────────────────────────────────────────
# PLANTNET-300K RECOMMENDED METRICS
# ──────────────────────────────────────────────


def top_k_accuracy(y_true, y_prob, k):
    """
    Top-k Accuracy (micro-average):
    Fraction of samples where the true label is among
    the k highest-predicted classes.

    This treats every sample equally regardless of class.
    Common classes dominate the score.
    """
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return correct.mean()


def macro_average_top_k_accuracy(y_true, y_prob, k):
    """
    Macro-average Top-k Accuracy:
    Compute top-k accuracy PER CLASS, then average across classes.

    This gives equal weight to every species regardless
    of how many samples it has. A species with 1 sample
    counts the same as a species with 50 samples.

    This is THE key metric from PlantNet-300K paper because
    it reveals performance on rare species.
    """
    classes = np.unique(y_true)
    per_class_acc = []
    per_class_detail = {}

    for c in classes:
        mask = y_true == c
        n_samples = mask.sum()
        if n_samples == 0:
            continue

        class_correct = 0
        for i in np.where(mask)[0]:
            top_k_preds = np.argsort(y_prob[i])[-k:]
            if y_true[i] in top_k_preds:
                class_correct += 1

        class_acc = class_correct / n_samples
        per_class_acc.append(class_acc)
        per_class_detail[c] = {
            "accuracy": class_acc,
            "n_samples": int(n_samples),
            "n_correct": class_correct,
        }

    return np.mean(per_class_acc), per_class_detail


def average_k_accuracy(y_true, y_prob, K_max=5):
    """
    Average-k Accuracy:
    Average of top-1, top-2, ..., top-K accuracies.

    Single number that captures how quickly the model
    "finds" the right answer as k increases.
    High average-k means the model consistently ranks
    the true class near the top.
    """
    accs = []
    for k in range(1, K_max + 1):
        acc = top_k_accuracy(y_true, y_prob, k)
        accs.append(acc)
    return np.mean(accs), accs


def compute_all_plantnet_metrics(y_true, y_prob, K_max=5):
    """
    Compute all PlantNet-300K recommended metrics at once.

    Returns a comprehensive dictionary of results.
    """
    results = {
        "n_samples": len(y_true),
        "n_classes": len(np.unique(y_true)),
    }

    # Top-k accuracy (micro) for k = 1 to K_max
    for k in range(1, K_max + 1):
        results[f"top_{k}_accuracy"] = round(top_k_accuracy(y_true, y_prob, k), 4)

    # Macro-average top-k accuracy for k = 1 to K_max
    for k in range(1, K_max + 1):
        macro_acc, per_class = macro_average_top_k_accuracy(y_true, y_prob, k)
        results[f"macro_top_{k}_accuracy"] = round(macro_acc, 4)
        if k == 1:
            results["per_class_detail"] = per_class

    # Average-k accuracy
    avg_k, per_k_accs = average_k_accuracy(y_true, y_prob, K_max)
    results["average_k_accuracy"] = round(avg_k, 4)
    results["per_k_accuracies"] = [round(a, 4) for a in per_k_accs]

    # Gap between micro and macro (reveals class imbalance impact)
    results["micro_macro_gap_top1"] = round(
        results["top_1_accuracy"] - results["macro_top_1_accuracy"], 4
    )

    return results
