import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

from collections import Counter

# Map class names to IDs
class_map = {"left_pole": 0, "right_pole": 1}

def parse_label_file(filepath):
    with open(filepath, 'r') as f:
        return [
            class_map.get(line.strip().split()[0], int(line.strip().split()[0]))
            for line in f if line.strip()
        ]

def load_all_labels(label_dir):
    all_labels = []
    for file in sorted(os.listdir(label_dir)):
        if file.endswith(".txt"):
            try:
                labels = parse_label_file(os.path.join(label_dir, file))
                all_labels.extend(labels)
            except Exception as e:
                print(f" Error reading {file}: {e}")
    return all_labels

def evaluate(gt_dir, pred_dir):
    print("Loading ground truth labels...")
    y_true = load_all_labels(gt_dir)
    
    print("Loading predicted labels...")
    y_pred = load_all_labels(pred_dir)

    # Safety check
    if not y_true or not y_pred:
        print("No labels found. Check the directories.")
        return

    print(f"Loaded {len(y_true)} GT labels and {len(y_pred)} predictions")

    # Align lengths if needed
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["left_pole", "right_pole"]))

    print(f"\n Precision: {precision_score(y_true, y_pred, average='weighted'):.2f}")
    print(f" Recall:    {recall_score(y_true, y_pred, average='weighted'):.2f}")
    print(f" F1 Score:  {f1_score(y_true, y_pred, average='weighted'):.2f}")

if __name__ == "__main__":
    evaluate("../data/labels/val", "predicted_labels")
