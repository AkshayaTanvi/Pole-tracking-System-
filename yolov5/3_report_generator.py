import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Class mapping
class_map = {"left_pole": 0, "right_pole": 1}
class_names = ["left_pole", "right_pole"]

def load_label_file(path):
    labels = []
    if not os.path.exists(path):
        return labels
    with open(path, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            cls_token = tokens[0]
            if cls_token in class_map:
                labels.append(class_map[cls_token])
            else:
                try:
                    labels.append(int(cls_token))
                except ValueError:
                    continue
    return labels

def evaluate_and_export(gt_dir, pred_dir, output_excel="classification_report.xlsx"):
    all_y_true = []
    all_y_pred = []

    common_files = sorted([
        f for f in os.listdir(gt_dir)
        if f.endswith(".txt") and os.path.exists(os.path.join(pred_dir, f))
    ])

    for file in common_files:
        gt_path = os.path.join(gt_dir, file)
        pred_path = os.path.join(pred_dir, file)

        y_true = load_label_file(gt_path)
        y_pred = load_label_file(pred_path)

        # Match by minimum length to avoid shape mismatch
        min_len = min(len(y_true), len(y_pred))
        if min_len == 0:
            continue
        all_y_true.extend(y_true[:min_len])
        all_y_pred.extend(y_pred[:min_len])

    if len(all_y_true) == 0 or len(all_y_pred) == 0:
        print("No valid data found for evaluation.")
        return

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    report = classification_report(
        all_y_true, all_y_pred, target_names=class_names, output_dict=True
    )

    precision = precision_score(all_y_true, all_y_pred, average='weighted')
    recall = recall_score(all_y_true, all_y_pred, average='weighted')
    f1 = f1_score(all_y_true, all_y_pred, average='weighted')

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(
        all_y_true, all_y_pred, target_names=class_names
    ))
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Save to Excel
    df_report = pd.DataFrame(report).transpose()
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    with pd.ExcelWriter(output_excel) as writer:
        df_report.to_excel(writer, sheet_name="Classification Report")
        df_cm.to_excel(writer, sheet_name="Confusion Matrix")

    print(f"Report saved to {output_excel}")

if __name__ == "__main__":
    evaluate_and_export("../data/labels/val", "predicted_labels")
