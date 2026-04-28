import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import seaborn as sns

from fusion.fusion_pipeline import FinalFusionPipeline
from utils.ct_inference import run_ct_inference


# =====================================================
# 📂 CONFIG
# =====================================================
DATASET_PATH = "dataset/chest ct scan lunng/test"

CLASSES = [
    "normal",
    "adenocarcinoma",
    "large.cell.carcinoma",
    "squamous.cell.carcinoma"
]


# =====================================================
# 📌 CLINICAL DATA
# =====================================================
def get_clinical():
    return {
        "age": 65,
        "stage": "II",
        "smoking": "Yes"
    }


# =====================================================
# 🔄 LABEL HELPERS
# =====================================================
def to_binary(label):
    return 0 if label == "normal" else 1


def normalize_label(label):
    return label.replace(".", " ").title()


# =====================================================
# 🚀 MAIN EVALUATION
# =====================================================
def evaluate():

    print("\n🚀 RUNNING FULL FUSION EVALUATION\n")

    pipeline = FinalFusionPipeline()

    y_true_bin = []
    y_pred_bin = []
    y_scores = []

    y_true_multi = []
    y_pred_multi = []

    risk_correct = 0
    total = 0

    for class_name in CLASSES:

        class_path = os.path.join(DATASET_PATH, class_name)

        for img_name in os.listdir(class_path):

            img_path = os.path.join(class_path, img_name)

            # =====================================================
            # 🫁 CT
            # =====================================================
            ct = run_ct_inference(img_path, username="Ajanya")

            if "error" in ct:
                continue

            # =====================================================
            # 🧠 FUSION
            # =====================================================
            fusion = pipeline.predict(ct, clinical_data=get_clinical())

            fusion_score = fusion["fusion_score"] / 100.0
            pred_type = fusion["cancer_type"]

            # =====================================================
            # 🎯 BINARY
            # =====================================================
            true_bin = to_binary(class_name)
            pred_bin = 1 if fusion_score >= 0.5 else 0

            y_true_bin.append(true_bin)
            y_pred_bin.append(pred_bin)
            y_scores.append(fusion_score)

            # =====================================================
            # 🧠 MULTI-CLASS
            # =====================================================
            true_multi = normalize_label(class_name)
            pred_multi = normalize_label(pred_type)

            y_true_multi.append(true_multi)
            y_pred_multi.append(pred_multi)

            # =====================================================
            # 📊 RISK VALIDATION
            # =====================================================
            if true_bin == 1 and fusion_score > 0.6:
                risk_correct += 1
            elif true_bin == 0 and fusion_score < 0.4:
                risk_correct += 1

            total += 1

            print(f"{img_name} → GT: {true_multi} | Pred: {pred_multi} | Score: {fusion_score:.2f}")

    # =====================================================
    # 📊 BINARY METRICS
    # =====================================================
    print("\n" + "="*60)
    print("📊 BINARY PERFORMANCE")
    print("="*60)

    print("Accuracy:", accuracy_score(y_true_bin, y_pred_bin))
    print("\nClassification Report:\n")
    print(classification_report(y_true_bin, y_pred_bin, target_names=["Normal", "Cancer"]))

    cm_bin = confusion_matrix(y_true_bin, y_pred_bin)

    # Save binary CM
    plt.figure()
    sns.heatmap(cm_bin, annot=True, fmt="d",
                xticklabels=["Normal", "Cancer"],
                yticklabels=["Normal", "Cancer"])
    plt.title("Binary Confusion Matrix")
    plt.savefig("results/binary_cm.png")
    plt.close()

    # =====================================================
    # 📈 ROC CURVE
    # =====================================================
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig("results/roc.png")
    plt.close()

    # Best threshold
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]

    print(f"\n🔥 ROC AUC: {roc_auc:.4f}")
    print(f"🎯 Best Threshold: {best_thresh:.3f}")

    # =====================================================
    # 🧠 MULTI-CLASS METRICS
    # =====================================================
    print("\n" + "="*60)
    print("🧠 MULTI-CLASS PERFORMANCE")
    print("="*60)

    print(classification_report(y_true_multi, y_pred_multi))

    cm_multi = confusion_matrix(y_true_multi, y_pred_multi)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_multi, annot=True, fmt="d")
    plt.title("Cancer Type Confusion Matrix")
    plt.savefig("results/multiclass_cm.png")
    plt.close()

    # =====================================================
    # 📊 RISK SCORE VALIDATION
    # =====================================================
    print("\n" + "="*60)
    print("📊 RISK MODEL VALIDATION")
    print("="*60)

    risk_acc = risk_correct / total
    print(f"Risk Alignment Accuracy: {risk_acc:.4f}")

    # =====================================================
    # 📁 DONE
    # =====================================================
    print("\n✅ ALL RESULTS SAVED IN /results/")
    print("="*60)


# =====================================================
# 🚀 RUN
# =====================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    evaluate()