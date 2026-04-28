# =====================================================
# 🔍 VERIFICATION + 🧪 VALIDATION (FINAL REALISTIC VERSION)
# =====================================================

import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from fusion.full_fusion import full_fusion


# =====================================================
# 🔍 VERIFICATION
# =====================================================
def verify_fusion():
    try:
        ct_logits = np.random.rand(4)
        clinical = np.random.rand()
        genomic = np.random.rand()

        result = full_fusion(ct_logits, clinical, genomic)

        assert "binary" in result
        assert "confidence" in result
        assert "type" in result
        assert result["binary"] in [0, 1]

        print("✅ Verification Passed: Fusion logic working correctly")

    except Exception as e:
        print("❌ Verification Failed:", e)


# =====================================================
# 🧪 REALISTIC TEST DATA (NO DATA LEAKAGE)
# =====================================================
def generate_realistic_data(n=300):
    data = []

    for _ in range(n):
        ct_logits = np.random.rand(4)
        clinical = np.random.rand()
        genomic = np.random.rand()

        # 🔥 DIFFERENT LOGIC FROM FUSION (IMPORTANT)
        ct_probs = np.exp(ct_logits) / np.sum(np.exp(ct_logits))
        ct_cancer_prob = ct_probs[0] + ct_probs[1] + ct_probs[3]

        noise = np.random.normal(0, 0.08)

        score = (
            0.5 * ct_cancer_prob +
            0.3 * clinical +
            0.2 * genomic +
            noise
        )

        label = 1 if score > 0.6 else 0

        data.append((ct_logits, clinical, genomic, label))

    return data


# =====================================================
# 🧪 VALIDATION
# =====================================================
def validate():
    data = generate_realistic_data()

    preds, targets, probs = [], [], []

    for ct_logits, clinical, genomic, label in data:
        result = full_fusion(ct_logits, clinical, genomic)

        pred = result["binary"]
        prob = result["confidence"]  # 🔥 correct probability

        preds.append(pred)
        targets.append(label)
        probs.append(prob)

    # =========================
    # 📊 METRICS
    # =========================
    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    auc = roc_auc_score(targets, probs)

    cm = confusion_matrix(targets, preds)

    # =========================
    # 📢 OUTPUT
    # =========================
    print("\n📊 Distribution")
    print("Pred:", Counter(preds))
    print("True:", Counter(targets))

    print("\n🧪 VALIDATION RESULTS")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    print("\n📉 Confusion Matrix")
    print(cm)

    print("\n📄 Classification Report")
    print(classification_report(targets, preds, zero_division=0))


# =====================================================
# 🚀 RUN
# =====================================================
if __name__ == "__main__":
    print("🔍 Running Verification...")
    verify_fusion()

    print("\n🧪 Running Validation...")
    validate()