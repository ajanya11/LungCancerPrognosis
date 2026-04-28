# =====================================================
# 🧠 RULE-BASED FUSION MODEL (FINAL VERSION)
# =====================================================

import numpy as np
import torch


# =====================================================
# 🔹 HELPERS
# =====================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clamp(x):
    return float(max(0.0, min(1.0, x)))


# =====================================================
# 🔹 TYPE MAP
# =====================================================
TYPE_MAP = {
    0: "Adenocarcinoma",
    1: "Large Cell Carcinoma",
    2: "Normal",
    3: "Squamous Cell Carcinoma"
}


# =====================================================
# 🩺 SUGGESTIONS ENGINE
# =====================================================
def generate_suggestions(final_label, cancer_type, risk):

    if final_label == "NORMAL":
        return [
            "Maintain a healthy lifestyle (balanced diet, exercise)",
            "Avoid smoking and second-hand smoke",
            "Routine lung screening every 6–12 months if high-risk",
            "Consult doctor if symptoms like persistent cough appear"
        ]

    suggestions = []

    # 🔴 HIGH RISK
    if risk == "HIGH":
        suggestions.extend([
            "Immediate consultation with an oncologist is strongly recommended",
            "Further diagnostic tests (biopsy, PET scan) required urgently",
            "Start treatment planning (surgery / chemotherapy / radiation)",
            "Regular monitoring and hospitalization may be required"
        ])

    # 🟠 MEDIUM RISK
    elif risk == "MEDIUM":
        suggestions.extend([
            "Consult a specialist for further evaluation",
            "Additional imaging tests (CT/PET) recommended",
            "Close monitoring required (every 1–3 months)",
            "Lifestyle changes (stop smoking, improve diet)"
        ])

    # 🟢 LOW RISK
    else:
        suggestions.extend([
            "Condition appears stable but requires monitoring",
            "Follow-up scan recommended in 3–6 months",
            "Maintain healthy lifestyle",
            "Consult doctor if symptoms worsen"
        ])

    # =========================
    # TYPE SPECIFIC
    # =========================
    if cancer_type == "Adenocarcinoma":
        suggestions.append("Targeted therapy may be effective (EGFR/ALK testing recommended)")

    elif cancer_type == "Squamous Cell Carcinoma":
        suggestions.append("Smoking cessation is critical for treatment success")

    elif cancer_type == "Large Cell Carcinoma":
        suggestions.append("Aggressive cancer type – early treatment planning is important")

    return suggestions


# =====================================================
# 🔥 MAIN FUSION FUNCTION
# =====================================================
def rule_based_fusion(
    ct_bin_logits,
    ct_type_logits,
    clinical_logit,
    genomic_score
):

    # =========================
    # 🔹 CT BINARY
    # =========================
    ct_probs = torch.softmax(torch.tensor(ct_bin_logits), dim=0)
    ct_normal = ct_probs[0].item()
    ct_cancer = ct_probs[1].item()

    # =========================
    # 🚫 HARD CT OVERRIDE
    # =========================
    if ct_normal > 0.7:
        return {
            "final_prediction": "NORMAL",
            "confidence": float(ct_normal),
            "type": "None",
            "type_confidence": 1.0,
            "risk": "LOW",
            "survival": 0.95,
            "suggestions": generate_suggestions("NORMAL", "None", "LOW"),
            "note": "CT override (normal)"
        }

    # =========================
    # 🔹 TYPE
    # =========================
    type_probs = torch.softmax(torch.tensor(ct_type_logits), dim=0)
    type_idx = int(torch.argmax(type_probs))
    type_conf = torch.max(type_probs).item()

    cancer_type = TYPE_MAP[type_idx]

    # =========================
    # 🔹 CLINICAL
    # =========================
    clinical_prob = sigmoid(clinical_logit)

    # =========================
    # 🔹 GENOMIC
    # =========================
    genomic_prob = clamp(genomic_score)

    # =========================
    # 🔹 WEIGHTS
    # =========================
    w_ct = 0.6
    w_clinical = 0.25
    w_genomic = 0.15

    # dynamic adjustment
    if clinical_prob < 0.3:
        w_ct += 0.1
        w_clinical -= 0.1

    if genomic_prob < 0.2:
        w_ct += 0.1
        w_genomic -= 0.1

    # normalize
    total = w_ct + w_clinical + w_genomic
    w_ct /= total
    w_clinical /= total
    w_genomic /= total

    # =========================
    # 🔥 FINAL SCORE
    # =========================
    final_score = (
        w_ct * ct_cancer +
        w_clinical * clinical_prob +
        w_genomic * genomic_prob
    )

    final_score = clamp(final_score)

    # =========================
    # 🔹 LABEL
    # =========================
    final_label = "CANCER" if final_score > 0.5 else "NORMAL"

    # =========================
    # 🔹 TYPE CORRECTION
    # =========================
    if final_label == "NORMAL":
        cancer_type = "None"
    elif cancer_type == "Normal":
        cancer_type = "Unknown Cancer Type"

    # =========================
    # 🔹 RISK
    # =========================
    if final_score > 0.75:
        risk = "HIGH"
    elif final_score > 0.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    survival = 1 - final_score

    # =========================
    # 🔹 SUGGESTIONS
    # =========================
    suggestions = generate_suggestions(final_label, cancer_type, risk)

    # =========================
    # 🔹 EXPLANATION
    # =========================
    explanation = f"""
CT Cancer: {ct_cancer:.2f}
Clinical Risk: {clinical_prob:.2f}
Genomic Risk: {genomic_prob:.2f}

Weights →
CT: {w_ct:.2f}, Clinical: {w_clinical:.2f}, Genomic: {w_genomic:.2f}
"""

    # =========================
    # 🔹 FINAL OUTPUT
    # =========================
    return {
        "final_prediction": final_label,
        "confidence": float(final_score),
        "type": cancer_type,
        "type_confidence": float(type_conf),
        "risk": risk,
        "survival": float(survival),
        "suggestions": suggestions,
        "ct_cancer_prob": float(ct_cancer),
        "explanation": explanation
    }