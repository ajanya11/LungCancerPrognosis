# =====================================================
# 🔥 FINAL PROGNOSIS REPORT (FULLY FIXED)
# =====================================================

import cv2
import numpy as np
import os


def draw_multiline_text(img, text, x, y, font, scale, color, thickness, max_width=700, line_height=35):
    words = text.split(" ")
    lines = []
    current = ""

    for word in words:
        test_line = current + " " + word if current else word
        (w, _), _ = cv2.getTextSize(test_line, font, scale, thickness)

        if w > max_width:
            lines.append(current)
            current = word
        else:
            current = test_line

    lines.append(current)

    for line in lines:
        cv2.putText(img, line, (x, y), font, scale, color, thickness)
        y += line_height

    return y


def generate_text_report(
    output_path,
    patient_name,
    patient_id,
    result,
    hospital_name="AI Multispeciality Hospital",
    doctor_name="Dr. John Smith",
    width=1200,
    height=1600
):

    output_path = os.path.abspath(output_path)

    if not output_path.lower().endswith((".png", ".jpg", ".jpeg")):
        output_path += ".png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # =========================
    # BASE IMAGE
    # =========================
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    black = (20, 20, 20)
    blue = (255, 120, 0)
    red = (0, 0, 255)
    green = (0, 150, 0)
    gray = (120, 120, 120)

    cv2.rectangle(img, (20, 20), (width-20, height-20), black, 3)

    # =========================
    # HEADER
    # =========================
    cv2.putText(img, hospital_name, (200, 80), font, 1.2, black, 3)
    cv2.putText(img, "LUNG CANCER PROGNOSIS REPORT", (180, 140), font, 1.0, gray, 2)
    cv2.line(img, (40, 180), (width-40, 180), black, 2)

    fusion = result.get("fusion", {})
    ct = result.get("ct", {})

    # =========================
    # 🔥 FIXED SAFE EXTRACTION
    # =========================
    cancer_type = ct.get("label", "Unknown")

    risk = fusion.get("risk_level", fusion.get("risk", "Unknown"))
    survival = fusion.get("survival_probability", fusion.get("survival", 0.0))

    tumor_area = result.get("tumor_area_percent", 0.0)

    # =========================
    # LOGIC
    # =========================
    if cancer_type == "None" or cancer_type == "Normal":
        prognosis_text = "No evidence of lung cancer detected."
        severity = "None"
        tumor_area = 0.0
        risk = "LOW"

        clinical_summary = "Lungs appear normal. No malignant patterns identified."

        recommendations = [
            "1. Maintain healthy lifestyle",
            "2. Avoid smoking",
            "3. Routine screening",
            "4. Follow-up in 6–12 months"
        ]
    else:
        prognosis_text = f"{cancer_type} detected with {risk.lower()} progression risk."

        if tumor_area < 10:
            severity = "Mild"
        elif tumor_area < 25:
            severity = "Moderate"
        else:
            severity = "Severe"

        clinical_summary = f"Tumor approx {tumor_area:.2f}% with {risk.lower()} risk."

        recommendations = [
            "1. Consult oncologist",
            "2. Biopsy required",
            "3. Imaging (PET/CT)",
            "4. Start treatment planning"
        ]

    # =========================
    # DRAW CONTENT
    # =========================
    y = 250

    cv2.putText(img, "PATIENT DETAILS", (60, y), font, 0.9, blue, 2)
    y += 60
    cv2.putText(img, f"Name: {patient_name}", (80, y), font, 0.8, black, 2)
    y += 50
    cv2.putText(img, f"Patient ID: {patient_id}", (80, y), font, 0.8, black, 2)

    y += 100
    cv2.putText(img, "PROGNOSIS SUMMARY", (60, y), font, 0.9, blue, 2)

    y += 70
    y = draw_multiline_text(img, prognosis_text, 80, y, font, 0.8, black, 2)

    y += 30
    cv2.putText(img, f"Type: {cancer_type}", (80, y), font, 0.9, red, 2)

    y += 60
    cv2.putText(img, f"Risk: {risk}", (80, y), font, 1.0, red, 3)

    y += 60
    cv2.putText(img, f"Survival: {float(survival):.2f}", (80, y), font, 1.0, green, 3)

    y += 60
    cv2.putText(img, f"Tumor Area: {float(tumor_area):.2f}%", (80, y), font, 1.0, black, 2)

    y += 80
    y = draw_multiline_text(img, clinical_summary, 80, y, font, 0.7, black, 2)

    y += 100
    cv2.putText(img, "RECOMMENDATIONS", (60, y), font, 0.9, blue, 2)

    y += 60
    for line in recommendations:
        cv2.putText(img, line, (80, y), font, 0.7, black, 2)
        y += 50

    # =========================
    # SAVE
    # =========================
    success = cv2.imwrite(output_path, img)

    if not success:
        raise RuntimeError(f"Report save failed: {output_path}")

    return output_path