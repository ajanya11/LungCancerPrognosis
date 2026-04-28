# =====================================================
# 🔥 MODEL INTEGRATION — FINAL FIXED
# =====================================================

import os
import sys
from typing import Optional, Dict, Any

THIS_FILE    = os.path.abspath(__file__)
FRONTEND_DIR = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.dirname(FRONTEND_DIR)

for p in (PROJECT_ROOT, FRONTEND_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.ct_inference import run_ct_inference
from fusion.fusion_pipeline import FinalFusionPipeline
from LungCancerPrognosis.utils.report_generator import generate_text_report

_pipeline: Optional[FinalFusionPipeline] = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = FinalFusionPipeline()
    return _pipeline


def run_prediction(
    image_path,
    age,
    gender,
    stage,
    smoking,
    genomic_path,
    patient_name,
    patient_id,
):

    ct_result = run_ct_inference(image_path, patient_name)

    fusion = get_pipeline().predict(
        ct_result=ct_result,
        clinical_data={},
        genomic_data=genomic_path,
    )

    # =========================
    # 🔥 REPORT GENERATION
    # =========================
    report_dir = os.path.join(FRONTEND_DIR, "static", "reports")
    os.makedirs(report_dir, exist_ok=True)

    report_filename = f"report_{patient_name}.png"
    report_path = os.path.join(report_dir, report_filename)

    try:
        generate_text_report(
            output_path=report_path,
            patient_name=patient_name,
            patient_id=patient_id,
            result={
                "fusion": {
                    "risk_level": fusion.get("risk_level"),
                    "survival_probability": fusion.get("survival_chance"),
                },
                "ct": {"label": fusion.get("cancer_type")},
                "tumor_area_percent": 0.0,
            },
        )
        report_rel = f"reports/{report_filename}"
    except Exception as e:
        print("Report error:", e)
        report_rel = None

    return {
        "risk_level": fusion.get("risk_level"),
        "final_score": fusion.get("fusion_score"),
        "survival_percentage": fusion.get("survival_chance"),
        "ct_score": ct_result.get("type_confidence"),

        "gradcam_image": ct_result.get("gradcam"),
        "gradcam_binary": ct_result.get("gradcam_binary"),
        "gradcam_type": ct_result.get("gradcam_type"),

        "report_image": report_rel,

        "cancer_type": fusion.get("cancer_type"),
        "model_confidence": fusion.get("model_confidence"),
        "suggestions": fusion.get("suggestions", []),
    }