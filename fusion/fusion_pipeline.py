# =====================================================
# 🧠 FUSION PIPELINE  (fusion/fusion_pipeline.py)
# =====================================================

from typing import Dict, Any, Optional


# ── Utils ──────────────────────────────────────────────────────────────

def clamp(x: float, low: float = 0.05, high: float = 0.95) -> float:
    return max(low, min(high, x))


def safe_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d and d[key] is not None else default


# ── Clinical scoring ───────────────────────────────────────────────────

def clinical_risk_score(age: int, stage: str, smoking: str) -> float:
    score = 0.0

    if age >= 65:
        score += 0.3
    elif age >= 50:
        score += 0.2
    else:
        score += 0.1

    stage_map = {"I": 0.1, "II": 0.3, "III": 0.6, "IV": 0.9}
    score += stage_map.get(str(stage).upper(), 0.3)

    if str(smoking).strip().lower() in ("yes", "y", "true", "1", "current", "former"):
        score += 0.3

    return clamp(score)


# ── Fusion pipeline ────────────────────────────────────────────────────

class FinalFusionPipeline:

    def __init__(self):
        print("🧠 Fusion Pipeline Ready")

    def predict(
        self,
        ct_result:     Dict[str, Any],
        clinical_data: Optional[Dict[str, Any]] = None,
        genomic_data:  Optional[Any]            = None,
    ) -> Dict[str, Any]:

        try:
            # ── Handle CT error ───────────────────────────────────
            if "error" in ct_result and not ct_result.get("cancer_type"):
                return {
                    "risk_level":       "ERROR",
                    "cancer_type":      "N/A",
                    "survival_chance":  0,
                    "model_confidence": 0,
                    "fusion_score":     0,
                    # ✅ Still pass gradcam even on partial error
                    "gradcam":          ct_result.get("gradcam"),
                    "gradcam_binary":   ct_result.get("gradcam_binary"),
                    "gradcam_type":     ct_result.get("gradcam_type"),
                    "suggestions":      [ct_result.get("error", "Unknown error")],
                }

            # ── CT data ───────────────────────────────────────────
            cancer_type = safe_get(ct_result, "cancer_type", "Unknown")
            cancer_prob = float(safe_get(ct_result, "cancer_probability", 0)) / 100.0
            type_conf   = float(safe_get(ct_result, "type_confidence",   0)) / 100.0

            # ✅ Always pass ALL gradcam paths through — Normal included
            gradcam        = safe_get(ct_result, "gradcam",         None)
            gradcam_binary = safe_get(ct_result, "gradcam_binary",  None)
            gradcam_type   = safe_get(ct_result, "gradcam_type",    None)

            # ── Clinical data ─────────────────────────────────────
            age     = 50
            stage   = "II"
            smoking = "No"

            if clinical_data:
                age     = int(clinical_data.get("age",     50))
                stage   = str(clinical_data.get("stage",   "II"))
                smoking = str(clinical_data.get("smoking", "No"))

            clinical_score = clinical_risk_score(age, stage, smoking)

            # ── Genomic ───────────────────────────────────────────
            genomic_score = 0.2 if genomic_data is not None else 0.0

            # ── Fusion score ──────────────────────────────────────
            # ✅ For Normal: cancer_prob is low → fusion score will be low → Low Risk
            fusion_score = clamp(
                0.6 * cancer_prob    +
                0.3 * clinical_score +
                0.1 * genomic_score
            )

            survival   = 1.0 - fusion_score
            confidence = clamp(pow(max(type_conf * 0.7 + cancer_prob * 0.3, 0.0), 0.75), 0.0, 1.0)

            # ── Risk level ────────────────────────────────────────
            if fusion_score < 0.3:
                risk_level = "Low Risk"
            elif fusion_score < 0.6:
                risk_level = "Medium Risk"
            else:
                risk_level = "High Risk"

            # ── Suggestions ───────────────────────────────────────
            # ✅ Normal gets its own suggestions WITH GradCAM note
            if cancer_type == "Normal":
                suggestions = [
                    "No cancer detected — lungs appear normal",
                    "Routine follow-up screening recommended",
                    "Maintain a healthy, smoke-free lifestyle",
                    "GradCAM shows regions the model focused on to confirm normality",
                ]
            else:
                suggestions = [
                    f"Detected: {cancer_type}",
                    "Consult oncologist immediately",
                    "Histopathological biopsy recommended",
                    "Consider PET/CT for staging",
                    "Begin treatment planning",
                ]

            return {
                "risk_level":       risk_level,
                "cancer_type":      cancer_type,
                "survival_chance":  round(survival    * 100, 2),
                "model_confidence": round(confidence  * 100, 2),
                "fusion_score":     round(fusion_score * 100, 2),
                # ✅ All three gradcam paths always returned
                "gradcam":          gradcam,
                "gradcam_binary":   gradcam_binary,
                "gradcam_type":     gradcam_type,
                "suggestions":      suggestions,
                "breakdown": {
                    "ct_cancer_prob":   round(cancer_prob    * 100, 2),
                    "ct_type_conf":     round(type_conf      * 100, 2),
                    "clinical_score":   round(clinical_score * 100, 2),
                    "genomic_score":    round(genomic_score  * 100, 2),
                    "age":              age,
                    "stage":            stage,
                    "smoking":          smoking,
                },
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "risk_level":       "ERROR",
                "cancer_type":      "N/A",
                "survival_chance":  0,
                "model_confidence": 0,
                "fusion_score":     0,
                "gradcam":          ct_result.get("gradcam"),
                "gradcam_binary":   ct_result.get("gradcam_binary"),
                "gradcam_type":     ct_result.get("gradcam_type"),
                "suggestions":      [f"System error: {e}"],
                "error":            str(e),
            }