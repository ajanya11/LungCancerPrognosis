# =====================================================
# 🧪 TEST + AUTO SAVE OUTPUT + GRAPHS
# =====================================================

import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from utils.ct_inference import run_ct_inference
from fusion.fusion_pipeline import FinalFusionPipeline


# =====================================================
# 📌 PATH CONFIG
# =====================================================
IMAGE_PATH = "samples/test.jpg"

OUTPUT_DIR = r"D:\Desktop\LungCancerPrognosis\LungCancerPrognosis\test_outputs\fusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# 📌 CLINICAL DATA
# =====================================================
clinical_data = {
    "age": 60,
    "stage": "III",
    "smoking": "Yes"
}

genomic_data = None


# =====================================================
# 📊 GRAPH FUNCTIONS
# =====================================================
def plot_breakdown(result, save_path):
    breakdown = result.get("breakdown", {})

    labels = ["CT Prob", "Type Conf", "Clinical", "Genomic"]
    values = [
        breakdown.get("ct_cancer_prob", 0),
        breakdown.get("ct_type_conf", 0),
        breakdown.get("clinical_score", 0),
        breakdown.get("genomic_score", 0)
    ]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Feature Contribution Breakdown")
    plt.xlabel("Components")
    plt.ylabel("Score (%)")
    plt.savefig(save_path)
    plt.close()


def plot_prediction_metrics(result, save_path):
    labels = ["Survival", "Confidence", "Fusion Score"]
    values = [
        result.get("survival_chance", 0),
        result.get("model_confidence", 0),
        result.get("fusion_score", 0)
    ]

    plt.figure()
    plt.plot(labels, values, marker='o')
    plt.title("Prediction Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Percentage")
    plt.savefig(save_path)
    plt.close()


# =====================================================
# 🚀 RUN TEST
# =====================================================
def main():
    print("\n🚀 STARTING TEST...\n")

    # Step 1: CT Inference
    ct_result = run_ct_inference(IMAGE_PATH)

    # Step 2: Fusion
    pipeline = FinalFusionPipeline()
    final_result = pipeline.predict(
        ct_result=ct_result,
        clinical_data=clinical_data,
        genomic_data=genomic_data
    )

    # =====================================================
    # 💾 SAVE OUTPUT
    # =====================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(OUTPUT_DIR, f"fusion_result_{timestamp}.json")
    graph1_path = os.path.join(OUTPUT_DIR, f"breakdown_{timestamp}.png")
    graph2_path = os.path.join(OUTPUT_DIR, f"metrics_{timestamp}.png")

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(final_result, f, indent=4)

    # Save Graphs
    plot_breakdown(final_result, graph1_path)
    plot_prediction_metrics(final_result, graph2_path)

    # =====================================================
    # 📢 PRINT OUTPUT
    # =====================================================
    print("\n✅ RESULT SAVED TO:")
    print(json_path)

    print("\n📊 GRAPHS SAVED:")
    print(graph1_path)
    print(graph2_path)

    print("\n🧠 FINAL RESULT:")
    print("=" * 50)
    print("Prediction :", final_result["cancer_type"])
    print("Risk       :", final_result["risk_level"])
    print("Survival   :", final_result["survival_chance"], "%")
    print("Confidence :", final_result["model_confidence"], "%")
    print("=" * 50)


# =====================================================
# ▶️ RUN
# =====================================================
if __name__ == "__main__":
    main()