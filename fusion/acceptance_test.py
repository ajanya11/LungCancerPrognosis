# =====================================================
# 🧪 ACCEPTANCE TESTING + SAVE + GRAPHS
# =====================================================

import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from utils.ct_inference import run_ct_inference
from fusion.fusion_pipeline import FinalFusionPipeline


# =====================================================
# 📁 OUTPUT DIR
# =====================================================
OUTPUT_DIR = r"D:\Desktop\LungCancerPrognosis\LungCancerPrognosis\test_outputs\acceptance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = FinalFusionPipeline()


# =====================================================
# 🧪 TEST CASES
# =====================================================
test_cases = [
    ("Normal_Case", "samples/normal.jpg", {"age":45,"stage":"I","smoking":"No"}),
    ("Cancer_Case", "samples/cancer.jpg", {"age":65,"stage":"IV","smoking":"Yes"}),
    ("No_Genomic", "samples/test.jpg", {"age":55,"stage":"II","smoking":"No"})
]


# =====================================================
# 📊 GRAPH FUNCTIONS
# =====================================================
def plot_metrics(result, save_path):
    labels = ["Survival", "Confidence", "Fusion"]
    values = [
        result["survival_chance"],
        result["model_confidence"],
        result["fusion_score"]
    ]

    plt.figure()
    plt.plot(labels, values, marker='o')
    plt.title("Prediction Metrics")
    plt.savefig(save_path)
    plt.close()


def plot_breakdown(result, save_path):
    b = result.get("breakdown", {})

    labels = ["CT", "Type", "Clinical", "Genomic"]
    values = [
        b.get("ct_cancer_prob", 0),
        b.get("ct_type_conf", 0),
        b.get("clinical_score", 0),
        b.get("genomic_score", 0)
    ]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Feature Contribution")
    plt.savefig(save_path)
    plt.close()


# =====================================================
# 🚀 RUN TESTS
# =====================================================
all_results = []

for name, img, clinical in test_cases:

    print(f"\n🔍 Running: {name}")

    ct = run_ct_inference(img)

    result = pipeline.predict(
        ct_result=ct,
        clinical_data=clinical,
        genomic_data=None
    )

    # =====================================================
    # 💾 SAVE JSON (DIFFERENT NAMES)
    # =====================================================
    json_path = os.path.join(OUTPUT_DIR, f"{name}_result.json")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    # =====================================================
    # 📊 SAVE GRAPHS
    # =====================================================
    metric_path = os.path.join(OUTPUT_DIR, f"{name}_metrics.png")
    breakdown_path = os.path.join(OUTPUT_DIR, f"{name}_breakdown.png")

    plot_metrics(result, metric_path)
    plot_breakdown(result, breakdown_path)

    # Collect for summary
    all_results.append({
        "name": name,
        "risk": result["risk_level"],
        "fusion": result["fusion_score"]
    })

    print(f"✅ Saved: {json_path}")


# =====================================================
# 📊 SUMMARY GRAPH
# =====================================================
names = [r["name"] for r in all_results]
fusion_scores = [r["fusion"] for r in all_results]

plt.figure()
plt.bar(names, fusion_scores)
plt.title("Fusion Score Comparison (Acceptance Testing)")
summary_path = os.path.join(OUTPUT_DIR, "summary_fusion.png")
plt.savefig(summary_path)
plt.close()

print("\n📊 Summary graph saved:", summary_path)

print("\n🎯 ACCEPTANCE TESTING COMPLETED")