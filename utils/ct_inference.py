# =====================================================
# 🫁 CT INFERENCE  (utils/ct_inference.py)
# =====================================================

import os
import sys
import shutil
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, Optional, cast

THIS_FILE    = os.path.abspath(__file__)
UTILS_DIR    = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.dirname(UTILS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ct_model import CTModel
from utils.gradcam import GradCAM, overlay_heatmap, get_target_layer

# ── Paths ──────────────────────────────────────────────────────────────
CKPT_TYPE   = os.path.join(PROJECT_ROOT, "checkpoints", "best_ct_type.pth")
CKPT_BINARY = os.path.join(PROJECT_ROOT, "checkpoints", "best_ct_binary.pth")

# Auto-detect static/gradcam directory (absolute path)
_FRONTEND = os.path.join(PROJECT_ROOT, "frontend", "static", "gradcam")
_ROOT     = os.path.join(PROJECT_ROOT, "static", "gradcam")
GRADCAM_DIR = _FRONTEND if os.path.isdir(os.path.join(PROJECT_ROOT, "frontend")) else _ROOT

# Also keep reference to the static root for URL building
STATIC_ROOT = os.path.join(PROJECT_ROOT, "frontend", "static") \
              if os.path.isdir(os.path.join(PROJECT_ROOT, "frontend")) \
              else os.path.join(PROJECT_ROOT, "static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ All 4 class names — index 2 is Normal
CLASS_NAMES = [
    "Adenocarcinoma",           # 0
    "Large Cell Carcinoma",     # 1
    "Normal",                   # 2
    "Squamous Cell Carcinoma",  # 3
]

# ✅ Labels considered "normal / no cancer" — GradCAM heatmap is skipped for these
NORMAL_LABELS = {"normal", "no cancer", "benign"}

ct_type_model:   Optional[CTModel] = None
ct_binary_model: Optional[CTModel] = None


# ── Model loading ──────────────────────────────────────────────────────

def _load_one(path: str, num_classes: int) -> CTModel:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(obj, dict):
        print(f"⚠️  state_dict load ({num_classes}-class): {os.path.basename(path)}")
        model = CTModel(num_classes=num_classes)
        model.load_state_dict(obj)
    else:
        print(f"✅  full model load ({num_classes}-class): {os.path.basename(path)}")
        model = obj
    model.eval()
    return model.to(DEVICE)


def load_models() -> None:
    global ct_type_model, ct_binary_model
    if ct_type_model   is None:
        ct_type_model   = _load_one(CKPT_TYPE,   num_classes=4)
    if ct_binary_model is None:
        ct_binary_model = _load_one(CKPT_BINARY, num_classes=2)


# ── Transform ──────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Copy original image into static/gradcam so it's web-accessible ────

def _copy_original_to_static(image_path: str, username: str) -> str:
    """
    Copies the uploaded CT image into the gradcam static folder so Flask
    can serve it as /static/gradcam/original_<username>.<ext>.
    Returns the absolute path of the copied file.
    """
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    ext = os.path.splitext(image_path)[1] or ".jpg"
    dest_filename = f"original_{username}{ext}"
    dest_path = os.path.join(GRADCAM_DIR, dest_filename)
    shutil.copy2(image_path, dest_path)
    print(f"✅ Original CT copied → {dest_path}")
    return dest_path


# ── GradCAM runner ─────────────────────────────────────────────────────

def _run_gradcam(
    model: CTModel,
    tensor: torch.Tensor,
    original_bgr: np.ndarray,
    class_idx: int,
    username: str,
    suffix: str,
    predicted_label: str = "",
) -> Optional[str]:
    """
    Runs GradCAM and saves the overlay image.

    For Normal / benign predictions the heatmap is intentionally skipped —
    GradCAM would highlight the 'most suspicious' region even on healthy
    scans, which is misleading.  In that case None is returned and the
    caller should fall back to the plain original image.

    Returns the ABSOLUTE filesystem path to the saved file, or None.
    """

    # ✅ Skip heatmap for normal predictions
    if predicted_label.strip().lower() in NORMAL_LABELS:
        print(f"⏭️  GradCAM [{suffix}] skipped — normal prediction: '{predicted_label}'")
        return None

    try:
        target_layer = get_target_layer(model)
        grad_cam     = GradCAM(model, target_layer)
        cam          = grad_cam.generate(tensor.clone(), class_idx=class_idx)
        overlay      = overlay_heatmap(original_bgr, cam)

        os.makedirs(GRADCAM_DIR, exist_ok=True)
        filename  = f"gradcam_{username}_{suffix}.jpg"
        full_path = os.path.join(GRADCAM_DIR, filename)
        cv2.imwrite(full_path, overlay)  # type: ignore
        print(f"✅ GradCAM [{suffix}] class_idx={class_idx} → {full_path}")

        # ✅ Return ABSOLUTE path — app.py to_web_url() handles URL conversion
        return full_path

    except Exception as e:
        import traceback
        print(f"[GradCAM {suffix}] Error: {e}")
        traceback.print_exc()
        return None


# ── Main inference ─────────────────────────────────────────────────────

def run_ct_inference(image_path: str, username: str = "user") -> Dict[str, Any]:
    try:
        load_models()
        assert ct_type_model   is not None
        assert ct_binary_model is not None

        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}

        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not read image: {image_path}"}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original  = cv2.resize(image, (224, 224))  # BGR for overlay

        tensor = cast(
            torch.Tensor,
            transform(Image.fromarray(image_rgb)),
        ).unsqueeze(0).to(DEVICE)

        # ── Copy original image into static folder so it's web-accessible ─
        static_original_path = _copy_original_to_static(image_path, username)

        # ── Inference ─────────────────────────────────────────────────
        with torch.no_grad():
            type_output = ct_type_model(tensor)
            type_out    = type_output[0] if isinstance(type_output, tuple) else type_output
            features    = type_output[1] if isinstance(type_output, tuple) else type_out
            type_probs  = torch.softmax(type_out, dim=1)

            binary_output = ct_binary_model(tensor)
            binary_out    = binary_output[0] if isinstance(binary_output, tuple) else binary_output
            binary_probs  = torch.softmax(binary_out, dim=1)

        type_idx    = int(torch.argmax(type_probs,   dim=1).item())
        cancer_type = CLASS_NAMES[type_idx]
        type_conf   = float(type_probs[0][type_idx].item())
        cancer_prob = float(binary_probs[0][1].item())

        is_normal = cancer_type.strip().lower() in NORMAL_LABELS

        print(f"🔍 Predicted: {cancer_type} (idx={type_idx}, conf={type_conf:.2%}, normal={is_normal})")

        # ── GradCAM ────────────────────────────────────────────────────
        # For Normal predictions both calls will return None (skipped).
        # The fallback chain in app.py will then use the plain original image.
        gradcam_type_path = _run_gradcam(
            ct_type_model, tensor, original,
            class_idx=type_idx,
            username=username,
            suffix="type",
            predicted_label=cancer_type,          # ← passed so skip logic works
        )

        # Binary GradCAM: for Normal we target class 0 (no-cancer),
        # but we still skip the heatmap to avoid misleading visuals.
        binary_class_idx = 0 if is_normal else 1
        gradcam_binary_path = _run_gradcam(
            ct_binary_model, tensor, original,
            class_idx=binary_class_idx,
            username=username,
            suffix="binary",
            predicted_label=cancer_type,          # ← same label drives the skip
        )

        gradcam_primary = gradcam_type_path or gradcam_binary_path

        print(f"✅ original_image (static) → {static_original_path}")
        print(f"✅ gradcam_binary → {gradcam_binary_path}")
        print(f"✅ gradcam_type   → {gradcam_type_path}")

        return {
            "cancer_type":        cancer_type,
            "type_confidence":    round(type_conf   * 100, 2),
            "cancer_probability": round(cancer_prob * 100, 2),
            # ✅ original_image now points to the STATIC copy — always web-accessible
            "original_image":     static_original_path,
            "gradcam":            gradcam_primary,
            "gradcam_binary":     gradcam_binary_path,
            "gradcam_type":       gradcam_type_path,
            "type_logits":        type_out.detach().cpu().numpy().tolist(),
            "binary_logits":      binary_out.detach().cpu().numpy().tolist(),
            "features":           features.detach().cpu().numpy().tolist(),
        }

    except Exception as e:
        import traceback
        print("Inference Error:", e)
        traceback.print_exc()
        return {
            "error":              str(e),
            "cancer_type":        "N/A",
            "type_confidence":    0,
            "cancer_probability": 0,
            "original_image":     image_path,
            "gradcam":            None,
            "gradcam_binary":     None,
            "gradcam_type":       None,
        }