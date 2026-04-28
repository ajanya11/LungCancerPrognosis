# =====================================================
# 🔥 GRAD-CAM (FULLY FIXED — NO ERRORS)
# =====================================================

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional


# ── Labels considered normal / healthy — heatmap is skipped for these ──
NORMAL_LABELS = {"normal", "no cancer", "benign"}


# =====================================================
# 🔥 GRADCAM CLASS
# =====================================================

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._fh = target_layer.register_forward_hook(self._forward_hook)
        self._bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self) -> None:
        if self._fh:
            self._fh.remove()
        if self._bh:
            self._bh.remove()

    def generate(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:

        try:
            self.gradients = None
            self.activations = None

            x = x.clone().detach().requires_grad_(True)

            output = self.model(x)
            if isinstance(output, tuple):
                output = output[0]

            if class_idx is None:
                class_idx = int(torch.argmax(output, dim=1).item())

            self.model.zero_grad()
            score = output[0, class_idx]
            score.backward()

            if self.gradients is None or self.activations is None:
                raise RuntimeError("GradCAM hooks not triggered")

            grads = self.gradients.detach().cpu()
            acts  = self.activations.detach().cpu()

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam     = (weights * acts).sum(dim=1).squeeze()

            cam = F.relu(cam).numpy()

            if float(cam.max()) > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            cam = cv2.resize(cam, (224, 224))  # type: ignore

            return cam.astype(np.float32)

        finally:
            self.remove_hooks()


# =====================================================
# 🔥 HEATMAP OVERLAY (FIXED)
# =====================================================
def overlay_heatmap(img_bgr: np.ndarray, cam: np.ndarray) -> np.ndarray:

    # ✅ FIX: explicit dtype conversion
    heatmap_input = (cam * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heatmap_input,
        cv2.COLORMAP_JET
    )

    overlay = heatmap.astype(np.float32) / 255.0 + img_bgr.astype(np.float32) / 255.0
    overlay = overlay / (overlay.max() + 1e-8)

    return (overlay * 255).astype(np.uint8)


# =====================================================
# 🔥 TARGET LAYER DETECTION (FIXED)
# =====================================================

def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:

    if hasattr(model, "layer4"):
        layer4 = getattr(model, "layer4")
        return layer4[-1]

    if hasattr(model, "target_layer"):
        tl = getattr(model, "target_layer")
        if isinstance(tl, torch.nn.Sequential):
            return tl[-1]
        return tl

    if hasattr(model, "features"):
        features = getattr(model, "features")
        return features[-1]

    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m

    raise RuntimeError("No Conv2d layer found for GradCAM")


# =====================================================
# 🔥 SAVE GRADCAM (FINAL)
# =====================================================

def save_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    image_path: str,
    username: str,
    class_idx: Optional[int] = None,
    predicted_label: Optional[str] = None,    # ✅ NEW: skip heatmap when Normal
    suffix: str = "type",
    save_dir: str = "frontend/static/gradcam"
) -> Optional[str]:
    """
    Generate and save a Grad-CAM heatmap overlay.

    If ``predicted_label`` is 'Normal', 'No Cancer', or 'Benign' (case-
    insensitive) the function returns None immediately without generating
    a heatmap.  Showing activation maps for healthy scans is misleading
    because GradCAM highlights the 'most suspicious' region regardless of
    whether cancer is actually present.

    Returns the relative path  gradcam/<filename>  on success, or None.
    """

    # ✅ Skip heatmap for normal / healthy predictions
    if predicted_label is not None and predicted_label.strip().lower() in NORMAL_LABELS:
        print(f"⏭️  save_gradcam [{suffix}] skipped — normal prediction: '{predicted_label}'")
        return None

    try:
        original = cv2.imread(image_path)
        if original is None:
            print("❌ Image not found:", image_path)
            return None

        original = cv2.resize(original, (224, 224))  # type: ignore

        target_layer = get_target_layer(model)

        cam_generator = GradCAM(model, target_layer)

        cam = cam_generator.generate(
            tensor,
            class_idx if class_idx is not None else None
        )

        overlay = overlay_heatmap(original, cam)

        os.makedirs(save_dir, exist_ok=True)

        filename = f"gradcam_{username}_{suffix}.jpg"
        full_path = os.path.join(save_dir, filename)

        cv2.imwrite(full_path, overlay)  # type: ignore

        print("✅ GradCAM saved:", full_path)

        return f"gradcam/{filename}"

    except Exception as e:
        import traceback
        print("❌ GradCAM error:", e)
        traceback.print_exc()
        return None