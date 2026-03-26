"""Grad-CAM implementation for HMSAN explainability."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM wrapper that hooks a target convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        output = self.model(x)
        logits = output["logits"]

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        target = logits[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        target.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay normalized CAM on top of an RGB image."""
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay
