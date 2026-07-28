import io

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.activations = None
        self.gradients = None
        self._forward_handle = None
        self._backward_handle = None

        for name, module in model.named_modules():
            if name == layer_name:
                self._forward_handle = module.register_forward_hook(self._capture_forward)
                self._backward_handle = module.register_full_backward_hook(self._capture_backward)
                break
        else:
            raise ValueError(f"Layer '{layer_name}' not found in model")

    def _capture_forward(self, module, input, output):
        self.activations = output.detach()

    def _capture_backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, target_class=None):
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        out = self.model(x)
        if target_class is None:
            target_class = out.argmax(dim=1).item()

        score = out[0, target_class]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("No gradients or activations captured")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def cleanup(self):
        if self._forward_handle is not None:
            self._forward_handle.remove()
        if self._backward_handle is not None:
            self._backward_handle.remove()


def generar_mapa_gradcam(model, spectrogram_tensor, layer_name='conv4', target_class=None):
    device = next(model.parameters()).device
    if spectrogram_tensor.dim() == 3:
        spectrogram_tensor = spectrogram_tensor.unsqueeze(0)
    x = spectrogram_tensor.to(device)
    model.train()
    x.requires_grad_(True)

    gradcam = GradCAM(model, layer_name)
    try:
        cam = gradcam.generate(x, target_class=target_class)
    finally:
        gradcam.cleanup()
    model.eval()
    return cam


def _figura_superposicion(spectrogram_2d, cam, sr=16000, hop_length=512):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    extent = [0, spectrogram_2d.shape[1] * hop_length / sr,
              0, spectrogram_2d.shape[0]]

    ax1.imshow(spectrogram_2d, aspect='auto', origin='lower',
               cmap='gray', extent=extent)
    ax1.set_title('Espectrograma Original')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Bandas Mel')

    ax2.imshow(spectrogram_2d, aspect='auto', origin='lower',
               cmap='gray', extent=extent)
    ax2.imshow(cam, aspect='auto', origin='lower', cmap='jet',
               alpha=0.5, extent=extent)
    ax2.set_title('Grad-CAM: Regiones críticas')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Bandas Mel')

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf