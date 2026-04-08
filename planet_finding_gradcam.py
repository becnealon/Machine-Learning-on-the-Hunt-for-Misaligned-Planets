import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# === import from the other script ===
from planet_finding_cnn import (
    Config,
    PlanetCubeFolderDataset,
    build_model
)

cfg = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Grad-CAM implementation
# =========================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx):
        self.model.zero_grad()

        logits = self.model(x)
        score = logits[:, class_idx]

        score.backward()

        grads = self.gradients          # (1, C, H', W')
        acts = self.activations         # (1, C, H', W')

        weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


# =========================
# Find last conv layer
# =========================

def find_last_conv_layer(model):
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise RuntimeError("No Conv2d layer found.")


# =========================
# Overlay helper
# =========================

def make_overlay(img, cam):
    """
    img: (C,H,W)
    cam: (H,W)
    """
    img2d = img.mean(axis=0)
    img2d = (img2d - img2d.min()) / (img2d.max() + 1e-8)

    heatmap = plt.get_cmap("jet")(cam)[:, :, :3]

    overlay = 0.5 * img2d[..., None] + 0.5 * heatmap
    overlay /= overlay.max()

    return img2d, overlay


# =========================
# Main
# =========================

def main():

    np.random.seed(42)

    test_dir = os.path.join(cfg.data_root, "test")

    ds = PlanetCubeFolderDataset(test_dir, normalize=cfg.normalize)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    # === Load model ===
    C, H, W = ds[0][0].shape

    model = build_model(cfg.model_name, in_channels=C, num_classes=2)
    model.load_state_dict(torch.load(cfg.save_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("Model loaded.")

    # =========================
    # Pass through test set
    # =========================

    all_preds = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)

            logits = model(X)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(y)
            all_inputs.append(X.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    X_full = torch.cat(all_inputs)   # (N,C,H,W)

    print(f"Collected {len(y_true)} samples.")

    # =========================
    # Choose examples
    # =========================

    idx_all = np.arange(len(y_true))
    corr_idx = idx_all[y_true == y_pred]
    inc_idx  = idx_all[y_true != y_pred]

    n_cam = 10
    chosen = []

    if len(inc_idx) > 0:
        k = min(n_cam // 2, len(inc_idx))
        chosen.extend(np.random.choice(inc_idx, k, replace=False).tolist())

    remaining = n_cam - len(chosen)

    if remaining > 0:
        pool = corr_idx if len(corr_idx) > 0 else idx_all
        chosen.extend(np.random.choice(pool, remaining, replace=False).tolist())

    print(f"Selected indices: {chosen}")

    # =========================
    # Grad-CAM setup
    # =========================

    target_layer = find_last_conv_layer(model)
    cam = GradCAM(model, target_layer)

    os.makedirs("gradcam_outputs", exist_ok=True)

    # =========================
    # Generate CAMs
    # =========================

    for idx in chosen:

        input_tensor = X_full[idx:idx+1].to(DEVICE)

        pred_class = int(y_pred[idx])
        true_class = int(y_true[idx])

        cam_map = cam(input_tensor, class_idx=pred_class)

        img = X_full[idx].numpy()

        img2d, overlay = make_overlay(img, cam_map)

        plt.figure(figsize=(8, 4))

        # original
        plt.subplot(1, 2, 1)
        plt.imshow(img2d, cmap="gray")
        plt.title(f"Original\ntrue={true_class}, pred={pred_class}")
        plt.axis("off")

        # gradcam
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM")
        plt.axis("off")

        out_path = f"gradcam_outputs/gradcam_{idx}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
