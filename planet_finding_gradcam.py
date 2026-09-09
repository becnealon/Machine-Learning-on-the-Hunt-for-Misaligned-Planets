import argparse
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from planet_finding_CNN import (
    Config,
    PlanetCubeFolderDataset,
    build_model
)

cfg = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIM_RE = re.compile(r"(sim\d+)")


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


def find_last_conv_layer(model):
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise RuntimeError("No Conv2d layer found.")


# =========================
# Simulation-ID handling
# =========================

def sim_id_from_path(path):
    """Extract the short 'simNN' identifier from a data filename.

    Matches how stack_channels.py groups files, so it works for both
    stacked cube filenames (simNN.pt) and raw per-channel filenames
    (simNN_mpX_incY..._raw_j_idxK.npy).
    """
    fn = os.path.basename(path)
    m = SIM_RE.search(fn)
    return m.group(1) if m else fn


def resolve_sim_arg(sim_arg, sim_ids):
    candidates = {sim_arg}
    if sim_arg.isdigit():
        candidates.add(f"sim{sim_arg}")

    matches = [i for i, s in enumerate(sim_ids) if s in candidates]
    if matches:
        return matches

    return [i for i, s in enumerate(sim_ids) if sim_arg in s]


# =========================
# Display helpers
# =========================

def _normalize01(arr2d):
    lo, hi = arr2d.min(), arr2d.max()
    return (arr2d - lo) / (hi - lo + 1e-8)


def overlay_on_channel(channel_img, cam):
    """channel_img, cam: (H,W), both used as-is (caller normalizes channel_img)."""
    heatmap = plt.get_cmap("jet")(cam)[:, :, :3]
    overlay = 0.5 * channel_img[..., None] + 0.5 * heatmap
    overlay /= (overlay.max() + 1e-8)
    return overlay


def save_two_panel(base_img, overlay, title_left, title_right, out_path):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(base_img, cmap="inferno")
    plt.title(title_left)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(title_right)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


# =========================
# Data / model loading
# =========================

def load_test_set():
    test_dir = os.path.join(cfg.data_root, "test")

    ds = PlanetCubeFolderDataset(test_dir, normalize=cfg.normalize)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    C, H, W = ds[0][0].shape

    model = build_model(cfg.model_name, in_channels=C, num_classes=2)
    model.load_state_dict(torch.load(cfg.save_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    all_preds, all_targets, all_inputs = [], [], []
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

    # ds is loaded with shuffle=False, so this ordering matches X_full/y_pred/y_true.
    sim_ids = [sim_id_from_path(p) for p, _ in ds.samples]

    print(f"Collected {len(y_true)} samples across {len(set(sim_ids))} simulations.")

    return model, X_full, y_true, y_pred, sim_ids


def compute_cam(cam_engine, X_full, idx, pred_class):
    input_tensor = X_full[idx:idx + 1].to(DEVICE)
    return cam_engine(input_tensor, class_idx=pred_class)


# =========================
# Output modes
# =========================

def run_single_channel(sim_id, idx, channel, X_full, y_true, y_pred, cam_engine, out_dir):
    img = X_full[idx].numpy()  # (C,H,W)
    C = img.shape[0]
    if not (0 <= channel < C):
        raise SystemExit(f"--channel {channel} out of range: {sim_id} cube has {C} channels (0-{C - 1}).")

    cam_map = compute_cam(cam_engine, X_full, idx, int(y_pred[idx]))

    ch_img = _normalize01(img[channel])
    overlay = overlay_on_channel(ch_img, cam_map)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sim_id}_channel{channel:02d}.png")
    save_two_panel(
        ch_img, overlay,
        f"{sim_id} channel {channel}\ntrue={int(y_true[idx])}, pred={int(y_pred[idx])}",
        "Grad-CAM",
        out_path,
    )


def run_all_channels(sim_id, idx, X_full, y_true, y_pred, cam_engine, out_dir):
    img = X_full[idx].numpy()  # (C,H,W)
    C = img.shape[0]

    # One CAM per sample: the target layer mixes all input channels together,
    # so the heatmap itself doesn't change per channel, only what it's drawn on.
    cam_map = compute_cam(cam_engine, X_full, idx, int(y_pred[idx]))

    sim_dir = os.path.join(out_dir, sim_id)
    os.makedirs(sim_dir, exist_ok=True)

    ncols = 8
    nrows = int(np.ceil(C / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes_flat = np.atleast_1d(axes).flatten()

    for c in range(nrows * ncols):
        ax = axes_flat[c]
        if c < C:
            ch_img = _normalize01(img[c])
            overlay = overlay_on_channel(ch_img, cam_map)

            save_two_panel(
                ch_img, overlay,
                f"{sim_id} channel {c}\ntrue={int(y_true[idx])}, pred={int(y_pred[idx])}",
                "Grad-CAM",
                os.path.join(sim_dir, f"channel{c:02d}.png"),
            )

            ax.imshow(overlay)
            ax.set_title(f"ch{c}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{sim_id}  true={int(y_true[idx])} pred={int(y_pred[idx])}  (all channels)")
    plt.tight_layout()
    overview_path = os.path.join(sim_dir, "_all_channels_overview.png")
    plt.savefig(overview_path)
    plt.close(fig)
    print(f"Saved overview {overview_path}")
    print(f"Saved {C} per-channel Grad-CAMs to {sim_dir}")


def choose_default_indices(n, y_true, y_pred, sim_ids, seed=42):
    """Prefer misclassified examples, but never more than one sample per simulation."""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y_true))
    inc_idx = idx_all[y_true != y_pred]
    corr_idx = idx_all[y_true == y_pred]
    rng.shuffle(inc_idx)
    rng.shuffle(corr_idx)

    chosen = []
    seen_sims = set()
    for i in list(inc_idx) + list(corr_idx):
        if len(chosen) >= n:
            break
        if sim_ids[i] in seen_sims:
            continue
        chosen.append(int(i))
        seen_sims.add(sim_ids[i])
    return chosen


def run_default_random(n, X_full, y_true, y_pred, sim_ids, cam_engine, out_dir):
    chosen = choose_default_indices(n, y_true, y_pred, sim_ids)
    print(f"Selected {len(chosen)} samples from distinct simulations: {[sim_ids[i] for i in chosen]}")

    os.makedirs(out_dir, exist_ok=True)

    for idx in chosen:
        sim_id = sim_ids[idx]
        img = X_full[idx].numpy()
        cam_map = compute_cam(cam_engine, X_full, idx, int(y_pred[idx]))

        img_mean = _normalize01(img.mean(axis=0))
        overlay = overlay_on_channel(img_mean, cam_map)

        out_path = os.path.join(out_dir, f"gradcam_{sim_id}.png")
        save_two_panel(
            img_mean, overlay,
            f"{sim_id} (channel-mean)\ntrue={int(y_true[idx])}, pred={int(y_pred[idx])}",
            "Grad-CAM",
            out_path,
        )


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM for the planet-finding CNN, labeled by source simulation."
    )
    parser.add_argument("--sim", type=str, default=None,
                         help="Simulation identifier (e.g. 'sim23' or '23') to generate Grad-CAM(s) for. "
                              "If omitted, --n random samples from distinct simulations are used instead.")
    parser.add_argument("--channel", type=int, default=None,
                         help="Specific velocity-channel index within --sim to generate a single Grad-CAM for. "
                              "Requires --sim. If omitted (with --sim set), all channels in that simulation's "
                              "cube are generated.")
    parser.add_argument("--n", type=int, default=10,
                         help="Number of random simulations to sample when --sim is not given (default: 10).")
    parser.add_argument("--out_dir", type=str, default="gradcam_outputs",
                         help="Output directory for saved figures (default: gradcam_outputs).")
    args = parser.parse_args()

    if args.channel is not None and args.sim is None:
        parser.error("--channel requires --sim to be specified.")

    np.random.seed(42)

    model, X_full, y_true, y_pred, sim_ids = load_test_set()
    target_layer = find_last_conv_layer(model)
    cam_engine = GradCAM(model, target_layer)

    if args.sim is not None:
        matches = resolve_sim_arg(args.sim, sim_ids)
        if not matches:
            available = ", ".join(sorted(set(sim_ids))[:20])
            raise SystemExit(
                f"No test-set sample found for sim '{args.sim}'. "
                f"Available (first 20 of {len(set(sim_ids))}): {available}"
            )
        if len(matches) > 1:
            print(f"Note: {len(matches)} test samples matched '{args.sim}'; using the first one "
                  f"({sim_ids[matches[0]]}).")
        idx = matches[0]
        sim_id = sim_ids[idx]

        if args.channel is not None:
            run_single_channel(sim_id, idx, args.channel, X_full, y_true, y_pred, cam_engine, args.out_dir)
        else:
            run_all_channels(sim_id, idx, X_full, y_true, y_pred, cam_engine, args.out_dir)
    else:
        run_default_random(args.n, X_full, y_true, y_pred, sim_ids, cam_engine, args.out_dir)


if __name__ == "__main__":
    main()
