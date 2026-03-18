import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt

# === IMPORT YOUR ORIGINAL CODE ===
from planet_finding_cnn import (
    Config,
    PlanetCubeFolderDataset,
    build_model
)

cfg = Config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Evaluation function
# =========================

def evaluate(model, loader):
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    return y_true, y_pred, y_prob


# =========================
# Plot helpers
# =========================

def plot_confusion_matrix(cm, save_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["No Planet", "Planet"]
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return auc


def plot_pr(y_true, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return ap


# =========================
# Main
# =========================

def main():

    test_dir = os.path.join(cfg.data_root, "test")

    test_ds = PlanetCubeFolderDataset(test_dir, normalize=cfg.normalize)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # Infer input channels
    C, H, W = test_ds[0][0].shape

    # Rebuild model
    model = build_model(cfg.model_name, in_channels=C, num_classes=2)
    model.load_state_dict(torch.load(cfg.save_path, map_location=DEVICE))
    model.to(DEVICE)

    print("Model loaded.")

    # === Evaluate ===
    y_true, y_pred, y_prob = evaluate(model, test_loader)

    # =========================
    # Metrics
    # =========================

    # Accuracy by class
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\n=== Accuracy by Class ===")
    print(f"No Planet (0): {class_acc[0]:.4f}")
    print(f"Planet (1):    {class_acc[1]:.4f}")

    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_prob)

    # PR AUC
    pr_auc = average_precision_score(y_true, y_prob)

    print("\n=== Global Metrics ===")
    print(f"ROC AUC : {roc_auc:.4f}")
    print(f"AUC-PR  : {pr_auc:.4f}")

    # =========================
    # Plots
    # =========================

    os.makedirs("plots", exist_ok=True)

    plot_confusion_matrix(cm, "plots/confusion_matrix.png")
    print("Saved confusion_matrix.png")

    plot_roc(y_true, y_prob, "plots/roc_curve.png")
    print("Saved roc_curve.png")

    plot_pr(y_true, y_prob, "plots/pr_curve.png")
    print("Saved pr_curve.png")


if __name__ == "__main__":
    main()
