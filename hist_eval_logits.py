# eval_evaluator_with_threshold.py
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, f1_score,
    confusion_matrix, balanced_accuracy_score
)

from simpleRL import run_to_completion, make_atari_env
from dataset import ReachabilityDataset
from agent import Embedder, Evaluator, TransitionModel, SimpleTransitionModel

def pick_threshold(probs, labels, mode: str):
    labels = labels.astype(int)

    if mode == "youden":
        fpr, tpr, thr = roc_curve(labels, probs)
        j = tpr - fpr
        i = np.argmax(j)
        return float(thr[i]), {"fpr": fpr, "tpr": tpr, "thr": thr, "idx": i}

    elif mode == "f1":
        # Use PR thresholds (note sklearn returns thresholds shorter by 1)
        prec, rec, thr = precision_recall_curve(labels, probs)
        # Evaluate F1 on a dense grid to be safe
        grid = np.linspace(0.0, 1.0, 1001)
        f1s = []
        for t in grid:
            preds = (probs >= t).astype(int)
            f1s.append(f1_score(labels, preds))
        i = int(np.argmax(f1s))
        return float(grid[i]), {"precision": prec, "recall": rec, "grid": grid, "f1s": np.array(f1s), "idx": i}

    elif mode == "balanced_accuracy":
        grid = np.linspace(0.0, 1.0, 1001)
        bals = []
        for t in grid:
            preds = (probs >= t).astype(int)
            bals.append(balanced_accuracy_score(labels, preds))
        i = int(np.argmax(bals))
        return float(grid[i]), {"grid": grid, "bal": np.array(bals), "idx": i}

    else:
        raise ValueError(f"Unknown threshold mode: {mode}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="model.pth")
    p.add_argument("--n_runs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--threshold_mode", type=str, default="youden",
                   choices=["youden", "f1", "balanced_accuracy"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    obs_shape = tuple(cfg["obs_shape"])
    embedding_dim = int(cfg["embedding_dim"])

    emb = Embedder(obs_shape, embedding_dim).to(device)
    eval_net = Evaluator(embedding_dim).to(device)
    tm = TransitionModel(cfg["n_actions"], embedding_dim).to(device)
    stm = SimpleTransitionModel(embedding_dim).to(device)

    emb.load_state_dict(ckpt["embedder_state_dict"])
    eval_net.load_state_dict(ckpt["evaluator_state_dict"])
    tm.load_state_dict(ckpt["transition_model_state_dict"])
    stm.load_state_dict(ckpt["simple_tm_state_dict"])

    emb.eval()
    eval_net.eval()

    # Fresh data
    env_fn = make_atari_env(cfg["env_id"], seed=args.seed)
    env = env_fn()

    def random_policy(obs_batch: torch.Tensor) -> torch.Tensor:
        B = obs_batch.shape[0]
        return torch.randint(0, cfg["n_actions"], (B,), device=obs_batch.device)

    S, A, R, TR = run_to_completion(env_fn=lambda: env, policy=random_policy,
                                    n_runs=args.n_runs, device="cpu")

    # Dataset / DataLoader
    datamodule = ReachabilityDataset(S, A, device=str(device))
    dataloader = torch.utils.data.DataLoader(datamodule, batch_size=args.batch_size, shuffle=False)

    # Collect logits and labels
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            states = batch["state"].to(device, non_blocking=True)
            comp_state = batch["compare_state"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy()

            e_state = emb(states)
            e_comp = emb(comp_state)

            logits = eval_net(e_comp, e_state)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels)

    if not all_logits:
        print("No data collected.")
        return

    logits_np = torch.cat(all_logits, dim=0).numpy().reshape(-1)
    labels_np = np.concatenate(all_labels).astype(int)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))

    # AUC
    try:
        auc = roc_auc_score(labels_np, probs_np)
        print(f"ROC AUC: {auc:.4f}")
    except ValueError:
        print("ROC AUC not defined (single-class labels in this sample).")

    # Histograms
    plt.figure(figsize=(8, 5))
    plt.hist(logits_np, bins=args.bins)
    plt.xlabel("Evaluator logits")
    plt.ylabel("Count")
    plt.title("Histogram of Evaluator logits")
    plt.tight_layout()
    plt.savefig("evaluator_logits_hist.png", dpi=150)

    plt.figure(figsize=(8, 5))
    plt.hist(probs_np, bins=args.bins, range=(0, 1))
    plt.xlabel("Evaluator probabilities (sigmoid)")
    plt.ylabel("Count")
    plt.title("Histogram of Evaluator sigmoid outputs")
    plt.tight_layout()
    plt.savefig("evaluator_probs_hist.png", dpi=150)

    # Choose optimal threshold
    thr, aux = pick_threshold(probs_np, labels_np, args.threshold_mode)
    print(f"Chosen threshold ({args.threshold_mode}): {thr:.4f}")

    # Confusion matrix (row-normalized percent)
    preds_np = (probs_np >= thr).astype(int)
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    cm_pct = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True) * 100.0
    print("Confusion matrix counts:\n", cm)
    print("Confusion matrix row-%:\n", np.round(cm_pct, 1))

    # Percent heatmap using matplotlib (no seaborn dependency)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_pct, interpolation="nearest")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (%) @ thr={thr:.3f} ({args.threshold_mode})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("evaluator_confusion_matrix_percent.png", dpi=150)

    # ROC + PR curves with chosen threshold marked
    # ROC
    fpr, tpr, roc_thr = roc_curve(labels_np, probs_np)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2)
    # Mark point closest to chosen threshold (by value difference)
    idx = np.argmin(np.abs(roc_thr - thr)) if roc_thr.size else None
    if idx is not None and idx < len(fpr):
        plt.scatter([fpr[idx]], [tpr[idx]])
        plt.annotate(f"thr={thr:.3f}", (fpr[idx], tpr[idx]))
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig("evaluator_roc.png", dpi=150)

    # PR
    prec, rec, pr_thr = precision_recall_curve(labels_np, probs_np)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2)
    # Mark point closest to chosen threshold
    if pr_thr.size:
        idx = np.argmin(np.abs(pr_thr - thr))
        # precision_recall_curve returns len(thr)+1 points; align annotation
        r = rec[idx]; p = prec[idx]
        plt.scatter([r], [p])
        plt.annotate(f"thr={thr:.3f}", (r, p))
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig("evaluator_pr.png", dpi=150)

if __name__ == "__main__":
    main()
