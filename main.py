from simpleRL import run_to_completion, make_atari_env


import torch
import torch.nn as nn
from dataset import SequenceDataset
from agent import Embedder, TransitionModel, DistanceFunction, RewardModel
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn.functional as F
env_id = "ALE/Qbert-v5"
env = make_atari_env(env_id, seed=42)()  # Create an environment instance
embedding_dim = 128

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

def random_policy(obs_batch):
    B = obs_batch.shape[0]
    return torch.randint(0, env.action_space.n, (B,), device=obs_batch.device)

S, A, R, TR = run_to_completion(env_fn=lambda: env, policy=random_policy, n_runs=40, device="cpu")

def sum_rewards(rewards):
    return sum(sum(ep_rewards) for ep_rewards in rewards)

print("Total rewards over random policy:", sum_rewards(R))

datamodule = SequenceDataset(S, A, R, device="cuda")

print("Dataset size (number of (state, action) pairs):", len(datamodule))

emb = Embedder(env.observation_space.shape,embedding_dim).to("cuda")
dist = DistanceFunction(embedding_dim).to("cuda")
transition = TransitionModel(env.action_space.n, embedding_dim).to("cuda")
reward_model = RewardModel(embedding_dim).to("cuda")

all_params = list(emb.parameters()) + list(dist.parameters()) + list(transition.parameters()) + list(reward_model.parameters())
opt = torch.optim.Adam(all_params, lr=1e-4)

ma_emb_mean = 0.0
ma_emb_std = 0.0

ma_trans_loss = 0.0

ma_dist_loss = 0.0
ma_triangle_loss = 0.0
ma_dist_pred_min = 0.0
ma_dist_pred_max = 0.0
ma_direct_loss = 0.0
ma_reward_loss = 0.0
ma_diff_minmax = 0.0

"""
        return {
            "s1": s1,
            "s2": s2,
            "sn": sn,
            "sm": sm,
            "a1": a1,
            "d1" : d1,
            "d2" : d2,
            "r1" : r1,
        }

"""

mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()

def mse_log_loss(pred, target):
    """

    :param pred:
    :param target:
    :return:
    """
    return F.mse_loss(torch.log1p(pred), torch.log1p(target))

def linear_loss(pred, target):
    #min(linear, mse)
    return torch.min(F.l1_loss(pred, target), F.mse_loss(pred, target))

n_epochs = 200
for i in range(n_epochs):
    dataloader = torch.utils.data.DataLoader(datamodule, batch_size=256, shuffle=True)

    eps = (1.0 - i/n_epochs) # goes from 1.0 to 0.0
    #make it exponentially decay
    #eps = eps * eps
    #make it go from 10 to 0.001
    eps = eps + 0.001

    for batch in dataloader:
        opt.zero_grad()
        s1 = batch["s1"].to("cuda", non_blocking=True)
        s2 = batch["s2"].to("cuda", non_blocking=True)
        s3 = batch["s3"].to("cuda", non_blocking=True)
        a1 = batch["a1"].to("cuda", non_blocking=True)
        a2 = batch["a2"].to("cuda", non_blocking=True)
        sn = batch["sn"].to("cuda", non_blocking=True)
        sm = batch["sm"].to("cuda", non_blocking=True)
        d1 = batch["d1"].to("cuda", non_blocking=True).float()
        d2 = batch["d2"].to("cuda", non_blocking=True).float()
        r1 = batch["r1"].to("cuda", non_blocking=True).float()
        e1 = emb(s1)
        e2 = emb(s2)
        e3 = emb(s3)
        en = emb(sn)
        em = emb(sm)

        e2_pred = transition(e1, a1)
        e3_pred = transition(e2_pred, a2)
        transition_loss = mse_loss(e2_pred, e2) + mse_loss(e3_pred, e3)

        ma_trans_loss = transition_loss.item() * 0.05 + ma_trans_loss * 0.95

        r1_pred = reward_model(e1).squeeze(-1)
        reward_loss = mse_loss(r1_pred, r1)
        ma_reward_loss = reward_loss.item() * 0.05 + ma_reward_loss * 0.95

        pd1_min, pd1_max = dist(e1, en)
        pd2_min, pd2_max = dist(en, em)
        pd_tot_min, pd_tot_max = dist(e1, em)
        pd_direct_min, pd_direct_max = dist(e1, e2)

        #the direct distance is at most 1 step
        loss_direct_min = (pd_direct_min - 1.0).clamp(min=0.0).pow(2).mean()
        #and at least 1 step for the max, right?
        loss_direct_max = (1.0 - pd_direct_max).clamp(min=0.0).pow(2).mean()
        loss_direct = loss_direct_min + loss_direct_max
        ma_direct_loss = loss_direct.item() * 0.05 + ma_direct_loss * 0.95

        #triangle inequality: pd_tot_min <= pd1_min + pd2_min
        triangle_loss_min = (pd_tot_min - (pd1_min + pd2_min)).clamp(min=0.0).mean()
        #for max side: pd_tot_max >= pd1_max + pd2_max
        triangle_loss_max = ((pd1_max + pd2_max) - pd_tot_max).clamp(min=0.0).mean()
        triangle_loss = triangle_loss_min + triangle_loss_max
        ma_triangle_loss = triangle_loss.item() * 0.05 + ma_triangle_loss * 0.95

        with torch.no_grad():
            twos = torch.ones_like(d1)
            hundred = 50 * twos
            target_tot_min = torch.min(pd_tot_min + eps, torch.min((torch.max(pd1_min + pd2_min,twos) + d1 + d2) * 0.5, d1 + d2))
            target_tot_max = torch.max(pd_tot_max - eps, torch.max((torch.min(pd1_max + pd2_max,hundred) + d1 + d2) * 0.5, d1 + d2))

        loss_dist = linear_loss(pd_tot_min, target_tot_min) + linear_loss(pd_tot_max, target_tot_max)

        ma_dist_loss = loss_dist.item() * 0.05 + ma_dist_loss * 0.95
        ma_dist_pred_min = pd_tot_min.mean().item() * 0.05 + ma_dist_pred_min * 0.95
        ma_dist_pred_max = pd_tot_max.mean().item() * 0.05 + ma_dist_pred_max * 0.95

        diff_minmax = (pd_tot_max - pd_tot_min).detach().mean().item()

        ma_diff_minmax = diff_minmax * 0.05 + ma_diff_minmax * 0.95

        #this loss forces the min to be smaller than the max
        loss_diff_minmax = (pd_tot_min - pd_tot_max).clamp(min=0.0).pow(2).mean()

        loss = loss_dist  + loss_direct + transition_loss * 0.01 + reward_loss + triangle_loss + loss_diff_minmax

        loss.backward()
        opt.step()

        ma_emb_mean = e1.mean().item() * 0.05 + ma_emb_mean * 0.95
        ma_emb_std = e1.std().mean().item() * 0.05 + ma_emb_std * 0.95

        print(
            f"Epoch {i:03d} "
            f"Distance: {ma_dist_loss:0.4f} "
            f"Triangle: {ma_triangle_loss:0.4f} "
            f"Direct: {ma_direct_loss:0.4f} "
            f"Transition: {ma_trans_loss:0.4f} "
            f"Reward: {ma_reward_loss:0.4f} "
            f"DiffMinMax: {ma_diff_minmax:0.4f} "
            f"Dist Pred: {ma_dist_pred_min:0.2f}-{ma_dist_pred_max:0.2f}\r",
            end="",
            flush=True
        )

#save the model
torch.save({
    'config' : {
        'env_id': env_id,
        'embedding_dim': embedding_dim,
        'n_actions': env.action_space.n,
        'obs_shape': env.observation_space.shape,
    },
    'embedder_state_dict': emb.state_dict(),
    'distance_state_dict': dist.state_dict(),
    'transition_model_state_dict': transition.state_dict(),
}, "model.pth")

import torch
import matplotlib.pyplot as plt

# --- collect embeddings as before ---
emb.eval()
infer_loader = torch.utils.data.DataLoader(datamodule, batch_size=512, shuffle=True)

emb_list = []
with torch.no_grad():
    for batch in infer_loader:
        s1 = batch["s1"].to("cuda", non_blocking=True)
        e1 = emb(s1)
        emb_list.append(e1.detach().cpu())
E = torch.cat(emb_list, dim=0)  # [N, D]

# --- center and compute covariance ---
E_centered = E - E.mean(dim=0, keepdim=True)
N = E_centered.shape[0]
C = (E_centered.T @ E_centered) / (N - 1)

# --- eigen decomposition ---
eigvals, _ = torch.linalg.eigh(C)  # ascending
eigvals = eigvals.flip(0)          # descending

# --- plot ---
plt.figure(figsize=(8,5))
plt.plot(eigvals.numpy(), marker="o", linestyle="-")
plt.title("Embedding covariance eigenvalue spectrum")
plt.xlabel("Principal component index")
plt.ylabel("Eigenvalue (variance explained)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("eigenvalue_spectrum.png", dpi=150)
#clean plt - dont show the plot
plt.close()

# --- visualize random (s1, sm) pairs with actual vs. predicted distances ---
import math

def _imshow_tensor(ax, t):
    """
    Show an image tensor t on a given matplotlib axis.
    Supports HxWxC, CxHxW, HxW, and handles 1- or 3-channel visuals.
    """
    x = t.detach().cpu()
    if x.dim() == 3:
        # Either CxHxW or HxWxC
        if x.shape[0] in (1, 3):  # CxHxW
            if x.shape[0] == 1:
                ax.imshow(x[0], cmap="gray")
            else:
                ax.imshow(x.permute(1, 2, 0))
        else:  # HxWxC
            if x.shape[-1] == 1:
                ax.imshow(x[..., 0], cmap="gray")
            else:
                ax.imshow(x)
    elif x.dim() == 2:  # HxW
        ax.imshow(x, cmap="gray")
    else:
        raise ValueError(f"Unsupported image shape: {tuple(x.shape)}")
    ax.axis("off")

def plot_sample_pairs_with_distances(emb, dist, datamodule, num_samples=6, batch_size=64):
    """
    Draw 'num_samples' random pairs (s1, sm) from the dataloader and annotate:
      - actual distance = d1 + d2
      - predicted distance = dist(emb(s1), emb(sm))
    """
    emb.eval()
    dist.eval()

    loader = torch.utils.data.DataLoader(datamodule, batch_size=batch_size, shuffle=True, drop_last=False)
    with torch.no_grad():
        batch = next(iter(loader))

        s1 = batch["s1"].to("cuda", non_blocking=True)
        sm = batch["sm"].to("cuda", non_blocking=True)
        d1 = batch["d1"].to("cuda", non_blocking=True).float()
        d2 = batch["d2"].to("cuda", non_blocking=True).float()

        e1 = emb(s1)
        em = emb(sm)
        pd_tot_l,pd_tot_u = dist(e1, em)  # predicted distance
        actual = (d1 + d2)     # actual distance proxy from dataset

        # choose a subset of indices to display
        n = min(num_samples, s1.shape[0])
        idx = torch.randperm(s1.shape[0], device=s1.device)[:n].cpu().tolist()

        fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 3 * n))
        if n == 1:
            axes = [axes]  # normalize shape to list of [left,right]

        for row, i in enumerate(idx):
            ax_left, ax_right = axes[row]

            _imshow_tensor(ax_left, s1[i][0])
            _imshow_tensor(ax_right, sm[i][0])

            a = float(actual[i].item())
            p = float(pd_tot_l[i].item())
            ax_left.set_title(f"Pair {row+1} — actual: {a:.2f}, predicted: {p:.2f}", fontsize=10)
            ax_right.set_title("target state (sm)", fontsize=10)
            ax_left.set_xlabel("source state (s1)")

        plt.tight_layout()
        plt.savefig("sample_state_pairs_with_distances.png", dpi=150)
        plt.close(fig)

# Call after the eigenvalue spectrum plot:
plot_sample_pairs_with_distances(emb, dist, datamodule, num_samples=6, batch_size=64)
