import torch
import matplotlib.pyplot as plt
from dataset import SequenceDataset
from agent import Embedder, DistanceFunction
from simpleRL import make_atari_env
import os

# ----------------- config -----------------
MODEL_PATH = "model.pth"
OUT_DIR = "sample_pairs"
NUM_SAMPLES = 10   # how many pairs to save
BATCH_SIZE = 64
# ------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# --- load model and config ---
ckpt = torch.load(MODEL_PATH, map_location="cuda", weights_only=False)
cfg = ckpt["config"]

env_id = cfg["env_id"]
embedding_dim = cfg["embedding_dim"]

env = make_atari_env(env_id, seed=42)()

# dummy run to get some trajectories for the dataset
from simpleRL import run_to_completion
def random_policy(obs_batch):
    B = obs_batch.shape[0]
    return torch.randint(0, env.action_space.n, (B,), device=obs_batch.device)

S, A, R, TR = run_to_completion(env_fn=lambda: env, policy=random_policy, n_runs=20, device="cpu")
datamodule = SequenceDataset(S, A, R, device="cuda")

# --- rebuild models ---
emb = Embedder(cfg["obs_shape"], embedding_dim).to("cuda")
dist = DistanceFunction(embedding_dim).to("cuda")

emb.load_state_dict(ckpt["embedder_state_dict"])
dist.load_state_dict(ckpt["distance_state_dict"])

emb.eval()
dist.eval()

# --- helper to plot tensors ---
def _imshow_tensor(ax, t):
    """
    Show an image tensor on ax.
    Supported:
      - (H, W) grayscale
      - (H, W, C) with C in {1,3,4}
      - (C, H, W) with C in {1,3,4}; if C==4 (Atari stack), show last frame as grayscale
    """
    x = t.detach().cpu()
    if x.dtype == torch.float32 or x.dtype == torch.float64:
        # Try to scale to [0,1] if it looks like 0..255
        if x.max() > 1.0:
            x = x / 255.0
        x = x.clamp(0, 1)

    if x.dim() == 2:
        img = x.numpy()
        ax.imshow(img, cmap="gray")
    elif x.dim() == 3:
        # HWC
        if x.shape[-1] in (1, 3, 4) and x.shape[0] != 1 and x.shape[0] != 3 and x.shape[0] != 4:
            C = x.shape[-1]
            if C == 1:
                ax.imshow(x[..., 0].numpy(), cmap="gray")
            else:
                ax.imshow(x.numpy())
        # CHW
        elif x.shape[0] in (1, 3, 4):
            C = x.shape[0]
            if C == 1:
                ax.imshow(x[0].numpy(), cmap="gray")
            elif C == 3:
                ax.imshow(x.permute(1, 2, 0).numpy())
            elif C == 4:
                # Atari stacked frames: show the most recent (last) frame as grayscale
                ax.imshow(x[-1].numpy(), cmap="gray")
        else:
            raise TypeError(f"Unsupported 3D image shape {tuple(x.shape)}")
    else:
        raise TypeError(f"Unsupported image shape {tuple(x.shape)}")

    ax.axis("off")

# --- main plotting ---
loader = torch.utils.data.DataLoader(datamodule, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
batch = next(iter(loader))

s1 = batch["s1"].to("cuda", non_blocking=True)
sm = batch["sm"].to("cuda", non_blocking=True)
d1 = batch["d1"].to("cuda", non_blocking=True).float()
d2 = batch["d2"].to("cuda", non_blocking=True).float()

with torch.no_grad():
    e1 = emb(s1)
    em = emb(sm)
    pd_tot = dist(e1, em).squeeze(-1)
    actual = (d1 + d2).squeeze(-1)

indices = torch.randperm(s1.shape[0], device=s1.device)[:NUM_SAMPLES].cpu().tolist()

for k, i in enumerate(indices):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    _imshow_tensor(axes[0], s1[i])
    _imshow_tensor(axes[1], sm[i])

    a = float(actual[i].item())
    p = float(pd_tot[i].item())
    axes[0].set_title(f"Pair {k+1}\nactual: {a:.2f}, pred: {p:.2f}")
    axes[1].set_title("target (sm)")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"pair_{k+1:02d}.png")
    plt.savefig(out_path)
    plt.close(fig)

print(f"Saved {NUM_SAMPLES} sample pairs to {OUT_DIR}/")
