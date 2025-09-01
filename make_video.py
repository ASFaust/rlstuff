# save as: viz_from_checkpoint.py
# Usage: python viz_from_checkpoint.py
# - Loads "model.pth" (written by your training script)
# - Recreates env and models, computes the action-dispersion metric along one episode
# - Writes a side-by-side video (800x450) with the newest Atari frame (left) and a static plot with a moving red dot (right)

import os
import cv2
import torch
import numpy as np

from simpleRL import run_to_completion, make_atari_env
from agent import Embedder, Evaluator, TransitionModel, SimpleTransitionModel


def compute_action_dispersion(emb, transition_model, seq_states, n_actions, device):
    """
    For each timestep t (except the last), compute:
      D_t = max_{a,a'} || f(e_t, a) - f(e_t, a') ||
    where e_t = emb(s_t), f is the transition model.
    Returns: np.ndarray of shape [T-1]
    """
    emb.eval()
    transition_model.eval()

    with torch.no_grad():
        # [T, C, H, W] -> embeddings [T, D]
        seq_tensor = torch.as_tensor(seq_states, dtype=torch.float32, device=device)
        seq_emb = emb(seq_tensor).cpu().numpy()

        T = len(seq_states)
        out = []
        for t in range(T - 1):
            state_emb = seq_emb[t]  # [D]
            state_emb_tensor = torch.as_tensor(state_emb, dtype=torch.float32, device=device).unsqueeze(0)  # [1, D]

            all_next_embs = []
            for a in range(n_actions):
                action_tensor = torch.as_tensor([[a]], dtype=torch.long, device=device)  # [1, 1]
                next_emb = transition_model(state_emb_tensor, action_tensor)  # [1, D]
                all_next_embs.append(next_emb.cpu().numpy()[0])
            all_next_embs = np.asarray(all_next_embs)  # [n_actions, D]

            # pairwise L2 distances among predicted next embeddings
            #diffs = all_next_embs[:, None, :] - all_next_embs[None, :, :]
            #dists = np.linalg.norm(diffs, axis=-1)
            #out.append(dists.mean())
            #instead, compute the variance of the next embeddings
            mean_emb = np.mean(all_next_embs, axis=0, keepdims=True)
            var = np.mean(np.sum((all_next_embs - mean_emb) ** 2, axis=-1))
            out.append(var)

    return np.asarray(out, dtype=np.float32)


def render_left_panel(state_4x84x84, H, LEFT_W):
    """
    Convert newest frame from stacked observation (4x84x84) into (H, LEFT_W, 3) BGR, letterboxed.
    """
    newest = state_4x84x84[-1]  # (84,84)
    if newest.dtype != np.uint8:
        arr = newest.astype(np.float32)
        arr = arr - arr.min()
        denom = arr.max() if arr.max() > 0 else 1.0
        newest_u8 = np.clip((arr / denom) * 255.0, 0, 255).astype(np.uint8)
    else:
        newest_u8 = newest

    frame_bgr = cv2.cvtColor(newest_u8, cv2.COLOR_GRAY2BGR)

    canvas = np.zeros((H, LEFT_W, 3), dtype=np.uint8)
    src_h, src_w = frame_bgr.shape[:2]  # 84x84
    scale = min(LEFT_W / src_w, H / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    top = (H - new_h) // 2
    left = (LEFT_W - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def build_plot_background(metric, H, RIGHT_W):
    """
    Build a static plot image (BGR) of size (H, RIGHT_W, 3) with axes and the full curve in light gray.
    Returns: (plot_bg, pts_np, margin, plot_h)
      - plot_bg: the static background image
      - pts_np:  Nx1x2 int polyline points for the curve
      - margin, plot_h: for drawing the moving indicator
    """
    plot_bg = np.zeros((H, RIGHT_W, 3), dtype=np.uint8)
    margin = 20
    plot_w = RIGHT_W - 2 * margin
    plot_h = H - 2 * margin

    if len(metric) == 0:
        cv2.rectangle(plot_bg, (margin - 1, margin - 1), (margin + plot_w, margin + plot_h), (80, 80, 80), 1)
        return plot_bg, np.empty((0, 1, 2), dtype=np.int32), margin, plot_h

    y_min = float(metric.min())
    y_max = float(metric.max())
    y_range = (y_max - y_min) if (y_max > y_min) else 1.0
    ys = (metric - y_min) / y_range

    N = len(metric)
    xs = np.linspace(0, plot_w - 1, N, dtype=np.float32) if N > 1 else np.array([0], dtype=np.float32)

    pts = []
    for i in range(N):
        x = int(margin + xs[i])
        y = int(margin + (plot_h - 1) - ys[i] * (plot_h - 1))
        pts.append([x, y])
    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

    # axes
    cv2.rectangle(plot_bg, (margin - 1, margin - 1), (margin + plot_w, margin + plot_h), (80, 80, 80), 1)
    # full curve
    if len(pts_np) > 1:
        cv2.polylines(plot_bg, [pts_np], isClosed=False, color=(180, 180, 180), thickness=1)

    # optional labels
    cv2.putText(plot_bg, "Action-dispersion", (margin, margin - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    return plot_bg, pts_np, margin, plot_h


def main():
    # --- Load checkpoint ---
    ckpt_path = "model.pth"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # in make_video.py -> main(), replace the load line with:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    env_id = cfg["env_id"]
    embedding_dim = int(cfg["embedding_dim"])
    n_actions_cfg = int(cfg["n_actions"])
    obs_shape = tuple(cfg["obs_shape"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Recreate env & models ---
    env = make_atari_env(env_id, seed=np.random.randint(0, 10000))()
    # Use saved n_actions if present; assert matches runtime env
    n_actions = env.action_space.n
    if n_actions != n_actions_cfg:
        # you asked for no extra suggestions; we just assert to avoid silent mismatch
        raise ValueError(f"n_actions mismatch: checkpoint={n_actions_cfg} env={n_actions}")

    emb = Embedder(obs_shape, embedding_dim).to(device)
    eval_m = Evaluator(embedding_dim).to(device)  # not used in viz; loaded for completeness
    transition_model = TransitionModel(n_actions, embedding_dim).to(device)
    simple_tm = SimpleTransitionModel(embedding_dim).to(device)  # not used in viz; loaded for completeness

    emb.load_state_dict(ckpt["embedder_state_dict"])
    eval_m.load_state_dict(ckpt["evaluator_state_dict"])
    transition_model.load_state_dict(ckpt["transition_model_state_dict"])
    simple_tm.load_state_dict(ckpt["simple_tm_state_dict"])

    emb.eval()
    eval_m.eval()
    transition_model.eval()
    simple_tm.eval()

    # --- Generate a fresh sequence to visualize (single run) ---
    def random_policy(obs_batch):
        B = obs_batch.shape[0]
        return torch.randint(0, n_actions, (B,), device=obs_batch.device)

    S, A, R, TR = run_to_completion(env_fn=lambda: env, policy=random_policy, n_runs=1, device="cpu")
    if len(S) == 0 or len(S[0]) < 2:
        raise RuntimeError("No valid sequence collected for visualization.")

    seq = S[0]  # [T, 4, 84, 84]
    T = len(seq)

    # --- Compute metric along the sequence ---
    metric = compute_action_dispersion(emb, transition_model, seq, n_actions, device)  # shape [T-1]
    T_vis = len(metric)  # frames to render

    # --- Video writer setup ---
    W, H = 800, 450       # 16:9
    LEFT_W = W // 2
    RIGHT_W = W - LEFT_W
    FPS = 15
    out_path = f"viz_from_{env_id.replace('/', '_')}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    # --- Static plot background ---
    plot_bg, pts_np, margin, plot_h = build_plot_background(metric, H, RIGHT_W)

    # --- Render frames ---
    for t in range(T_vis):
        left_img = render_left_panel(seq[t], H, LEFT_W)

        # right: copy static plot, then draw locator & red dot
        right_img = plot_bg.copy()
        cx, cy = int(pts_np[t][0][0]), int(pts_np[t][0][1]) if len(pts_np) > 0 else (margin, margin + plot_h // 2)

        cv2.line(right_img, (cx, margin), (cx, margin + plot_h - 1), (60, 60, 200), 1)  # vertical locator line
        cv2.circle(right_img, (cx, cy), 4, (0, 0, 255), thickness=-1)                    # current red dot

        frame = cv2.hconcat([left_img, right_img])  # (H, W, 3)
        writer.write(frame)

    writer.release()
    print(f"Wrote {out_path} ({W}x{H} @ {FPS} fps) using checkpoint '{ckpt_path}'")


if __name__ == "__main__":
    main()
