# goal_seek.py
#
# Greedy goal-seeking in ALE Atari using learned embeddings/distances and a learned transition model.
# Now with video capture via OpenCV's VideoWriter:
#   - Left: current observation (upscaled)
#   - Right: goal observation (upscaled)
#   - HUD overlay with t, action, distances, reward, total reward
#   - Bottom bar: recent distance history (curr->goal)
#
# Usage:
#   python3 goal_seek.py --model model.pth --env ALE/MontezumaRevenge-v5 --video out.mp4 --fps 30 --scale 5
#
# Notes:
# - Expects your 'agent.py' to provide: Embedder, DistanceFunction, TransitionModel
# - Expects your 'simpleRL.py' to provide: make_atari_env
# - Works with stacked frames (4,84,84) or HWC; converts to a 4D batch tensor for Embedder.
# - OpenCV expects BGR uint8 frames shaped (H, W, 3). We convert Atari frames accordingly.

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from agent import Embedder, DistanceFunction, TransitionModel
from simpleRL import make_atari_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="model.pth")
    p.add_argument("--seed", type=int, default=234753)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--warmup", type=int, default=1, help="random steps before snapshot (s1)")
    p.add_argument("--rand-steps", type=int, default=200, help="random steps after snapshot to define goal")
    p.add_argument("--max-steps", type=int, default=300, help="max greedy steps when pursuing goal")
    p.add_argument("--threshold", type=float, default=0.01, help="stop when dist(curr, goal) <= threshold")
    p.add_argument("--no-normalize", action="store_true", help="(kept for compatibility; Embedder normalizes internally)")
    # ---- video options ----
    p.add_argument("--video", type=str, default="", help="output video path (e.g., out.mp4). If empty, no video is saved.")
    p.add_argument("--fps", type=int, default=30, help="video frames per second")
    p.add_argument("--scale", type=int, default=5, help="upscale factor for Atari frames before tiling")
    p.add_argument("--codec", type=str, default="mp4v", help="fourcc codec, e.g. mp4v, avc1, XVID")
    return p.parse_args()


def torch_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def to_bgr_uint8(img_like):
    """
    Convert observation to 3-channel BGR uint8.
    Handles (H,W), (H,W,1), (H,W,3) RGB, or (C,H,W) stacked (uses last channel as gray).
    """
    arr = np.array(img_like, copy=False)
    if arr.ndim == 2:
        g = arr.astype(np.uint8)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):  # CHW
            c, h, w = arr.shape
            if c == 1:
                g = arr[0].astype(np.uint8)
                return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            # stacked grayscale -> take last channel as gray
            g = arr[-1].astype(np.uint8)
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        else:
            # HWC
            h, w, c = arr.shape
            if c == 1:
                g = arr[:, :, 0].astype(np.uint8)
                return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            if c == 3:
                rgb = arr.astype(np.uint8)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Unsupported obs format for BGR conversion: shape={arr.shape}, dtype={arr.dtype}")

def upscale(img_bgr, scale=5, interpolation=cv2.INTER_NEAREST):
    return cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=interpolation)

def tile_side_by_side(left_bgr, right_bgr, gap_px=8, bg_color=(0,0,0)):
    """
    Side-by-side panel with a gap between images and no overlays.
    Returns a panel whose height is the max of inputs, padded with bg_color.
    """
    hL, wL = left_bgr.shape[:2]
    hR, wR = right_bgr.shape[:2]
    H = max(hL, hR)

    if hL < H:
        pad = np.full((H - hL, wL, 3), bg_color, dtype=np.uint8)
        left_bgr = np.vstack([left_bgr, pad])
    if hR < H:
        pad = np.full((H - hR, wR, 3), bg_color, dtype=np.uint8)
        right_bgr = np.vstack([right_bgr, pad])

    gap = np.full((H, gap_px, 3), bg_color, dtype=np.uint8)
    return np.hstack([left_bgr, gap, right_bgr])

def render_hud_band(
    width,
    height,
    lines,
    dist_history=None,
    reach_history=None,          # NEW: list/array of reachability probabilities in [0,1]
    margins=8,
    bg=(24, 24, 24),
    fg=(230, 230, 230),
):
    """
    Render a separate HUD band (H x W) with:
      - Text lines on the left
      - Optional distance history plot (top-right)
      - Optional reachability history plot (bottom-right, in [0,1])
    Returns an image of shape (height, width, 3), uint8.
    """
    hud = np.full((height, width, 3), bg, dtype=np.uint8)

    # --- Text block (left) ---
    x = margins
    y = margins + 18
    for line in lines:
        cv2.putText(hud, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2, cv2.LINE_AA)
        y += 22

    # --- Right-side plots area (distance + reachability) ---
    has_dist = dist_history is not None and len(dist_history) > 0
    has_reach = reach_history is not None and len(reach_history) > 0

    if has_dist or has_reach:
        # Reserve right half for plots
        plot_w = max(160, width // 2)
        plot_h = height - 2 * margins
        x0 = width - plot_w - margins
        y0 = margins
        cv2.rectangle(hud, (x0, y0), (x0 + plot_w, y0 + plot_h), (60, 60, 60), -1)

        # Decide layout: if both -> two stacked panels; if one -> single panel full height
        if has_dist and has_reach:
            h_top = (plot_h - 6) // 2
            h_bot = plot_h - 6 - h_top
            # Distance (top)
            _plot_series_polyline(
                hud,
                x0 + 3,
                y0 + 3,
                plot_w - 6,
                h_top,
                series=dist_history,
                label="dist",
                label_fmt=lambda vmin, vmax: f"dist [{vmin:.2f}, {vmax:.2f}]",
                y_range=None,            # autoscale
                bg_col=(70, 70, 70),
                frame_col=(90, 90, 90),
                line_col=(210, 210, 210),
            )
            # Reachability (bottom), fixed [0,1]
            _plot_series_polyline(
                hud,
                x0 + 3,
                y0 + 3 + h_top + 6,
                plot_w - 6,
                h_bot,
                series=reach_history,
                label="reach",
                label_fmt=lambda vmin, vmax: f"reach [{max(0,vmin):.2f}, {min(1,vmax):.2f}]",
                y_range=(0.0, 1.0),      # fixed scale
                bg_col=(70, 70, 70),
                frame_col=(90, 90, 90),
                line_col=(210, 210, 210),
            )
        elif has_dist:
            _plot_series_polyline(
                hud,
                x0 + 3,
                y0 + 3,
                plot_w - 6,
                plot_h - 6,
                series=dist_history,
                label="dist",
                label_fmt=lambda vmin, vmax: f"dist [{vmin:.2f}, {vmax:.2f}]",
                y_range=None,
                bg_col=(70, 70, 70),
                frame_col=(90, 90, 90),
                line_col=(210, 210, 210),
            )
        else:  # only reachability
            _plot_series_polyline(
                hud,
                x0 + 3,
                y0 + 3,
                plot_w - 6,
                plot_h - 6,
                series=reach_history,
                label="reach",
                label_fmt=lambda vmin, vmax: f"reach [{max(0,vmin):.2f}, {min(1,vmax):.2f}]",
                y_range=(0.0, 1.0),
                bg_col=(70, 70, 70),
                frame_col=(90, 90, 90),
                line_col=(210, 210, 210),
            )

    return hud


def _plot_series_polyline(
    canvas,
    x,
    y,
    w,
    h,
    series,
    label="",
    label_fmt=None,
    y_range=None,                 # None -> autoscale, else (ymin,ymax)
    bg_col=(70, 70, 70),
    frame_col=(90, 90, 90),
    line_col=(210, 210, 210),
    draw_y_ticks=None,           # e.g., [0.0, 0.5, 1.0] for reachability
):
    """
    Draw a time series polyline into a rectangle on 'canvas' at (x,y,w,h).
    - series: list/array of floats
    - y_range: None for autoscale, or (ymin,ymax) for fixed scaling
    - draw_y_ticks: optional list of y tick values to annotate on right edge
    """
    # background + frame
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_col, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), frame_col, 1)

    # pick slice length ~ one pixel per sample
    max_pts = max(2, w - 20)
    data = np.asarray(series[-max_pts:], dtype=np.float32)
    if data.size < 2:
        return

    if y_range is None:
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.min(data)), float(np.max(data) + 1.0)
    else:
        vmin, vmax = y_range
        if vmax <= vmin:
            vmax = vmin + 1.0

    # map to pixel coords
    inner = 8
    x0, y0 = x + inner, y + inner
    ww = max(2, w - 2 * inner)
    hh = max(2, h - 2 * inner)
    xs = np.linspace(x0, x0 + ww - 1, num=data.size, dtype=np.int32)
    # higher value -> higher on plot; invert for image coordinates
    frac = (data - vmin) / (vmax - vmin + 1e-12)
    ys = (y0 + hh - 1 - (frac * (hh - 1))).astype(np.int32)

    pts = np.stack([xs, ys], axis=1)
    cv2.polylines(canvas, [pts], isClosed=False, color=line_col, thickness=2, lineType=cv2.LINE_AA)

    # label
    if label_fmt is None:
        label_text = f"{label}"
    else:
        label_text = label_fmt(vmin, vmax)
    cv2.putText(canvas, label_text, (x0, y0 - 4 if y0 - 4 > y else y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # optional y ticks
    if draw_y_ticks:
        for tv in draw_y_ticks:
            # clamp to [vmin, vmax]
            tvc = min(max(tv, vmin), vmax)
            ty = int(y0 + hh - 1 - ( (tvc - vmin) / (vmax - vmin + 1e-12) * (hh - 1) ))
            cv2.line(canvas, (x0 + ww + 2, ty), (x0 + ww + 6, ty), (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{tv:.1f}", (x0 + ww + 8, ty + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)


def compose_frame(curr_obs, goal_obs, dist_hist, reach_hist, lines, scale=5, hud_h=400):
    """
    Build the final frame with:
      [top]  current || goal (no overlays)
      [bottom] HUD band (text + plot)
    """
    left = upscale(to_bgr_uint8(curr_obs), scale=scale)
    right = upscale(to_bgr_uint8(goal_obs), scale=scale)
    panel = tile_side_by_side(left, right, gap_px=12, bg_color=(0,0,0))

    H, W = panel.shape[:2]
    hud = render_hud_band(width=W, height=hud_h, lines=lines, dist_history=dist_hist, reach_history=reach_hist, margins=10)

    # stack vertically
    frame = np.vstack([panel, hud])
    return frame



def obs_to_tensor(obs, device):
    """
    Convert env observation to [1, C, H, W] on device.
    No normalization here; Embedder does state.float()/255.0 internally.
    """
    x = torch.from_numpy(np.array(obs, copy=False))
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)                 # (1,1,H,W)
    elif x.ndim == 3:
        # (H,W,C) -> (1,C,H,W) or already (C,H,W) -> (1,C,H,W)
        if x.shape[0] in (1, 3, 4):                     # CHW
            x = x.unsqueeze(0)
        else:                                           # HWC
            x = x.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported observation shape: {tuple(x.shape)}")
    return x.to(device, non_blocking=True)              # keep uint8; Embedder normalizes

def upscale(img_bgr, scale=5, interpolation=cv2.INTER_NEAREST):
    return cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=interpolation)


def clone_state(env):
    if hasattr(env, "clone_state"):
        return env.clone_state()
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "clone_state"):
        return env.unwrapped.clone_state()
    raise AttributeError("Environment does not support clone_state().")


def restore_state(env, state_bytes):
    if hasattr(env, "restore_state"):
        return env.restore_state(state_bytes)
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "restore_state"):
        return env.unwrapped.restore_state(state_bytes)
    raise AttributeError("Environment does not support restore_state().")


def main():
    args = parse_args()
    device = torch_device(args.device)

    # -------- load trained models --------
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    embedding_dim = cfg["embedding_dim"]
    n_actions = cfg["n_actions"]
    obs_shape = cfg["obs_shape"]

    emb = Embedder(obs_shape, embedding_dim).to(device)
    dist_fn = DistanceFunction(embedding_dim).to(device)
    trans = TransitionModel(n_actions, embedding_dim).to(device)

    emb.load_state_dict(ckpt["embedder_state_dict"])
    dist_fn.load_state_dict(ckpt["distance_state_dict"])
    trans.load_state_dict(ckpt["transition_model_state_dict"])

    for m in (emb, dist_fn, trans):
        m.eval()



    # -------- env --------
    env_fn = make_atari_env(cfg["env_id"], seed=args.seed)
    env = env_fn()
    obs, info = env.reset(seed=args.seed)

    # -------- random warmup to s1 --------
    step = 0
    for _ in range(args.warmup):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        step += 1
        if terminated or truncated:
            obs, info = env.reset()

    # Save snapshot at s1
    snapshot = clone_state(env)
    s1_obs = obs  # keep for reference

    # -------- continue randomly to define goal --------
    last_obs = None
    for _ in range(args.rand_steps):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        last_obs = obs
        step += 1
        if terminated or truncated:
            break

    if last_obs is None:
        last_obs = obs  # fallback

    # Goal embedding (for control) and goal image (for video)
    with torch.no_grad():
        goal_tensor = obs_to_tensor(last_obs, device)
        e_goal = emb(goal_tensor)

    goal_img_raw = last_obs

    # -------- restore to s1 and pursue goal greedily --------
    restore_state(env, snapshot)
    obs, info = env.reset()  # some wrappers require reset() to sync; if this resets too far, comment this line

    # If reset() restarted the episode, re-apply snapshot
    try:
        restore_state(env, snapshot)
    except Exception:
        pass

    # Initialize from current obs after restore
    obs, _, _, _, _ = env.step(0) if hasattr(env.action_space, "sample") else (s1_obs, 0, False, False, {})

    total_reward = 0.0
    reached = False

    # -------- video writer init --------
    writer = None
    dist_hist = []
    reachability_hist = []

    if args.video:
        sample = compose_frame(
            curr_obs=obs,
            goal_obs=last_obs,  # your computed goal image
            dist_hist=[],
            reach_hist=[],  # empty at start
            lines=["Initializing..."],
            scale=args.scale
        )
        H, W = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        writer = cv2.VideoWriter(args.video, fourcc, float(args.fps), (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {args.video} with codec {args.codec}")

    # Precompute a visualized goal panel (right side) once for efficiency
    goal_bgr_up = upscale(to_bgr_uint8(goal_img_raw), scale=args.scale, interpolation=cv2.INTER_NEAREST)

    for t in range(args.max_steps):
        with torch.no_grad():
            curr = obs_to_tensor(obs, device)
            e_curr = emb(curr)  # [1, D]

            # 1) First-step predictions for all a1
            A1 = torch.arange(n_actions, device=device, dtype=torch.long)
            e_curr_rep = e_curr.repeat(n_actions, 1)  # [A, D]
            e_after_a1 = trans(e_curr_rep, A1)  # [A, D]

            # 2) Second-step predictions for all (a1, a2) pairs
            A2 = torch.arange(n_actions, device=device, dtype=torch.long)
            e_after_a1_rep = e_after_a1.unsqueeze(1).repeat(1, n_actions, 1)  # [A, A, D]
            e_after_a1a2_in = e_after_a1_rep.reshape(n_actions * n_actions, -1)  # [A*A, D]
            A2_tiled = A2.repeat(n_actions)  # [A*A]

            e_after_a1a2 = trans(e_after_a1a2_in, A2_tiled)  # [A*A, D]

            # 3) Distances + reachability for all pairs
            e_goal_rep_pairs = e_goal.expand(e_after_a1a2.size(0), -1)  # [A*A, D]
            d_pairs, reach_logits = dist_fn(e_after_a1a2, e_goal_rep_pairs)  # each [A*A, 1]
            d_pairs = d_pairs.squeeze(-1)  # [A*A]
            reach_probs = torch.sigmoid(reach_logits.squeeze(-1))  # [A*A]

            # Reshape back to [A1, A2]
            d_after_a1a2 = d_pairs.view(n_actions, n_actions)  # [A, A] raw distances
            reach_after_a1a2 = reach_probs.view(n_actions, n_actions)  # [A, A]

            # --- New: threshold by a global reachability baseline (no division) ---
            baseline_reach = reach_after_a1a2.max()  # scalar
            reach_thresh = 0.9 * baseline_reach

            # Best achievable distance after two steps if we start with a1,
            # but only over (a1,a2) pairs whose reachability >= 90% of baseline.
            d_two_step_best = torch.empty(n_actions, device=device)
            best_a2_per_a1 = torch.empty(n_actions, dtype=torch.long, device=device)

            for a1 in range(n_actions):
                row_reach = reach_after_a1a2[a1]  # [A]
                row_dist = d_after_a1a2[a1]  # [A]
                mask = row_reach >= reach_thresh  # [A] bool

                if mask.any():
                    # Restrict to sufficiently reachable a2
                    filtered_dist = row_dist[mask]
                    min_idx_local = torch.argmin(filtered_dist)
                    d_two_step_best[a1] = filtered_dist[min_idx_local]
                    # Map back to original a2 index
                    best_a2_per_a1[a1] = mask.nonzero(as_tuple=False).view(-1)[min_idx_local]
                else:
                    # Fallback: if nothing passes the threshold, use the best raw distance in this row
                    best_a2_per_a1[a1] = torch.argmin(row_dist)
                    d_two_step_best[a1] = row_dist[best_a2_per_a1[a1]]

            # 5) Pick a1 with 10% near-optimal tie-break
            d_min = float(d_two_step_best.min().item())
            near_min = (d_two_step_best <= d_min * 1.01).nonzero(as_tuple=False).view(-1)
            a1_star = int(near_min[torch.randint(len(near_min), (1,)).item()])

            # For logging: the corresponding best a2 given chosen a1
            a2_star = int(best_a2_per_a1[a1_star].item())
            pred_min = float(d_two_step_best[a1_star].item())  # two-step best under reachability threshold

            # Current raw distance to goal (for HUD/plotting)
            d_curr_val, reachability_logit_curr_val = dist_fn(e_curr, e_goal)
            d_curr_val = float(d_curr_val.item())
            reachability_curr_val = float(torch.sigmoid(reachability_logit_curr_val).item())
            reachability_hist.append(reachability_curr_val)
            dist_hist.append(d_curr_val)

            # Optional logging
            avg_reach_prob = float(reach_after_a1a2[a1_star].mean().item())

        # Compose and write video frame BEFORE stepping (so frame reflects 'curr' state)

        # 2) Each step BEFORE env.step(a_star), write a frame without drawing over the game panels:
        if writer is not None:
            hud_lines = [
                f"t={t:04d}   a*={a1_star:2d}",
                f"dist(curr,goal)={d_curr_val:.3f}",
                f"pred_dist(next,goal)={pred_min:.3f}",
                f"total_reward={total_reward:.2f}",
            ]
            frame = compose_frame(
                curr_obs=obs,
                goal_obs=goal_img_raw,  # cached goal image
                dist_hist=dist_hist,
                reach_hist=reachability_hist,
                lines=hud_lines,
                scale=args.scale
            )
            writer.write(frame)

        if d_curr_val <= args.threshold:
            print(f"[t={t}] Reached goal proximity: dist={d_curr_val:.3f} <= {args.threshold:.3f}")
            reached = True
            break

        # Step env using the chosen action
        obs, r, terminated, truncated, info = env.step(a1_star)
        total_reward += float(r)

        print(f"[t={t}] a*={a1_star:2d} | dist(curr,goal)={d_curr_val:.3f} "
              f"| pred_dist(next,goal)={pred_min:.3f} | reward={r}")

        if terminated or truncated:
            print(f"Episode ended (terminated={terminated}, truncated={truncated}) at t={t}.")
            break

    # Final frame after loop (optional)
    if writer is not None:
        final_lines = [
            f"FINAL reached={reached} steps={t if reached else t + 1}",
            f"dist(curr,goal)={dist_hist[-1]:.3f}" if dist_hist else "dist(curr,goal)=n/a",
            f"total_reward={total_reward:.2f}",
        ]
        frame = compose_frame(
            curr_obs=obs,
            goal_obs=goal_img_raw,
            dist_hist=dist_hist,
            reach_hist=reachability_hist,
            lines=final_lines,
            scale=args.scale
        )
        writer.write(frame)
        writer.release()

    print(f"\nDone. Reached={reached}, steps_taken={t if reached else t+1}, total_reward={total_reward:.2f}")


if __name__ == "__main__":
    main()
