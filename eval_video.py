import numpy as np
import torch

from minatar import Environment

_CACHED_CMAP = {}


def _flatten_obs(obs):
    # (H, W, C) -> (C, H, W), float32
    obs = np.transpose(obs, (2, 0, 1)).astype(np.float32)
    return obs


def _minatar_state_to_rgb(state, cmap_name="cubehelix"):
    """Convert MinAtar boolean state (H, W, C) to RGB uint8 image."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import cm

    h, w, c = state.shape
    # match MinAtar display: highest channel index wins
    channel_ids = np.arange(c, dtype=np.int32) + 1
    numerical_state = np.amax(state * channel_ids.reshape(1, 1, -1), axis=2)

    cache_key = (cmap_name, c + 1)
    colors = _CACHED_CMAP.get(cache_key)
    if colors is None:
        cmap = cm.get_cmap(cmap_name, c + 1)
        colors = (cmap(np.arange(c + 1))[:, :3] * 255).astype(np.uint8)
        _CACHED_CMAP[cache_key] = colors
    rgb = colors[numerical_state]
    return rgb


def _scale_image(img, scale):
    if scale == 1:
        return img
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def _render_value_plot(values, width, height, line_color="black"):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float32)
    n = len(values)

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(n), values, color=line_color, linewidth=2)

    if n == 1:
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.set_xlim(0, n - 1)

    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        pad = 1.0
    else:
        pad = 0.05 * (vmax - vmin)
    ax.set_ylim(vmin - pad, vmax + pad)

    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.canvas.draw()

    base = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    base = base.reshape(height, width, 3)

    # Precompute pixel coordinates for red line and axis bounds.
    trans = ax.transData
    bbox = ax.get_window_extent()
    x_pixels = []
    y_for_x = vmin
    for i in range(n):
        x_pix = int(round(trans.transform((i, y_for_x))[0]))
        x_pixels.append(x_pix)

    x_pixels = np.array(x_pixels, dtype=np.int32)
    x_pixels = np.clip(x_pixels, 0, width - 1)

    # Bbox is in display coords (origin bottom-left); convert to image coords (origin top-left).
    y0 = int(round(bbox.y0))
    y1 = int(round(bbox.y1))
    img_y0 = max(0, height - y1)
    img_y1 = min(height, height - y0)

    plt.close(fig)

    return base, x_pixels, img_y0, img_y1


def _value_for_obs(agent, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        values, _ = agent._values_and_regrets(obs_t)
    return float(values.item())


def make_eval_video_frames(agent, game, seed=0, max_steps=2000, fps=4, scale=12):
    """
    Returns frames in (T, C, H, W) uint8 format, suitable for wandb.Video.
    """
    env = Environment(game)
    env.seed(seed)

    obs = _flatten_obs(env.state())
    frames = []
    values = []

    done = False
    steps = 0

    while not done and steps < max_steps:
        state = env.state()
        rgb = _minatar_state_to_rgb(state)
        rgb = _scale_image(rgb, scale)

        value = _value_for_obs(agent, obs)
        frames.append(rgb)
        values.append(value)

        action = agent.act(obs, greedy=True)
        _, done = env.act(action)
        obs = _flatten_obs(env.state())
        steps += 1

    if len(frames) == 0:
        return np.zeros((0, 3, 1, 1), dtype=np.uint8)

    frame_h, frame_w, _ = frames[0].shape
    graph_h = frame_h
    graph_w = frame_h * 2

    base_graph, x_pixels, y0, y1 = _render_value_plot(values, graph_w, graph_h)

    out_frames = []
    for i, frame in enumerate(frames):
        graph = base_graph.copy()
        x = x_pixels[i]
        # draw a 2px red line
        x0 = max(0, x - 1)
        x1 = min(graph_w, x + 1)
        graph[y0:y1, x0:x1] = np.array([255, 0, 0], dtype=np.uint8)

        combined = np.zeros((graph_h, frame_w + graph_w, 3), dtype=np.uint8)
        combined[:, :frame_w] = frame
        combined[:, frame_w:] = graph
        out_frames.append(combined)

    frames_np = np.stack(out_frames, axis=0)
    frames_np = np.transpose(frames_np, (0, 3, 1, 2))  # (T, C, H, W)

    return frames_np
