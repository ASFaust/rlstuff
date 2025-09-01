import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
import torch
from torch.utils.data import TensorDataset, DataLoader

# --- Register ALE envs with Gymnasium (required for "ALE/..." ids) ---
gym.register_envs(ale_py)

def make_atari_env(env_id: str, seed: int, render_mode=None):
    """
    Factory for a single Atari env with standard preprocessing:
      - grayscale, resize to 84x84
      - frame-skip=4 with max-pool over last two frames
      - frame stack of 4 (channel-first: 4x84x84)
    """
    def thunk():
        env = gym.make(env_id, frameskip=1, render_mode=render_mode)
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            grayscale_obs=True,
            grayscale_newaxis=False,
            terminal_on_life_loss=False,
            scale_obs=False,
        )
        env = FrameStackObservation(env, stack_size=4)
        env.reset(seed=seed)
        return env
    return thunk



def run_to_completion(env_fn, policy, n_runs: int, *, device="cpu", seed=None, max_steps=None):
    """
    Minimal, no-replay rollout: runs `n_runs` episodes to completion and returns ragged sequences.
    Stops an episode immediately on `terminated=True`. Tracks whether an episode ended via truncation.

    Args:
        env_fn: zero-arg factory returning a Gymnasium env (e.g. from your make_atari_env(...)).
        n_runs: number of episodes to run.
        device: torch device for the policy forward pass.
        seed: optional base seed; if given, per-episode seeds are derived as seed + i.
        max_steps: optional hard cap on steps per episode.

    Returns:
        states:    List[np.ndarray]  (per-episode arrays of shape (T_i, *obs_shape))
        actions:   List[np.ndarray]  (per-episode arrays of shape (T_i,))
        rewards:   List[np.ndarray]  (per-episode arrays of shape (T_i,))
        truncated: List[bool]        (length n_runs)
    """
    env = env_fn()
    states, actions, rewards, truncated_flags = [], [], [], []

    for i in range(n_runs):
        # (re)seed per-episode if requested
        ep_seed = None if seed is None else (int(seed) + i)
        obs, info = env.reset(seed=ep_seed)

        ep_states, ep_actions, ep_rewards = [], [], []
        ep_truncated = False
        steps = 0

        while True:
            # Save current observation (copy to avoid later mutation)
            ep_states.append(np.array(obs, copy=True))

            # Policy forward -> logits -> sample action
            # Note: keep obs layout as produced by env. Your policy must accept that layout.
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)  # shape: (1, *obs_shape)
            a = policy(obs_t)

            # Step env
            next_obs, r, terminated, truncated, info = env.step(a)

            ep_actions.append(a)
            ep_rewards.append(float(r))

            steps += 1
            if truncated:
                ep_truncated = True

            # Episode end conditions
            if terminated or truncated or (max_steps is not None and steps >= max_steps):
                break

            obs = next_obs

        # Stack per-episode sequences (ragged across episodes by list-of-arrays)
        states.append(np.stack(ep_states, axis=0))           # (T_i, *obs_shape)
        actions.append(np.asarray(ep_actions, dtype=np.int64))
        rewards.append(np.asarray(ep_rewards, dtype=np.float32))
        truncated_flags.append(bool(ep_truncated))

    env.close()
    return states, actions, rewards, truncated_flags


# --- Example usage (assuming you already have make_atari_env from your code above) ---
# env_fn = make_atari_env("ALE/Pong-v5", seed=123)
# def policy(obs_batch):
#     # obs_batch: torch.Tensor with shape (B, *obs_shape)
#     # return logits of shape (B, n_actions)
#     return torch.zeros((obs_batch.shape[0], 6), device=obs_batch.device)  # uniform dummy logits for Pong
# S, A, R, TR = run_to_completion(env_fn, policy, n_runs=3, device="cpu")
