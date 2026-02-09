import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import wandb

from minatar import Environment
from DiscreteAgent import DiscreteAgent
from ReplayBuffer import ReplayBuffer
from eval_video import make_eval_video_frames

torch.set_float32_matmul_precision('high')

def make_env(game, seed=None):
    env = Environment(game)
    if seed is not None:
        env.seed(seed)
    return env


def flatten_obs(obs):
    #print(f"Original obs shape: {obs.shape}, dtype: {obs.dtype}") #(10,10,n_channels), bool
    #we dont want to flatten, we want to rearrange to (n_channels, 10, 10) and convert to float32
    obs = np.transpose(obs, (2, 0, 1)) # (n_channels, 10, 10)
    obs = obs.astype(np.float32)
    return obs
    #return obs #.astype(np.float32).reshape(-1)


def greedy_eval(agent, game, seeds, max_steps=2000):
    returns = []

    for seed in seeds:
        env = make_env(game, seed)
        obs = flatten_obs(env.state())

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(obs, greedy=True)
            reward, done = env.act(action)
            obs = flatten_obs(env.state())
            total_reward += reward
            steps += 1

        returns.append(total_reward)

    returns = np.array(returns)

    ret = {
        "eval/mean": returns.mean(),
        "eval/std": returns.std(),
        "eval/min": returns.min(),
        "eval/max": returns.max(),
    }

    return ret

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)

    env = make_env(cfg.env.game, cfg.seed)
    obs = flatten_obs(env.state())
    obs_dim = obs.shape[0]
    n_actions = env.num_actions()

    buffer = ReplayBuffer(capacity = cfg.agent.buffer_size, obs_shape=obs.shape, device=device)

    agent = DiscreteAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        buffer=buffer,
        target_entropy=cfg.agent.target_entropy,
        gamma=cfg.agent.gamma,
        batch_size=cfg.agent.batch_size,
        device=device,
        lr=cfg.agent.lr,
        lr_alpha=cfg.agent.lr_alpha,
        subtract_min=cfg.agent.subtract_min,
        alpha_init=cfg.agent.alpha_init,
        learn_alpha=cfg.agent.learn_alpha,
        optimizer=cfg.agent.optimizer,
        positivity_transform=cfg.agent.positivity_transform,
        loss_weights=cfg.agent.loss_weights,
        terminal_value=cfg.agent.terminal_value,
        on_off_policy_lambda=cfg.agent.on_off_policy_lambda,
        tau=cfg.agent.tau,
    )

    wandb.init(project=cfg.project, config=OmegaConf.to_container(cfg, resolve=True))

    global_step = 0
    episode_return = 0
    episode = 0
    obs = flatten_obs(env.state())
    eval_seeds = list(range(10))

    while global_step < cfg.training.total_steps:

        action = agent.act(obs)
        reward, done = env.act(action)
        next_obs = flatten_obs(env.state())

        agent.store(obs, action, reward, next_obs, done)

        for _ in range(cfg.training.updates_per_step):
            update_stats = agent.update()

        obs = next_obs
        episode_return += reward
        global_step += 1

        if update_stats:
            wandb.log(update_stats, step=global_step)

        if done:
            wandb.log(
                {
                    "train/episode_return": episode_return,
                    "train/episode": episode,
                },
                step=global_step,
            )
            episode += 1
            episode_return = 0
            env = make_env(cfg.env.game)
            obs = flatten_obs(env.state())

        if global_step % cfg.training.eval_interval == 0:
            eval_stats = greedy_eval(agent, cfg.env.game, eval_seeds)
            wandb.log(eval_stats, step=global_step)
            print(f"[{global_step}] eval:", eval_stats)
            video_frames = make_eval_video_frames(
                agent,
                cfg.env.game,
                seed=eval_seeds[0],
                max_steps=2000,
                fps=4,
            )
            if video_frames.shape[0] > 0:
                wandb.log(
                    {"eval/rollout": wandb.Video(video_frames, format="mp4", fps=4)},
                    step=global_step,
                )

    wandb.finish()


if __name__ == "__main__":
    main()


#example usage:
# python train.py agent=discrete env=breakout training.total_steps=1000000
# with overrides for any of the config parameters, e.g. to use a different optimizer:
# python train.py agent=discrete env=breakout training.total_steps=1000000 agent.optimizer=sgd
