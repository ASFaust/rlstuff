import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import wandb

from minatar import Environment
from agent import DiscreteAgent
from ReplayBuffer import ReplayBuffer


def make_env(game, seed=None):
    env = Environment(game)
    if seed is not None:
        env.seed(seed)
    return env


def flatten_obs(obs):
    return obs.astype(np.float32).reshape(-1)


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

    return {
        "eval/mean": returns.mean(),
        "eval/std": returns.std(),
        "eval/min": returns.min(),
        "eval/max": returns.max(),
    }


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

    buffer = ReplayBuffer(cfg.agent.buffer_size, obs_dim, device=device)

    agent = DiscreteAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        buffer=buffer,
        target_entropy=cfg.agent.target_entropy,
        gamma=cfg.agent.gamma,
        batch_size=cfg.agent.batch_size,
        device=device,
        lr_value=cfg.agent.lr_value,
        lr_regret=cfg.agent.lr_regret,
        lr_alpha=cfg.agent.lr_alpha,
        subtract_min=cfg.agent.subtract_min,
        alpha_init=cfg.agent.alpha_init,
        learn_alpha=cfg.agent.learn_alpha,
        detach_rhs=cfg.agent.detach_rhs,
        optimizer=cfg.agent.optimizer,
        positivity_transform=cfg.agent.positivity_transform,
        use_per=cfg.agent.use_per,
        per_clamp=cfg.agent.per_clamp,
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

    wandb.finish()


if __name__ == "__main__":
    main()
