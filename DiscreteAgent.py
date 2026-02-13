import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ReplayBuffer import ReplayBuffer
from contextlib import nullcontext
from network import StateEmbedder, DuelingRegretValueHead, DiscreteBellmanOracle


def softmin(g, tau=1.0, dim=-1):
    """
    g: [Batch, n_actions]
    returns: [Batch] soft minimum over actions
    """
    return -tau * torch.logsumexp(-g / tau, dim=dim)

def get_optimizer(name, params, lr):
    if name == "adam":
        return optim.Adam(params, lr=lr, betas=(0.8, 0.999), eps=1e-8)
    elif name == "sgd":
        return optim.SGD(params, lr=lr)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def get_positivity_transform(name):
    if name == "softplus":
        return F.softplus
    elif name == "relu":
        return F.relu
    elif name == "exp":
        return torch.exp
    elif name == "square":
        return torch.square
    elif name == "abs":
        return torch.abs
    elif name == "identity":
        return lambda x: x
    else:
        raise ValueError(f"Unsupported positivity transform: {name}")

class DiscreteAgent:
    def __init__(
        self,
        obs_dim,
        n_actions,
        buffer,
        target_entropy,
        loss_weights,
        terminal_value,
        lr,
        lr_alpha,
        gamma,
        batch_size,
        subtract_min,
        learn_alpha,
        alpha_init,
        optimizer,
        positivity_transform,
        device="cpu"
    ):
        self.device = device
        self.loss_weights = loss_weights
        self.terminal_value = terminal_value
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = buffer
        self.n_actions = n_actions
        self.target_entropy = target_entropy * np.log(n_actions)

        self.sqrt_alpha = torch.tensor(np.sqrt(alpha_init), requires_grad=True, device=device)
        if learn_alpha:
            self.alpha_optimizer = get_optimizer(optimizer, [self.sqrt_alpha], lr=lr_alpha)
        self.learn_alpha = learn_alpha

        self.net_embed = StateEmbedder(in_channels=obs_dim, embedding_dim=128).to(device)
        self.net_value_regret = DuelingRegretValueHead(
            num_actions=n_actions,
            embedding_dim=128,
            positivity_transform=get_positivity_transform(positivity_transform),
            subtract_min=subtract_min
        ).to(device)

        self.net_optimizer = get_optimizer(
            optimizer,
            list(self.net_embed.parameters()) +
            list(self.net_value_regret.parameters()),
            lr=lr
        )

    @property
    def alpha(self):
        with torch.no_grad():
            return self.sqrt_alpha ** 2

    def act(self, obs, epsilon=0.1, greedy=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value, regrets = self.net_value_regret(self.net_embed(obs))
            if greedy:
                action = regrets.argmin(dim=1).item()
            else:
                action_probs = F.softmax(-regrets / self.alpha, dim=1)
                action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)

        obs = batch["obs"]
        actions = batch["actions"].long().view(-1)
        rewards = batch["rewards"].view(-1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].view(-1)

        mask = (1 - dones)
        


        emb_obs = self.net_embed(obs)

        value, regrets = self.net_value_regret(emb_obs)        
        
        taken_regret = regrets.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            emb_next_obs = self.net_embed(next_obs)
            next_value, _ = self.net_value_regret(emb_next_obs)
            next_value_target = next_value * mask + self.terminal_value * (1 - mask)
            bellman_target = rewards + self.gamma * next_value_target

        # Bellman fit
        q_taken = value - taken_regret
        value_regret_loss = F.mse_loss(q_taken, bellman_target)
        value_regularization_loss = value.max()

        # hinge: min_a regret -> 0
        min_regret, _ = regrets.min(dim=1)
        regret_hinge_loss = (min_regret ** 2).mean()

        loss = (
            self.loss_weights.value_regret * value_regret_loss +
            self.loss_weights.regret * regret_hinge_loss +
            self.loss_weights.value_regularization * value_regularization_loss 
        )

        self.net_optimizer.zero_grad()
        loss.backward()
        #clip gradients to prevent explosion, especially with large positivity transforms
        torch.nn.utils.clip_grad_norm_(
            list(self.net_embed.parameters()) + list(self.net_value_regret.parameters()),
            max_norm=1.0
        )
        self.net_optimizer.step()

        action_probs = F.softmax(-regrets / self.alpha, dim=1)

        ret_dict = {
            "loss/value_regret_loss": value_regret_loss.item(),
            "loss/value_regularization": value_regularization_loss.item(),
            "loss/regret_hinge": regret_hinge_loss.item(),
            "loss/total": loss.item(),
            "stats/value_mean": value.mean().item(),
            "stats/value_min": value.min().item(),
            "stats/value_max": value.max().item(),
            "stats/regret_min": regrets.min().item(),
            "stats/regret_max": regrets.max().item(),
            "stats/regret_min_max": regrets.min(dim=1)[0].max().item(),
        }

        if self.learn_alpha:
            with torch.no_grad():
                log_action_probs = torch.log(action_probs + 1e-8)
                entropy = -(action_probs * log_action_probs).sum(dim=1).mean()

            alpha_loss = (self.sqrt_alpha ** 2) * (entropy - self.target_entropy)

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            ret_dict["exploration/alpha"] = self.alpha.item()
            ret_dict["exploration/entropy_diff"] = (entropy - self.target_entropy).item()
        ret_dict["exploration/max_action_prob"] = action_probs.max(dim=1)[0].mean().item()
        ret_dict["exploration/min_action_prob"] = action_probs.min(dim=1)[0].mean().item()
        ret_dict["exploration/taken_action_prob"] = action_probs.gather(1, actions.unsqueeze(1)).mean().item()

        return ret_dict
