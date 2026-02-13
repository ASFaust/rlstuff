import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ReplayBuffer import ReplayBuffer
from contextlib import nullcontext
from network import StateEmbedder, DuelingRegretValueHead, DiscreteBellmanOracle 

#this implements:
#V(s) = max_a Q(s,a)
#Q(s,a) = V(s) + R(s,a)
#so bellman becomes:
#V(s) - R(s,a) = r + gamma * (1-done) * V(next_s)
#the regret net is strictly positive.
#so regret of optimal action is zero
#and suboptimal actions have positive regret, which is subtracted from the optimal value to get the Q value.

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
            num_actions=n_actions, embedding_dim=128, 
            positivity_transform=get_positivity_transform(positivity_transform), 
            subtract_min=subtract_min).to(device)
        self.net_oracle = DiscreteBellmanOracle(num_actions=n_actions, embedding_dim=128).to(device)
        self.net_optimizer = get_optimizer(optimizer, 
                                           list(self.net_embed.parameters()) + 
                                           list(self.net_value_regret.parameters()) +
                                           list(self.net_oracle.parameters()), lr=lr)
        self.step = 0
        self.n_oracle_init_steps = 5000

    @property
    def alpha(self):
        with torch.no_grad():
            return self.sqrt_alpha ** 2

    def act(self, obs, epsilon=0.1, greedy=False):
        #entropy regularized action selection
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
        min_reward, max_reward = self.buffer.get_reward_range()
        obs = batch["obs"]
        actions = batch["actions"].long().view(-1)
        rewards = batch["rewards"].view(-1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].view(-1)

        emb_obs = self.net_embed(obs) #[B, emb_dim]
        emb_next_obs = self.net_embed(next_obs) #[B, emb_dim]
        value, regrets = self.net_value_regret(emb_obs) #[B], [B, n_actions]
        next_value, _ = self.net_value_regret(emb_next_obs) # [B], [B, n_actions]
        oracle_rewards, oracle_next_values = self.net_oracle(emb_obs) #[B, n_actions], [B, n_actions]
        taken_regrets = regrets.gather(1, actions.unsqueeze(1)).squeeze(1) #[B] #this is g(s,a) for taken actions
        taken_oracle_rewards = oracle_rewards.gather(1, actions.unsqueeze(1)).squeeze(1) #[B]
        taken_oracle_next_values = oracle_next_values.gather(1, actions.unsqueeze(1)).squeeze(1) #[B]

        mask = (1 - dones)

        with torch.no_grad():
            real_oracle_rewards = torch.clamp(oracle_rewards, min=min_reward, max=max_reward).detach()
            next_value_target = (next_value * mask + self.terminal_value * (1 - mask)).detach()
            #f(s) - g(s,a) = r + gamma * (1-done) * f(next_s)
            #g(s,a) = f(s) - r - gamma * (1-done) * f(next_s)
            value_regret_target = (real_oracle_rewards + self.gamma * oracle_next_values * mask.unsqueeze(1) + \
                self.terminal_value * (1 - mask.unsqueeze(1))).detach()
            #has shape [B, n_actions]
            #this gives us targets for the sum of value and regret for every action

        oracle_value_loss = F.mse_loss(taken_oracle_next_values, next_value_target)
        oracle_reward_loss = F.mse_loss(taken_oracle_rewards, rewards)
        next_value_loss = F.mse_loss(next_value, next_value_target)
        value_regret_loss = F.mse_loss(value.unsqueeze(1) - regrets, value_regret_target)

        #now to policy improvement: we need g(s,a) = 0 for optimal action in order to generate policy improvement signal.
        #we add this in the oracle again:
        #g(s,a) = f(s) - r - gamma * (1-done) * f(next_s) 
        #unsure wether we should detach oracle next values here
        #but for standard bellman, we usually do detach the next value, so maybe that's the right thing to do here as well.
        #but it is also true that oracle next values are already E[V(s) | s,a], so it's not like we are averaging
        #V(s') over its predecsessors, which is the reason why we usually detach next values in bellman updates.
        #observation: if we detach oracle next values, we get the old overestimation problem, 
        #and it vanishes completely if we don't detach them
        #the big question is wether it is principled to not detach them
        #i think it is fine, im not 100% sure
        oracle_regrets = value.unsqueeze(1) - real_oracle_rewards - self.gamma * oracle_next_values * mask.unsqueeze(1) - self.terminal_value * (1 - mask.unsqueeze(1))
        #oracle_regrets has shape [B, A]
        #we select min along dim 1
        #oracle_regrets_targets = torch.relu(oracle_regrets)
        oracle_regrets_targets = oracle_regrets - oracle_regrets.min(dim=1, keepdim=True)[0]
        #min_oracle_regrets, _ = oracle_regrets.min(dim=1, keepdim=True) #[B, 1]
        #we want those to be = 0, so we add them to the loss:
        #regret_loss = min_oracle_regrets.pow(2).mean()
        regret_loss = F.mse_loss(regrets, oracle_regrets_targets.detach())

        if self.step < self.n_oracle_init_steps:
            actual_regret_weight = 0.0
        else:
            actual_regret_weight = self.loss_weights.regret
        self.step += 1

        loss = self.loss_weights.oracle_value * oracle_value_loss + \
            self.loss_weights.oracle_reward * oracle_reward_loss + \
            self.loss_weights.value_regret * value_regret_loss + \
            actual_regret_weight * regret_loss + \
            self.loss_weights.next_value * next_value_loss

        ret_dict = {
            "loss/value_loss": oracle_value_loss.item(),
            "loss/regret_loss": regret_loss.item(),
            "loss/reward_loss": oracle_reward_loss.item(),
            "loss/value_regret_loss": value_regret_loss.item(),
            "loss/next_value_loss": next_value_loss.item(),
            "loss/total_loss": loss.item(),
            "stats1/mean_value": value.mean().item(),
            "stats1/min_value": value.min().item(),
            "stats1/max_value": value.max().item(),
            "stats1/min_regret": regrets.min().item(),
            "stats1/max_regret": regrets.max().item(),
            "stats2/mean_taken_regret": taken_regrets.mean().item(),
            "stats2/max_oracle_reward_diff": (taken_oracle_rewards - rewards).abs().max().item(),
            "stats2/max_value_diff": (taken_oracle_next_values - next_value).abs().max().item(),
        }

        self.net_optimizer.zero_grad()
        loss.backward()
        self.net_optimizer.step()

        action_probs = F.softmax(-regrets / self.alpha, dim=1)

        if self.learn_alpha:
            #update alpha   
            with torch.no_grad():
                log_action_probs = torch.log(action_probs + 1e-8)
                entropy = -(action_probs * log_action_probs).sum(dim=1).mean()

            alpha_loss = (self.sqrt_alpha) ** 2 * (entropy - self.target_entropy)

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            ret_dict["alpha_loss"] = alpha_loss.item()
            ret_dict["alpha"] = self.alpha.item()
            ret_dict["entropy/diff to target"] = (entropy - self.target_entropy).mean().item()

        return ret_dict

