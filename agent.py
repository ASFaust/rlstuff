import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ReplayBuffer import ReplayBuffer
from contextlib import nullcontext

#this implements:
#V(s) = max_a Q(s,a)
#Q(s,a) = V(s) + R(s,a)
#so bellman becomes:
#V(s) - R(s,a) = r + gamma * (1-done) * V(next_s)
#the regret net is strictly positive.
#so regret of optimal action is zero
#and suboptimal actions have positive regret, which is added to the optimal value to get the Q value.

def get_optimizer(name, params, lr):
    if name == "adam":
        return optim.Adam(params, lr=lr)
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
    else:
        raise ValueError(f"Unsupported positivity transform: {name}")

class DiscreteAgent:
    def __init__(
        self,
        obs_dim,
        n_actions,
        buffer,
        target_entropy,
        lr_value=1e-3,
        lr_regret=1e-3,
        lr_alpha=1e-3,
        gamma=0.99,
        batch_size=256,
        device="cpu",
        subtract_min=False,
        learn_alpha=False,
        alpha_init=1.0,
        detach_rhs=False,
        optimizer="adam",
        positivity_transform="softplus",
        use_per=False,
        per_clamp=10.0,
    ):
        self.device = device
        self.use_per = use_per
        self.per_clamp = per_clamp
        self.gamma = gamma
        self.detach_rhs = detach_rhs
        self.subtract_min = subtract_min
        self.batch_size = batch_size
        self.buffer = buffer
        self.n_actions = n_actions
        self.steps = 0
        self.target_entropy = target_entropy * np.log(n_actions)
        self.sqrt_alpha = torch.tensor(np.sqrt(alpha_init), requires_grad=True, device=device)
        if learn_alpha:
            self.alpha_optimizer = get_optimizer(optimizer, [self.sqrt_alpha], lr=lr_alpha)

        self.learn_alpha = learn_alpha
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1 ),
        ).to(device) 
        self.regret_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        ).to(device)
        self.positivity_transform = get_positivity_transform(positivity_transform)
        self.value_optimizer = get_optimizer(optimizer, self.value_net.parameters(), lr=lr_value)
        self.regret_optimizer = get_optimizer(optimizer, self.regret_net.parameters(), lr=lr_regret)

    @property
    def alpha(self):
        return self.sqrt_alpha ** 2.0 

    def act(self, obs, epsilon=0.1, greedy=False):
        #entropy regularized action selection
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            regrets = self.positivity_transform(self.regret_net(obs))
            value = self.value_net(obs)
            if self.subtract_min:
                regrets = regrets - regrets.min(dim=1, keepdim=True)[0] #optional: subtract min regret to improve stability
            if greedy:
                action = torch.argmin(regrets).item()
            else:
                action_probs = F.softmax((value - regrets) / self.alpha, dim=1)
                action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        if self.use_per:
            batch = self.buffer.sample_prioritized(self.batch_size, ratio=self.per_clamp)
        else:
            batch = self.buffer.sample(self.batch_size)

        obs = batch["obs"]
        actions = batch["actions"].long().view(-1)
        rewards = batch["rewards"].view(-1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].view(-1)

        self.steps += 1

        #compute targets
        #with torch.no_grad():
        with torch.no_grad() if self.detach_rhs else nullcontext():
            #compute V(next_obs) using value net
            next_values = self.value_net(next_obs).squeeze()
            #target = r + gamma * (1-done) * V(next_obs)
            targets = rewards + self.gamma * (1.0 - dones) * next_values

        #compute current estimates
        values = self.value_net(obs).squeeze()
        regrets = self.positivity_transform(self.regret_net(obs))
        if self.subtract_min:
            regrets = regrets - regrets.min(dim=1, keepdim=True)[0] #optional: subtract min regret to improve stability
        #select the regrets for the taken actions
        action_regrets = regrets.gather(1, actions.unsqueeze(1)).squeeze()
        #compute Q values for taken actions
        q_values = values - action_regrets
        #compute loss (includes regret net since Q depends on it)
        squared_errors = (q_values - targets) ** 2
        if self.use_per:
            self.buffer.update_errors(batch["indices"], squared_errors.detach().cpu().numpy())
        loss = squared_errors.mean()

        self.value_optimizer.zero_grad()
        self.regret_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        self.regret_optimizer.step()

        with torch.no_grad():#values has shape [batch_size], regrets has shape [batch_size, n_actions]
            action_probs = F.softmax((values.unsqueeze(1) - regrets) / self.alpha, dim=1)

        ret_dict = {
            "loss": loss.item(),
            "value/max": values.max().item(),
            "value/min": values.min().item(),
            "value/mean": values.mean().item(),
            #max regret over the _actions_ for each sample, then mean over the batch
            "regret/max": regrets.max(dim=1)[0].mean().item(),
            "regret/min": regrets.min(dim=1)[0].mean().item(),
            "regret/mean": regrets.mean().item(),
            "action prob/max": action_probs.max(dim=1)[0].mean().item(),
            "action prob/min": action_probs.min(dim=1)[0].mean().item(),
            "action prob/taken": action_probs.gather(1, actions.unsqueeze(1)).mean().item(),
        }

        if self.learn_alpha:
            #update alpha   
            with torch.no_grad():
                log_action_probs = F.log_softmax((values.unsqueeze(1)-regrets) / self.alpha, dim=1)
                entropy = -(action_probs * log_action_probs).sum(dim=1)

            alpha_loss = (action_probs * (
                - self.alpha * (log_action_probs + self.target_entropy)
                )).sum(dim=-1).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            ret_dict["alpha_loss"] = alpha_loss.item()
            ret_dict["alpha"] = self.alpha.item()
            ret_dict["entropy/diff to target"] = (entropy - self.target_entropy).mean().item()

        return ret_dict

