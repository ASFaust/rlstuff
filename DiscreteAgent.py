import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ReplayBuffer import ReplayBuffer
from contextlib import nullcontext
from network import DiscreteNetwork 

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
        on_off_policy_lambda,
        tau,
        device="cpu"
    ):
        self.device = device
        self.loss_weights = loss_weights
        self.terminal_value = terminal_value
        self.on_off_policy_lambda = on_off_policy_lambda
        self.gamma = gamma
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
        self.net = DiscreteNetwork(obs_dim, n_actions).to(device)
        self.positivity_transform = get_positivity_transform(positivity_transform)
        self.net_optimizer = get_optimizer(optimizer, self.net.parameters(), lr=lr)
        self.tau = tau
        if tau != 1.0:
            self.target_net = DiscreteNetwork(obs_dim, n_actions).to(device)
            self.target_net.load_state_dict(self.net.state_dict())


    @property
    def alpha(self):
        with torch.no_grad():
            return self.sqrt_alpha ** 2

    def act(self, obs, epsilon=0.1, greedy=False):
        #entropy regularized action selection
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value, regrets = self._values_and_regrets(obs)
            if greedy:
                action = regrets.argmin(dim=1).item()
            else:
                action_probs = F.softmax((value - regrets) / self.alpha, dim=1)
                action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def _values_and_regrets(self, obs):
        values, regrets = self.net(obs) #values has shape [batch_size], regrets has shape [batch_size, n_actions]
        regrets = self.positivity_transform(regrets) #ensure regrets are positive
        if self.subtract_min:
            regrets = regrets - regrets.min(dim=1, keepdim=True)[0] #optional: subtract min regret to improve stability
        return values, regrets

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)

        obs = batch["obs"]
        actions = batch["actions"].long().view(-1)
        rewards = batch["rewards"].view(-1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].view(-1)

        self.steps += 1

        #no grad for next_values
        with torch.no_grad():
            if self.tau != 1.0:
                next_values, _ = self.target_net(next_obs)
            else:
                next_values, _ = self.net(next_obs)

        values, regrets = self._values_and_regrets(obs)
        taken_regret = regrets.gather(1, actions.unsqueeze(1)).squeeze()

        mask = (1.0 - dones)

        # raw Bellman residual
        # this is the regret target.
        bellman_residual = values - (rewards + self.gamma * mask * next_values)

        # -------------------------
        # backward Bellman (standard TD)
        # -------------------------
        with torch.no_grad():
            bw_target = rewards + self.gamma * mask * next_values + taken_regret.detach() * (1.0 - self.on_off_policy_lambda)

        bw_loss = (values - bw_target).pow(2).mean() 
        
        # experimental forward bellman loss, only valid in expectation, since we can't assume deterministic transitions, so we use the mean value over the batch.
        fw_target = rewards + self.gamma * mask * next_values + taken_regret * (1.0 - self.on_off_policy_lambda)
        fw_loss = (values.mean() - fw_target.mean()).pow(2)

        # gauge fixing:
        with torch.no_grad():
            anchor_actions = regrets.argmin(dim=1) #anchor action is the action with lowest regret, should be the optimal action, we want its regret to be zero, this fixes the gauge freedom of the regret. we can also use a random action as anchor, but this is more stable and makes more sense.

        anchor_regret = regrets.gather(1, anchor_actions.unsqueeze(1)).squeeze()
        gauge_loss = anchor_regret.pow(2).mean()   # or .mean()

        # -------------------------
        # regret hinge
        # -------------------------
        regret_target = torch.clamp(bellman_residual, min=0).detach()
        regret_loss = (taken_regret - regret_target).pow(2).mean()

        # -------------------------
        # forward consistency hinge
        # only active when Bellman violated (bellman_residual < 0)
        # can only act on the mean, not per sample, 
        # since we can't assume deterministic transitions
        # -------------------------
        cons_loss = torch.clamp(bellman_residual.mean(), max=0).pow(2)

        # and a loss for the terminals to be close to zero
        if dones.any():
            # optional: encourage terminal states to have zero value by penalizing the value of terminal states
            terminal_loss = (values[dones.bool()] - self.terminal_value).pow(2).mean()
        else:
            terminal_loss = torch.tensor(0.0, device=self.device)

        # -------------------------
        loss = (
            self.loss_weights.bw * bw_loss +
            self.loss_weights.regret * regret_loss +
            self.loss_weights.consistency * cons_loss +
            self.loss_weights.terminal * terminal_loss +
            self.loss_weights.fw * fw_loss + 
            self.loss_weights.gauge * gauge_loss
        )

        self.net_optimizer.zero_grad()
        loss.backward()
        self.net_optimizer.step()

        #update target network with soft updates
        if self.tau != 1.0:
            with torch.no_grad():
                for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        with torch.no_grad():#values has shape [batch_size], regrets has shape [batch_size, n_actions]
            action_probs = F.softmax((values.unsqueeze(1) - regrets) / self.alpha, dim=1)

        ret_dict = {
            "loss/bw": bw_loss.item(),
            "loss/regret": regret_loss.item(),
            "loss/consistency": cons_loss.item(),
            "loss/terminal": terminal_loss.item(),
            "loss/fw": fw_loss.item(),
            "loss/gauge": gauge_loss.item(),
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

