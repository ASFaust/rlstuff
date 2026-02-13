import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from collections import deque

from minatar import Environment


# ---------------------------
# Config
# ---------------------------

config = dict(
    game="breakout",
    seed=0,
    total_steps=200_000,
    gamma=0.99,
    eps=0.1,                  # inflate step
    batch_size=256,
    buffer_size=10000,
    start_learning=100,
    train_freq=1,
    epsilon_start=0.4,
    epsilon_final=0.05,
    epsilon_decay=50_000,
    lr=1e-3,
)

wandb.init(project="inflate-balloon-rl", config=config)
cfg = wandb.config

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def softmin(a, b, tau=0.05):
    stacked = torch.stack([-a/tau, -b/tau], dim=0)
    return -tau * torch.logsumexp(stacked, dim=0)

# ---------------------------
# Replay buffer
# ---------------------------

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch):
        batch = random.sample(self.buffer, batch)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------
# Network
# ---------------------------

class Net(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, 3, stride=1, padding=0)
        self.act = nn.SiLU()

        conv_out = 16 * 8 * 8  # 10x10 -> 8x8
        self.fc = nn.Linear(conv_out, 64)

        self.q_head = nn.Linear(64, n_actions)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.act(self.conv(x))
        x = x.flatten(1)
        x = self.act(self.fc(x))
        q_raw = self.q_head(x)
        q = self.softplus(q_raw)   # enforce positivity
        return q


# ---------------------------
# Env setup
# ---------------------------

env = Environment(cfg.game)
n_actions = env.num_actions()
obs = env.state_shape()

def preprocess(s):
    return np.transpose(s.astype(np.float32), (2,0,1))

net = Net(obs[2], n_actions).to(device)
opt = optim.Adam(net.parameters(), lr=cfg.lr)
rb = ReplayBuffer(cfg.buffer_size)


# ---------------------------
# Epsilon schedule
# ---------------------------

def epsilon(step):
    frac = min(1.0, step / cfg.epsilon_decay)
    return cfg.epsilon_start + frac * (cfg.epsilon_final - cfg.epsilon_start)


# ---------------------------
# Training loop
# ---------------------------

s = preprocess(env.state())
episode_return = 0
episode = 0

for step in range(cfg.total_steps):

    with torch.no_grad():
        q = net(torch.tensor(s).unsqueeze(0).to(device))
        q = q.cpu().numpy()[0]

    if random.random() < epsilon(step):
        a = random.randrange(n_actions)
    else:
        a = int(np.argmax(q))

    r, done = env.act(a)
    s2 = preprocess(env.state())

    rb.add(s, a, r, s2, done)

    episode_return += r
    s = s2

    if done:
        wandb.log({"episode_return": episode_return, "episode": episode})
        episode_return = 0
        episode += 1
        env.reset()
        s = preprocess(env.state())

    # ---------------------------
    # Learning
    # ---------------------------

    if step > cfg.start_learning and step % cfg.train_freq == 0 and len(rb) >= cfg.batch_size:
        S, A, R, S2, D = rb.sample(cfg.batch_size)
        S = S.to(device)
        S2 = S2.to(device)
        A = A.to(device)
        R = R.to(device)
        D = D.to(device)

        Q = net(S)
        Q_next = net(S2)

        V_next = Q_next.max(dim=1).values.detach()
        target = R + cfg.gamma * (1 - D) * V_next

        Qa = Q.gather(1, A.unsqueeze(1)).squeeze(1)

        inflated = Qa + cfg.eps
        new_Q = torch.minimum(inflated, target * 0.1 + inflated * 0.9)

        loss = ((Qa - new_Q.detach()) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"\rStep: {step}, Loss: {loss.item():.4f}, mean Q: {Qa.mean().item():.4f}, max Q: {Qa.max().item():.4f}, epsilon: {epsilon(step):.4f}", end="")

        wandb.log({
            "loss": loss.item(),
            "mean_q": Qa.mean().item(),
            "max_q": Qa.max().item(),
            "epsilon": epsilon(step),
            "step": step,
        })

env.close()
wandb.finish()
