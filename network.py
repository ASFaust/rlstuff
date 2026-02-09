import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()

        self.num_actions = num_actions

        # --- conv backbone (unchanged spirit) ---
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        self.fc_hidden = nn.Linear(num_linear_units, 128)

        # --- regret head ---
        self.head = nn.Linear(128, 2 + num_actions)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"DiscreteNetwork params={total_params}")

    def forward(self, x):
        # backbone
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_hidden(x))

        # structured head
        out = self.head(x)

        values = out[:, 0]
        regret_baseline = out[:, 1]
        deltas = out[:, 2:]

        # dueling-style coupling
        deltas = deltas - deltas.mean(dim=1, keepdim=True)
        q_like = regret_baseline.unsqueeze(1) + deltas

        regrets = values.unsqueeze(1) - q_like

        return values, regrets
