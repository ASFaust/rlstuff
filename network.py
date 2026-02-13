import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEmbedder(nn.Module):
    def __init__(self, in_channels, embedding_dim = 128):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0)

        def size_linear_unit(size, kernel_size=3, stride=1, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
            
        num_linear_units = 16 * size_linear_unit(10, 3, 1, 0) * size_linear_unit(10, 3, 1, 0)
        
        self.act = nn.SiLU()

        self.fc_1 = nn.Linear(num_linear_units, embedding_dim)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"StateEmbedder initialized with {total_params} parameters. Linear units: {num_linear_units}. Embedding dim: {embedding_dim}.")

    def forward(self, x):
        x = self.act(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc_1(x))
        return x


class DuelingRegretValueHead(nn.Module):
    def __init__(self, num_actions, embedding_dim = 128, positivity_transform = torch.exp, subtract_min = False):
        super().__init__()

        self.l1 = nn.Linear(embedding_dim,num_actions + 1) # num_actions for regrets, 1 for value, 1 for baseline
        self.num_actions = num_actions
        self.positivity_transform = positivity_transform

    def forward(self, x):
        out = self.l1(x)
        value = out[:, 0] 
        #baseline = out[:, 1].unsqueeze(1)  # (batch_size, 1)
        #q_values = out[:, 2:]  # (batch_size, num_actions)
        #q_values = q_values - baseline  # (batch_size, num_actions)
        #regrets = value.unsqueeze(1) - q_values  # (batch_size, num_actions)
        regrets = self.positivity_transform(out[:, 1:])  # (batch_size, num_actions)
        return value, regrets

    
class DiscreteBellmanOracle(nn.Module):
    #bellman oracle predicts only 2 values: reward and next_value, per action taken
    #it also uses dueling architecture: baseline for both reward and next value, and advantages for each
    def __init__(self, num_actions, embedding_dim = 128):
        super().__init__()

        self.head = nn.Linear(embedding_dim, 2 * num_actions)
        self.num_actions = num_actions
    
    def forward(self, x):
        out = self.head(x)
        rewards = out[:, :self.num_actions]
        next_values = out[:, self.num_actions:]
        return rewards, next_values
