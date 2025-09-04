import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, in_shape, embedding_dim):
        super().__init__()
        c, h, w = in_shape
        assert c == 4, "Expected input with 4 channels (stacked frames)"

        # convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4),   # (32, 20, 20) for 84x84 input
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (64, 9, 9)
            nn.SiLU(),
        )

        # compute flattened size after convs
        with torch.no_grad():
            dummy = torch.zeros(1, *in_shape)
            n_flat = self.conv(dummy).view(1, -1).size(1)

        self.emb_head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, embedding_dim), # output an embedding vector
            nn.Tanh(), # keep embeddings bounded (-1, 1
        )

    def forward(self, state):
        """Run input through conv+fc and return embedding."""
        x = (state.float() / 255.0).to(next(self.parameters()).device)
        x = self.conv(x).view(x.size(0), -1)
        return self.emb_head(x)

class TransitionModel(nn.Module):
    """
    Given an embedding and an action, predict the next embedding
    """
    def __init__(self, n_actions, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_actions = n_actions

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + n_actions, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, embedding_dim),
            nn.Tanh()
        )


    def forward(self, emb, action):
        action = action.squeeze(-1)
        action_onehot = F.one_hot(action, num_classes=self.n_actions).float()
        x = torch.cat([emb, action_onehot], dim=1)
        return self.fc(x)


class DistanceFunction(nn.Module):
    """
    Given two embeddings, predict a scalar distance
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 2) #distance + wether it is reachable
        )
        self.softplus = nn.Softplus()

    def forward(self, emb1, emb2):
        x = torch.cat([emb1, emb2], dim=1)
        distance, reachable_logits = self.fc(x).squeeze(-1).split(1, dim=-1)
        distance = self.softplus(distance) + 1.0
        return distance.squeeze(-1), reachable_logits.squeeze(-1)

class RewardModel(nn.Module):
    """
    Given an embedding, predict the reward
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(self, emb):
        reward = self.fc(emb).squeeze(-1)
        return reward