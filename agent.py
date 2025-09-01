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
            nn.BatchNorm1d(embedding_dim, affine=False)
        )

    def forward(self, state):
        """Run input through conv+fc and return embedding."""
        x = (state.float() / 255.0).to(next(self.parameters()).device)
        x = self.conv(x).view(x.size(0), -1)
        embedding = self.emb_head(x)
        #normalize to unit sphere
        return embedding

class Evaluator(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),  # include embedding
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1) # output probability logit
        )

    def forward(self, state1, state2):
        x = torch.cat([state1, state2], dim=1)
        logits = self.fc(x).squeeze(-1)  # shape [B]
        return logits

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
            nn.Linear(256, embedding_dim), # output next embedding
            nn.BatchNorm1d(embedding_dim, affine=False)
        )

    def forward(self, emb, action):
        action = action.squeeze(-1)
        action_onehot = F.one_hot(action, num_classes=self.n_actions).float()
        x = torch.cat([emb, action_onehot], dim=1)
        next_emb = self.fc(x)
        #normalize to unit sphere
        return next_emb


class SimpleTransitionModel(nn.Module):
    """
    Given just an embedding, predict the next embedding
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=False)  # enforce mean=0, var=1
        )

    def forward(self, emb):
        next_emb = self.fc(emb)
        return next_emb

class RewardPredictor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1) # output reward prediction
        )

    def forward(self, emb):
        reward = self.fc(emb).squeeze(-1)  # shape [B]
        return reward