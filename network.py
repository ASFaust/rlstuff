import torch 

class DiscreteNetwork(torch.nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=256):
        super(DiscreteNetwork, self).__init__()
        print(f"Initializing DiscreteNetwork with obs_dim={obs_dim}, n_actions={n_actions}, hidden_dim={hidden_dim}")
        self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.g1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.g2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l_out = torch.nn.Linear(hidden_dim, 1 + n_actions)  # output: [value, regret for each action]
        self.n_actions = n_actions
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        h1 = self.fc1(x)
        g1 = self.sigmoid(self.g1(x))
        h1 = h1 * g1
        h2 = self.fc2(h1)
        g2 = self.sigmoid(self.g2(h1))
        h2 = h2 * g2  
        out = self.l_out(h2)
        value = out[:, 0]  # state value
        regret = out[:, 1:]  # regret for each action
        return value, regret
