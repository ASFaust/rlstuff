import torch 

class DiscreteNetwork(torch.nn.Module):
    def __init__(self, n_channels, n_actions):
        super(DiscreteNetwork, self).__init__()
        #output dim is 1 + n_actions, since we have a value and a regret for each action
        conv_dim = n_channels * 2
        n_position_channels = 2
        #position grid is a learnable parameter for each pixel. it starts at 0 and can learn to encode positional information. we add it as additional channels to the input.
        self.conv1 = torch.nn.Conv2d(n_channels + n_position_channels, conv_dim, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1)
        self.act = torch.nn.SiLU()
        hidden_dim = max(conv_dim, n_actions) * 4 
        self.fc = torch.nn.Linear(conv_dim * 2, hidden_dim)
        self.l_out = torch.nn.Linear(hidden_dim, 1 + n_actions)

        #add a fixed position grid that goes from 0 to 1 in x and y direction, to help the learnable position grid
        #x = torch.linspace(-1, 1, steps=10)
        #y = torch.linspace(-1, 1, steps=10)
        # xx, yy = torch.meshgrid(x, y, indexing='ij')
        # position_grid_init = torch.stack([xx, yy], dim=0).unsqueeze(0) # (1, 2, 10, 10)
        self.position_grid = torch.nn.Parameter(torch.zeros(1, n_position_channels, 10, 10)) # (1, n_position_channels, 10, 10)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"DiscreteNetwork initialized with {total_params} parameters, conv_dim={conv_dim}, n_position_channels={n_position_channels}, hidden_dim={hidden_dim}, obs_dim={n_channels}, n_actions={n_actions}")

    def forward(self, obs):
        #obs is (batch, n_channels, 10, 10)
        batch_size = obs.shape[0]
        x = torch.cat([obs, self.position_grid.expand(batch_size, -1, -1, -1)], dim=1) # (batch, n_channels + n_position_channels, 10, 10)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x_mean = x.mean(dim=[2,3]) # global average pooling, (batch, conv_dim)
        x_max = x.amax(dim=[2,3]) # global max pooling, (batch, conv_dim)
        x = torch.cat([x_mean, x_max], dim=1) # (batch, conv_dim * 2)
        x = self.act(self.fc(x))
        out = self.l_out(x) # (batch, 1 + n_actions)
        values = out[:, 0] # (batch,)
        regrets = out[:, 1:] # (batch, n_actions)
        return values, regrets

    def get_position_grid(self):
        return self.position_grid.detach().cpu().numpy()[0] # (n_position_channels, 10, 10)
