import torch
import torch.nn as nn

# DQN class - neural network - CNN -> linear layers -> action

# Network Architecture V1
# First Layer:        Input: 4x84x84,     Output: 32x20x20
# Second Layer:       Input: 32x20x20,    Output: 64x9x9
# Third Layer:        Input: 64x9x9,      Output: 64x7x7
# Fully Connected 1:  Input: 64*7*7       Output: 512
# Fully Connected 2:  Input: 512          Output: num_actions (6 for Pong, 4 for Breakout)

# Network Architecture V2
# First Layer:        Input: 4x80x80,     Output: 32x19x19
# Second Layer:       Input: 32x19x19,    Output: 64x8x8
# Third Layer:        Input: 64x8x8,      Output: 64x6x6
# Fully Connected 1:  Input: 64*6*6       Output: 512
# Fully Connected 2:  Input: 512          Output: num_actions (6 for Pong, 4 for Breakout)


class DQN(nn.Module):

    def __init__(self, num_actions = 4, input_size = 84):
        super().__init__()

        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Change linear module respect input size
        if input_size == 80:  
            self.fc = nn.Sequential(
                nn.Linear(64 * 6 * 6, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions)
            )
        
        else:
            self.fc = nn.Sequential(
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions)
            )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).cuda()
        if len(x.size()) == 3:
            x = x.unsqueeze(dim=0)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Target network update function
def soft_update(local_model, target_model, tau):
    """
    Soft update model parameters with a tau weight.
    
    θ_target = τ*θ_local + (1 - τ)*θ_target
    
    local_model (PyTorch model): weights will be copied from
    target_model (PyTorch model): weights will be copied to
    tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
