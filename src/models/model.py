import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from typing import Optional

#  ---------------  Model  ---------------
class MLP(nn.Module):
    def __init__(self, D_in, H=15, D_out=1):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze()
    
#  ---------------  Model  ---------------
class FullyConnected(nn.Module):
    def __init__(self, channels_in, channels_out=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(channels_in, 128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, channels_out),
            nn.Softmax(dim=1),
        )
                
    def forward(self, x):
        probs = self.net(x)
        logits = torch.log(probs)
        return probs, logits

def get_loss_function(type: str = 'BCE'):
    if type == 'BCE':
        return nn.BCELoss()
    elif type == 'NLL':
        return nn.NLLLoss()
    else:
        raise NotImplemented("The specified loss criterion is yet to be implemented...")

def get_optimizer(network, type: str = 'SGD', lr: float = 0.001, momentum: Optional[float] = None):
    """Return the optimizer to use during training.
    
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    if type == 'SGD':
        return optim.SGD(network.parameters(), lr=lr, momentum=momentum)
    elif type == 'Adam':
        return optim.Adam(network.parameters(), lr=lr)
    else:
        raise NotImplemented("The specified optimizer is yet to be implemented...")