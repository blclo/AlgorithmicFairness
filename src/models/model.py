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

        self.net1 = nn.Sequential(
            nn.Linear(channels_in, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.1),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.1),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(p=0.1),

            nn.Linear(128, 64),
            nn.Linear(64, channels_out),
            nn.Softmax(dim=1) if channels_out == 1 else nn.Softmax(dim=1),
        )

        self.net2 = nn.Sequential(
            nn.Linear(channels_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, channels_out),
            nn.Sigmoid() if channels_out == 1 else nn.Softmax(dim=1),
        )

        self.net3 = nn.Sequential(
            nn.Linear(channels_in, 1),
            nn.Sigmoid() if channels_out == 1 else nn.Softmax(dim=1),
        )
        
        self.net1.apply(self.init_weights)
        #self.net2.apply(self.init_weights)
        #self.net3.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net2(x)

def get_loss_function(type: str = 'BCE'):
    if type == 'BCE':
        return nn.BCELoss()
    elif type == 'NLL':
        return nn.NLLLoss()
    elif type == 'CrossEntropy':
        return nn.CrossEntropyLoss()
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