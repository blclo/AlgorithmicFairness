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
    def __init__(self, channels_in, channels_out=1):
        super().__init__()

        # Store input parameters
        self.input_params = [channels_in, channels_out]

        self.net = nn.Sequential(
            nn.Linear(channels_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, channels_out),
            nn.Sigmoid() if channels_out == 1 else nn.Softmax(dim=1),
        )
       
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
    
    
#  ---------------  Model  ---------------
class AutoEncoder(nn.Module):
    def __init__(self, channels_in, z_dim=32, channels_out=1):
        super().__init__()
        
        # Store input parameters
        self.input_params = [channels_in, z_dim, channels_out]

        self.encoder = nn.Sequential(
            nn.Linear(channels_in, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(32),

            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(32),

            nn.Linear(32, z_dim),
        )
       
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, channels_out),
            nn.Sigmoid() if channels_out == 1 else nn.Softmax(dim=1),
        )

        self.encoder.apply(self.init_weights)
        self.classifier.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        z = self.encoder(x)
        pred = self.classifier(z)
        return {'pred': pred, 'z': z}
    

def get_loss_function(type: str = 'BCE'):
    if type == 'BCE':
        return nn.BCELoss()
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
    
def get_model(model_name: str, channels_in: int):
    if model_name == 'FullyConnected':
        return FullyConnected(channels_in=channels_in, channels_out=1)
    elif model_name == 'AutoEncoder':
        model = AutoEncoder(channels_in=channels_in, z_dim=32, channels_out=1)
        model.net = [model.encoder, model.classifier]
        return model
    else:
        raise NotImplementedError(f"No such model class exists... {(model_name)}")
