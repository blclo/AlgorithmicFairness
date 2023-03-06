import torch
from src.models.model import FullyConnected, AutoEncoder

def load_model(experiment: dict):
    input_parameters = experiment['model']['input_parameters']

    if experiment['model']['name'] == 'AutoEncoder':
        model = AutoEncoder(channels_in=input_parameters[0], z_dim=input_parameters[1], channels_out=input_parameters[2])
        model.encoder = experiment['model']['net'][0]
        model.classifier = experiment['model']['net'][1]
        model.load_state_dict(experiment['state_dict'])
        return model
    
    elif experiment['model']['name'] == 'FullyConnected':
        model = FullyConnected(channels_in=input_parameters[0], channels_out=input_parameters[1])
        model.net = experiment['model']['net']
        model.load_state_dict(experiment['state_dict'])
        return model
    
    else:
        raise NotImplementedError("Yet to be implemented...")
    
def set_seed(seed: int):
    torch.manual_seed(seed)