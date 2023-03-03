#!/usr/bin/python
# 

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import trange
import matplotlib.pyplot as plt

from src.models.model import MLP, FullyConnected, get_loss_function, get_optimizer
from src.data.dataloader import CatalanJuvenileJustice

#  ---------------  Training  ---------------
def train(
        datafolder_path: str,
        batch_size: int = 128, num_workers: int = 1, test_proportion: float = 0.2, val_proportion: float = 0.2, split_type: str = 'random',
        lr=1e-3, epochs: int = 100,
        experiment_name: str = '',
    ):
    """
    Trains the model.
    
    Args:
        csv_file (str): Absolute path of the dataset used for training.
        n_epochs (int): Number of epochs to train.
    """

    # Load dataset
    dataset = CatalanJuvenileJustice(
        data_path=f"{datafolder_path}/processed/catalan_dataset.pth"
    )

    # Split into training and test
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=1, 
        test_size=test_proportion, 
        val_size=val_proportion, 
        split_type=split_type,
    )

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model, loss criterion and optimizer
    model = FullyConnected(channels_in = dataset.n_attributes, channels_out = 2).to(device)
    criterion = get_loss_function(type='NLL')
    optimizer = get_optimizer(model, type='Adam', lr=lr)

    print("MLP Architecture:")
    print(model)

    writer = SummaryWriter(f"logs/{experiment_name}")
    with trange(epochs) as t:
        for epoch in t:
            running_loss_train, running_loss_val    = 0.0, 0.0
            running_acc_train,  running_acc_val     = 0.0, 0.0

            for batch in iter(train_loader):
                # Extract data                
                inputs, labels = batch['data'].to(device), batch['label'].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward
                probs, log_probs = model(inputs)
                loss = criterion(log_probs, labels)
                running_loss_train += loss.item()
                loss.backward()

                # Store accuracy
                _, predictions = probs.topk(1, dim=1)
                equals = predictions == labels.view(*predictions.shape)
                running_acc_train += torch.mean(equals.type(torch.FloatTensor))

                # Optimize
                optimizer.step()

            # Validation
            with torch.no_grad():
                for batch in iter(val_loader):
                    inputs, labels = batch['data'].to(device), batch['label'].to(device)

                    # Get predictions
                    probs, log_probs = model(inputs)
                    _, predictions = probs.topk(1, dim=1)

                    # Compute loss and accuracy
                    running_loss_val += criterion(log_probs, labels)
                    equals = predictions == labels.view(*predictions.shape)
                    running_acc_val += torch.mean(equals.type(torch.FloatTensor))

            # Update progress bar
            train_loss_descr = (
                f"Train loss: {running_loss_train / len(train_loader):.3f}"
            )
            val_loss_descr = (
                f"Validation loss: {running_loss_val / len(val_loader):.3f}"
            )
            val_acc_descr = (
                f"Validation accuracy: {running_acc_val / len(val_loader):.3f}"
            )
            t.set_description(
                f"EPOCH [{epoch}/{epochs}] --> {train_loss_descr} | {val_loss_descr} | {val_acc_descr} | Progress: "
            )

            writer.add_scalar('NLLLoss/train',  running_loss_train  / len(train_loader),    epoch)
            writer.add_scalar('accuracy/train', running_acc_train   / len(train_loader),    epoch)
            writer.add_scalar('NLLLoss/validation',    running_loss_val    / len(val_loader),      epoch)
            writer.add_scalar('accuracy/validation',   running_acc_val     / len(val_loader),      epoch)

if __name__ == '__main__':
    import time

    train(
        datafolder_path = 'data',
        batch_size = 128, 
        epochs = 100, 
        lr=1e-3,
        experiment_name=f'test-{int(round(time.time()))}'
    )