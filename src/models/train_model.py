#!/usr/bin/python
# 

import torch
from torch.utils.tensorboard import SummaryWriter

import time
from tqdm import trange

from src.models.model import MLP, FullyConnected, get_loss_function, get_optimizer
from src.data.dataloader import CatalanJuvenileJustice

def set_seed(seed: int):
    torch.manual_seed(seed)

#  ---------------  Training  ---------------
def train(
        datafolder_path: str,
        batch_size: int = 128, num_workers: int = 1, test_proportion: float = 0.2, val_proportion: float = 0.2, split_type: str = 'random',
        lr=1e-3, epochs: int = 100, loss_type: str = 'BCE', optimizer: str = 'SGD', momentum: float = 0.9,
        experiment_name: str = str(int(round(time.time()))),
        seed: int = 42,
    ):
    """
    Trains the model.

    Args:
        datafolder_path (str): _description_
        batch_size (int, optional): _description_. Defaults to 128.
        num_workers (int, optional): _description_. Defaults to 1.
        test_proportion (float, optional): _description_. Defaults to 0.2.
        val_proportion (float, optional): _description_. Defaults to 0.2.
        split_type (str, optional): _description_. Defaults to 'random'.
        lr (_type_, optional): _description_. Defaults to 1e-3.
        epochs (int, optional): _description_. Defaults to 100.
        experiment_name (str, optional): _description_. Defaults to str(int(round(time.time()))).
    """
    # Set seed
    set_seed(seed)

    # Load dataset
    dataset = CatalanJuvenileJustice(
        data_path=f"{datafolder_path}/processed/catalan_dataset.pth"
    )

    # Split into training and test
    train_loader, val_loader, _ = dataset.get_loaders(
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
    model = FullyConnected(channels_in = dataset.n_attributes, channels_out = 1).to(device)
    criterion = get_loss_function(type=loss_type)
    optimizer = get_optimizer(model, type=optimizer, lr=lr, momentum=momentum)

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
                y_pred = model(inputs)

                loss = criterion(y_pred.float(), labels.float())
                running_loss_train += loss.item()
                loss.backward()

                # Optimize
                optimizer.step()
                
                # Store accuracy
                equals = (y_pred >= 0.5) == labels.view(*y_pred.shape)
                running_acc_train += torch.mean(equals.type(torch.FloatTensor))

            # Validation
            with torch.no_grad():
                for batch in iter(val_loader):
                    inputs, labels = batch['data'].to(device), batch['label'].to(device)

                    # Get predictions
                    y_pred = model(inputs)
        
                    # Compute loss and accuracy
                    running_loss_val += criterion(y_pred.float(), labels.float())
                    equals = (y_pred >= 0.5) == labels.view(*y_pred.shape)
                    running_acc_val += torch.mean(equals.type(torch.FloatTensor))

            # Update progress bar
            train_loss_descr = (
                f"Train loss: {running_loss_train / len(train_loader):.3f}"
            )
            val_loss_descr = (
                f"Validation loss: {running_loss_val / len(val_loader):.3f}"
            )
            train_acc_descr = (
                f"Train accuracy: {running_acc_train / len(train_loader):.3f}"
            )
            val_acc_descr = (
                f"Validation accuracy: {running_acc_val / len(val_loader):.3f}"
            )
            t.set_description(
                f"EPOCH [{epoch + 1}/{epochs}] --> {train_loss_descr} | {val_loss_descr} | {train_acc_descr} | {val_acc_descr} | Progress: "
            )

            writer.add_scalar(f'{loss_type}/train',         running_loss_train  / len(train_loader),    epoch)
            writer.add_scalar('accuracy/train',             running_acc_train   / len(train_loader),    epoch)
            writer.add_scalar(f'{loss_type}/validation',    running_loss_val    / len(val_loader),      epoch)
            writer.add_scalar('accuracy/validation',        running_acc_val     / len(val_loader),      epoch)


if __name__ == '__main__':

    train(
        #datafolder_path = 'projects/AlgorithmicFairness/data',
        datafolder_path = 'data',
        batch_size = 128, 
        epochs = 100, 
        lr=1e-3,
        loss_type='BCE',
        optimizer='Adam',
        experiment_name=f'net3.lr1e-4.BCE.Adam.{int(round(time.time()))}'
    )