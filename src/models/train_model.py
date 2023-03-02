#!/usr/bin/python
# 
from model import MLP, get_loss_function, get_optimizer
from src.data.dataloader import *
import matplotlib.pyplot as plt
import torch
import torchvision

print("CNN Architecture:")
print(cnn_network)

criterion = get_loss_function()  # get loss function
optimizer = get_optimizer(cnn_network, lr=0.001, momentum=0.9)  # get optimizer

#  ---------------  Training  ---------------
def train(csv_file, n_epochs=100):
    """Trains the model.
    Args:
        csv_file (str): Absolute path of the dataset used for training.
        n_epochs (int): Number of epochs to train.
    """
    batch_size = 100          # Number of entries in each batch

    # Load dataset
    data = CatalanJuvenileJustice("./AlgorithmicFairness/catalan-juvenile-recidivism-subset-numeric.csv")
    data.get_loaders(batch_size, split_type = 'random')

 # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Dataloaders
    trainloader = DataLoader(trainset, batch_size=200, shuffle=True)
    testloader = DataLoader(testset, batch_size=200, shuffle=False)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model
    D_in, H = 19, 15
    net = MLP(D_in, H).to(device)

    # Loss function
    criterion = get_loss_function

    # Optimizer
    optimizer = get_optimizer(net)

    # Train the net
    loss_per_iter = []
    loss_per_batch = []


    num_epochs = 15           # Number of passes over the entire dataset
    print_every_iters = 100  # Print training loss every X mini-batches

    for epoch in range(num_epochs):

            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                # get the inputs
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = net(inputs.float())
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                # Save loss to plot
                running_loss += loss.item()
                loss_per_iter.append(loss.item())

                if (i + 1) % print_every_iters == 0:
                    print(
                        f'[Epoch: {epoch + 1} / {num_epochs},'
                        f' Iter: {i + 1:5d} / {len(trainloader)}]'
                        f' Training loss: {running_loss / (i + 1):.3f}'
                    )

            loss_per_batch.append(running_loss / (i + 1))
            running_loss = 0.0
            mean_loss = running_loss / len(trainloader)
            training_loss_per_epoch.append(mean_loss)

    # Comparing training to test
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = net(inputs.float())
    print("Root mean squared error")
    print("Training:", np.sqrt(loss_per_batch[-1]))
    print("Test", np.sqrt(criterion(labels.float(), outputs).detach().cpu().numpy()))

    # Plot training loss curve
    plt.plot(np.arange(len(loss_per_iter)), loss_per_iter, "-", alpha=0.5, label="Loss per epoch")
    plt.plot(np.arange(len(loss_per_iter), step=4) + 3, loss_per_batch, ".-", label="Loss per mini-batch")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()