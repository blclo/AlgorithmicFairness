#!/usr/bin/python
# 
from model import CNN, get_loss_function, get_optimizer
from model import get_transforms_val, get_transforms_train, CNN, get_loss_function, get_optimizer
from src.data.dataloader import *
import torch
import torchvision

cnn_network = CNN()  # create CNN model

print("CNN Architecture:")
print(cnn_network)

criterion = get_loss_function()  # get loss function
optimizer = get_optimizer(cnn_network, lr=0.001, momentum=0.9)  # get optimizer

# *********************************************************** #
# set all training parameters. You can play around with these
# *********************************************************** #

batch_size = 100          # Number of images in each batch
learning_rate = 0.001    # Learning rate in the optimizer
momentum = 0.9           # Momentum in SGD
num_epochs = 15           # Number of passes over the entire dataset
print_every_iters = 100  # Print training loss every X mini-batches


# *********************************************************** #
# Initialize all the training components, e.g. model, cost function
# *********************************************************** #

# Get transforms
transform_train = get_transforms_train()
transform_val = get_transforms_val()

# Generate our data loaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size, transform_train, transform_val)

# create CNN model
cnn_network = CNN()  

# Get optimizer and loss functions
criterion = get_loss_function() 
optimizer = get_optimizer(cnn_network, lr=learning_rate, momentum=momentum) 

# *********************************************************** #
# The main training loop. You dont need to change this
# *********************************************************** #
training_loss_per_epoch = []
val_loss_per_epoch = []
for epoch in range(num_epochs):  # loop over the dataset multiple times
    # First we loop over training dataset
    running_loss = 0.0

    # Set network to train mode before training
    cnn_network.train()
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()  # zero the gradients from previous iteration

        # forward + backward + optimize
        outputs = cnn_network(inputs)  # forward pass to obtain network outputs
        loss = criterion(outputs, labels)  # compute loss with respect to labels
        loss.backward()  # compute gradients with backpropagation (autograd)
        optimizer.step()  # optimize network parameters

        # print statistics
        running_loss += loss.item()
        if (i + 1) % print_every_iters == 0:
            print(
                f'[Epoch: {epoch + 1} / {num_epochs},'
                f' Iter: {i + 1:5d} / {len(train_loader)}]'
                f' Training loss: {running_loss / (i + 1):.3f}'
            )
    
    mean_loss = running_loss / len(train_loader)
    training_loss_per_epoch.append(mean_loss)

    # Next we loop over validation dataset
    running_loss = 0.0

    # Set network to eval mode before validation
    cnn_network.eval()
    for i, data in enumerate(val_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # on validation dataset, we only do forward, without computing gradients
        with torch.no_grad():
            outputs = cnn_network(inputs)  # forward pass to obtain network outputs
            loss = criterion(outputs, labels)  # compute loss with respect to labels
        
        # print statistics
        running_loss += loss.item()

    mean_loss = running_loss / len(val_loader)
    val_loss_per_epoch.append(mean_loss)

    print(
        f'[Epoch: {epoch + 1} / {num_epochs}]'
        f' Validation loss: {mean_loss:.3f}'
    )

print('Finished Training')

# Plot the training curves
plt.figure()
plt.plot(np.array(training_loss_per_epoch))
plt.plot(np.array(val_loss_per_epoch))
plt.legend(['Training loss', 'Val loss'])
plt.xlabel('Epoch')
plt.show()
plt.close()
'''