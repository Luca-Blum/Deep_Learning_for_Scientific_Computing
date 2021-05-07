import pandas as pd
import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from Network1 import Network1

"""
Adapted from https://averdones.github.io/reading-tabular-data-with-pytorch-and-training-a-multilayer-perceptron/
"""
class Datahandler():
    def __init__(self, txt_file: str, type: str):
        """
        :param txt_file(str): Path to the txt file with the data
        :param type(str): tf0 or ts0
        """
        self.df = pd.read_csv(txt_file)
        self.target = type

        print(self.df.dtypes)

        self.X = self.df['t']
        self.Y = self.df[self.target]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return [self.X.iloc[idx].values, self.Y[idx]]

if __name__ == "__main__":
    dataset = Datahandler('TrainingData.txt','ts0')

    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=20, shuffle=True)
    testloader = DataLoader(testset, batch_size=20, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = Network1(input_dimension=1,
                           output_dimension=1,
                           n_hidden_layers=5,
                           neurons=20,
                           regularization_param=10e-5,
                           regularization_exp=2)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    n_epochs = 4

    # Train the net
    loss_per_iter = []
    loss_per_batch = []
    for epoch in range(n_epochs):

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()
            loss_per_iter.append(loss.item())

        loss_per_batch.append(running_loss / (i + 1))
        running_loss = 0.0

    # Comparing training to test
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs.float())
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