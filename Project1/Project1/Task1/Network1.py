import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Network1(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param,
                 regularization_exp):
        """
        Creating a pytorch dense neural network
        :param input_dimension: dimension of the predictors
        :param output_dimension: dimensiont of the target
        :param n_hidden_layers: number of hidden layers
        :param neurons: number of neurons in each hidden layer
        :param regularization_param: strength of regularization
        :param regularization_exp: norm for regularization
        """
        super(Network1, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = nn.Tanh()
        #
        self.regularization_param = regularization_param
        #
        self.regularization_exp = regularization_exp

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)


def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            # torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


def fit(model, training_set, x_validation_, y_validation_, num_epochs, optimizer, p, verbose=True):
    history = [[], []]
    regularization_param = model.regularization_param
    regularization_exp = model.regularization_exp

    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose:
            print("################################ ", epoch, " ################################")

        running_loss = list([0])

        # Loop over batches
        for j, (x_train_, u_train_) in enumerate(training_set):
            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                u_pred_ = model(x_train_)
                loss_u = torch.mean((u_pred_.reshape(-1, ) - u_train_.reshape(-1, )) ** p)
                loss_reg = regularization(model, regularization_exp)
                loss = loss_u + regularization_param * loss_reg
                loss.backward()
                # Compute average training loss over batches for the current epoch
                running_loss[0] += loss.item() / len(training_set)
                return loss

            optimizer.step(closure=closure)

        y_validation_pred_ = model(x_validation_)
        validation_loss = torch.mean((y_validation_pred_.reshape(-1, ) - y_validation_.reshape(-1, )) ** p).item()
        history[0].append(running_loss[0])
        history[1].append(validation_loss)

        if verbose:
            print('Training Loss: ', np.round(running_loss[0], 8))
            print('Validation Loss: ', np.round(validation_loss, 8))

    print('Final Training Loss: ', np.round(history[0][-1], 8))
    print('Final Validation Loss: ', np.round(history[1][-1], 8))
    return history


def fit_custom(model, training_set, validation_set, num_epochs, optimizer, meta, p=2, output_step=0):
    """
    Adapted from fit function provided by the Deep Learning for scientific computing team @ ETHZ in FS 2021
    :param model: pytorch neural network
    :param training_set:  compatible pytorch tensor to train model
    :param validation_set: compatible pytorch tensor to evaluate model
    :param num_epochs: number of epochs to train model
    :param optimizer: type of optimizer [torch.optim.Adam, torch.optim.LBFGS]
    :param meta: dictionary with    total number of configurations to run,
                                    the current configuration,
                                    total number of folds,
                                    current fold
    :param p: p-norm use to calculate the loss
    :param output_step: epoch interval to output current training loss. No output = 0
    :return: trainings loss and validation loss
    """

    history = [[], []]
    regularization_param = model.regularization_param
    regularization_exp = model.regularization_exp

    # Loop over epochs

    description = f"Configuration: [{meta['current_conf']+1}/{meta['total_confs']}] " \
                  f"Folds: [{meta['current_fold']+1}/{meta['total_folds']}] " \
                  f"Epoch Progress: "

    for epoch in tqdm(range(num_epochs), desc=description):

        running_loss = list([0])

        # Loop over batches
        for j, (x_train_, u_train_) in enumerate(training_set):
            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                u_pred_ = model(x_train_)
                loss_u = torch.mean((u_pred_.reshape(-1, ) - u_train_.reshape(-1, )) ** p)
                loss_reg = regularization(model, regularization_exp)
                loss = loss_u + regularization_param * loss_reg
                loss.backward()
                # Compute average training loss over batches for the current epoch
                running_loss[0] += loss.item() * x_train_.size(0)
                return loss

            optimizer.step(closure=closure)

        running_loss[0] /= len(training_set.sampler)

        if output_step != 0 and epoch != 0 and epoch % output_step == 0:
            print(" LOSS: ", running_loss[0])

        history[0].append(running_loss[0])

    # Evaluation
    model.eval()

    validation_loss = 0

    # Iterate over the test data and generate predictions
    for i, data in enumerate(validation_set, 0):
        # Get inputs
        inputs, targets = data

        # Generate outputs
        prediction = model(inputs)

        validation_loss += torch.mean((prediction.reshape(-1, )
                                      - targets.reshape(-1, )) ** p).item() * inputs.size(0)

    validation_loss /= len(validation_set.sampler)
    model.train()
    '''
    print('Fold Training Loss: ', running_loss[0])
    print('Fold Validation Loss', validation_loss)

    configuration_number = np.linspace(start=0.0, stop=len(history[0]), num=len(history[0]), endpoint=False)

    plt.plot(configuration_number, history[0], label="Training Error")
    plt.legend()
    plt.xlabel("Configuration")
    plt.ylabel("Loss")
    # plt.show()
    '''

    return running_loss[0], validation_loss
