import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

# https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_prob=0.0, regularization_param=0.0,
                 regularization_exp=2):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.n_layers = n_layers
        #
        self.regularization_param = regularization_param
        #
        self.regularization_exp = regularization_exp

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):

        r_out, hidden = self.lstm(x, hidden)

        out_drop = self.dropout(r_out)

        out = out_drop.contiguous().view(-1, self.hidden_dim)

        output = self.fc(out)

        return output, hidden
    """
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)

        # Initialize cell state
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.dropout(out)

        out = out.contiguous().view(-1, self.hidden_dim)

        output = self.fc(out)

        return output
    """

    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    reg_loss = torch.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += torch.norm(param, p)
    return reg_loss



def fit_custom(model, training_set, validation_set, num_epochs, optimizer, meta, batch_size=1, p=2, output_step=0):
    """
    Adapted from fit function provided by the Deep Learning for scientific computing team @ ETHZ in FS 2021
    :param model: pytorch neural network
    :param training_set:  compatible pytorch tensor to train model
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

    pbar = tqdm(range(num_epochs), desc=description)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    for epoch in pbar:

        hidden = list([model.init_hidden(batch_size, device)])

        running_loss = list([0])

        # Loop over batches
        for j, (x_train_, u_train_) in enumerate(training_set):
            x_train_ = x_train_.to(device)
            u_train_ = u_train_.to(device)

            def closure():

                hidden[0] = tuple([each.data for each in hidden[0]])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                u_pred_, hidden[0] = model(x_train_, hidden[0])

                # u_pred_ = model(x_train_)

                loss_u = torch.mean((u_pred_.reshape(-1, ) - u_train_.reshape(-1, )) ** p)
                loss_reg = regularization(model, regularization_exp)
                loss = loss_u + regularization_param * loss_reg
                loss.backward()
                # Compute average training loss over batches for the current epoch
                running_loss[0] += loss.item() * x_train_.size(0)
                return loss

            optimizer.step(closure=closure)

        history[0].append(running_loss[0] / len(training_set))

        # Evaluation
        model.eval()

        running_validation_loss = 0

        # Iterate over the test data and generate predictions
        hidden_val = model.init_hidden(1, device)
        for i, data in enumerate(validation_set, 0):
            hidden_val = tuple([each.data for each in hidden_val])
            # Get inputs
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Generate outputs
            prediction, hidden_val = model(inputs, hidden_val)

            running_validation_loss += torch.mean((prediction.reshape(-1, )
                                                   - targets.reshape(-1, )) ** p).item() * inputs.size(0)

        history[1].append(running_validation_loss / len(validation_set))

        model.train()

        if output_step != 0 and epoch % output_step == 0:
            pbar.set_postfix({'Training loss': history[0][-1], 'Validation loss': history[1][-1]})


    '''
    # Plot trainings process
    
    print('Fold Training Loss: ', history[0])
    print('Fold Validation Loss', history[1])

    configuration_number = np.linspace(start=0.0, stop=len(history[0]), num=len(history[0]), endpoint=False)

    plt.plot(configuration_number[1:], history[0][1:], label="Training Error")
    plt.plot(configuration_number[1:], history[1][1:], label="Validation Error")
    plt.legend()
    plt.xlabel("Configuration")
    plt.ylabel("Loss")
    plt.show()
    '''

    return history[0][-1], history[1][-1]
