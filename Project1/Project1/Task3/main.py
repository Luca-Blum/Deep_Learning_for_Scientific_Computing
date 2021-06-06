import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import KFold, train_test_split
from pipeline3 import Datahandler, LSTM, init_xavier, fit_custom, IOHandler, LSTM_stateless, fit_stateless, RNN, GRU
from os import path


def run_configuration(conf_dict, x, y, meta_info, io_handler, k_folds=5):
    """
    run a k-fold cross validation with a given set of parameters
    :param conf_dict: contains the parameters for the network and the training
    :param x: time series features
    :param y: time series labels
    :param meta_info: dictionary with total number of configurations to run and the current configuration
    :param io_handler: keeps track of best model during training
    :param k_folds: number of folds for cross validation
    :return: training loss and validation loss
    Adapted from skeleton code from Deep Learning for Scientific Computing lecture @ ETHZ in FS2021
    """

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Get the confgiuration to test
    opt_type = conf_dict["optimizer"]
    n_epochs = conf_dict["epochs"]
    n_hidden_layers = conf_dict["hidden_layers"]
    neurons = conf_dict["neurons"]
    regularization_param = conf_dict["regularization_param"]
    regularization_exp = conf_dict["regularization_exp"]
    retrain = conf_dict["init_weight_seed"]
    batch_size = conf_dict["batch_size"]
    activation_type = conf_dict["activation"]
    dropout = conf_dict["dropout"]
    state = conf_dict["state"]

    if activation_type == "sigmoid":
        activation = nn.Sigmoid()
    elif activation_type == "tanh":
        activation = nn.Tanh()
    else:
        activation = nn.ReLU()

    if state == "stateless":
        model = LSTM_stateless(input_size=1,
                               output_size=1,
                               hidden_dim=neurons,
                               n_layers=n_hidden_layers,
                               regularization_param=regularization_param,
                               regularization_exp=regularization_exp,
                               activation=activation,
                               dropout=dropout)
    elif state == "rnn":
        model = RNN(input_size=1,
                               output_size=1,
                               hidden_dim=neurons,
                               n_layers=n_hidden_layers,
                               regularization_param=regularization_param,
                               regularization_exp=regularization_exp,
                               activation=activation,
                               dropout=dropout)
    elif state == "gru":
        model = GRU(input_size=1,
                               output_size=1,
                               hidden_dim=neurons,
                               n_layers=n_hidden_layers,
                               regularization_param=regularization_param,
                               regularization_exp=regularization_exp,
                               activation=activation,
                               dropout=dropout)
    else:
        model = LSTM(input_size=1,
                     output_size=1,
                     hidden_dim=neurons,
                     n_layers=n_hidden_layers,
                     regularization_param=regularization_param,
                     regularization_exp=regularization_exp,
                     activation=activation,
                     dropout=dropout)

    if opt_type == "ADAM":
        optimizer_ = optim.Adam(model.parameters(), lr=0.0001)
    elif opt_type == "LBFGS":
        optimizer_ = optim.LBFGS(model.parameters(), lr=0.1, max_iter=1, max_eval=50000,
                                 tolerance_change=1.0 * np.finfo(float).eps)
    elif opt_type == "SGD":
        optimizer_ = optim.SGD(model.parameters(), lr=0.1)
    else:
        raise ValueError("Optimizer not recognized")

    training_loss_total = 0.0
    validation_loss_total = 0.0

    meta_info['total_folds'] = k_folds
    meta_info['current_fold'] = 0

    incr = 0.5 / k_folds

    splits = [0.5 + (fold+1) * incr for fold in range(k_folds)]

    for i, split in enumerate(splits):

        meta_info['current_fold'] = i

        cut = int(len(x) * split)

        x_temp = x[:cut]
        y_temp = y[:cut]

        x_train, x_test, y_train, y_test = train_test_split(x_temp, y_temp, test_size=0.1, random_state=42,
                                                            shuffle=False)

        training_set = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                  shuffle=False, drop_last=True)

        validation_set = DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False,
                                    drop_last=False)

        init_xavier(model, retrain)

        if state in ["stateless", "rnn", "gru"]:
            fold_training_loss, fold_validation_loss = fit_stateless(model, training_set, validation_set, n_epochs,
                                                                     optimizer_, meta_info, batch_size=batch_size,
                                                                     p=2, output_step=1)
        else:
            fold_training_loss, fold_validation_loss = fit_custom(model, training_set, validation_set, n_epochs,
                                                                  optimizer_, meta_info, batch_size=batch_size,
                                                                  p=2, output_step=1)

        training_loss_total += fold_training_loss
        validation_loss_total += fold_validation_loss

        print(f"Fold {i + 1} Training Loss: {fold_training_loss}")
        print(f"Fold {i + 1} Validation Loss: {fold_validation_loss}")

    training_loss_total = training_loss_total / len(splits)
    validation_loss_total = validation_loss_total / len(splits)

    print('K-Fold Crossvalidation')
    print('-------------------------------')
    print('training loss: \t\t', training_loss_total)
    print('validation loss: \t', validation_loss_total, '\n')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    io_handler.write_running(training_loss_total, validation_loss_total, conf_dict, model)

    return training_loss_total, validation_loss_total


def train_predictor(iohandler, network_properties, debug=False):
    """
    :param iohandler: handler for the input and output of the models
    :param network_properties: dictionary of different configurations
    :param debug: plot losses of different configurations
    :return: trains the NN with the given configurations and calculates the training and validation loss
    """

    settings = list(itertools.product(*network_properties.values()))

    train_err_conf = list()
    val_err_conf = list()

    number_of_conf = len(settings)

    meta = {'total_confs': number_of_conf,
            'current_conf': 0}

    for set_num, setup in enumerate(settings):
        setup_properties = {
            "hidden_layers": setup[0],
            "neurons": setup[1],
            "regularization_exp": setup[2],
            "regularization_param": setup[3],
            "batch_size": setup[4],
            "epochs": setup[5],
            "optimizer": setup[6],
            "init_weight_seed": setup[7],
            "activation": setup[8],
            "dropout": setup[9],
            "state": setup[10]
        }

        meta['current_conf'] = set_num

        print(setup_properties)

        x, y = datahandler.get_data(iohandler.get_name())

        relative_error_train_, relative_error_val_ = run_configuration(setup_properties, x, y, meta, iohandler, k_folds=1)

        train_err_conf.append(relative_error_train_)
        val_err_conf.append(relative_error_val_)

    # print(train_err_conf, val_err_conf)

    iohandler.finalize()

    train_err_conf = np.array(train_err_conf)
    val_err_conf = np.array(val_err_conf)

    if debug:
        configuration_number = np.linspace(start=0.0, stop=len(train_err_conf), num=len(train_err_conf), endpoint=False)

        plt.plot(configuration_number, train_err_conf, label="Training Error")
        plt.plot(configuration_number, val_err_conf, label="Validation Error")
        plt.legend()
        plt.xlabel("Configuration")
        plt.ylabel("Loss")
        plt.show()


if __name__ == "__main__":

    dirname = path.dirname(__file__)
    training_filename = path.join(dirname, 'data/TrainingData.txt')
    testing_filename = path.join(dirname, 'data/TestingData.txt')

    datahandler = Datahandler(training_filename, testing_filename)

    iohandler_tf0 = IOHandler('tf0')
    iohandler_ts0 = IOHandler('ts0')

    network_properties_tf0 = {
        "hidden_layers": [4],
        "neurons": [100],
        "regularization_exp": [2],
        "regularization_param": [1e-5],
        "batch_size": [1],
        "epochs": [100],
        "optimizer": ["ADAM"],
        "init_weight_seed": [34],
        "activation": ['relu'],
        "dropout": [0.0],
        "state": ['stateful']
    }

    network_properties_ts0 = {
        "hidden_layers": [4],
        "neurons": [100],
        "regularization_exp": [2],
        "regularization_param": [0],
        "batch_size": [1],
        "epochs": [100],
        "optimizer": ["ADAM"],
        "init_weight_seed": [1],
        "activation": ['relu'],
        "dropout": [0.0],
        "state": ['stateful']
    }

    train_predictor(iohandler_tf0, network_properties_tf0)
    train_predictor(iohandler_ts0, network_properties_ts0)

    datahandler.create_submission(iohandler_tf0.load_best_model(), iohandler_ts0.load_best_model(), 'stateful')
