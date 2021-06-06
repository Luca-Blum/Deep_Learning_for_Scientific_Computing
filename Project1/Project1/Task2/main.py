import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import KFold
from pipeline2 import Datahandler, Network1, init_xavier, fit_custom, IOHandler
from os import path


def run_configuration(conf_dict, x, y, meta_info, io_handler, k_folds=5):
    """
    run a k-fold cross validation with a given set of parameters
    :param conf_dict: contains the parameters for the network and the training
    :param x: predictors
    :param y: targets
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

    if activation_type == "sigmoid":
        activation = nn.Sigmoid()
    elif activation_type == "tanh":
        activation = nn.Tanh()
    else:
        activation = nn.ReLU()

    model = Network1(input_dimension=8,
                     output_dimension=1,
                     n_hidden_layers=n_hidden_layers,
                     neurons=neurons,
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

    kfold = KFold(n_splits=k_folds, shuffle=True)
    # K-fold Cross Validation model evaluation

    training_loss_total = 0.0
    validation_loss_total = 0.0

    meta_info['total_folds'] = k_folds
    meta_info['current_fold'] = 0

    for fold, (train_ids, test_ids) in enumerate(kfold.split(x, y)):

        meta_info['current_fold'] = fold

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        training_set = DataLoader(torch.utils.data.TensorDataset(x, y),
                                  batch_size=batch_size,
                                  sampler=train_subsampler)

        validation_set = DataLoader(torch.utils.data.TensorDataset(x, y),
                                    batch_size=batch_size,
                                    sampler=test_subsampler)

        init_xavier(model, retrain)

        fold_training_loss, fold_validation_loss = fit_custom(model, training_set, validation_set,
                                                              n_epochs, optimizer_, meta_info, p=2,
                                                              output_step=1)
        training_loss_total += fold_training_loss
        validation_loss_total += fold_validation_loss

        # print(f"Fold {fold + 1} Training Loss: {fold_training_loss}")
        # print(f"Fold {fold + 1} Validation Loss: {fold_validation_loss}")

    training_loss_total = training_loss_total / k_folds
    validation_loss_total = validation_loss_total / k_folds

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
            "dropout": setup[9]
        }

        meta['current_conf'] = set_num

        print(setup_properties)

        predictors, targets = datahandler.get_frame(iohandler.get_name())

        relative_error_train_, relative_error_val_ = run_configuration(setup_properties,
                                                                       predictors,
                                                                       targets,
                                                                       meta, iohandler)

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
    training_filename_101 = path.join(dirname, 'data/TrainingData_101.txt')
    training_filename_401 = path.join(dirname, 'data/TrainingData_401.txt')
    training_filename_1601 = path.join(dirname, 'data/TrainingData_1601.txt')
    testing_filename = path.join(dirname, 'data/TestingData.txt')

    datahandler = Datahandler([training_filename_101, training_filename_401, training_filename_1601],
                              ['101', '401', '1601'],
                              testing_filename)

    properties_101 = {
        "hidden_layers": [4],
        "neurons": [100],
        "regularization_exp": [1],
        "regularization_param": [1e-5],
        "batch_size": [16],
        "epochs": [1000],
        "optimizer": ["ADAM"],
        "init_weight_seed": [70],
        "activation": ['tanh'],
        "dropout": [0.1]
    }
    properties_401 = {
        "hidden_layers": [12],
        "neurons": [120, 180],
        "regularization_exp": [2],
        "regularization_param": [1e-4],
        "batch_size": [8],
        "epochs": [1000],
        "optimizer": ["ADAM"],
        "init_weight_seed": [19],
        "activation": ['tanh', 'sigmoid'],
        "dropout": [0.0]
    }
    properties_1601 = {
        "hidden_layers": [16],
        "neurons": [100, 200],
        "regularization_exp": [2],
        "regularization_param": [1e-5],
        "batch_size": [4, 32],
        "epochs": [1000],
        "optimizer": ["ADAM"],
        "init_weight_seed": [1],
        "activation": ['tanh', 'sigmoid'],
        "dropout": [0.1]
    }

    iohandler_101 = IOHandler('101')
    iohandler_401 = IOHandler('401')
    iohandler_1601 = IOHandler('1601')

    train_predictor(iohandler_101, properties_101)
    train_predictor(iohandler_401, properties_401)
    train_predictor(iohandler_1601, properties_1601)

    datahandler.create_submission([iohandler_101.load_best_model(),
                                   iohandler_401.load_best_model(),
                                   iohandler_1601.load_best_model()])
