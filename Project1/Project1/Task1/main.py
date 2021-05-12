import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
from Network1 import Network1, init_xavier, fit_custom
from sklearn.model_selection import KFold
from Datahandler import Datahandler


def run_configuration(conf_dict, x, y, meta, k_folds=5):
    """
    run a k-fold cross validation with a given set of parameters
    :param conf_dict: contains the parameters for the network and the training
    :param x: predictors
    :param y: targets
    :param meta: dictionary with total number of configurations to run and the current configuration
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

    model = Network1(input_dimension=1,
                     output_dimension=1,
                     n_hidden_layers=n_hidden_layers,
                     neurons=neurons,
                     regularization_param=regularization_param,
                     regularization_exp=regularization_exp)

    # Xavier weight initialization

    if opt_type == "ADAM":
        optimizer_ = optim.Adam(model.parameters(), lr=0.001)
    elif opt_type == "LBFGS":
        optimizer_ = optim.LBFGS(model.parameters(), lr=0.1, max_iter=1, max_eval=50000,
                                 tolerance_change=1.0 * np.finfo(float).eps)
    else:
        raise ValueError("Optimizer not recognized")

    kfold = KFold(n_splits=k_folds, shuffle=True)
    # K-fold Cross Validation model evaluation

    training_loss_total = 0.0
    validation_loss_total = 0.0

    meta['total_folds'] = k_folds
    meta['current_fold'] = 0

    for fold, (train_ids, test_ids) in enumerate(kfold.split(x)):

        meta['current_fold'] = fold

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
                                                              n_epochs, optimizer_, meta, p=2, output_step=n_epochs/10)
        training_loss_total += fold_training_loss
        validation_loss_total += fold_validation_loss

        print(f"Fold {fold} Training Loss: {fold_training_loss/ k_folds}")
        print(f"Fold {fold} Validation Loss: {fold_validation_loss/ k_folds}")

    training_loss_total = training_loss_total / k_folds
    validation_loss_total = validation_loss_total / k_folds

    print('K-Fold Crossvalidation')
    print('-------------------------------')
    print('training loss')
    print(training_loss_total)
    print('validation loss')
    print(validation_loss_total)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: Write parameters and training + validation loss to file. Name file with date and time (dd:mm:yy_hh:mm:ss)

    return training_loss_total, validation_loss_total


if __name__ == "__main__":

    datahandler = Datahandler('TrainingData.txt')

    network_properties = {
        "hidden_layers": [16],
        "neurons": [40],
        "regularization_exp": [2],
        "regularization_param": [0],
        "batch_size": [20],
        "epochs": [100],
        "optimizer": ["LBFGS"],
        "init_weight_seed": [567]
    }

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
            "init_weight_seed": setup[7]
        }

        meta['current_conf'] = set_num

        print(setup_properties)
        relative_error_train_, relative_error_val_ = run_configuration(setup_properties,
                                                                       datahandler.get_predictors(),
                                                                       datahandler.get_targets('tf0'),
                                                                       meta)

        train_err_conf.append(relative_error_train_)
        val_err_conf.append(relative_error_val_)

    print(train_err_conf, val_err_conf)

    # TODO: create predictions from Testing Data and create final file

    # TODO: create a class to handle IO
    # TODO: get best validation error and store in best temp file
    # TODO: compare best validation error with overall best validation error and update file if needed
    # TODO: save best model
    """
    - initial setup to create all files
    - update function for loss_running_best_model and loss_best_model
        update_best_running(training_loss, validation_loss, model)
    - update function to compare and evaluate best model overall
        update_best_model()
        read both loss_running_best_model.txt and loss_best_model
        compare them
        update if necessary
    - load stored model
    - save model
    
    """

    """
    dir: files
    
    run_dd:mm:yy_hh:mm:ss.txt:
            file for run_conf to write down all parameters and corresponding losses
    loss_running_best_model.txt
        file for run_conf to write down the current best model(param + losses). 
        Can be evaluated during training
    running_best_model.pt
        file to store the best running model
    loss_best_model.txt
        file for main to write down the best model overall (param + losses). 
        Need to compare this file to the loss_running_best_model.txt and update if needed
    best_model.pt
        file to store the best overall model
    """

    train_err_conf = np.array(train_err_conf)
    val_err_conf = np.array(val_err_conf)

    configuration_number = np.linspace(start=0.0, stop=len(train_err_conf), num=len(train_err_conf), endpoint=False)

    print(configuration_number)

    plt.plot(configuration_number, train_err_conf, label="Training Error")
    plt.plot(configuration_number, val_err_conf, label="Validation Error")
    plt.legend()
    plt.xlabel("Configuration")
    plt.ylabel("Loss")
    # plt.show()
