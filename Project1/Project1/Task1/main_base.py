from Network1 import Network1, fit, init_xavier
import torch
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import itertools


# Define the exact solution
def exact_solution(x):
    return torch.sin(x)


def run_configuration(conf_dict, x, y):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print(conf_dict)

    # Get the confgiuration to test
    opt_type = conf_dict["optimizer"]
    n_epochs = conf_dict["epochs"]
    n_hidden_layers = conf_dict["hidden_layers"]
    neurons = conf_dict["neurons"]
    regularization_param = conf_dict["regularization_param"]
    regularization_exp = conf_dict["regularization_exp"]
    retrain = conf_dict["init_weight_seed"]
    batch_size = conf_dict["batch_size"]

    validation_size = int(20 * x.shape[0] / 100)
    training_size = x.shape[0] - validation_size
    x_train = x[:training_size, :]
    y_train = y[:training_size, :]

    x_val = x[training_size:, :]
    y_val = y[training_size:, :]

    training_set = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    my_network = Network1(input_dimension=x.shape[1],
                           output_dimension=y.shape[1],
                           n_hidden_layers=n_hidden_layers,
                           neurons=neurons,
                           regularization_param=regularization_param,
                           regularization_exp=regularization_exp)

    # Xavier weight initialization
    init_xavier(my_network, retrain)

    if opt_type == "ADAM":
        optimizer_ = optim.Adam(my_network.parameters(), lr=0.001)
    elif opt_type == "LBFGS":
        optimizer_ = optim.LBFGS(my_network.parameters(), lr=0.1, max_iter=1, max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps)
    else:
        raise ValueError("Optimizer not recognized")

    history = fit(my_network, training_set, x_val, y_val, n_epochs, optimizer_, p=2, verbose=False)

    x_test = torch.linspace(0, 2 * np.pi, 10000).reshape(-1, 1)
    y_test = exact_solution(x_test).reshape(-1, )
    y_val = y_val.reshape(-1, )
    y_train = y_train.reshape(-1, )

    y_test_pred = my_network(x_test).reshape(-1, )
    y_val_pred = my_network(x_val).reshape(-1, )
    y_train_pred = my_network(x_train).reshape(-1, )

    # Compute the relative validation error
    relative_error_train = torch.mean((y_train_pred - y_train) ** 2) / torch.mean(y_train ** 2)
    print("Relative Training Error: ", relative_error_train.detach().numpy() ** 0.5 * 100, "%")

    # Compute the relative validation error
    relative_error_val = torch.mean((y_val_pred - y_val) ** 2) / torch.mean(y_val ** 2)
    print("Relative Validation Error: ", relative_error_val.detach().numpy() ** 0.5 * 100, "%")

    # Compute the relative L2 error norm (generalization error)
    relative_error_test = torch.mean((y_test_pred - y_test) ** 2) / torch.mean(y_test ** 2)
    print("Relative Testing Error: ", relative_error_test.detach().numpy() ** 0.5 * 100, "%")

    return relative_error_train.item(), relative_error_val.item(), relative_error_test.item()


# Random Seed for dataset generation
sampling_seed = 78
torch.manual_seed(sampling_seed)

# Number of training samples
n_samples = 100
# Noise level
sigma = 0.0

x = 2 * np.pi * torch.rand((n_samples, 1))
y = exact_solution(x) + sigma * torch.randn(x.shape)

print(x,y)

network_properties = {
    "hidden_layers": [2, 4],
    "neurons": [5, 20],
    "regularization_exp": [2],
    "regularization_param": [0, 1e-4],
    "batch_size": [n_samples],
    "epochs": [1000],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [567, 34, 134]
}

settings = list(itertools.product(*network_properties.values()))

i = 0

train_err_conf = list()
val_err_conf = list()
test_err_conf = list()
for set_num, setup in enumerate(settings):
    print("###################################", set_num, "###################################")
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

    relative_error_train_, relative_error_val_, relative_error_test_ = run_configuration(setup_properties, x, y)
    train_err_conf.append(relative_error_train_)
    val_err_conf.append(relative_error_val_)
    test_err_conf.append(relative_error_test_)

print(train_err_conf, val_err_conf, test_err_conf)

train_err_conf = np.array(train_err_conf)
val_err_conf = np.array(val_err_conf)
test_err_conf = np.array(test_err_conf)

plt.figure(figsize=(16, 8))
plt.grid(True, which="both", ls=":")
plt.scatter(np.log10(train_err_conf), np.log10(test_err_conf), marker="*", label="Training Error")
plt.scatter(np.log10(val_err_conf), np.log10(test_err_conf), label="Validation Error")
plt.xlabel("Selection Criterion")
plt.ylabel("Generalization Error")
plt.title(r'Validation - Training Error VS Generalization error ($\sigma=0.0$)')
plt.legend()
plt.savefig("sigma.png", dpi=400)
plt.show()






