from os import path
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import torch


class IOHandler:
    def __init__(self, name: str):

        self.name = name

        basepath = path.dirname(__file__)

        self.new_training_cycle = True

        self.output_path = path.abspath(path.join(basepath, "..", "output", self.name))

        self.loss_best_model_path = path.join(self.output_path, 'loss_best_model_' + self.name + '.dat')
        self.best_model_path = path.join(self.output_path, 'best_model_' + self.name + '.pt')

        self.loss_best_running_model_path = path.join(self.output_path, 'loss_best_running_model_' + self.name + '.dat')
        self.best_running_model_path = path.join(self.output_path, 'best_running_model_' + self.name + '.pt')

        self.loss_running_model_log_path = path.join(self.output_path, 'log')
        self.loss_running_model_path = self.loss_running_model_log_path

        # Create file hierarchy if not exist
        if not Path(self.output_path).is_dir():
            Path(self.output_path).mkdir(parents=True, exist_ok=True)

        if not Path(self.loss_best_model_path).is_file():
            with open(self.loss_best_model_path, "w+") as f:
                f.write("Training and validation loss of the best model\n\n")
                f.write("Configuration:\n")
                f.write("Training Loss:\t\t\t" + str(np.finfo(float).max) + "\n")
                f.write("Validation Loss:\t\t" + str(np.finfo(float).max) + "\n")

        if not Path(self.loss_best_running_model_path).is_file():
            with open(self.loss_best_running_model_path, "w+") as f:
                f.write("Training and validation loss of the currently best running model\n\n")
                f.write("Configuration:\n")
                f.write("Training Loss: \t \t \t" + str(np.finfo(float).max) + "\n")
                f.write("Validation Loss: \t \t" + str(np.finfo(float).max) + "\n")

        self.validation_loss_running = np.finfo(float).max

    def write_running(self, training_loss, validation_loss, configuration, model):

        # Create new log file for the following training cycle
        if self.new_training_cycle:
            self.new_training_cycle = False

            if not Path(self.loss_running_model_log_path).is_dir():
                Path(self.loss_running_model_log_path).mkdir(parents=True, exist_ok=True)

            date_time = datetime.now()
            filename = str(date_time.day) + '-' + str(date_time.month) + '-' + str(date_time.year) + '_' + \
                       str(date_time.hour) + ':' + str(date_time.minute) + ':' + str(date_time.second) + ':' + \
                       str(date_time.microsecond) + '_' + self.name + '.dat'

            self.loss_running_model_path = path.join(self.loss_running_model_log_path, filename)

            with open(self.loss_running_model_path, "w+") as f:
                f.write("Training and validation loss of all running models\n\n")

        # Append configuration, training and validation loss to log file
        with open(self.loss_running_model_path, "a") as f:
            f.write("Configuration:\t\t\t")
            f.write(json.dumps(configuration))
            f.write("\n")
            f.write("Training Loss:\t\t\t" + str(training_loss) + "\n")
            f.write("Validation Loss:\t\t" + str(validation_loss) + "\n\n")

        # Store best model for current configuration set
        if self.validation_loss_running > validation_loss:
            with open(self.loss_best_running_model_path, "w+") as f:
                f.write("Training and validation loss of the currently best running model\n\n")
                f.write("Configuration: \t \t \t")
                f.write(json.dumps(configuration))
                f.write("\n")
                # f.write("Configuration: \t \t \t" + str(configuration) + "\n")
                f.write("Training Loss: \t \t \t" + str(training_loss) + "\n")
                f.write("Validation Loss: \t \t" + str(validation_loss) + "\n\n")

            self.validation_loss_running = validation_loss
            # store new model
            torch.save(model, self.best_running_model_path)

    def finalize(self):

        self.new_training_cycle = True

        with open(self.loss_best_model_path, "r+") as f_best:
            # Read header
            f_best.readline()
            f_best.readline()

            # Read configuration
            f_best.readline()

            # Read training and validation loss
            f_best.readline()
            validation_loss = float(f_best.readline().split()[2])

            # Compare if the running model is better
            if validation_loss > self.validation_loss_running:
                print("New best Model")
                with open(self.loss_best_running_model_path, "r+") as f_run:
                    f_run.readline()
                    f_run.readline()

                    f_best.seek(0)
                    # pointer to beginning
                    f_best.write("Training and validation loss of the best model\n\n")
                    f_best.write(f_run.read())

            # Store new best model
            torch.save(torch.load(self.best_running_model_path), self.best_model_path)

    def load_best_model(self):
        return torch.load(self.best_model_path)

    def get_name(self):
        return self.name
    """
    run_dd:mm:yy_hh:mm:ss.txt:
            file for run_configuration to write down all parameters and corresponding losses

    running_best_model.pt OTF
        file to store the best currently running model
    best_model.pt OTF
        file to store the best overall model
        
    loss_running_best_model.txt
        file for run_configuration to write down the current best running model(param + losses). 
 
    loss_best_model.txt
        file to write down the best model overall (param + losses). 
    """


if __name__ == '__main__':
    network_properties = {
        "hidden_layers": 16,
        "neurons": 40,
        "regularization_exp": 2,
        "regularization_param": 0,
        "batch_size": 20,
        "epochs": 100,
        "optimizer": "LBFGS",
        "init_weight_seed": 567
    }

    iohandler = IOHandler('ts0')
    iohandler.write_running(0,0,{'test':0},None)
    iohandler.finalize()
    m = iohandler.load_best_model()

    print(m)
