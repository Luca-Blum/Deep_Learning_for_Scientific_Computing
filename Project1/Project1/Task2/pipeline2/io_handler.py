from os import path
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import torch


class IOHandler:
    def __init__(self, name: str):
        """
        IO handler to store and load the currently best model that is running, the best model overall
        and log files for the corresponding parameters of the models and losses

        Every output is stored in a subfolder of the folder "output"
        The folder "output" contains subfolders for every newly specified iohandler name.
        This subfolder contains a "log" folder which stores the parameters and losses of every tested model.
        Additionally the subfolder contains the overall best model and the currently running best model
        and the corresponding parameters and losses
        """

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
        """
        Create new log file for the following training cycle
        :param training_loss: loss of training set
        :param validation_loss: loss of validation set
        :param configuration: configuration parameters
        :param model: neural network
        """

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
        """
        Finalizes the current test run. Compares the current best model with the past best model
        and updates it if necessary.
        """

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
        """
        :return: best tested model
        """
        print("loading from: ", self.best_model_path)
        return torch.load(self.best_model_path)

    def load_best_running_model(self):
        """
        :return: best tested model of the current testing run
        """
        print("loading from: ", self.best_running_model_path)
        return torch.load(self.best_running_model_path)

    def get_name(self):
        """
        :return: name of handler
        """
        return self.name
