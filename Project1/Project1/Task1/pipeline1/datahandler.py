import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
from pipeline1 import IOHandler
from pathlib import Path
from os import path
import matplotlib.pyplot as plt


class Datahandler:
    def __init__(self, training_txt_file: str, testing_txt_file: str = None):
        """
        :param training_txt_file(str): Path to the training txt file with the data
        :param testing_txt_file(str): Path to the testing txt file with the data
        :param type(str): tf0 or ts0
        """

        training_df = pd.read_csv(training_txt_file)

        self.t_scaler = preprocessing.StandardScaler()
        self.tf0_scaler = preprocessing.StandardScaler()
        self.ts0_scaler = preprocessing.StandardScaler()

        training_df[['t']] = self.t_scaler.fit_transform(training_df[['t']])
        training_df[['tf0']] = self.tf0_scaler.fit_transform(training_df[['tf0']])
        training_df[['ts0']] = self.ts0_scaler.fit_transform(training_df[['ts0']])

        self.tfo_training = torch.tensor(training_df['tf0'].values.astype(np.float32).reshape((-1, 1)))
        self.tso_training = torch.tensor(training_df['ts0'].values.astype(np.float32).reshape((-1, 1)))
        self.t_training = torch.tensor(training_df['t'].values.astype(np.float32).reshape((-1, 1)))

        self.output_path = None

        if testing_txt_file is not None:
            testing_df = pd.read_csv(testing_txt_file)

            self.t_testing_unscaled = testing_df[['t']].values
            self.submission = pd.DataFrame(self.t_testing_unscaled, columns=['t'])

            testing_df[['t']] = self.t_scaler.transform(testing_df[['t']])

            self.t_testing = torch.tensor(testing_df['t'].values.astype(np.float32).reshape((-1, 1)))

            basepath = path.dirname(__file__)

            output_dir_path = path.abspath(path.join(basepath, "..", "submission"))
            self.output_path = path.join(output_dir_path, "submission.txt")

            # Create directory for submission
            if not Path(output_dir_path).is_dir():
                Path(output_dir_path).mkdir(parents=True, exist_ok=True)

    def get_data(self, target_type: str):
        """
        :param target_type: specify target variable ['tf0', 'ts0']
        :return: tensor with predictors and tensor with either target variable 'tf0' or 'ts0'
        """

        if target_type == 'ts0':
            return self.t_training, self.tso_training
        else:
            return self.t_training, self.tfo_training

    def create_submission(self, model_tf0, model_ts0):
        """
        Creates prediction for testing data with trained model and writes result to text file
        """

        if self.output_path is None:
            raise ValueError("testing file was not specified during initialization")

        if model_tf0 is None or model_ts0 is None:
            raise ValueError("please provide valid models")

        predictions_tf0 = model_tf0(self.t_testing)
        predictions_ts0 = model_ts0(self.t_testing)

        self.submission['tf0'] = predictions_tf0.detach().numpy()

        self.submission['ts0'] = predictions_ts0.detach().numpy()

        self.submission['tf0'] = self.tf0_scaler.inverse_transform(self.submission['tf0'])
        self.submission['ts0'] = self.ts0_scaler.inverse_transform(self.submission['ts0'])

        self.submission.to_csv(self.output_path, index=False)

    def plot_data(self):
        """
        plot training time vs training temperature fluid and training time vs training temperature solid
        """

        plt.plot(self.t_scaler.inverse_transform(self.t_training.detach().numpy()),
                 self.tf0_scaler.inverse_transform(self.tfo_training.detach().numpy()), label="tf0")
        plt.plot(self.t_scaler.inverse_transform(self.t_training.detach().numpy()),
                 self.ts0_scaler.inverse_transform(self.tso_training.detach().numpy()), label="ts0")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.show()

    def plot_submission(self):
        """
        plot testing time vs predicted temperature of solid and fluid
        """

        plt.plot(self.t_scaler.inverse_transform(self.t_testing.detach().numpy()),
                 self.submission['tf0'].values, label="Predicted tf0")
        plt.plot(self.t_scaler.inverse_transform(self.t_testing.detach().numpy()),
                 self.submission['ts0'].values, label="Predicted ts0")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.show()

    def plot_all(self):
        """
        plot training time vs training temperature fluid and training time vs training temperature solid
        plot testing time vs predicted temperature of solid and fluid
        """

        plt.plot(self.t_training, self.tf0_scaler.inverse_transform(self.tfo_training.detach().numpy()), label="Training tf0")
        plt.plot(self.t_training, self.ts0_scaler.inverse_transform(self.tso_training.detach().numpy()), label="Training ts0")
        plt.plot(self.t_testing, self.submission['tf0'].values, label="Predicted tf0")
        plt.plot(self.t_testing, self.submission['ts0'].values, label="Predicted ts0")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.show()


if __name__ == "__main__":

    iohandler_tf0 = IOHandler('tf0')
    iohandler_ts0 = IOHandler('ts0')

    dirname = path.dirname(__file__)

    testing_filename = path.abspath(path.join(dirname, "..", "data", "TestingData.txt"))
    training_filename = path.abspath(path.join(dirname, "..", "data", "TrainingData.txt"))

    datahandler = Datahandler(training_filename, training_filename)

    datahandler.create_submission(iohandler_tf0.load_best_model(), iohandler_ts0.load_best_model())
    datahandler.plot_data()
    datahandler.plot_submission()
    datahandler.plot_all()