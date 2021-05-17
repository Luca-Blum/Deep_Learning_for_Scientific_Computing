from typing import List

import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
from pathlib import Path
from os import path
import matplotlib.pyplot as plt

from pipeline2 import IOHandler


class Datahandler:
    def __init__(self, training_txt_file: List[str], training_names: List[str], testing_txt_file: str = None,
                 debug=False):
        """
        :param training_txt_file: List of paths to the training txt file with the data
        :param training_names: List of names for the training file
        :param testing_txt_file: Path to the testing txt file with the data
        :param debug: used to plot predictions for a testing file that is the same as a training file
        """

        self.training_dfs = {}
        self.names = training_names
        self.scalers_pred = {}
        self.scalers_target = {}

        self.header = ['ps', 'pf', 'Cs', 'Cf', 'mf', 'd', 'D', 'V', 'y']
        self.predictors = self.header[0:8]
        self.target = ['y']

        original_dfs = {}
        for idx, file in enumerate(training_txt_file):
            original_dfs[self.names[idx]] = pd.read_csv(file, names=self.header, sep=' ')
            self.scalers_pred[self.names[idx]] = preprocessing.StandardScaler()
            self.scalers_target[self.names[idx]] = preprocessing.StandardScaler()

        previous = self.names[0]
        self.training_dfs[previous] = original_dfs[previous]

        # Differences for target
        for name in self.names[1:]:
            ns = original_dfs[name].shape[0]
            obs_diff = original_dfs[name].iloc[:ns, -1] - original_dfs[previous].iloc[:ns, -1]
            self.training_dfs[name] = pd.DataFrame(np.concatenate([original_dfs[name].iloc[:ns, :len(self.predictors)],
                                                                   obs_diff.values.reshape(-1, 1)], 1),
                                                   columns=self.header)
            previous = name

        # scaling predictors and target
        for idx, key in enumerate(original_dfs):
            # print(self.training_dfs[key])
            self.training_dfs[key][self.predictors] = self.scalers_pred[self.names[idx]].fit_transform(
                self.training_dfs[key][self.predictors])
            self.training_dfs[key][self.target] = self.scalers_target[self.names[idx]].fit_transform(
                self.training_dfs[key][self.target])

        # Create testing frame

        self.output_path = None
        self.submission = None
        self.predictions = {}

        if testing_txt_file is not None:

            testing_df = pd.read_csv(testing_txt_file, names=self.predictors, sep=' ')
            # print(testing_df)

            self.testing_dfs = {}

            # Create different scalings
            for idx, scaler in enumerate(self.scalers_pred):
                copy = testing_df.copy()
                copy[self.predictors] = self.scalers_pred[self.names[idx]].transform(testing_df[self.predictors])
                self.testing_dfs[self.names[idx]] = torch.tensor(copy.values.astype(np.float32).reshape((-1, 8)))

            basepath = path.dirname(__file__)

            output_dir_path = path.abspath(path.join(basepath, "..", "submission"))
            self.output_path = path.join(output_dir_path, "submission.txt")

            # Create directory for submission
            if not Path(output_dir_path).is_dir():
                Path(output_dir_path).mkdir(parents=True, exist_ok=True)

        # store initial testing frame
        if debug:
            self.debug = debug

            testing_df = pd.read_csv(testing_txt_file, names=self.header, sep=' ')

            self.testing_dfs = {}
            # Create different scalings
            for idx, scaler in enumerate(self.scalers_pred):
                copy = testing_df.copy()
                copy[self.predictors] = self.scalers_pred[self.names[idx]].transform(testing_df[self.predictors])
                self.testing_dfs[self.names[idx]] = torch.tensor(copy[self.predictors].
                                                                 values.astype(np.float32).reshape((-1, 8)))

            self.debug_df = testing_df

    def get_frame(self, name: str):
        """
        :param name: name of the training dataframe
        :return: tensor with predictors and targets
        """
        if name not in self.names:
            raise ValueError("name unknown")

        return torch.tensor(self.training_dfs[name][self.predictors].values.astype(np.float32).reshape((-1, 8))), \
               torch.tensor(self.training_dfs[name][self.target].values.astype(np.float32).reshape((-1, 1)))

    def create_submission(self, models):
        """
        Creates prediction for the testing file and stores the result in the submission folder
        :param models: List of the trained models for the different grids
        :return: None
        """

        if self.output_path is None:
            raise ValueError("testing file was not specified during initialization")

        combination = np.zeros((self.testing_dfs[self.names[0]].shape[0], 1))

        for idx, model in enumerate(models):
            prediction = model(self.testing_dfs[self.names[idx]])
            self.predictions[self.names[idx]] = prediction.detach().numpy()
            scaled = self.scalers_target[self.names[idx]].inverse_transform(prediction.detach().numpy())
            combination = combination + scaled

        self.submission = pd.DataFrame(combination)

        self.submission.to_csv(self.output_path, index=False, header=False, float_format='%.18e')

    def plot_data(self):
        """
        Plots training data
        :return: None
        """
        for name in self.names:
            plt.plot(range(self.training_dfs[name][['y']].shape[0]), self.training_dfs[name][['y']], label="y")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()

    def plot_debug(self):
        """
        Plots the prediction for a testing file that is the same as a training file
        :return:
        """
        if not self.debug:
            raise ValueError('please enable debug mode')

        plt.plot(range(self.debug_df.shape[0]), self.debug_df[['y']], label="y_training")
        plt.plot(range(self.submission.shape[0]), self.submission, label="y_predict")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


if __name__ == "__main__":
    dirname = path.dirname(__file__)
    training_filename_101 = path.join(dirname, '..', 'data/TrainingData_101.txt')
    training_filename_401 = path.join(dirname, '..', 'data/TrainingData_401.txt')
    training_filename_1601 = path.join(dirname, '..', 'data/TrainingData_1601.txt')
    testing_filename = path.join(dirname, '..', 'data/TestingData.txt')

    datahandler = Datahandler([training_filename_101, training_filename_401, training_filename_1601],
                              ['101', '401', '1601'],
                              training_filename_1601, debug=True)

    iohandler_101 = IOHandler('101')
    iohandler_401 = IOHandler('401')
    iohandler_1601 = IOHandler('1601')

    datahandler.create_submission(
        [iohandler_101.load_best_model(), iohandler_401.load_best_model(), iohandler_1601.load_best_model()])

    # datahandler.plot_data()
    datahandler.plot_debug()
