from typing import List

import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
from pathlib import Path
from os import path
import matplotlib.pyplot as plt

from pipeline import IOHandler


class Datahandler:
    def __init__(self, training_txt_file: List[str], training_names: List[str], testing_txt_file: str = None):
        """
        :param training_txt_file(str): List of paths to the training txt file with the data
        :param training_names (List[str]): List of names for the training file
        :param testing_txt_file(List[str]): Path to the testing txt file with the data
        :param type(str): tf0 or ts0
        """

        # TODO: scale targets
        # TODO: plot all

        self.training_dfs = {}
        self.names = training_names
        self.scalers = {}

        self.header = ['ps', 'pf', 'Cs', 'Cf', 'mf', 'd', 'D', 'V', 'y']
        self.predictors = self.header[0:8]
        self.target = ['y']

        original_dfs = {}
        for idx, file in enumerate(training_txt_file):
            original_dfs[self.names[idx]] = pd.read_csv(file, names=self.header, sep=' ')
            self.scalers[self.names[idx]] = preprocessing.StandardScaler()

        for idx, key in enumerate(original_dfs):
            # print(self.training_dfs[key])
            # original_dfs[key][self.predictors] = self.scalers[self.names[idx]].fit_transform(original_dfs[key][self.predictors])
            pass

        # TODO: create differences for observables

        previous = self.names[0]
        self.training_dfs[previous] = original_dfs[previous]

        for name in self.names[1:]:
            ns = original_dfs[name].shape[0]
            obs_diff = original_dfs[name].iloc[:ns, -1] - original_dfs[previous].iloc[:ns, -1]
            self.training_dfs[name] = pd.DataFrame(np.concatenate([original_dfs[name].iloc[:ns, :len(self.predictors)], obs_diff.values.reshape(-1, 1)], 1), columns=self.header)
            previous = name

        self.output_path = None

        if testing_txt_file is not None:

            testing_df = pd.read_csv(testing_txt_file, names=self.predictors, sep=' ')
            # print(testing_df)

            self.testing_dfs = []

            for scaler in self.scalers:
                copy = testing_df.copy()
                # copy[self.predictors] = scaler.transform(testing_df[self.predictors])
                # print(copy)
                self.testing_dfs.append(torch.tensor(copy.values.astype(np.float32).reshape((-1, 8))))

            basepath = path.dirname(__file__)

            output_dir_path = path.abspath(path.join(basepath, "..", "submission"))
            self.output_path = path.join(output_dir_path, "submission.txt")

            # Create directory for submission
            if not Path(output_dir_path).is_dir():
                Path(output_dir_path).mkdir(parents=True, exist_ok=True)


    def get_frame(self, name:str):
        """
        :param name(str): name of the training dataframe
        :return: tensor with predictors and targets
        """
        if name not in self.names:
            raise ValueError("name unknown")

        return torch.tensor(self.training_dfs[name][self.predictors].values.astype(np.float32).reshape((-1, 8))),\
               torch.tensor(self.training_dfs[name][self.target].values.astype(np.float32).reshape((-1, 1)))

    def create_submission(self, models):

        if self.output_path is None:
            raise ValueError("testing file was not specified during initialization")

        """
        Creates prediction for Testing data with trained model and writes result to text file
        """

        combination = torch.zeros((self.testing_dfs[0].shape[0],1))

        for idx, model in enumerate(models):
            combination = combination + model(self.testing_dfs[idx])

        submission = pd.DataFrame(combination.detach().numpy())

        submission.to_csv(self.output_path, index=False, header=None, float_format='%.18e')

    def plot_data(self):
        for name in self.names:
            plt.plot(range(self.training_dfs[name][['y']].shape[0]), self.training_dfs[name][['y']], label="y")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()

    def plot_all(self):
        plt.plot(self.t_training, self.tf0_scaler.inverse_transform(self.tfo_training.detach().numpy()), label="tf0")
        plt.plot(self.t_training, self.ts0_scaler.inverse_transform(self.tso_training.detach().numpy()), label="ts0")
        plt.plot(self.t_testing, self.submission['tf0'].values, label="testing tf0")
        plt.plot(self.t_testing, self.submission['ts0'].values, label="testing ts0")
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
                              testing_filename)

    iohandler_101 = IOHandler('101')
    iohandler_401 = IOHandler('401')
    iohandler_1601 = IOHandler('1601')


    datahandler.create_submission([iohandler_101.load_best_model(), iohandler_401.load_best_model(), iohandler_1601.load_best_model()])

    datahandler.plot_data()