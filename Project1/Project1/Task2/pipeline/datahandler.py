from typing import List

import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
from pipeline import predict, IOHandler
from pathlib import Path
from os import path
import matplotlib.pyplot as plt


class Datahandler:
    def __init__(self, training_txt_file: List[str], training_names: List[str], testing_txt_file: str = None):
        """
        :param training_txt_file(str): List of paths to the training txt file with the data
        :param training_names (List[str]): List of names for the training file
        :param testing_txt_file(List[str]): Path to the testing txt file with the data
        :param type(str): tf0 or ts0
        """

        # TODO: scale targets

        self.training_dfs = {}
        self.names = training_names
        self.scalers = {}

        self.header = ['ps', 'pf', 'Cs', 'Cf', 'mf', 'd', 'D', 'V', 'y']
        self.predictors = self.header[0:8]
        self.target = ['y']

        for idx, file in enumerate(training_txt_file):
            self.training_dfs[self.names[idx]] = pd.read_csv(file, names=self.header, sep=' ')
            self.scalers[self.names[idx]] = preprocessing.MinMaxScaler()

        for idx, key in enumerate(self.training_dfs):
            # print(self.training_dfs[key])
            # self.training_dfs[key][self.predictors] = self.scalers[self.names[idx]].fit_transform(self.training_dfs[key][self.predictors])
            pass

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
        predictions = []
        for idx, model in enumerate(models):
            predictions.append(predict(model, self.testing_dfs[idx]))

        # TODO: combine predicionts
        combination = predictions[0]
        for prediction in predictions[1:]:
            combination += prediction
        combination /= len(predictions)

        submission = pd.DataFrame(combination.detach().numpy())

        submission.to_csv(self.output_path, index=False, header=None, float_format='%.18e')


if __name__ == "__main__":

    iohandler_tf0 = IOHandler('tf0')
    iohandler_ts0 = IOHandler('ts0')

    dirname = path.dirname(__file__)

    testing_filename = path.abspath(path.join(dirname, "..", "data", "TestingData.txt"))
    training_filename1 = path.abspath(path.join(dirname, "..", "data", "TrainingData_101.txt"))
    training_filename2 = path.abspath(path.join(dirname, "..", "data", "TrainingData_401.txt"))
    training_filename3 = path.abspath(path.join(dirname, "..", "data", "TrainingData_1601.txt"))

    datahandler = Datahandler([training_filename1, training_filename2, training_filename3], ['101','401', '1601'], testing_filename)

    datahandler.create_submission([None,None,None])