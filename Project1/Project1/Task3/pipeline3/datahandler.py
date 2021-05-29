import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing

from pipeline3 import IOHandler
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

        self.training_df = pd.read_csv(training_txt_file)

        self.t_scaler = preprocessing.MinMaxScaler()
        self.tf0_scaler = preprocessing.MinMaxScaler()
        self.ts0_scaler = preprocessing.MinMaxScaler()

        self.training_df[['t']] = self.t_scaler.fit_transform(self.training_df[['t']])
        self.training_df[['tf0']] = self.tf0_scaler.fit_transform(self.training_df[['tf0']])
        self.training_df[['ts0']] = self.ts0_scaler.fit_transform(self.training_df[['ts0']])

        self.tf0_training = self.training_df['tf0'].values.astype(np.float32)
        self.ts0_training = self.training_df['ts0'].values.astype(np.float32)
        self.t_training = torch.tensor(self.training_df['t'].values.astype(np.float32).reshape((-1, 1)))

        self.train_window = 25
        self.prediction_offset = 1

        if self.train_window <= self.prediction_offset:
            raise ValueError("training window needs to be larger than prediction offset")

        self.tf0_x = []
        self.tf0_y = []

        l = len(self.tf0_training)
        for i in range(l - self.train_window):
            '''
            train_seq = tso_training[i: i + self.train_window]
            train_label = tso_training[i + self.train_window: i + self.train_window + self.prediction_window]
            '''

            train_seq = self.tf0_training[i: i + self.train_window]
            train_label = self.tf0_training[i + self.prediction_offset: i + self.train_window + self.prediction_offset]

            self.tf0_x.append(train_seq.reshape((-1, 1)))
            self.tf0_y.append(train_label)

        self.tf0_x = torch.from_numpy(np.array(self.tf0_x))
        self.tf0_y = torch.from_numpy(np.array(self.tf0_y))

        self.ts0_x = []
        self.ts0_y = []

        l = len(self.ts0_training)
        for i in range(l - self.train_window):
            '''
            train_seq = tso_training[i: i + self.train_window]
            train_label = tso_training[i + self.train_window: i + self.train_window + self.prediction_window]
            '''

            train_seq = self.ts0_training[i: i + self.train_window]
            train_label = self.ts0_training[i + self.prediction_offset: i + self.train_window + self.prediction_offset]

            self.ts0_x.append(train_seq.reshape((-1, 1)))
            self.ts0_y.append(train_label)

        self.ts0_x = torch.from_numpy(np.array(self.ts0_x))
        self.ts0_y = torch.from_numpy(np.array(self.ts0_y))

        self.output_path = None

        if testing_txt_file is not None:
            testing_df = pd.read_csv(testing_txt_file)

            self.t_testing_unscaled = testing_df[['t']].values
            self.submission = pd.DataFrame(self.t_testing_unscaled, columns=['t'])

            testing_df[['t']] = self.t_scaler.transform(testing_df[['t']])

            self.t_testing = torch.tensor(testing_df['t'].values.astype(np.float32).reshape((-1, 1, 1, 1)))

            basepath = path.dirname(__file__)

            output_dir_path = path.abspath(path.join(basepath, "..", "submission"))
            self.output_path = path.join(output_dir_path, "submission.txt")

            # Create directory for submission
            if not Path(output_dir_path).is_dir():
                Path(output_dir_path).mkdir(parents=True, exist_ok=True)

    def get_data(self, target_type: str):
        """
        :param target_type: specify target variable ['tf0', 'ts0']
        :return: tensor with either target variable 'tf0' or 'ts0'
        """
        if target_type == 'ts0':
            return self.ts0_x, self.ts0_y
        else:
            return self.tf0_x, self.tf0_y

    def get_raw(self, target_type: str):
        """
        :param target_type: specify target variable ['tf0', 'ts0']
        :return: tensor with either target variable 'tf0' or 'ts0'
        """

        if target_type == 'ts0':
            return self.ts0_training
        else:
            return self.tf0_training

    def create_submission(self, model_tf0, model_ts0):

        if self.output_path is None:
            raise ValueError("testing file was not specified during initialization")

        """
        Creates prediction for Testing data with trained model and writes result to text file
        """

        previous_tf0, hidden_tf0 = self.priming(model_tf0, 'tf0')
        previous_ts0, hidden_ts0 = self.priming(model_ts0, 'ts0')

        predictions_tf0 = []
        predictions_ts0 = []

        print(self.t_testing)

        for t in self.t_testing:

            hidden_tf0 = tuple([each.data for each in hidden_tf0])
            previous_tf0, hidden_tf0 = model_tf0(t, hidden_tf0)
            predictions_tf0.append(previous_tf0.detach().numpy()[0, 0])

            hidden_ts0 = tuple([each.data for each in hidden_ts0])
            previous_ts0, hidden_ts0 = model_tf0(t, hidden_ts0)
            predictions_ts0.append(previous_ts0.detach().numpy()[0, 0])

        self.submission['tf0'] = np.array(predictions_tf0)
        self.submission['ts0'] = np.array(predictions_ts0)

        self.submission['tf0'] = self.tf0_scaler.inverse_transform(self.submission[['tf0']])
        self.submission['ts0'] = self.ts0_scaler.inverse_transform(self.submission[['ts0']])

        self.submission.to_csv(self.output_path, index=False)

    def priming(self, model, target_type):
        model.eval()

        prime = self.get_raw(target_type)

        hidden = model.init_hidden(1)
        for x in prime:
            feature = torch.from_numpy(np.array([[[x]]]))
            hidden = tuple([each.data for each in hidden])
            out, hidden = model(feature, hidden)

        return out, hidden

    def plot_data(self):
        plt.plot(self.t_training, self.tf0_scaler.inverse_transform(self.tfo_training.detach().numpy()), label="tf0")
        plt.plot(self.t_training, self.ts0_scaler.inverse_transform(self.tso_training.detach().numpy()), label="ts0")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def plot_submission(self):
        plt.plot(self.t_testing, self.submission['tf0'].values, label="testing tf0")
        plt.plot(self.t_testing, self.submission['ts0'].values, label="testing ts0")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def plot_all(self):

        tf0 = self.tf0_scaler.inverse_transform(self.training_df[['tf0']])
        ts0 = self.ts0_scaler.inverse_transform(self.training_df[['ts0']])

        tf0 = np.append(tf0.flatten(), self.submission[['tf0']].to_numpy().flatten())
        ts0 = np.append(ts0.flatten(), self.submission[['ts0']].to_numpy().flatten())

        plt.plot(range(len(tf0)), tf0, label="tf0")
        plt.plot(range(len(ts0)), ts0, label="ts0")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        """
        sub_tf0 = torch.tensor(self.submission['tf0'].values.astype(np.float32).reshape((-1, 1)))
        print(torch.mean((sub_tf0.reshape(-1, ) - self.tfo_training.reshape(-1, )) ** 2))
        """


if __name__ == "__main__":

    iohandler_tf0 = IOHandler('tf0')
    iohandler_ts0 = IOHandler('ts0')

    dirname = path.dirname(__file__)

    testing_filename = path.abspath(path.join(dirname, "..", "data", "TestingData.txt"))
    training_filename = path.abspath(path.join(dirname, "..", "data", "TrainingData.txt"))

    datahandler = Datahandler(training_filename, training_filename)

    datahandler.create_submission(iohandler_tf0.load_best_model(), iohandler_ts0.load_best_model())

    datahandler.plot_all()
