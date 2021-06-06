import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader

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

        self.train_window_tf0 = 35
        self.train_window_ts0 = 34
        self.prediction_offset = 1

        self.tf0_x = []
        self.tf0_y = []

        length = len(self.tf0_training)
        for i in range(length - self.train_window_tf0):
            '''
            train_seq = tso_training[i: i + self.train_window]
            train_label = tso_training[i + self.train_window: i + self.train_window + self.prediction_window]
            '''

            train_seq = self.tf0_training[i: i + self.train_window_tf0]
            train_label = self.tf0_training[i + self.prediction_offset: i + self.train_window_tf0 + self.prediction_offset]

            self.tf0_x.append(train_seq.reshape((-1, 1)))
            self.tf0_y.append(train_label)

        self.tf0_x = torch.from_numpy(np.array(self.tf0_x))
        self.tf0_y = torch.from_numpy(np.array(self.tf0_y))

        self.ts0_x = []
        self.ts0_y = []

        length = len(self.ts0_training)
        for i in range(length - self.train_window_ts0):
            '''
            train_seq = tso_training[i: i + self.train_window]
            train_label = tso_training[i + self.train_window: i + self.train_window + self.prediction_window]
            '''

            train_seq = self.ts0_training[i: i + self.train_window_ts0]
            train_label = self.ts0_training[i + self.prediction_offset: i + self.train_window_ts0 + self.prediction_offset]

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
            self.output_path = path.join(output_dir_path, "task3_submission.txt")

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
        :return: numpy ndarray full dataframe with either target variable 'tf0' or 'ts0'
        """

        if target_type == 'ts0':
            return self.ts0_training
        else:
            return self.tf0_training

    def create_submission(self, model_tf0, model_ts0, state):
        """
        Creates prediction for Testing data with trained model and writes result to text file
        Networks will be primed before making prediction
        """

        if self.output_path is None:
            raise ValueError("testing file was not specified during initialization")
        if model_tf0 is None or model_ts0 is None:
            raise ValueError("provide 2 valid neural networks")

        if state in ["stateless", "rnn", "gru"]:
            return self.create_submission_stateless(model_tf0, model_ts0)

        previous_tf0, hidden_tf0 = self.priming(model_tf0, 'tf0')
        previous_ts0, hidden_ts0 = self.priming(model_ts0, 'ts0')

        predictions_tf0 = []
        predictions_ts0 = []

        for t in self.t_testing:
            hidden_tf0 = tuple([each.data for each in hidden_tf0])
            previous_tf0, hidden_tf0 = model_tf0(previous_tf0.reshape((1, -1, 1)), hidden_tf0)
            predictions_tf0.append(previous_tf0.detach().numpy()[-1, 0])

            hidden_ts0 = tuple([each.data for each in hidden_ts0])
            previous_ts0, hidden_ts0 = model_ts0(previous_ts0.reshape((1, -1, 1)), hidden_ts0)
            predictions_ts0.append(previous_ts0.detach().numpy()[-1, 0])

        self.submission['tf0'] = np.array(predictions_tf0)
        self.submission['ts0'] = np.array(predictions_ts0)

        self.submission['tf0'] = self.tf0_scaler.inverse_transform(self.submission[['tf0']])
        self.submission['ts0'] = self.ts0_scaler.inverse_transform(self.submission[['ts0']])

        self.submission.to_csv(self.output_path, index=False)

        """
        for t in self.t_testing:

            hidden_tf0 = tuple([each.data for each in hidden_tf0])
            previous_tf0, hidden_tf0 = model_tf0(t, hidden_tf0)
            predictions_tf0.append(previous_tf0.detach().numpy()[0, 0])

            hidden_ts0 = tuple([each.data for each in hidden_ts0])
            previous_ts0, hidden_ts0 = model_ts0(t, hidden_ts0)
            predictions_ts0.append(previous_ts0.detach().numpy()[0, 0])

        self.submission['tf0'] = np.array(predictions_tf0)
        self.submission['ts0'] = np.array(predictions_ts0)

        self.submission['tf0'] = self.tf0_scaler.inverse_transform(self.submission[['tf0']])
        self.submission['ts0'] = self.ts0_scaler.inverse_transform(self.submission[['ts0']])

        self.submission.to_csv(self.output_path, index=False)
        
        """

    def create_submission_stateless(self, model_tf0, model_ts0):
        """
        Creates prediction for Testing data with trained model and writes result to text file
        """
        if self.output_path is None:
            raise ValueError("testing file was not specified during initialization")
        if model_tf0 is None or model_ts0 is None:
            raise ValueError("provide 2 valid neural networks")

        predictions_tf0 = []
        predictions_ts0 = []

        last_tf0 = torch.FloatTensor([self.tf0_training[-self.train_window_tf0:]]).reshape((-1, 1))
        last_ts0 = torch.FloatTensor([self.ts0_training[-self.train_window_ts0:]]).reshape((-1, 1))

        for _ in self.t_testing:
            last_tf0 = model_tf0(last_tf0.reshape((-1, 1, 1)))
            predictions_tf0.append(last_tf0.detach().numpy()[-1, 0])

            last_ts0 = model_ts0(last_ts0.reshape((-1, 1, 1)))
            predictions_ts0.append(last_ts0.detach().numpy()[-1, 0])

        self.submission['tf0'] = np.array(predictions_tf0)
        self.submission['ts0'] = np.array(predictions_ts0)

        self.submission['tf0'] = self.tf0_scaler.inverse_transform(self.submission[['tf0']])
        self.submission['ts0'] = self.ts0_scaler.inverse_transform(self.submission[['ts0']])

        self.submission.to_csv(self.output_path, index=False)

    def priming(self, model, target_type):
        """
        :param model: trained neural network
        :param target_type: define type ['tf0', 'ts0']
        primes the neural network
        """
        model.eval()

        times, preds = self.get_data(target_type)

        training_set = DataLoader(torch.utils.data.TensorDataset(times, preds), batch_size=1,
                                  shuffle=False, drop_last=True)

        hidden = model.init_hidden(1)

        out = None

        for t, pred in training_set:
            hidden = tuple([each.data for each in hidden])
            out, hidden = model(t, hidden)
            break

        return out, hidden

    def plot_data(self):
        """
        Plots training data
        """

        tf0 = self.tf0_scaler.inverse_transform(self.training_df[['tf0']])
        ts0 = self.ts0_scaler.inverse_transform(self.training_df[['ts0']])

        plt.plot(self.t_scaler.inverse_transform(self.t_training), tf0, label="tf0")
        plt.plot(self.t_scaler.inverse_transform(self.t_training), ts0, label="ts0")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.show()

    def plot_submission(self):
        """
        Plots predicted submission
        """
        plt.plot(self.submission[['t']].to_numpy().flatten(), self.submission['tf0'].values, label="testing tf0")
        plt.plot(self.submission[['t']].to_numpy().flatten(), self.submission['ts0'].values, label="testing ts0")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.show()

    def plot_all(self):
        """
        Plots training data and the time series prediction afterwards
        """

        tf0 = self.tf0_scaler.inverse_transform(self.training_df[['tf0']])
        ts0 = self.ts0_scaler.inverse_transform(self.training_df[['ts0']])

        tf0 = np.append(tf0.flatten(), self.submission[['tf0']].to_numpy().flatten())
        ts0 = np.append(ts0.flatten(), self.submission[['ts0']].to_numpy().flatten())

        time = np.append(self.t_scaler.inverse_transform(self.t_training).flatten(),
                         self.submission[['t']].to_numpy().flatten())

        plt.plot(time, tf0, label="tf0")
        plt.plot(time, ts0, label="ts0")
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

    datahandler = Datahandler(training_filename, testing_filename)

    datahandler.create_submission(iohandler_tf0.load_best_running_model(), iohandler_ts0.load_best_running_model(),
                                  "stateful")

    datahandler.plot_all()
    datahandler.plot_submission()
