import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
from torch import optim
from tqdm import tqdm
from pipeline4 import IOHandler
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

        training_df = pd.read_csv(training_txt_file, names=['t', 'u', 'T'], sep=' ')

        self.t_scaler = preprocessing.MinMaxScaler()
        self.u_scaler = preprocessing.MinMaxScaler()
        self.T_scaler = preprocessing.MinMaxScaler()

        training_df[['t']] = self.t_scaler.fit_transform(training_df[['t']])
        training_df[['u']] = self.u_scaler.fit_transform(training_df[['u']])
        training_df[['T']] = self.T_scaler.fit_transform(training_df[['T']])

        self.t_training = torch.tensor(training_df['t'].values.astype(np.float32).reshape((-1, 1)))
        self.u_training = torch.tensor(training_df['u'].values.astype(np.float32).reshape((-1, 1)))
        self.features_training = torch.tensor(training_df[['t', 'u']].values.astype(np.float32).reshape((-1, 2)))
        self.T_training = torch.tensor(training_df['T'].values.astype(np.float32).reshape((-1, 1)))

        self.output_path = None

        if testing_txt_file is not None:
            testing_df = pd.read_csv(testing_txt_file, names=['t', 'T'], sep=' ')

            testing_df[['t']] = self.t_scaler.transform(testing_df[['t']])
            testing_df[['T']] = self.T_scaler.transform(testing_df[['T']])

            self.t_testing = torch.tensor(testing_df['t'].values.astype(np.float32).reshape((-1, 1)))
            self.T_testing = torch.tensor(testing_df['T'].values.astype(np.float32).reshape((-1, 1)))

            self.submission = pd.DataFrame([0], columns=['u'])

            basepath = path.dirname(__file__)

            output_dir_path = path.abspath(path.join(basepath, "..", "submission"))
            self.output_path = path.join(output_dir_path, "submission.txt")

            # Create directory for submission
            if not Path(output_dir_path).is_dir():
                Path(output_dir_path).mkdir(parents=True, exist_ok=True)

    def get_data(self):
        """
        :return: tensor with predictors and target
        """
        return self.features_training, self.T_training

    def create_submission(self, model, debug=False):
        """
        Finds the optimal velocity that created the measured data
        :param model: trained neural network
        :param debug: output so stats during optimization
        """

        if self.output_path is None:
            raise ValueError("testing file was not specified during initialization")
        if model is None:
            raise ValueError("provide a valid neural network")

        u_opts = []

        iterations = 10

        # take average of different seeds
        pbar = tqdm(total=iterations, desc=f"find optimal velocity")

        for seed in range(iterations):

            torch.manual_seed(seed)
            u_opt = torch.rand(1,).reshape((-1, 1)).requires_grad_(True)

            optimizer = optim.LBFGS([u_opt], lr=float(0.00001), max_iter=50000, max_eval=50000, history_size=100,
                                    line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)

            optimizer.zero_grad()
            cost = list([0])

            def closure():
                optimizer.zero_grad()
                features_temp = torch.cat((self.t_testing, u_opt.repeat(
                    (len(self.t_testing), 1)).reshape(-1, 1).requires_grad_(True)), dim=1)

                g = ((model(features_temp) - self.T_testing) ** 2).sum()
                cost[0] = g
                g.backward(retain_graph=True)
                return g

            optimizer.step(closure=closure)
            features = torch.cat((self.t_testing, u_opt.repeat(
                (len(self.t_testing), 1)).reshape(-1, 1).requires_grad_(True)), dim=1)

            u_opt = u_opt.reshape((-1, 1))

            u_opt = self.u_scaler.inverse_transform(u_opt.detach().numpy())

            u_opts.append(u_opt)

            pbar.update(1)

            if debug:
                print("Minimizer: ", u_opt)
                print("Correspodning T values: ", self.T_scaler.inverse_transform(model(features).detach().numpy()))
                print("Value of final cost function: ", cost[0])

                print("u_opt unscaled", u_opt)

                print("u_opt unscaled", u_opt)

        pbar.close()
        print("optimal velocity: ", np.mean(u_opts))

        self.submission['u'] = np.mean(u_opts)
        self.submission.to_csv(self.output_path, index=False, header=False)

    def plot_data(self):
        """
        Plots training time vs. training Temperature and colors the crosses with the corresponding velocity
        Additionally the measured  time and measured temperature is plotted
        """
        u_unscaled = self.u_scaler.inverse_transform(self.u_training.detach().numpy())

        colors = ['black', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']
        labels = ['u0 = ' + str(u_unscaled[0]),
                  'u1 = ' + str(u_unscaled[128]),
                  'u2 = ' + str(u_unscaled[256]),
                  'u3 = ' + str(u_unscaled[384]),
                  'u4 = ' + str(u_unscaled[512]),
                  'u5 = ' + str(u_unscaled[640]),
                  'u6 = ' + str(u_unscaled[768]),
                  'u7 = ' + str(u_unscaled[896])]

        plt.scatter(self.t_scaler.inverse_transform(self.t_testing.detach().numpy()),
                    self.T_scaler.inverse_transform(self.T_testing.detach().numpy()),
                    color='lime', marker='+', label="measured")

        for i in range(8):
            plt.scatter(self.t_scaler.inverse_transform(self.t_training[i*128:(i+1)*128].detach().numpy()),
                        self.T_scaler.inverse_transform(self.T_training.detach().numpy()[i*128:(i+1)*128]),
                        color=colors[i], marker='+', label=labels[i])

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 2, 3, 4, 5, 6, 7, 8, 0]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.show()

    def plot_submission(self, model):
        """
        Plots the measured time and measured temperature
        Additionally plots the submission => the predicted temperature of the trained model
        given the measured time and the optimized velocity
        """
        u_opt = torch.tensor(self.u_scaler.transform([self.submission['u']]).repeat(len(self.t_testing)).reshape(-1, 1))

        features = torch.cat((self.t_testing, u_opt), dim=1)

        model.eval()

        plt.scatter(self.t_scaler.inverse_transform(self.t_testing.detach().numpy()),
                    self.T_scaler.inverse_transform(self.T_testing.detach().numpy()),
                    color='lime', marker='+', label="T testing")
        plt.scatter(self.t_scaler.inverse_transform(self.t_testing.detach().numpy()),
                    self.T_scaler.inverse_transform(model(features.float()).detach().numpy()),
                    color='indigo', marker='+', label="T model")

        model.train()

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.show()

    def plot_all(self, model):
        """
        Plots training time vs. training Temperature and colors the crosses with the corresponding velocity
        Additionally the measured training time and training Temperature is plotted
        Additionally plots the submission => the predicted temperature of the trained model
        given the measured time and the optimized velocity
        """
        u_unscaled = self.u_scaler.inverse_transform(self.u_training.detach().numpy())

        colors = ['black', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']
        labels = [f'u_0 = {u_unscaled[0,0]:.2f}',
                  f'u_1 = {u_unscaled[128,0]:.2f}',
                  f'u_2 = {u_unscaled[256,0]:.2f}',
                  f'u_3 = {u_unscaled[384,0]:.2f}',
                  f'u_4 = {u_unscaled[512, 0]:.2f}',
                  f'u_5 = {u_unscaled[640, 0]:.2f}',
                  f'u_6 = {u_unscaled[768, 0]:.2f}',
                  f'u_7 = {u_unscaled[896, 0]:.2f}']

        ax = plt.subplot(111)

        ax.scatter(self.t_scaler.inverse_transform(self.t_testing.detach().numpy()),
                   self.T_scaler.inverse_transform(self.T_testing.detach().numpy()),
                   color='lime', marker='+', label="measured")

        for i in range(8):
            ax.scatter(self.t_scaler.inverse_transform(self.t_training[i*128:(i+1)*128].detach().numpy()),
                       self.T_scaler.inverse_transform(self.T_training.detach().numpy()[i*128:(i+1)*128]),
                       color=colors[i], marker='+', label=labels[i])

        u_opt = torch.tensor(self.u_scaler.transform([self.submission['u']]).repeat(len(self.t_testing)).reshape(-1, 1))

        features = torch.cat((self.t_testing, u_opt), dim=1)

        model.eval()

        t_model = model(features.float())

        u_unscaled = self.submission['u'][0]

        ax.scatter(self.t_scaler.inverse_transform(self.t_testing.detach().numpy()),
                   self.T_scaler.inverse_transform(t_model.detach().numpy()),
                   color='indigo', marker='+', label=f'u_* = {u_unscaled:.2f}')
        model.train()

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 2, 3, 4, 5, 6, 7, 8, 0, 9]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature")
        plt.show()


if __name__ == "__main__":

    iohandler = IOHandler('task4')

    dirname = path.dirname(__file__)

    testing_filename = path.abspath(path.join(dirname, "..", "data", "MeasuredData.txt"))
    training_filename = path.abspath(path.join(dirname, "..", "data", "TrainingData.txt"))

    datahandler = Datahandler(training_filename, testing_filename)

    datahandler.create_submission(iohandler.load_best_model())

    datahandler.plot_data()
    datahandler.plot_submission(iohandler.load_best_model())

    datahandler.plot_all(iohandler.load_best_model())
