import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
from torch import optim
from tqdm import tqdm
from pipeline5 import IOHandler
from pathlib import Path
from os import path
import matplotlib.pyplot as plt


class Datahandler:
    def __init__(self, training_txt_file: str):
        """
        :param training_txt_file(str): Path to the training txt file with the data
        :param testing_txt_file(str): Path to the testing txt file with the data
        :param type(str): tf0 or ts0
        """

        training_df = pd.read_csv(training_txt_file, names=['D', 'v', 'CF'], sep=' ')

        self.d_scaler = preprocessing.MinMaxScaler()
        self.v_scaler = preprocessing.MinMaxScaler()
        self.cf_scaler = preprocessing.MinMaxScaler()

        training_df[['D']] = self.d_scaler.fit_transform(training_df[['D']])
        training_df[['v']] = self.v_scaler.fit_transform(training_df[['v']])
        training_df[['CF']] = self.cf_scaler.fit_transform(training_df[['CF']])

        self.d_training = torch.tensor(training_df['D'].values.astype(np.float32).reshape((-1, 1)))
        self.v_training = torch.tensor(training_df['v'].values.astype(np.float32).reshape((-1, 1)))
        self.features_training = torch.tensor(training_df[['D', 'v']].values.astype(np.float32).reshape((-1, 2)))

        self.cf_training = torch.tensor(training_df['CF'].values.astype(np.float32).reshape((-1, 1)))

        self.output_path = None

        basepath = path.dirname(__file__)

        output_dir_path = path.abspath(path.join(basepath, "..", "submission"))
        self.output_path = path.join(output_dir_path, "submission.txt")

        self.submission = pd.DataFrame(columns=['D', 'v'])

        # Create directory for submission
        if not Path(output_dir_path).is_dir():
            Path(output_dir_path).mkdir(parents=True, exist_ok=True)

    def get_data(self):
        """
        :return: tensor with predictors and targets
        """
        return self.features_training, self.cf_training

    def create_submission(self, model, cf_ref=0.45, debug=False):
        """
        Finds 1000 samples for the diameter and volume
        so that the prediction of the model is equal to the reference capacity factor
        :param model: trained neural network
        :param cf_ref: capacity factor reference
        :param debug: output so stats during optimization
        """
        if model is None:
            raise ValueError("please provide a neural network")

        features_opt = []
        cf_values = []
        samples = 1000

        seed = 0

        cf_ref_scaled = self.cf_scaler.transform([[cf_ref]])[0, 0]

        # take average of different seeds
        pbar = tqdm(total=samples, desc=f"optimizing to create {samples} samples")
        while len(cf_values) != samples:

            torch.manual_seed(seed)
            features = torch.rand(2, ).requires_grad_(True)

            optimizer = optim.LBFGS([features], lr=float(0.001), max_iter=50000, max_eval=50000, history_size=100,
                                    line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)

            optimizer.zero_grad()
            cost = list([0])

            def closure():
                optimizer.zero_grad()

                g = (model(features) - cf_ref_scaled) ** 2
                cost[0] = g
                g.backward(retain_graph=True)
                return g

            optimizer.step(closure=closure)

            # add feature if it is in domain
            if not (features[0] < 0 or features[0] > 1 or features[1] < 0 or features[1] > 1):

                cf_value = self.cf_scaler.inverse_transform([model(features).detach().numpy()])[0, 0]
                cf_values.append(cf_value)

                d_opt = self.d_scaler.inverse_transform([[features[0].detach().numpy()]])
                v_opt = self.v_scaler.inverse_transform([[features[1].detach().numpy()]])

                features_opt.append([d_opt[0, 0], v_opt[0, 0]])

                if debug:
                    print("Minimizer: ", features)
                    print("Correspodning CF value: ", cf_value)
                    print("Value of final cost function: ", cost[0])
                    print("features unscaled", [d_opt[0, 0], v_opt[0, 0]])

                pbar.update(1)

            seed += 1

        pbar.close()

        print("CF mean: ", np.mean(cf_values))

        self.submission = pd.DataFrame(features_opt, columns=['D', 'v'])
        self.submission['CF'] = cf_values

        self.submission.to_csv(self.output_path, columns=['D', 'v'], index=False, header=False, sep=' ')

    def plot_data(self):
        """
        Plots the training capacity factor and a constant reference capacity factor
        """

        constant = np.empty(50)
        constant.fill(0.45)

        plt.plot(range(len(self.cf_training.detach().numpy())),
                 self.cf_scaler.inverse_transform(self.cf_training.detach().numpy()), label="CF")
        plt.plot(range(len(self.cf_training.detach().numpy())),
                 constant, label="CF")

        plt.legend()
        plt.xlabel("x")
        plt.ylabel("CF")
        plt.show()

    def plot_submission(self):
        """
        Plots the diameter vs. the volume colored with the corresponding capacity factor
        of the training data and the submitted data
        """
        d_sub = self.submission['D'].values.astype(np.float32)
        v_sub = self.submission['v'].values.astype(np.float32)
        cf_sub = self.submission['CF'].values.astype(np.float32)

        d_train = self.d_scaler.inverse_transform(self.d_training.detach().numpy()).flatten()
        v_train = self.v_scaler.inverse_transform(self.v_training.detach().numpy()).flatten()
        cf_train = self.cf_scaler.inverse_transform(self.cf_training.detach().numpy()).flatten()

        zs = np.concatenate([cf_sub, cf_train], axis=0)
        min_, max_ = zs.min(), zs.max()

        plt1 = plt.scatter(v_train, d_train, c=cf_train, cmap='viridis', label="training")
        plt.clim(min_, max_)
        plt.scatter(v_sub, d_sub, c=cf_sub, s=20, marker='+', cmap='viridis', label="optimized")
        plt.clim(min_, max_)
        cbar = plt.colorbar(plt1)

        cbar.ax.set_ylabel("CF", rotation=0)

        plt.legend()
        plt.xlabel("v")
        plt.ylabel("D", rotation=0)
        plt.show()


if __name__ == "__main__":

    iohandler = IOHandler('task5')

    dirname = path.dirname(__file__)

    training_filename = path.abspath(path.join(dirname, "..", "data", "TrainingData.txt"))

    datahandler = Datahandler(training_filename)

    datahandler.create_submission(iohandler.load_best_model())

    datahandler.plot_data()
    datahandler.plot_submission()
