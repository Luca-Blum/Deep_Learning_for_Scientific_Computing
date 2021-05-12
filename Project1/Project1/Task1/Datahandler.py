import pandas as pd
import torch
import numpy as np


class Datahandler:
    def __init__(self, txt_file: str):
        """
        :param txt_file(str): Path to the txt file with the data
        :param type(str): tf0 or ts0
        """
        df = pd.read_csv(txt_file)
        self.tfo = torch.tensor(df['tf0'].values.astype(np.float32).reshape((-1, 1)))
        self.tso = torch.tensor(df['ts0'].values.astype(np.float32).reshape((-1, 1)))
        self.t = torch.tensor(df['t'].values.astype(np.float32).reshape((-1, 1)))

    def get_predictors(self):
        """
        :return: tensor with predictors
        """
        return self.t

    def get_targets(self, target_type: str):
        """
        :param target_type: specify target varialbe ['tf0', 'ts0']
        :return: tensor with either target variable 'tf0' or 'ts0'
        """
        if target_type == 'ts0':
            return self.tso
        else:
            return self.tfo
