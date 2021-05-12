import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing


# TODO: create predictions from Testing Data and create final file

class Testing:
    def __init__(self, txt_file: str, model: torch.nn.Module):
        """
        Handles the testing dataset to create the final submisssion file
        :param txt_file(str): Path to the txt file with the data
        :param model: trained pytorch neural network
        """
        df = pd.read_csv(txt_file)

        min_max_scaler = preprocessing.MinMaxScaler()
        df_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)

        self.t = torch.tensor(df_scaled['t'].values.astype(np.float32).reshape((-1, 1)))
        self.model = model

    def create_predictions(self):
        pass

    def create_submission(self):
        """
        Creates prediciton for Testing data with trained model and writes result to text file
        """
        predictions = self.create_predictions()
