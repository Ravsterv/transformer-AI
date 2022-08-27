import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
# Kindve basis to start off, where exactly to go from here is uncertain


class FeatureDataset(Dataset):
    def __init__(self, file_name):

        # read csv file into pandas
        # Don't need this anymore
        conversions = {"Age":np.int32,
                       "CO":np.float64,
                       "CO2":np.float64,
                       "H2":np.float64,
                       "CH4":np.float64,
                       "C2H4":np.float64,
                       "C2H6": np.float64,
                       "C2H2": np.float64,
                       "H2O": np.int32,
                       "ACIDITY": np.float64,
                       "BREAKDOWN": np.int32,
                       "FURFURAL": np.float64,
                       "DDF":np.float64
                       }
        file_out = pd.read_csv(file_name)
        total_rows = 184
        dropping_rows = []
        for i, data in enumerate(file_out.iloc[0:184, 4:15].values):
            for entry in data:
                if pd.isna(entry):
                    # print(i, data)
                    dropping_rows.append(i)
                    total_rows -= 1

        file_out.drop(
            labels=dropping_rows,
            axis=0,
            inplace=True
        )

        self.data = file_out.iloc[0:total_rows, 4:17].values

        self.target = file_out.iloc[0:total_rows, 17].values

        # Feature Scaling? Not sure what this is but lets use it for now.
        # I guess we will change it to SMOTEcross after this

        print(self.target)
        # Convert to Torch tensors

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # Required
        return self.data[idx], self.target[idx]

