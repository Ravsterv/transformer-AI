import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Kindve basis to start off, where exactly to go from here is uncertain


class FeatureDataset(Dataset):
    def __init__(self, file_name):

        # read csv file into pandas
        file_out = pd.read_csv(file_name)

        x = file_out.iloc[2:184, 4:15].values
        y = file_out.iloc[2:184, 12].values

        # Feature Scaling? Not sure what this is but lets use it for now.
        # I guess we will change it to SMOTEcross after this
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # Convert to Torch tensors
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        # Required
        return self.x_train[idx], self.y_train[idx]

