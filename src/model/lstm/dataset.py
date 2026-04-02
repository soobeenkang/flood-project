import torch
from torch.utils.data import Dataset
import numpy as np
import os

class LSTMDataset(Dataset):
    def __init__(self, data_dir, prefix="train"):
        self.X, self.y = [], []

        files = sorted(os.listdir(data_dir))

        X_files = [f for f in files if f.startswith(f"{prefix}_X")]
        y_files = [f for f in files if f.startswith(f"{prefix}_y")]

        for xf, yf in zip(X_files, y_files):
            X = np.load(f"{data_dir}/{xf}")
            y = np.load(f"{data_dir}/{yf}")

            self.X.append(X)
            self.y.append(y)

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axif=0)

        print(f"{prefix} 데이터:", self.X.shape)

    
    def __len__(self):
        return len(self.X_files)
    
    def __getitem__(self, idx):
        X = np.load(f"{self.data_dir}/{self.X_files[idx]}")
        y = np.load(f"{self.data_dir}/{self.y_files[idx]}")

        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )
    