from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd

import librosa

def load_wav_file(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    return TensorDataset(X, y)

def get_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)