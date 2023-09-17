from torch.utils.data import Dataset, DataLoader, random_split
import torch
from data_loader import AudioMixtureDataset

def preprocess(RAW_DATA_PATH, TEST_SIZE):
    directory = "Dataset/data"
    dataset = AudioMixtureDataset(RAW_DATA_PATH)

    # Split dataset into training, validation, and test sets
    train_size = float(1.0-TEST_SIZE)
    test_size = TEST_SIZE
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    torch.save(train_dataset, "Preprocessed_dataset/train_dataset.pt")
    torch.save(test_dataset, "Preprocessed_dataset/test_dataset.pt")