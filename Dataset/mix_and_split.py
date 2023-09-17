import os
import random
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class AudioMixtureDataset(Dataset):
    def __init__(self, directory):
        self.audio_files = self.load_audio_files(directory)
        self.class_keys = list(self.audio_files.keys())
        random.shuffle(self.class_keys)

    def load_audio_files(self, directory):
        audio_files = {}
        for class_dir in os.listdir(directory):
            audio_files[class_dir] = []
            for filename in os.listdir(os.path.join(directory, class_dir)):
                filepath = os.path.join(directory, class_dir, filename)
                audio_data, _ = librosa.load(filepath)
                audio_files[class_dir].append(audio_data)
        return audio_files

    def mix_audio_files(self, class1_data, class2_data):
        min_length = min(len(class1_data), len(class2_data))
        return class1_data[:min_length] + class2_data[:min_length]

    def __len__(self):
        return sum([len(data) for data in self.audio_files.values()])

    def __getitem__(self, idx):
        # Randomly choose two classes
        class1, class2 = random.sample(self.class_keys, 2)
        
        # Randomly choose one audio clip per class
        audio1 = random.choice(self.audio_files[class1])
        audio2 = random.choice(self.audio_files[class2])

        # Mix the two audio clips
        mixed_audio = self.mix_audio_files(audio1, audio2)

        # Convert to PyTorch tensor and return
        return torch.tensor(mixed_audio, dtype=torch.float32)
    
    # Main code
if __name__ == "__main__":
    directory = "Dataset/data/classes"
    dataset = AudioMixtureDataset(directory)

    # Split dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    torch.save(train_dataset, "Preprocessed_dataset/train_dataset.pt")
    torch.save(test_dataset, "Preprocessed_dataset/test_dataset.pt")
    
    # Now you can use train_loader, val_loader, and test_loader in your training and evaluation loops

