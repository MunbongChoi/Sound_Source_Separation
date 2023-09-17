from hyperparameter import (
    RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH,
    PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH,
    RESULT_FILE_NAME,
    TEST_SIZE, BATCH_SIZE, MODEL_NAME, INPUT_DIM,
    LEARNING_RATE, NUM_EPOCHS
)
from Dataset.data_loader import load_data, get_data_loader, AudioMixtureDataset
from Dataset.preprocess import preprocess
from Dataset.split_data import split_data
from Model.model_loader import load_model
from Training.train_methods import train_model
from Evaluation.evaluation_methods import evaluate_model
from Result.result_saver import save_results
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, random_split

    
if __name__ == "__main__" :
    preprocess(RAW_DATA_PATH, TEST_SIZE)
    
    train_dataset = torch.load("Preprocessed_dataset/train_dataset.pt") 
    test_dataset = torch.load("Preprocessed_dataset/test_dataset.pt")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    for model in ['conv_tasnet', 'biocppnet'] :
        model = load_model(model, input_dim=INPUT_DIM)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # Train Model
        train_model(train_loader, model, optimizer, criterion, num_epochs=NUM_EPOCHS)

        # Evaluate Model
        test_dataset = load_data(PROCESSED_TEST_DATA_PATH)
        test_loader = get_data_loader(test_dataset, batch_size=BATCH_SIZE)
        evaluation_metrics = evaluate_model(test_loader, model)

        # Save Results
        save_results(evaluation_metrics, RESULT_FILE_NAME)
