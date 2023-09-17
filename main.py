from hyperparameter import (
    RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH,
    PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH,
    RESULT_FILE_NAME,
    TEST_SIZE, BATCH_SIZE, MODEL_NAME, INPUT_DIM,
    LEARNING_RATE, NUM_EPOCHS
)
from Dataset.data_loader import load_data, get_data_loader
from Dataset.preprocess import preprocess
from Dataset.split_data import split_data
from Model.model_loader import load_model
from Training.train_methods import train_model
from Evaluation.evaluation_methods import evaluate_model
from Result.result_saver import save_results
import torch.optim as optim
import torch.nn as nn

# Split Data into Train and Test
split_data(RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, test_size=TEST_SIZE)

# Preprocess Data
preprocess(TRAIN_DATA_PATH, PROCESSED_TRAIN_DATA_PATH)
preprocess(TEST_DATA_PATH, PROCESSED_TEST_DATA_PATH)

# Load Data
train_dataset = load_data(PROCESSED_TRAIN_DATA_PATH)
train_loader = get_data_loader(train_dataset, batch_size=BATCH_SIZE)

# Load Model
model = load_model(MODEL_NAME, input_dim=INPUT_DIM)
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
