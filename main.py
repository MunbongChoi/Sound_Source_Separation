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
from Model.conv_tasnet.conv_tasnet import ConvTasNet
from Model.conv_tasnet.solver import Solver


if __name__ == "__main__" :
    preprocess(RAW_DATA_PATH, TEST_SIZE)
    
    train_dataset = torch.load("Preprocessed_dataset/train_dataset.pt") 
    test_dataset = torch.load("Preprocessed_dataset/test_dataset.pt")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    data = {'tr_loader': train_loader, 'cv_loader': test_loader}
    
    # Load Model
    for model in ['conv_tasnet', 'biocppnet'] :
        if model == 'conv_tasnet':
            N = 256	
            L = 20	
            B = 256	
            H = 512	
            P = 3	
            X = 8	
            R = 4
            C = 2
            use_cuda = True
            optimizer = 'sgd'
            norm_type = 'gLN'
            lr = 0.0001
            momentum = 0.0
            l2 = 0.0
            model = ConvTasNet(N, L, B, H, P, X, R,
                            C, norm_type=norm_type, causal=False,
                            mask_nonlinear='relu')
            print(model)
            if use_cuda:
                model = torch.nn.DataParallel(model)
                model.cuda()
            # optimizer
            if optimizer == 'sgd':
                optimizier = torch.optim.SGD(model.parameters(),
                                            lr=LEARNING_RATE,
                                            momentum=momentum,
                                            weight_decay=l2)
            elif optimizer == 'adam':
                optimizier = torch.optim.Adam(model.parameters(),
                                            lr=LEARNING_RATE,
                                            weight_decay=l2)
            # solver
            solver = Solver(data, model, optimizier)
            solver.train()
        else :
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
