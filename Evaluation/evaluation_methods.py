from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import torch
import numpy as np

def si_sdr(reference, estimation):
    # Implement SI-SDR here
    pass

def evaluate_model(test_loader, model):
    predictions, labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.tolist())
            labels.extend(target.tolist())

    si_sdr_value = si_sdr(labels, predictions)
    mae_value = mean_absolute_error(labels, predictions)
    mse_value = mean_squared_error(labels, predictions)
    acc_value = accuracy_score(labels, predictions)

    return {
        "SI-SDR": si_sdr_value,
        "MAE": mae_value,
        "MSE": mse_value,
        "Accuracy": acc_value
    }