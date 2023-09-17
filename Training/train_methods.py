import torch
import os
def train_model(train_loader, model, optimizer, criterion, num_epochs=10):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    print("Training complete.")