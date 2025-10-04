##### model_trainer.py (VGGNet)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.cuda.amp import autocast


# Training function with AMP
def train(my_model, optimizer, loss_fn, dataloaders, device, scaler):  # dataloaders = dataloaders['train']
    my_model.train()
    train_loss = 0.0
    for data, targets, descriptors in dataloaders['train']:
        data, targets, descriptors = data.to(device), targets.to(device), descriptors.to(device)  # to cuda
        optimizer.zero_grad()

        # autocast context for forward calculation
        with autocast():
            outputs = my_model(data, descriptors)
            outputs = outputs[:, 0]
            loss = loss_fn(outputs, targets)

            # scaler for backward and param update
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * len(data)

    avg_train_loss = train_loss / len(dataloaders['train'].dataset)
    return avg_train_loss


# Validation function with AMP (autocast only)
def validate(my_model, loss_fn, dataloaders, device):  # dataloaders = dataloaders['val']
    my_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, targets, descriptors in dataloaders['val']:
            data, targets, descriptors = data.to(device), targets.to(device), descriptors.to(device)  # to cuda
            with autocast():
                outputs = my_model(data, descriptors)
                outputs = outputs[:, 0]
                loss = loss_fn(outputs, targets)

            val_loss += loss.item() * len(data)

    avg_val_loss = val_loss / len(dataloaders['val'].dataset)
    return avg_val_loss


# Test function using various metrics with AMP (autocast only)
def test(my_model, dataloaders, device, plot_save_folder, model_name, target_name):  # dataloaders = dataloaders['test']
    my_model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data, targets, descriptors in dataloaders['test']:
            data, targets, descriptors = data.to(device), targets.to(device), descriptors.to(device)  # to cuda

            with autocast():
                outputs = my_model(data, descriptors)
                outputs = outputs[:, 0]

            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all outputs and targets into a single tensor, then convert to numpy arrays
    all_outputs = torch.cat(all_outputs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Compute metrics using scikit-learn
    mse = mean_squared_error(all_targets, all_outputs)
    mae = mean_absolute_error(all_targets, all_outputs)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_outputs)

    # Add Plotting and Saving
    # Create the scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_outputs, alpha=0.5, color='blue', label='Predictions')

    # Add a line of perfect prediction (y=x) for reference
    min_val = min(min(all_targets), min(all_outputs))
    max_val = max(max(all_targets), max(all_outputs))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Perfect Prediction')

    # Add labels, title, and a legend
    plt.xlabel(f'Actual {target_name}', fontsize=12)
    plt.ylabel(f'Predicted {target_name}', fontsize=12)
    plt.title(f'{model_name}: Actual vs. Predicted ({target_name}) \n(MAE: {mae:.4f}, R2: {r2:.4f})', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plot_save_path = os.path.join(plot_save_folder, f'{model_name}_MAE_plot_{target_name}.png')
    plt.savefig(plot_save_path, bbox_inches='tight')
    plt.close()
    print(f'{model_name} MAE plot saved to {plot_save_path}')

    return {'test_mse': mse, 'test_mae': mae, 'test_rmse': rmse, 'test_r2': r2}
