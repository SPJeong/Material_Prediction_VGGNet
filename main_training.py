##### VGGNet_main_training.py
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # for AMP, float 16 or 32 calculation

import CONFIG  # custom.py
import chemical_feature_extraction  # custom.py
import data_extraction  # custom.py
import model  # custom.py
import my_utils  # custom.py
from model_trainer import train, validate, test  # custom.py

# parameter setting
filtered_num = CONFIG.filtered_num
random_pick_num = CONFIG.random_pick_num
ecfp_radius = CONFIG.ECFP_radius
ecfp_nbits = CONFIG.ECFP_nBits
data_extraction_folder = CONFIG.data_extraction_folder
chemical_feature_extraction_folder = CONFIG.chemical_feature_extraction_folder
plot_save_folder = CONFIG.plot_save_folder
model_save_folder = CONFIG.model_save_folder
os.makedirs(chemical_feature_extraction_folder, exist_ok=True)
os.makedirs(plot_save_folder, exist_ok=True)
os.makedirs(model_save_folder, exist_ok=True)
model_name = 'VGGNet'
batch_size = CONFIG.batch_size
learning_rate = CONFIG.learning_rate
epochs = CONFIG.epochs
device = CONFIG.device
ROnPlateauLR_factor = CONFIG.ROnPlateauLR_factor
ROnPlateauLR_patience = CONFIG.ROnPlateauLR_patience

Y_total_list = ['Cp', 'Tg', 'Tm', 'Td', 'LOI',
                'YM', 'TSy', 'TSb', 'epsb', 'CED',
                'Egc', 'Egb', 'Eib', 'Ei', 'Eea', 'nc', 'ne',
                'permH2', 'permHe', 'permCH4', 'permCO2', 'permN2', 'permO2',
                'Eat', 'rho', 'Xc', 'Xe']

# load file
file_folder = chemical_feature_extraction_folder
file_name = f'chemical_feature_extraction_len_{filtered_num}_num_{random_pick_num}_scaled_True_ECFP_False_desc_True.csv'
file_raw_path = os.path.join(file_folder, file_name)

if os.path.exists(file_raw_path):
    print(f"Loading existing file from: {file_raw_path}")
    file_raw = pd.read_csv(file_raw_path)

else:
    print(f"File not found. Generating data and saving to: {file_raw_path}")
    file_raw = chemical_feature_extraction.run_feature_extraction(filtered_num=filtered_num,
                                                                  random_pick_num=random_pick_num,
                                                                  data_extraction_folder=data_extraction_folder,
                                                                  ecfp=False,
                                                                  descriptors=True,
                                                                  scale_descriptors=True,
                                                                  ecfp_radius=None,
                                                                  ecfp_nbits=None,
                                                                  chemical_feature_extraction_folder=chemical_feature_extraction_folder,
                                                                  inference_mode=False,
                                                                  new_smiles_list=None)

# descriptors for X
start_column_index = file_raw.columns.get_loc('BalabanJ')
end_column_index = file_raw.columns.get_loc('CalcNumBridgeheadAtoms')
descriptor_df_columns_list = file_raw.columns[start_column_index:end_column_index + 1].tolist()
descriptor_df = file_raw.iloc[:, start_column_index:end_column_index + 1].copy()

# ECFP -> 2D for X
X_file_smile_2d_fps = chemical_feature_extraction.generate_ecfp_2d(df=file_raw, radius=ecfp_radius, nbits=ecfp_nbits)

# total targets for Y
start_column_index = file_raw.columns.get_loc('Egc')
end_column_index = file_raw.columns.get_loc('Tm')
Y_total_file = file_raw.iloc[:, start_column_index:end_column_index + 1]

for i, target_name in tqdm(enumerate(Y_total_list), total=len(Y_total_list)):

    y_data = Y_total_file[str(target_name)]
    concat_df_raw = pd.concat([X_file_smile_2d_fps, descriptor_df, y_data], axis=1)

    # load dataloaders
    my_dataloaders = my_utils.prepare_cnn_data_loaders(combined_df=concat_df_raw,
                                                       descriptor_columns=descriptor_df_columns_list,
                                                       target_name=target_name,
                                                       batch_size=batch_size,
                                                       test_size=0.2,
                                                       random_state=777)

    # # my_dataloaders check
    # for data in my_dataloaders["train"]:
    #     x, y, desc = data
    #     print("print(x.shape, y.shape, desc.shape)")
    #     print(x.shape, y.shape, desc.shape)
    #     print("_" *40)
    #     print("print(x[0].shape, y[0].shape, desc[0].shape)")
    #     print(x[0].shape, y[0].shape, desc[0].shape)
    #     print("_" *40)
    #     print("print(type(x[0]), type(y[0]), type(desc[0]))")
    #     print(type(x[0]), type(y[0]), type(desc[0]))
    #     print("_" *40)
    #     print("print(x[0][0][0])")
    #     print(x[0][0][0])
    #     print("_" *40)
    #     print("print(x[0][0][0].shape)")
    #     print(x[0][0][0].shape)
    #     print("_" *40)
    #     print("print(y[0])")
    #     print(y[0])
    #     print("_" *40)
    #     print("print(desc[0][:10])")
    #     print(desc[0][:10])
    #     print("_" *40)
    #     print("_" *40)
    #     print("_" *40)

    # model, loss fn and optimizer define
    my_model = model.VGG11(descriptor_size=descriptor_df.shape[1])
    my_model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=my_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = GradScaler()  # GradScaler for float16 calculation

    # ROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=ROnPlateauLR_factor,
                                                           patience=ROnPlateauLR_patience)

    # Main training and validation loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # initiate val loss as infinite
    best_model_state = None  # for best model save during training

    for epoch in range(0, epochs):
        # load train, val function
        train_loss = train(my_model, optimizer, loss_fn, my_dataloaders, device,
                           scaler)  # my_dataloaders = my_dataloaders['train']
        val_loss = validate(my_model, loss_fn, my_dataloaders, device)  # my_dataloaders = my_dataloaders['val']

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check if current validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = my_model.state_dict()
            print(f"New best model found at epoch {epoch} with Val Loss: {best_val_loss:.4f}. Model state saved.")

        if epoch % 1 == 0:
            print(
                f"Target: {target_name} | Epoch {epoch} | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")

    # Final evaluation on the test set after training is complete
    final_metrics = test(my_model, my_dataloaders, device, plot_save_folder, model_name,
                         target_name)  # my_dataloaders = cnn_dataloaders['test']
    print(f'\nFinal Metrics on Test Set:')
    for metric_name, value in final_metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")

    # Dictionary to save all parameters and metrics
    results = {'model_type': model_name,
               'target_variable': target_name,
               'model_state_dict': best_model_state,
               'optimizer_state_dict': optimizer.state_dict(),
               'train_losses': train_losses,
               'val_losses': val_losses,
               'final_test_metrics': final_metrics,
               'epochs': epoch, }

    # Save the entire package (best model + metadata)
    model_file_name = f'{model_name}_model_len_{filtered_num}_num_{random_pick_num}_{target_name}.pth'
    model_save_file_name = os.path.join(model_save_folder, model_file_name)
    torch.save(results, model_save_file_name)
    print(f'Best model and training results saved to {model_save_file_name}')