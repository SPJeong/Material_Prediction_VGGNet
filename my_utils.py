##### my_utils.py (VGGNet)

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List
import torch.nn.functional as F


class my_CNN_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, descriptor_df: pd.DataFrame, target_name: str):
        self.x = list(df['2d_fps'])
        self.y = list(df[str(target_name)].values)
        self.descriptors = list(descriptor_df.to_numpy()) # descriptor -> numpy -> list

        self.image_size = 64
        self.new_size = 224
        self.channels = 3

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_img = torch.tensor(self.x[index], dtype=torch.float32)  # self.x[index]: (1, 64, 64)

        # calculate padding size (center position is original_img 64x64)
        pad_size_x = (self.new_size - self.image_size) // 2
        pad_size_y = (self.new_size - self.image_size) // 2
        padded_img = F.pad(original_img, (pad_size_x, pad_size_x, pad_size_y, pad_size_y)) # F.pad order (left, right, top, bottom)
        final_img = padded_img.repeat(self.channels, 1, 1)  # (1, 224, 224) -> (3, 224, 224)

        y_tensor = torch.tensor(self.y[index], dtype=torch.float32)
        desc_tensor = torch.tensor(self.descriptors[index], dtype=torch.float32)

        return final_img, y_tensor, desc_tensor # return x_tensor, y_tensor, desc_tensor


def prepare_cnn_data_loaders(combined_df: pd.DataFrame,
                         descriptor_columns: List[str],
                         target_name: str,
                         batch_size: int,
                         test_size: float = 0.2,
                         random_state: int = 777) -> Dict[str, DataLoader]:
    """
    Args:
        combined_df: DataFrame including 'smiles', '2d_fps', descriptor columns, target_name columns.
        descriptor_columns: list for descriptor df.columns
        target_name: single target name (e.g. Tg)
        batch_size: DataLoader batch size
        test_size: test size ratio (e.g., 0.2 -> 20%).
        random_state: for reproducibility.

    Returns:
        DataLoader Dictionary containing 'train', 'val', 'test' key
    """

    train, val_test_temp = train_test_split(combined_df, test_size= test_size, random_state= random_state)
    val, test = train_test_split(val_test_temp, test_size= 0.5, random_state= random_state)

    dataset_dfs = {'train': pd.concat([train[['smiles', '2d_fps']], train[[str(target_name)]]], axis=1),
                   'val': pd.concat([val[['smiles', '2d_fps']], val[[str(target_name)]]], axis=1),
                   'test': pd.concat([test[['smiles', '2d_fps']], test[[str(target_name)]]], axis=1)}

    descriptor_datasets = {'train': train[descriptor_columns],
                           'val': val[descriptor_columns],
                           'test': test[descriptor_columns]}

    # my_CNN_Dataset
    cnn_data = {'train': my_CNN_Dataset(dataset_dfs['train'], descriptor_datasets['train'], target_name),
                'val': my_CNN_Dataset(dataset_dfs['val'], descriptor_datasets['val'], target_name),
                'test': my_CNN_Dataset(dataset_dfs['test'], descriptor_datasets['test'], target_name)}

    # DataLoader
    cnn_dataloaders = {'train': DataLoader(cnn_data['train'], batch_size=batch_size, shuffle=True),
                       'val': DataLoader(cnn_data['val'], batch_size=batch_size, shuffle=False),
                       'test': DataLoader(cnn_data['test'], batch_size=batch_size, shuffle=False)}

    return cnn_dataloaders
