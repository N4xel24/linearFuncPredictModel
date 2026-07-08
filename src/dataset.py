import torch
from torch.utils.data import Dataset
import pandas as pd

class PointDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=None)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = torch.tensor([row[0]], dtype=torch.float32)
        target = torch.tensor([row[1]], dtype=torch.float32)

        return x, target